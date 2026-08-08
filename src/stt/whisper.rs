use anyhow::{Context, Result, anyhow};
use tracing::{debug, info};
use whisper_rs::{
  FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters,
};

use crate::audio::{SAMPLE_RATE, ms_to_samples};
use crate::config::expand_tilde;
use crate::stt::Transcriber;

/// Whisper degrades badly on very short clips, so pad up to this.
const MIN_AUDIO_MS: usize = 1000;

pub struct WhisperTranscriber {
  context: WhisperContext,
  prompt: String,
  threads: i32,
}

impl WhisperTranscriber {
  /// `vocabulary` biases decoding towards the configured command
  /// phrases, which matters a lot for one-word utterances.
  pub fn load(model_path: &str, vocabulary: &[String]) -> Result<Self> {
    let path = expand_tilde(model_path);

    if !path.exists() {
      return Err(anyhow!(
        "whisper model {} not found - download a ggml model \
         (see README)",
        path.display()
      ));
    }

    let context = WhisperContext::new_with_params(
      path.to_str().context("whisper model path is not UTF-8")?,
      WhisperContextParameters::default(),
    )
    .map_err(|why| anyhow!("loading {}: {why}", path.display()))?;

    let prompt = build_prompt(vocabulary);
    debug!(prompt = %prompt, "whisper decoding bias");

    // Leave headroom; these clips are ~2s and latency is dominated by
    // model load, not thread count.
    let threads = std::thread::available_parallelism()
      .map_or(4, |n| n.get() as i32)
      .min(8);

    info!(model = %path.display(), threads, "loaded whisper model");

    Ok(Self {
      context,
      prompt,
      threads,
    })
  }
}

impl Transcriber for WhisperTranscriber {
  fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
    let mut state = self
      .context
      .create_state()
      .map_err(|why| anyhow!("creating whisper state: {why}"))?;

    let mut padded;
    let audio = if audio.len() < ms_to_samples(MIN_AUDIO_MS) {
      padded = audio.to_vec();
      padded.resize(ms_to_samples(MIN_AUDIO_MS), 0.0);
      &padded[..]
    } else {
      audio
    };

    let mut params =
      FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(self.threads);
    params.set_language(Some("en"));
    params.set_translate(false);
    // Each command stands alone; carrying context between them only
    // invents continuations.
    params.set_no_context(true);
    params.set_single_segment(true);
    params.set_suppress_blank(true);
    params.set_initial_prompt(&self.prompt);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    let started = std::time::Instant::now();

    state
      .full(params, audio)
      .map_err(|why| anyhow!("whisper inference failed: {why}"))?;

    let mut text = String::new();

    for i in 0..state.full_n_segments() {
      let Some(segment) = state.get_segment(i) else {
        break;
      };
      text.push_str(&segment.to_string());
    }

    debug!(
      ms = started.elapsed().as_millis(),
      samples = audio.len(),
      audio_ms = audio.len() * 1000 / SAMPLE_RATE as usize,
      "transcribed"
    );

    Ok(text.trim().to_string())
  }
}

fn build_prompt(vocabulary: &[String]) -> String {
  let mut prompt =
    String::from("Short voice commands addressed to computa. ");
  prompt.push_str(&vocabulary.join(", "));
  prompt.push('.');
  prompt
}
