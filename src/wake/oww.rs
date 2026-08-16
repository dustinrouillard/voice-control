//! [openWakeWord][oww] inference.
//!
//! Three ONNX graphs in series, scored once per 80 ms hop:
//!
//! ```text
//! 1280 samples ─► melspectrogram ─► 8 mel frames of 32 bins
//!                                      │  (ring of 80)
//!                        last 76 frames ▼
//!                                   embedding ─► 96 floats
//!                                      │  (ring of 16)
//!                                      ▼
//!                                    wake ─► one probability
//! ```
//!
//! The first two are openWakeWord's own pretrained feature
//! extractors and are the same for every wake word; only the third is
//! trained per word. That split is why this replaced rustpotter: the
//! embedding model is a speech representation trained on a very large
//! corpus, so the classifier sees "what was said" rather than "how
//! close is this waveform to my twelve recordings", and dialogue off a
//! pair of speakers stops looking like the wake word.
//!
//! The buffering follows openWakeWord's streaming path, which the
//! Rust port in [oww_rs][port] states more plainly than the original.
//!
//! [oww]: https://github.com/dscripka/openWakeWord
//! [port]: https://github.com/skoky/oww_rs

use anyhow::{Result, anyhow, bail};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use tracing::{info, warn};

use crate::config::expand_tilde;
use crate::wake::{Detection, Detector};

/// Samples per scored hop: 80 ms at 16 kHz. Not adjustable - the
/// models were trained on this hop.
pub const CHUNK: usize = 1280;

/// Samples between mel frames, i.e. one 10 ms mel hop.
const HOP: usize = 160;
/// Tail of the previous chunk prepended before the melspectrogram, so
/// the mel windows either side of a chunk boundary are the ones the
/// model would have seen over a continuous stream. Three hops is what
/// the graph needs to produce a whole number of frames per chunk.
const LOOKBACK: usize = HOP * 3;
const MEL_INPUT: usize = LOOKBACK + CHUNK;
const MEL_BINS: usize = 32;
/// Mel frames the graph emits per chunk: `MEL_INPUT / HOP - 3` = 8.
const MELS_PER_CHUNK: usize = MEL_INPUT / HOP - 3;
/// Mel frames the embedding model consumes: 760 ms of context.
const MEL_WINDOW: usize = 76;
/// Mel frames kept. A whole number of chunks and at least a window.
const MEL_HISTORY: usize = 80;
/// Floats per embedding.
const EMBEDDING: usize = 96;
/// Embeddings the wake model consumes. At one per chunk, its view
/// reaches back 16 * 80 ms plus the window each embedding covers.
const EMBEDDINGS: usize = 16;

const _: () = assert!(MEL_HISTORY.is_multiple_of(MELS_PER_CHUNK));
const _: () = assert!(MEL_HISTORY >= MEL_WINDOW);

pub struct OpenWakeWordDetector {
  mel: Session,
  embedding: Session,
  wake: Session,
  name: String,
  threshold: f32,
  patience: usize,

  /// Tail of the last chunk, for the next melspectrogram call.
  lookback: Vec<f32>,
  /// `MEL_HISTORY` frames of `MEL_BINS`, oldest first.
  mels: Vec<f32>,
  /// `EMBEDDINGS` embeddings of `EMBEDDING`, oldest first.
  embeddings: Vec<f32>,
  /// Consecutive hops over the threshold.
  hits: usize,
}

impl OpenWakeWordDetector {
  pub fn load(
    model: &str,
    melspectrogram: &str,
    embedding: &str,
    threshold: f32,
    patience: usize,
  ) -> Result<Self> {
    if patience == 0 {
      bail!("wake patience must be at least 1");
    }

    let name = expand_tilde(model)
      .file_stem()
      .map(|stem| stem.to_string_lossy().into_owned())
      .unwrap_or_else(|| "wake".to_string());

    let mel = session(melspectrogram, "melspectrogram")?;
    let embedding = session(embedding, "embedding")?;
    let wake = session(model, "wake word")?;

    info!(model = %expand_tilde(model).display(), threshold, patience, "loaded wake word model");

    Ok(Self {
      mel,
      embedding,
      wake,
      name,
      threshold,
      patience,
      lookback: vec![0.0; LOOKBACK],
      mels: vec![0.0; MEL_HISTORY * MEL_BINS],
      embeddings: vec![0.0; EMBEDDINGS * EMBEDDING],
      hits: 0,
    })
  }

  /// What the model is called, taken from its filename. Reported with
  /// every detection, and what the overlay labels itself with.
  pub fn name(&self) -> &str {
    &self.name
  }

  /// Scores one hop and reports both the probability and whether it
  /// completed a detection.
  ///
  /// [`Detector::push`] is this with the probability dropped. `score`
  /// and `record` want the number for every hop, not just the ones
  /// that fire, and taking both from here keeps what those print in
  /// step with what the daemon actually does.
  pub fn advance(&mut self, frame: &[f32]) -> Result<(f32, bool)> {
    let score = self.score(frame)?;

    // Consecutive hops rather than one, because the cheapest false
    // positives to lose are the single-frame ones: a syllable off a
    // speaker that momentarily lines up scores one hop and is gone by
    // the next, where someone actually saying the word holds the score
    // up across several. The cost is `patience` hops of latency, which
    // the pre-roll covers.
    if score <= self.threshold {
      self.hits = 0;
      return Ok((score, false));
    }

    self.hits += 1;

    if self.hits < self.patience {
      return Ok((score, false));
    }

    self.hits = 0;

    Ok((score, true))
  }

  /// The wake model's probability for the hop ending at `frame`.
  ///
  /// Every stage is stateful, so this has to be called for each hop in
  /// order even when the caller only cares about some of them.
  fn score(&mut self, frame: &[f32]) -> Result<f32> {
    if frame.len() != CHUNK {
      bail!("expected {CHUNK} samples, got {}", frame.len());
    }

    let mut input = Vec::with_capacity(MEL_INPUT);
    input.extend_from_slice(&self.lookback);
    input.extend_from_slice(frame);
    self.lookback.copy_from_slice(&frame[CHUNK - LOOKBACK..]);

    let mels: Vec<f32> = {
      let tensor =
        Tensor::from_array((vec![1_i64, MEL_INPUT as i64], input))?;
      let outputs = self.mel.run(ort::inputs![tensor])?;
      let (_, values) = outputs[0].try_extract_tensor::<f32>()?;

      if values.len() != MELS_PER_CHUNK * MEL_BINS {
        bail!(
          "melspectrogram returned {} values, expected {}",
          values.len(),
          MELS_PER_CHUNK * MEL_BINS
        );
      }

      // The graph emits raw log mels; the embedding model was trained
      // on this rescaling of them, which openWakeWord applies between
      // the two.
      values.iter().map(|value| value / 10.0 + 2.0).collect()
    };

    self.mels.drain(..mels.len());
    self.mels.extend_from_slice(&mels);

    let window =
      self.mels[self.mels.len() - MEL_WINDOW * MEL_BINS..].to_vec();

    let embedding: Vec<f32> = {
      let tensor = Tensor::from_array((
        vec![1_i64, MEL_WINDOW as i64, MEL_BINS as i64, 1_i64],
        window,
      ))?;
      let outputs = self.embedding.run(ort::inputs![tensor])?;
      let (_, values) = outputs[0].try_extract_tensor::<f32>()?;

      if values.len() != EMBEDDING {
        bail!(
          "embedding model returned {} values, expected {EMBEDDING}",
          values.len()
        );
      }

      values.to_vec()
    };

    self.embeddings.drain(..EMBEDDING);
    self.embeddings.extend_from_slice(&embedding);

    let tensor = Tensor::from_array((
      vec![1_i64, EMBEDDINGS as i64, EMBEDDING as i64],
      self.embeddings.clone(),
    ))?;
    let outputs = self.wake.run(ort::inputs![tensor])?;
    let (_, values) = outputs[0].try_extract_tensor::<f32>()?;

    values
      .first()
      .copied()
      .ok_or_else(|| anyhow!("wake model returned nothing"))
  }
}

impl Detector for OpenWakeWordDetector {
  fn frame_size(&self) -> usize {
    CHUNK
  }

  fn push(&mut self, frame: Vec<f32>) -> Option<Detection> {
    match self.advance(&frame) {
      Ok((score, true)) => Some(Detection {
        name: self.name.clone(),
        score,
      }),
      Ok((_, false)) => None,
      Err(why) => {
        warn!(error = ?why, "wake word inference failed");
        None
      }
    }
  }

  fn reset(&mut self) {
    // Everything here is a rolling view of the last two seconds of
    // audio, and reset is called precisely when that audio stops being
    // contiguous - after a command, or on resuming from a pause. Left
    // alone the buffers would splice two unrelated moments together
    // and score the seam. Clearing them costs a hop-and-a-bit of
    // deafness while they refill, which is inside the refractory
    // period the pipeline is already holding.
    self.lookback.fill(0.0);
    self.mels.fill(0.0);
    self.embeddings.fill(0.0);
    self.hits = 0;
  }
}

fn session(path: &str, what: &str) -> Result<Session> {
  let path = expand_tilde(path);

  if !path.exists() {
    return Err(anyhow!(
      "{what} model {} not found - see the README",
      path.display()
    ));
  }

  Session::builder()
    .and_then(|builder| {
      builder
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        // These graphs are small and run every 80 ms; handing each one
        // to a thread pool costs more in scheduling than it saves.
        .with_intra_threads(1)?
        .with_inter_threads(1)
    })
    .and_then(|builder| builder.commit_from_file(&path))
    .map_err(|why| anyhow!("loading {}: {why}", path.display()))
}
