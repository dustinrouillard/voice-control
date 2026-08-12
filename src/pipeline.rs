use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::audio::ring::PreRoll;
use crate::audio::vad::{Endpoint, Endpointer, Timing, WINDOW};
use crate::audio::{ms_to_samples, samples_to_ms};
use crate::commands::{Command, match_command};
use crate::dispatch::{Dispatcher, try_run};
use crate::feedback::{Cue, Feedback};
use crate::obs::ObsConfig;
use crate::status::{Outcome, Status};
use crate::stt::Transcriber;
use crate::wake::Detector;

/// How much audio before the detection to keep.
///
/// Two effects stack here. "computa, mute" is one breath, so the
/// command is already partly spoken when the wake word starts
/// scoring - and rustpotter then needs several more frames before it
/// will commit to a detection. Between them the whole utterance can
/// be in the past by the time we are told about it, so the window
/// reaches back far enough to recover the wake word itself. That the
/// wake word lands in the clip is fine: the matcher works over
/// suffixes and drops it.
const PRE_ROLL_MS: usize = 900;
/// Ring capacity. Comfortably more than PRE_ROLL_MS.
const RING_MS: usize = 3000;
/// Ignore wake detections for a moment after handling one, so a single
/// utterance cannot fire twice.
const REFRACTORY: Duration = Duration::from_millis(1000);

enum State {
  Idle,
  Listening,
}

pub struct Pipeline {
  detector: Box<dyn Detector>,
  endpointer: Endpointer,
  transcriber: Arc<Mutex<Box<dyn Transcriber>>>,
  commands: Vec<Command>,
  dispatcher: Dispatcher,
  feedback: Feedback,
  status: Arc<Status>,

  pre_roll: PreRoll,
  pending: Vec<f32>,
  captured: Vec<f32>,
  state: State,
  quiet_until: Option<Instant>,
  was_paused: bool,
}

impl Pipeline {
  pub fn new(
    detector: Box<dyn Detector>,
    transcriber: Box<dyn Transcriber>,
    commands: Vec<Command>,
    feedback: Feedback,
    obs: ObsConfig,
    timing: &Timing,
    status: Arc<Status>,
  ) -> Result<Self> {
    Ok(Self {
      detector,
      endpointer: Endpointer::new(timing)?,
      transcriber: Arc::new(Mutex::new(transcriber)),
      commands,
      dispatcher: Dispatcher::new(obs, feedback.clone())?,
      feedback,
      status,
      pre_roll: PreRoll::new(ms_to_samples(RING_MS)),
      pending: Vec::new(),
      captured: Vec::new(),
      state: State::Idle,
      quiet_until: None,
      was_paused: false,
    })
  }

  pub async fn run(
    mut self,
    mut audio: mpsc::Receiver<Vec<f32>>,
  ) -> Result<()> {
    info!("listening for the wake word");
    self.status.idle();

    while let Some(chunk) = audio.recv().await {
      self.feed(&chunk).await;
    }

    warn!("audio stream ended");
    self.status.stopped();

    Ok(())
  }

  /// Drives the whole pipeline over one buffer of 16 kHz mono audio.
  /// Split out from [`Pipeline::run`] so `replay` can push a file
  /// through the same path without a microphone.
  pub async fn feed(&mut self, chunk: &[f32]) {
    // Reported for every buffer, before anything can return early:
    // the level meter and both "am I actually hearing anything"
    // watchdogs are driven off this.
    self.status.audio(peak(chunk));

    self.pending.extend_from_slice(chunk);

    match self.state {
      State::Idle => {
        // Keep the ring warm while paused - resuming mid-sentence
        // should not cost the pre-roll - but do not score.
        self.pre_roll.extend(chunk);

        if self.status.paused() {
          self.pending.clear();
          self.was_paused = true;
          return;
        }

        // Resume from a clean slate: partial scores banked before the
        // pause would otherwise combine with frames from after it.
        if std::mem::take(&mut self.was_paused) {
          self.detector.reset();
        }

        self.poll_wake_word();
      }
      // A capture already in flight is allowed to finish: pausing is
      // about not being woken, not about dropping the word you are
      // halfway through saying.
      State::Listening => self.poll_endpoint().await,
    }
  }

  /// Ends any in-flight capture, e.g. when a replayed file runs out
  /// mid-command.
  pub async fn flush(&mut self) {
    if matches!(self.state, State::Listening) {
      self.finish(true).await;
    }
  }

  fn poll_wake_word(&mut self) {
    let frame_size = self.detector.frame_size();

    if let Some(until) = self.quiet_until {
      if Instant::now() < until {
        // Keep the ring warm, but do not score.
        self.pending.clear();
        return;
      }

      self.quiet_until = None;
      self.detector.reset();
    }

    while self.pending.len() >= frame_size {
      let frame: Vec<f32> = self.pending.drain(..frame_size).collect();

      if let Some(detection) = self.detector.push(frame) {
        info!(
          wakeword = %detection.name,
          score = detection.score,
          "wake word detected"
        );

        self.feedback.play(Cue::Wake);
        self.status.wake();
        self.begin_listening();
        return;
      }
    }
  }

  fn begin_listening(&mut self) {
    // Everything still in `pending` is already in the ring, so the
    // tail covers it. Starting from a clean buffer keeps the VAD
    // windows aligned.
    let mut seed = self.pre_roll.tail(ms_to_samples(PRE_ROLL_MS));

    // Drop the oldest samples rather than the newest, so the seed ends
    // flush with "now" and every later window stays aligned.
    seed.drain(..seed.len() % WINDOW);

    self.pending.clear();
    self.endpointer.reset();
    self.captured.clear();

    // Run the pre-roll through the VAD as well, not just into the
    // buffer. Said briskly, "computa mute" is over before the detector
    // commits, so the command lives entirely in the pre-roll - and a
    // VAD that starts from scratch here sees nothing but silence and
    // hands whisper a clip that is mostly nothing.
    //
    // `prime` rather than `push`, so none of this counts as the
    // command having been started: the seed always ends with the wake
    // word, and a hangover measured from there would end the capture
    // before someone who paused after "computa" had said anything.
    for window in seed.chunks(WINDOW) {
      self.captured.extend_from_slice(window);
      self.endpointer.prime(window.to_vec());
    }

    self.state = State::Listening;
  }

  async fn poll_endpoint(&mut self) {
    while self.pending.len() >= WINDOW {
      let window: Vec<f32> = self.pending.drain(..WINDOW).collect();
      self.captured.extend_from_slice(&window);

      match self.endpointer.push(window) {
        Endpoint::Continue => {}
        Endpoint::Done => {
          self.finish(true).await;
          return;
        }
        Endpoint::NoSpeech => {
          // The command may have landed entirely in the pre-roll, so
          // still transcribe - but stay silent if it comes to
          // nothing, since this is also what a false trigger looks
          // like and a chirp every time would be maddening.
          debug!("no speech after the wake word, trying the pre-roll");
          self.finish(false).await;
          return;
        }
      }
    }
  }

  async fn finish(&mut self, noisy: bool) {
    let raw = std::mem::take(&mut self.captured);
    let audio = self.endpointer.crop(&raw);
    self.pending.clear();
    self.state = State::Idle;
    self.pre_roll.clear();
    self.quiet_until = Some(Instant::now() + REFRACTORY);

    debug!(
      captured_ms = samples_to_ms(raw.len()),
      speech_ms = samples_to_ms(audio.len()),
      "captured command audio"
    );

    self.status.thinking();

    let transcript = match self.transcribe(audio).await {
      Ok(transcript) => transcript,
      Err(why) => {
        warn!(error = ?why, "transcription failed");
        self.give_up(noisy);

        return;
      }
    };

    if transcript.is_empty() {
      debug!("empty transcript");
      self.give_up(noisy);

      return;
    }

    match match_command(&self.commands, &transcript) {
      Some(hit) => {
        info!(
          %transcript,
          command = %hit.command.name,
          score = hit.score,
          "matched"
        );

        let name = hit.command.name.clone();
        // A command that plays its own tone has already answered by
        // the time it succeeds, so the generic chirp is left off.
        let answered = hit.command.answers_for_itself();

        let (cue, outcome) =
          if try_run(&self.dispatcher, hit.command).await {
            ((!answered).then_some(Cue::Ok), Outcome::Dispatched(name))
          } else {
            (Some(Cue::Fail), Outcome::Failed(name))
          };

        if let Some(cue) = cue {
          self.feedback.play(cue);
        }

        self.status.finished(&transcript, outcome);
      }
      None => {
        // Logged at info on purpose: grepping these is how the
        // phrase lists in commands.toml get grown. They are in the
        // menu for the same reason.
        info!(%transcript, "no matching command");
        self.status.finished(&transcript, Outcome::NoMatch);

        if noisy {
          self.feedback.play(Cue::Fail);
        }
      }
    }
  }

  /// Woken but nothing usable came of it. A silent give-up is what a
  /// false trigger looks like, and those are not worth a line in the
  /// menu - there would be one every time the room said something
  /// vaguely like "computa".
  fn give_up(&self, noisy: bool) {
    if noisy {
      self.feedback.play(Cue::Fail);
      self.status.finished("", Outcome::Unheard);
    } else {
      self.status.idle();
    }
  }

  async fn transcribe(&self, audio: Vec<f32>) -> Result<String> {
    let transcriber = Arc::clone(&self.transcriber);

    tokio::task::spawn_blocking(move || {
      let mut transcriber =
        transcriber.lock().expect("transcriber mutex poisoned");
      transcriber.transcribe(&audio)
    })
    .await
    .context("transcription task panicked")?
  }
}

fn peak(chunk: &[f32]) -> f32 {
  chunk.iter().fold(0.0_f32, |peak, s| peak.max(s.abs()))
}
