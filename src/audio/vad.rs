use anyhow::{Context, Result};
use serde::Deserialize;
use voice_activity_detector::VoiceActivityDetector;

use crate::audio::SAMPLE_RATE;

/// Silero V5 only accepts a 512-sample window at 16 kHz.
pub const WINDOW: usize = 512;

const WINDOW_MS: usize = WINDOW * 1000 / SAMPLE_RATE as usize;

const fn windows_for(ms: usize) -> usize {
  ms / WINDOW_MS
}

const SPEECH_PROBABILITY: f32 = 0.5;

/// How long a capture runs for, in milliseconds.
#[derive(Debug, Clone, Deserialize)]
pub struct Timing {
  /// Trailing silence that ends a command, once one has been heard.
  #[serde(default = "default_silence_ms")]
  pub silence_ms: usize,
  /// How long after the wake word the command may still be starting.
  ///
  /// Nothing ends the capture before this except the ceiling, because
  /// until something is said live there is nothing to have finished
  /// saying. "computa" and then a beat to think is normal, and
  /// `silence_ms` alone would have closed the window before the first
  /// syllable - the wake word is in the pre-roll, so a capture that
  /// began in silence looks exactly like one that just ended.
  #[serde(default = "default_grace_ms")]
  pub grace_ms: usize,
  /// Absolute ceiling on one capture, pre-roll included.
  #[serde(default = "default_max_ms")]
  pub max_ms: usize,
}

fn default_silence_ms() -> usize {
  700
}

fn default_grace_ms() -> usize {
  2500
}

fn default_max_ms() -> usize {
  8000
}

impl Default for Timing {
  fn default() -> Self {
    Self {
      silence_ms: default_silence_ms(),
      grace_ms: default_grace_ms(),
      max_ms: default_max_ms(),
    }
  }
}

pub enum Endpoint {
  /// Still listening.
  Continue,
  /// Speech ended, or the hard cap was reached.
  Done,
  /// The wake word fired but nothing was said after it.
  NoSpeech,
}

/// Kept either side of the speech when cropping.
const CROP_PAD: usize = windows_for(200);

/// Decides when the user has finished speaking a command.
pub struct Endpointer {
  vad: VoiceActivityDetector,
  hangover: usize,
  grace: usize,
  max: usize,
  /// Speech since the pre-roll ended, which is the only kind that says
  /// the command has actually started.
  live_speech: bool,
  silence_run: usize,
  windows: usize,
  /// Windows pushed since the pre-roll ended.
  live_windows: usize,
  /// Which windows held speech, in the order they were pushed.
  mask: Vec<bool>,
}

impl Endpointer {
  pub fn new(timing: &Timing) -> Result<Self> {
    let vad = VoiceActivityDetector::builder()
      .sample_rate(SAMPLE_RATE as i64)
      .chunk_size(WINDOW)
      .build()
      .context("building the Silero VAD")?;

    Ok(Self {
      vad,
      hangover: windows_for(timing.silence_ms),
      grace: windows_for(timing.grace_ms),
      max: windows_for(timing.max_ms),
      live_speech: false,
      silence_run: 0,
      windows: 0,
      live_windows: 0,
      mask: Vec::new(),
    })
  }

  pub fn reset(&mut self) {
    self.vad.reset();
    self.live_speech = false;
    self.silence_run = 0;
    self.windows = 0;
    self.live_windows = 0;
    self.mask.clear();
  }

  /// Feed a window of pre-roll: it counts towards the clip and primes
  /// the VAD, but not towards the command having been spoken.
  ///
  /// The distinction is the whole of how the grace period works. The
  /// pre-roll always ends with the wake word, so a capture seeded with
  /// it starts out having "heard speech" whatever the user does next -
  /// and a hangover measured from there ends the capture while they
  /// are still drawing breath.
  pub fn prime(&mut self, window: Vec<f32>) {
    self.consume(window, false);
  }

  /// Feed exactly [`WINDOW`] samples of live audio.
  pub fn push(&mut self, window: Vec<f32>) -> Endpoint {
    self.consume(window, true)
  }

  fn consume(&mut self, window: Vec<f32>, live: bool) -> Endpoint {
    debug_assert_eq!(window.len(), WINDOW);

    let speech = self.vad.predict(window) >= SPEECH_PROBABILITY;

    self.decide(speech, live)
  }

  /// The decision, once the VAD has been reduced to a yes or a no -
  /// which is the half a test can drive.
  fn decide(&mut self, speech: bool, live: bool) -> Endpoint {
    self.windows += 1;
    self.mask.push(speech);

    if live {
      self.live_windows += 1;
    }

    if speech {
      self.silence_run = 0;
      self.live_speech |= live;
    } else {
      self.silence_run += 1;
    }

    if self.windows >= self.max {
      return Endpoint::Done;
    }

    if !self.live_speech {
      return if self.live_windows >= self.grace {
        Endpoint::NoSpeech
      } else {
        Endpoint::Continue
      };
    }

    if self.silence_run >= self.hangover {
      Endpoint::Done
    } else {
      Endpoint::Continue
    }
  }

  /// Crops a captured clip down to the speech it contains, plus a
  /// little padding.
  ///
  /// The clip is deliberately over-captured at the front (pre-roll has
  /// to reach back past the detector's latency) and at the back (the
  /// hangover has to elapse before we know the command ended). Handing
  /// whisper the raw buffer means most of what it decodes is silence,
  /// and it answers `[BLANK_AUDIO]` — so give it just the speech.
  ///
  /// `audio` must be the same windows that were pushed, in order.
  pub fn crop(&self, audio: &[f32]) -> Vec<f32> {
    let Some(first) = self.mask.iter().position(|&s| s) else {
      return audio.to_vec();
    };
    let last = self
      .mask
      .iter()
      .rposition(|&s| s)
      .unwrap_or(self.mask.len() - 1);

    let start = first.saturating_sub(CROP_PAD) * WINDOW;
    let end = ((last + 1 + CROP_PAD) * WINDOW).min(audio.len());

    if start >= end {
      return audio.to_vec();
    }

    audio[start..end].to_vec()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn endpointer_with(mask: Vec<bool>) -> Endpointer {
    let mut endpointer = Endpointer::new(&Timing::default()).unwrap();
    endpointer.mask = mask;
    endpointer
  }

  fn endpointer() -> Endpointer {
    Endpointer::new(&Timing::default()).unwrap()
  }

  /// The pre-roll always ends with the wake word, so every capture
  /// starts out having heard speech. Measuring the hangover from there
  /// is what used to end the capture while the user was still drawing
  /// breath.
  #[test]
  fn silence_after_the_wake_word_does_not_end_the_capture() {
    let mut endpointer = endpointer();

    // The pre-roll: the wake word, then the gap before the detector
    // committed.
    for _ in 0..10 {
      endpointer.decide(true, false);
    }

    // A beat longer than the hangover, and nothing said yet.
    for _ in 0..endpointer.hangover + 2 {
      assert!(matches!(
        endpointer.decide(false, true),
        Endpoint::Continue
      ));
    }
  }

  /// The command lands inside the grace period, and from then on the
  /// hangover is what ends the capture - not the grace, which would
  /// cut off anyone who paused mid-sentence.
  #[test]
  fn a_late_command_still_ends_on_its_own_silence() {
    let mut endpointer = endpointer();

    endpointer.decide(true, false);

    for _ in 0..endpointer.grace - 1 {
      endpointer.decide(false, true);
    }

    endpointer.decide(true, true);

    // Well past the grace period now, and still listening.
    for _ in 0..endpointer.hangover - 1 {
      assert!(matches!(
        endpointer.decide(false, true),
        Endpoint::Continue
      ));
    }

    assert!(matches!(endpointer.decide(false, true), Endpoint::Done));
  }

  /// A wake word with nothing after it is what a false trigger looks
  /// like, so it ends quietly rather than with the failure tone.
  #[test]
  fn a_wake_word_with_nothing_after_it_gives_up_quietly() {
    let mut endpointer = endpointer();

    endpointer.decide(true, false);

    for _ in 0..endpointer.grace - 1 {
      assert!(matches!(
        endpointer.decide(false, true),
        Endpoint::Continue
      ));
    }

    assert!(matches!(endpointer.decide(false, true), Endpoint::NoSpeech));
  }

  /// Said briskly, the command is already in the pre-roll and only its
  /// tail arrives live. That tail is what has to keep the capture from
  /// waiting out the whole grace period for a command already said.
  #[test]
  fn a_brisk_command_ends_without_waiting_out_the_grace() {
    let mut endpointer = endpointer();

    for _ in 0..10 {
      endpointer.decide(true, false);
    }

    endpointer.decide(true, true);

    for _ in 0..endpointer.hangover - 1 {
      endpointer.decide(false, true);
    }

    assert!(matches!(endpointer.decide(false, true), Endpoint::Done));
    assert!(endpointer.live_windows < endpointer.grace);
  }

  #[test]
  fn crop_keeps_speech_and_drops_surrounding_silence() {
    // Long enough that the padding fits either side and the crop is
    // genuinely exercised rather than clamped to the whole clip.
    let windows = CROP_PAD * 4;
    let (first, last) = (windows / 2, windows / 2 + 1);

    let mut mask = vec![false; windows];
    mask[first] = true;
    mask[last] = true;

    let audio = vec![0.0_f32; windows * WINDOW];
    let cropped = endpointer_with(mask).crop(&audio);

    // 2 speech windows plus CROP_PAD either side.
    assert_eq!(cropped.len(), (2 + 2 * CROP_PAD) * WINDOW);
    assert!(cropped.len() < audio.len());
  }

  #[test]
  fn crop_passes_through_when_no_speech_was_seen() {
    let audio = vec![0.0_f32; 4 * WINDOW];
    let cropped = endpointer_with(vec![false; 4]).crop(&audio);

    assert_eq!(cropped.len(), audio.len());
  }

  #[test]
  fn crop_clamps_at_the_end_of_the_clip() {
    // Speech runs right up to the last window, so the trailing pad
    // has nothing to expand into.
    let audio = vec![0.0_f32; 3 * WINDOW];
    let cropped = endpointer_with(vec![false, true, true]).crop(&audio);

    assert_eq!(cropped.len(), audio.len());
  }
}
