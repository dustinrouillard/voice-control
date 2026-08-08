use anyhow::{Context, Result};
use voice_activity_detector::VoiceActivityDetector;

use crate::audio::SAMPLE_RATE;

/// Silero V5 only accepts a 512-sample window at 16 kHz.
pub const WINDOW: usize = 512;

const WINDOW_MS: usize = WINDOW * 1000 / SAMPLE_RATE as usize;

const fn windows_for(ms: usize) -> usize {
  ms / WINDOW_MS
}

/// Trailing silence that ends a command.
const HANGOVER: usize = windows_for(700);
/// Give up if the wake word was not actually followed by speech.
const NO_SPEECH_TIMEOUT: usize = windows_for(1500);
/// Absolute ceiling on one command.
const MAX_WINDOWS: usize = windows_for(5000);

const SPEECH_PROBABILITY: f32 = 0.5;

pub enum Endpoint {
  /// Still listening.
  Continue,
  /// Speech ended, or the hard cap was reached.
  Done,
  /// The wake word fired but no speech followed.
  NoSpeech,
}

/// Kept either side of the speech when cropping.
const CROP_PAD: usize = windows_for(200);

/// Decides when the user has finished speaking a command.
pub struct Endpointer {
  vad: VoiceActivityDetector,
  speech_seen: bool,
  silence_run: usize,
  windows: usize,
  /// Which windows held speech, in the order they were pushed.
  mask: Vec<bool>,
}

impl Endpointer {
  pub fn new() -> Result<Self> {
    let vad = VoiceActivityDetector::builder()
      .sample_rate(SAMPLE_RATE as i64)
      .chunk_size(WINDOW)
      .build()
      .context("building the Silero VAD")?;

    Ok(Self {
      vad,
      speech_seen: false,
      silence_run: 0,
      windows: 0,
      mask: Vec::new(),
    })
  }

  pub fn reset(&mut self) {
    self.vad.reset();
    self.speech_seen = false;
    self.silence_run = 0;
    self.windows = 0;
    self.mask.clear();
  }

  /// Feed exactly [`WINDOW`] samples.
  pub fn push(&mut self, window: Vec<f32>) -> Endpoint {
    debug_assert_eq!(window.len(), WINDOW);

    let speech = self.vad.predict(window) >= SPEECH_PROBABILITY;
    self.windows += 1;
    self.mask.push(speech);

    if speech {
      self.speech_seen = true;
      self.silence_run = 0;
    } else {
      self.silence_run += 1;
    }

    if self.windows >= MAX_WINDOWS {
      return Endpoint::Done;
    }

    if !self.speech_seen {
      return if self.windows >= NO_SPEECH_TIMEOUT {
        Endpoint::NoSpeech
      } else {
        Endpoint::Continue
      };
    }

    if self.silence_run >= HANGOVER {
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
    let mut endpointer = Endpointer::new().unwrap();
    endpointer.mask = mask;
    endpointer
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
