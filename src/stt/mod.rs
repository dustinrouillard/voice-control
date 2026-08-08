pub mod whisper;

use anyhow::Result;

/// Swappable speech-to-text backend, fed 16 kHz mono f32.
pub trait Transcriber: Send {
  fn transcribe(&mut self, audio: &[f32]) -> Result<String>;
}
