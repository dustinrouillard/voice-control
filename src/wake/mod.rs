pub mod rustpotter;

/// A wake word detection.
#[derive(Debug, Clone)]
pub struct Detection {
  pub name: String,
  pub score: f32,
}

/// Swappable wake word backend.
///
/// Everything upstream deals in 16 kHz mono f32, and every engine
/// wants its own frame size, so implementations buffer internally and
/// return `None` until they have enough audio to score.
pub trait Detector: Send {
  /// Samples per call to [`Detector::push`].
  fn frame_size(&self) -> usize;

  /// Feed exactly [`Detector::frame_size`] samples.
  fn push(&mut self, frame: Vec<f32>) -> Option<Detection>;

  /// Drop internal state, e.g. after a command has been handled.
  fn reset(&mut self);
}
