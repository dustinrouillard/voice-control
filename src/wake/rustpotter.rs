use anyhow::{Context, Result, anyhow};
use rustpotter::{Rustpotter, RustpotterConfig};
use tracing::info;

use crate::audio::SAMPLE_RATE;
use crate::config::expand_tilde;
use crate::wake::{Detection, Detector};

pub struct RustpotterDetector {
  inner: Rustpotter,
  frame_size: usize,
}

impl RustpotterDetector {
  pub fn load(
    model_path: &str,
    threshold: f32,
    avg_threshold: f32,
    eager: bool,
  ) -> Result<Self> {
    let path = expand_tilde(model_path);

    if !path.exists() {
      return Err(anyhow!(
        "wake word model {} not found - train one with \
         `rustpotter-cli build` (see README)",
        path.display()
      ));
    }

    let mut config = RustpotterConfig::default();
    // Defaults are already 16 kHz mono f32, but the pipeline depends
    // on that so state it rather than inherit it silently.
    config.fmt.sample_rate = SAMPLE_RATE as usize;
    config.fmt.channels = 1;
    config.detector.threshold = threshold;
    config.detector.avg_threshold = avg_threshold;
    config.detector.eager = eager;

    let mut inner = Rustpotter::new(&config)
      .map_err(|why| anyhow!("configuring rustpotter: {why}"))?;

    inner
      .add_wakeword_from_file(
        "wake",
        path.to_str().context("wake model path is not UTF-8")?,
      )
      .map_err(|why| anyhow!("loading {}: {why}", path.display()))?;

    let frame_size = inner.get_samples_per_frame();
    info!(model = %path.display(), frame_size, "loaded wake word model");

    Ok(Self { inner, frame_size })
  }
}

impl Detector for RustpotterDetector {
  fn frame_size(&self) -> usize {
    self.frame_size
  }

  fn push(&mut self, frame: Vec<f32>) -> Option<Detection> {
    self
      .inner
      .process_samples(frame)
      .map(|detection| Detection {
        name: detection.name,
        score: detection.score,
      })
  }

  fn reset(&mut self) {
    self.inner.reset();
  }
}
