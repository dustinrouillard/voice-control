use std::path::PathBuf;

use tracing::{debug, warn};

use crate::config::expand_tilde;

#[derive(Clone, Copy)]
pub enum Cue {
  /// Wake word heard, listening for the command.
  Wake,
  /// Command recognised and dispatched.
  Ok,
  /// Not understood, or the API call failed.
  Fail,
}

impl Cue {
  fn file(self) -> &'static str {
    match self {
      Cue::Wake => "wake.wav",
      Cue::Ok => "ok.wav",
      Cue::Fail => "fail.wav",
    }
  }
}

/// Plays short cues through `afplay`.
///
/// Spawning a process per cue is wasteful, but it routes to whatever
/// the system output currently is without holding an output device
/// open, which matters when the default device changes mid-session.
#[derive(Clone)]
pub struct Feedback {
  dir: Option<PathBuf>,
}

impl Feedback {
  pub fn new(sounds_dir: &str) -> Self {
    if sounds_dir.is_empty() {
      debug!("SOUNDS_DIR unset, audio feedback disabled");
      return Self { dir: None };
    }

    let dir = expand_tilde(sounds_dir);

    if !dir.is_dir() {
      warn!(dir = %dir.display(), "sounds dir missing, feedback disabled");
      return Self { dir: None };
    }

    Self { dir: Some(dir) }
  }

  pub fn play(&self, cue: Cue) {
    let Some(dir) = &self.dir else {
      return;
    };

    let path = dir.join(cue.file());

    if !path.exists() {
      warn!(path = %path.display(), "cue file missing");
      return;
    }

    // Detached: the pipeline should never wait on audio playback.
    match std::process::Command::new("afplay").arg(&path).spawn() {
      Ok(mut child) => {
        std::thread::spawn(move || {
          let _ = child.wait();
        });
      }
      Err(why) => warn!(error = %why, "failed to spawn afplay"),
    }
  }
}
