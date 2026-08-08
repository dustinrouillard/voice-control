use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Recent commands kept for the menu.
const HISTORY: usize = 10;

/// How long the outcome glyph stays in the menu bar after a command.
const FLASH: Duration = Duration::from_millis(1500);

/// A peak below this is not a quiet room, it is nothing at all -
/// the same threshold `voice-control listen` uses to call out a
/// missing microphone grant.
const SILENCE: f32 = 1e-5;

/// Dead silence for this long, while idle, means we are not really
/// hearing the room: the grant was revoked, or the device is muted
/// in hardware.
const DEAF_AFTER: Duration = Duration::from_secs(15);

/// No audio buffers arriving at all for this long means the stream
/// itself is gone, which is a different fault from silence.
///
/// Only ever evaluated while idle. Transcription blocks the feed loop
/// for a couple of hundred milliseconds and the incoming audio queues
/// up behind it, which would otherwise look like a stall.
const STALL_AFTER: Duration = Duration::from_secs(5);

/// How fast the level meter falls back. Applied per buffer (~21 ms),
/// so a peak decays over roughly 200 ms - slow enough to see, fast
/// enough to track speech.
const LEVEL_DECAY: f32 = 0.85;

/// What the daemon is doing, as far as the menu bar is concerned.
#[derive(Clone, PartialEq)]
pub enum Phase {
  /// Models loading, or no audio seen yet.
  Starting,
  /// Waiting for the wake word.
  Idle,
  /// Wake word heard, capturing the command.
  Hearing,
  /// Transcribing and dispatching.
  Thinking,
  /// Wake word scoring suspended from the menu.
  Paused,
  /// Audio is arriving but it is all silence.
  Deaf(Duration),
  /// No audio has arrived for a while.
  Stalled(Duration),
  /// The capture stream ended and is not coming back.
  Stopped,
}

/// How one utterance turned out.
#[derive(Clone)]
pub enum Outcome {
  /// Matched a command and the target accepted it.
  Dispatched(String),
  /// Matched a command but the HTTP call or OBS switch failed.
  Failed(String),
  /// Transcribed fine, matched nothing. These are the interesting
  /// ones: each is a phrase list waiting to be grown.
  NoMatch,
  /// Woken but nothing usable came back.
  Unheard,
}

impl Outcome {
  pub fn ok(&self) -> bool {
    matches!(self, Outcome::Dispatched(_))
  }
}

pub struct Entry {
  pub ago: Duration,
  pub transcript: String,
  pub outcome: Outcome,
}

/// Everything the tray needs, read in one lock.
pub struct Snapshot {
  pub phase: Phase,
  /// Set briefly after a command, `true` for success.
  pub flash: Option<bool>,
  pub level: f32,
  pub device: String,
  pub last_wake: Option<Duration>,
  pub history: Vec<Entry>,
}

struct Inner {
  phase: Phase,
  level: f32,
  device: String,
  /// When the first buffer arrived, when the last one did, and when
  /// one last held any sound. The first is what lets a microphone
  /// that has been dead since startup be called out: there is no
  /// "last sound" to measure the silence from.
  first_audio: Option<Instant>,
  last_audio: Option<Instant>,
  last_sound: Option<Instant>,
  last_wake: Option<Instant>,
  flash: Option<(bool, Instant)>,
  history: VecDeque<(Instant, String, Outcome)>,
}

/// Shared state between the pipeline and the menu bar.
///
/// A plain mutex rather than a channel: the tray polls on a timer and
/// only ever wants the current picture, so there is nothing to queue.
/// The pipeline's hot path takes this lock once per audio buffer,
/// roughly fifty times a second, which is nothing.
pub struct Status {
  inner: Mutex<Inner>,
  paused: AtomicBool,
}

impl Status {
  pub fn new() -> Self {
    Self {
      inner: Mutex::new(Inner {
        phase: Phase::Starting,
        level: 0.0,
        device: String::new(),
        first_audio: None,
        last_audio: None,
        last_sound: None,
        last_wake: None,
        flash: None,
        history: VecDeque::new(),
      }),
      paused: AtomicBool::new(false),
    }
  }

  pub fn paused(&self) -> bool {
    self.paused.load(Ordering::Relaxed)
  }

  /// Returns the new state, so the caller can log it.
  pub fn toggle_pause(&self) -> bool {
    !self.paused.fetch_xor(true, Ordering::Relaxed)
  }

  pub fn set_device(&self, name: &str) {
    self.lock().device = name.to_string();
  }

  /// Called for every buffer of audio, whether or not it is scored.
  /// Feeds the level meter and both watchdogs.
  pub fn audio(&self, peak: f32) {
    let now = Instant::now();
    let mut inner = self.lock();

    inner.level = peak.max(inner.level * LEVEL_DECAY);
    inner.first_audio.get_or_insert(now);
    inner.last_audio = Some(now);

    if peak >= SILENCE {
      inner.last_sound = Some(now);
    }
  }

  pub fn wake(&self) {
    let now = Instant::now();
    let mut inner = self.lock();

    inner.phase = Phase::Hearing;
    inner.last_wake = Some(now);
    inner.flash = None;
  }

  pub fn thinking(&self) {
    self.lock().phase = Phase::Thinking;
  }

  pub fn idle(&self) {
    self.lock().phase = Phase::Idle;
  }

  pub fn stopped(&self) {
    self.lock().phase = Phase::Stopped;
  }

  /// Records how an utterance turned out and returns to idle.
  pub fn finished(&self, transcript: &str, outcome: Outcome) {
    let now = Instant::now();
    let mut inner = self.lock();

    inner.phase = Phase::Idle;
    inner.flash = Some((outcome.ok(), now));
    inner
      .history
      .push_front((now, transcript.to_string(), outcome));
    inner.history.truncate(HISTORY);
  }

  pub fn snapshot(&self) -> Snapshot {
    let now = Instant::now();
    let inner = self.lock();

    Snapshot {
      phase: self.derive_phase(&inner, now),
      flash: inner
        .flash
        .filter(|(_, at)| now.duration_since(*at) < FLASH)
        .map(|(ok, _)| ok),
      level: inner.level,
      device: inner.device.clone(),
      last_wake: inner.last_wake.map(|at| now.duration_since(at)),
      history: inner
        .history
        .iter()
        .map(|(at, transcript, outcome)| Entry {
          ago: now.duration_since(*at),
          transcript: transcript.clone(),
          outcome: outcome.clone(),
        })
        .collect(),
    }
  }

  /// The fault states are derived rather than stored, so nothing has
  /// to run a timer to notice them - whoever asks does the arithmetic.
  fn derive_phase(&self, inner: &Inner, now: Instant) -> Phase {
    // A real fault outranks a pause: paused is also silent, but a
    // revoked microphone grant is still worth showing.
    if !matches!(inner.phase, Phase::Idle | Phase::Paused) {
      return inner.phase.clone();
    }

    let Some(last_audio) = inner.last_audio else {
      return Phase::Starting;
    };

    let quiet = now.duration_since(last_audio);

    if quiet > STALL_AFTER {
      return Phase::Stalled(quiet);
    }

    // With no sound ever heard there is nothing to measure the
    // silence from, so fall back to how long audio has been arriving.
    // Measuring from the newest buffer instead would restart the
    // clock fifty times a second and never trip.
    let silent = match inner.last_sound.or(inner.first_audio) {
      Some(at) => now.duration_since(at),
      None => quiet,
    };

    if silent > DEAF_AFTER {
      return Phase::Deaf(silent);
    }

    if self.paused() {
      return Phase::Paused;
    }

    Phase::Idle
  }

  fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
    // Nothing held across a panic point, but a poisoned lock would
    // otherwise take the menu bar down with it.
    self.inner.lock().unwrap_or_else(|e| e.into_inner())
  }
}

impl Default for Status {
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn starts_before_any_audio_arrives() {
    let status = Status::new();

    assert!(matches!(status.snapshot().phase, Phase::Starting));
  }

  #[test]
  fn quiet_room_is_not_deaf() {
    let status = Status::new();
    status.idle();
    status.audio(0.02);

    assert!(matches!(status.snapshot().phase, Phase::Idle));
  }

  #[test]
  fn a_pause_shows_through() {
    let status = Status::new();
    status.idle();
    status.audio(0.02);

    assert!(status.toggle_pause());
    assert!(matches!(status.snapshot().phase, Phase::Paused));
  }

  #[test]
  fn capture_states_outrank_the_watchdogs() {
    let status = Status::new();
    status.wake();

    // No audio has ever arrived, which would otherwise read as
    // Starting - but a capture in flight is the more useful truth.
    assert!(matches!(status.snapshot().phase, Phase::Hearing));
  }

  #[test]
  fn a_run_of_silence_reads_as_deaf() {
    let status = Status::new();
    status.idle();
    status.audio(0.0);

    // Buffers are still arriving, they just hold nothing.
    let stale = Instant::now() - (DEAF_AFTER + Duration::from_secs(1));
    status.lock().last_sound = Some(stale);

    let Phase::Deaf(since) = status.snapshot().phase else {
      panic!("expected Deaf");
    };

    assert!(since > DEAF_AFTER);
  }

  #[test]
  fn silence_that_has_never_broken_is_measured_from_the_first_buffer() {
    let status = Status::new();
    status.idle();
    status.audio(0.0);

    // Buffers are arriving normally - the newest one landed just now -
    // but not one of them has ever held sound. Measured from the
    // newest buffer this would never trip, so it has to be measured
    // from the first.
    let stale = Instant::now() - (DEAF_AFTER + Duration::from_secs(1));
    let mut inner = status.lock();
    inner.first_audio = Some(stale);
    inner.last_audio = Some(Instant::now());
    inner.last_sound = None;
    drop(inner);

    assert!(matches!(status.snapshot().phase, Phase::Deaf(_)));
  }

  #[test]
  fn buffers_drying_up_reads_as_stalled() {
    let status = Status::new();
    status.idle();
    status.audio(0.5);

    let stale = Instant::now() - (STALL_AFTER + Duration::from_secs(1));
    status.lock().last_audio = Some(stale);

    // Louder than silence and more recent than DEAF_AFTER, so the
    // only thing wrong is that the stream stopped.
    assert!(matches!(status.snapshot().phase, Phase::Stalled(_)));
  }

  #[test]
  fn a_fault_outranks_a_pause() {
    let status = Status::new();
    status.idle();
    status.audio(0.0);
    status.toggle_pause();

    let stale = Instant::now() - (STALL_AFTER + Duration::from_secs(1));
    status.lock().last_audio = Some(stale);

    // Paused is also silent, but a stream that has gone away is
    // worth saying out loud either way.
    assert!(matches!(status.snapshot().phase, Phase::Stalled(_)));
  }

  #[test]
  fn history_is_newest_first_and_bounded() {
    let status = Status::new();

    for i in 0..HISTORY + 3 {
      status.finished(&format!("command {i}"), Outcome::NoMatch);
    }

    let history = status.snapshot().history;

    assert_eq!(history.len(), HISTORY);
    assert_eq!(history[0].transcript, format!("command {}", HISTORY + 2));
  }

  #[test]
  fn the_outcome_flash_expires() {
    let status = Status::new();
    status.finished("mute", Outcome::Dispatched("mute".into()));

    assert_eq!(status.snapshot().flash, Some(true));

    status.lock().flash = Some((true, Instant::now() - FLASH));

    assert_eq!(status.snapshot().flash, None);
  }
}
