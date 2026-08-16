//! Publishes what the daemon is doing to an HTTP endpoint.
//!
//! For the OBS overlay, which wants to light up the moment the wake
//! word lands. That rules out having it poll: a quarter of a second of
//! lag is visible on stream, and polling fast enough to hide it means
//! hammering an endpoint that has nothing new to say all day.
//!
//! Nothing in the pipeline calls this. Every state change already goes
//! through [`Status`] on its way to the menu bar, so this watches that
//! instead and posts when the picture changes - which also picks up
//! the fault states, `Deaf` and `Stalled`, that are derived at read
//! time and never "happen" anywhere a hook could sit.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio::time::{MissedTickBehavior, interval};
use tracing::{debug, info, warn};

use crate::status::{Outcome, Phase, Status};

/// How often the status is examined. Fast enough that the overlay
/// lights up with the chirp rather than after it.
const POLL: Duration = Duration::from_millis(100);
/// Posted even when nothing has changed, so the overlay can tell a
/// quiet daemon from a dead one.
const HEARTBEAT: Duration = Duration::from_secs(5);
/// A stuck endpoint must not back up the queue behind it.
const TIMEOUT: Duration = Duration::from_secs(2);
/// How long a command's result stays in the payload. The overlay runs
/// its own fade; this is only the backstop for a page that connects
/// while one is on screen.
const RESULT_FOR: Duration = Duration::from_secs(6);

/// What the daemon is doing, as far as anything downstream cares.
#[derive(Debug, Serialize, PartialEq, Clone)]
pub struct Event {
  /// The wake word in play, so the overlay can label itself without
  /// being told separately.
  pub wake_word: String,
  pub state: &'static str,
  pub device: String,
  /// How long the fault behind `deaf` or `stalled` has been going on.
  #[serde(skip_serializing_if = "Option::is_none")]
  pub fault_ms: Option<u128>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub result: Option<CommandResult>,
}

/// How the last utterance turned out, while it is still recent.
#[derive(Debug, Serialize, PartialEq, Clone)]
pub struct CommandResult {
  /// Which utterance this is, counting from startup. Saying "next
  /// track" twice produces two identical results, and whatever draws
  /// them has to be able to tell that the second one is new.
  pub id: u64,
  pub transcript: String,
  pub outcome: &'static str,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub command: Option<String>,
}

/// Starts the publisher on the current runtime. Returns immediately;
/// a failing endpoint is never allowed to matter to anything else.
pub fn spawn(status: Arc<Status>, url: String, wake_word: String) {
  tokio::spawn(async move {
    if let Err(why) = run(status, &url, &wake_word).await {
      warn!(error = ?why, "status publisher stopped");
    }
  });
}

async fn run(
  status: Arc<Status>,
  url: &str,
  wake_word: &str,
) -> anyhow::Result<()> {
  let client = reqwest::Client::builder().timeout(TIMEOUT).build()?;

  let mut ticker = interval(POLL);
  // The default would try to catch up after the machine sleeps, which
  // means a burst of posts describing moments that have already gone.
  ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

  info!(url, "publishing status");

  let mut last: Option<Event> = None;
  let mut sent_at: Option<Instant> = None;
  // Whether the endpoint was reachable last time. Tracked so that an
  // overlay which is simply not running costs one line in the log
  // rather than one every five seconds.
  let mut healthy = true;

  loop {
    ticker.tick().await;

    let event = build(&status, wake_word);
    let stale = sent_at.is_none_or(|at| at.elapsed() >= HEARTBEAT);

    if last.as_ref() == Some(&event) && !stale {
      continue;
    }

    match client.post(url).json(&event).send().await {
      Ok(response) if response.status().is_success() => {
        if !healthy {
          info!(url, "status endpoint is back");
          healthy = true;
        }
      }
      Ok(response) => {
        if healthy {
          warn!(url, status = %response.status(), "status endpoint refused the post");
          healthy = false;
        }
      }
      Err(why) => {
        if healthy {
          debug!(url, error = ?why, "status endpoint unreachable");
          healthy = false;
        }
      }
    }

    last = Some(event);
    sent_at = Some(Instant::now());
  }
}

fn build(status: &Status, wake_word: &str) -> Event {
  let snapshot = status.snapshot();

  let (state, fault_ms) = match &snapshot.phase {
    Phase::Starting => ("starting", None),
    Phase::Idle => ("idle", None),
    Phase::Hearing => ("listening", None),
    Phase::Thinking => ("thinking", None),
    Phase::Paused => ("paused", None),
    Phase::Deaf(since) => ("deaf", Some(since.as_millis())),
    Phase::Stalled(since) => ("stalled", Some(since.as_millis())),
    Phase::Stopped => ("stopped", None),
  };

  // Only the newest entry, and only while it is recent: this is the
  // one being shown, not a history.
  let result = snapshot
    .history
    .first()
    .filter(|entry| entry.ago < RESULT_FOR)
    .map(|entry| {
      let (outcome, command) = match &entry.outcome {
        Outcome::Dispatched(name) => ("dispatched", Some(name.clone())),
        Outcome::Failed(name) => ("failed", Some(name.clone())),
        Outcome::NoMatch => ("no_match", None),
        Outcome::Unheard => ("unheard", None),
      };

      CommandResult {
        id: snapshot.utterances,
        transcript: entry.transcript.clone(),
        outcome,
        command,
      }
    });

  Event {
    wake_word: wake_word.to_string(),
    state,
    device: snapshot.device,
    fault_ms,
    result,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The overlay reads these names, and it lives in another repo, so
  /// renaming a field here is a breaking change rather than a rename.
  #[test]
  fn the_wire_shape_is_what_the_overlay_reads() {
    let status = Status::new();
    status.audio(0.2);
    status
      .finished("next track", Outcome::Dispatched("next track".into()));

    let json = serde_json::to_value(build(&status, "alexa")).unwrap();

    assert_eq!(
      json,
      serde_json::json!({
        "wake_word": "alexa",
        "state": "idle",
        "device": "",
        "result": {
          "id": 1,
          "transcript": "next track",
          "outcome": "dispatched",
          "command": "next track"
        }
      })
    );

    // Absent rather than null, so the overlay can test for presence.
    status.idle();
    let quiet =
      serde_json::to_value(build(&Status::new(), "alexa")).unwrap();
    assert!(quiet.get("result").is_none());
    assert!(quiet.get("fault_ms").is_none());
  }

  #[test]
  fn reports_the_phase_the_menu_bar_would_show() {
    let status = Status::new();
    status.idle();
    status.audio(0.2);

    assert_eq!(build(&status, "alexa").state, "idle");

    status.wake();
    assert_eq!(build(&status, "alexa").state, "listening");

    status.thinking();
    assert_eq!(build(&status, "alexa").state, "thinking");
  }

  #[test]
  fn carries_the_last_command_while_it_is_fresh() {
    let status = Status::new();
    status.audio(0.2);
    status
      .finished("next track", Outcome::Dispatched("next track".into()));

    let event = build(&status, "alexa");
    let result = event.result.expect("a result");

    assert_eq!(event.state, "idle");
    assert_eq!(result.outcome, "dispatched");
    assert_eq!(result.command.as_deref(), Some("next track"));
    assert_eq!(result.transcript, "next track");
  }

  #[test]
  fn a_failed_command_is_distinguishable_from_one_that_matched_nothing() {
    let status = Status::new();
    status.audio(0.2);
    status.finished("mute", Outcome::Failed("mute".into()));
    assert_eq!(build(&status, "alexa").result.unwrap().outcome, "failed");

    status.finished("what time is it", Outcome::NoMatch);
    let result = build(&status, "alexa").result.unwrap();
    assert_eq!(result.outcome, "no_match");
    assert_eq!(result.command, None);
  }

  #[test]
  fn the_same_command_twice_is_two_results() {
    let status = Status::new();
    status.audio(0.2);

    status.finished("next", Outcome::Dispatched("next track".into()));
    let first = build(&status, "alexa").result.unwrap();

    status.finished("next", Outcome::Dispatched("next track".into()));
    let second = build(&status, "alexa").result.unwrap();

    assert_ne!(first.id, second.id);
    assert_ne!(first, second);
  }

  /// The glue rather than the payload: that `spawn` reaches the
  /// network at all, and posts JSON to the path it was given.
  #[tokio::test]
  async fn spawn_posts_to_the_configured_url() {
    use tokio::io::AsyncReadExt;
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url =
      format!("http://{}/api/voice", listener.local_addr().unwrap());

    let status = Arc::new(Status::new());
    status.audio(0.2);
    status.wake();

    spawn(Arc::clone(&status), url, "alexa".to_string());

    let (mut socket, _) =
      tokio::time::timeout(Duration::from_secs(5), listener.accept())
        .await
        .expect("the publisher never connected")
        .unwrap();

    let mut request = vec![0; 1024];
    let read = tokio::time::timeout(
      Duration::from_secs(5),
      socket.read(&mut request),
    )
    .await
    .expect("the publisher never sent anything")
    .unwrap();

    let request = String::from_utf8_lossy(&request[..read]);

    assert!(request.starts_with("POST /api/voice "), "{request}");
    assert!(
      request.contains("content-type: application/json"),
      "{request}"
    );
    assert!(request.contains(r#""state":"listening""#), "{request}");
    assert!(request.contains(r#""wake_word":"alexa""#), "{request}");
  }

  #[test]
  fn heartbeats_are_identical_so_only_changes_post() {
    let status = Status::new();
    status.idle();
    status.audio(0.2);

    let before = build(&status, "alexa");
    assert_eq!(before, build(&status, "alexa"));

    status.wake();
    assert_ne!(before, build(&status, "alexa"));
  }
}
