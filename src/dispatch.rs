use std::time::Duration;

use anyhow::{Context, Result, bail};
use tracing::{info, warn};

use crate::commands::{Action, Command, Step};
use crate::devices;
use crate::exec;
use crate::feedback::Feedback;
use crate::media;
use crate::obs::{self, ObsConfig};

/// The APIs are on localhost or the LAN; if one is not answering
/// promptly it is down, and blocking the pipeline helps nobody.
const TIMEOUT: Duration = Duration::from_secs(3);

pub struct Dispatcher {
  client: reqwest::Client,
  obs: ObsConfig,
  feedback: Feedback,
}

impl Dispatcher {
  pub fn new(obs: ObsConfig, feedback: Feedback) -> Result<Self> {
    let client = reqwest::Client::builder()
      .timeout(TIMEOUT)
      .build()
      .context("building the http client")?;

    Ok(Self {
      client,
      obs,
      feedback,
    })
  }

  /// Runs a command's steps in order, stopping at the first failure.
  ///
  /// Stopping matters for a flow that animates: if the scene switch
  /// did not happen there is no point enabling the move that was meant
  /// to play on it, and carrying on would leave the source shown
  /// somewhere you cannot see it.
  pub async fn run(&self, command: &Command) -> Result<()> {
    let steps = command.steps();

    for (index, step) in steps.iter().enumerate() {
      self.step(command, step).await.with_context(|| {
        // Only worth saying which step when there is more than one.
        if steps.len() > 1 {
          format!("step {} of {}", index + 1, steps.len())
        } else {
          String::new()
        }
      })?;
    }

    Ok(())
  }

  async fn step(&self, command: &Command, step: &Step) -> Result<()> {
    match step.action() {
      Action::Http { method, url } => {
        self.http(command, method, url).await
      }
      Action::Wait(delay) => {
        tokio::time::sleep(delay).await;

        Ok(())
      }
      // Two event posts and no round trip, so there is nothing here
      // worth handing to a blocking pool.
      Action::Media(key) => {
        media::press(key)
          .with_context(|| format!("pressing the {} key", key.as_str()))?;

        info!(command = %command.name, key = key.as_str(), "pressed");

        Ok(())
      }
      // A handful of HAL property reads and one write, all of them
      // answered from coreaudiod's own state - there is no round trip
      // here worth handing to a blocking pool either.
      Action::Device { direction, pattern } => {
        let switch = devices::set_default(direction, pattern)
          .with_context(|| {
            format!(
              "switching the {} device to {pattern:?}",
              direction.as_str()
            )
          })?;

        info!(
          command = %command.name,
          direction = direction.as_str(),
          device = switch.name,
          changed = switch.changed,
          "set the default audio device"
        );

        Ok(())
      }
      // The only step that hands control to something we did not
      // write, so it is also the only one with a timeout of its own.
      Action::Run(program) => {
        let bin = program.bin;

        let output = exec::run(program)
          .await
          .with_context(|| format!("running {bin}"))?;

        info!(
          command = %command.name,
          program = bin,
          output = %output,
          "ran"
        );

        Ok(())
      }
      Action::Sound(name) => {
        self
          .feedback
          .play_named(name)
          .with_context(|| format!("playing {name}.wav"))?;

        info!(command = %command.name, sound = name, "played");

        Ok(())
      }
      Action::Scene(scene) => {
        obs::set_scene(&self.obs, scene)
          .await
          .with_context(|| format!("switching to scene {scene:?}"))?;

        info!(command = %command.name, scene, "switched obs scene");

        Ok(())
      }
      Action::Source(action) => {
        let name = action.source;
        let visibility = action.visibility;

        let visible =
          obs::run_source(&self.obs, action).await.with_context(|| {
            format!("setting source {name:?} to {}", visibility.as_str())
          })?;

        info!(
          command = %command.name,
          source = name,
          visible,
          "set obs source visibility"
        );

        Ok(())
      }
    }
  }

  async fn http(
    &self,
    command: &Command,
    method: &str,
    url: &str,
  ) -> Result<()> {
    let method: reqwest::Method = method
      .parse()
      .with_context(|| format!("bad method {method:?}"))?;

    let response = self
      .client
      .request(method, url)
      .send()
      .await
      .with_context(|| format!("calling {url}"))?;

    let status = response.status();

    if !status.is_success() {
      let body = response.text().await.unwrap_or_default();
      bail!("{url} returned {status}: {body}");
    }

    info!(command = %command.name, %status, "dispatched");

    Ok(())
  }
}

/// Logs rather than propagates: a failed command should not take the
/// daemon down, and the caller still needs to play the failure tone.
pub async fn try_run(dispatcher: &Dispatcher, command: &Command) -> bool {
  match dispatcher.run(command).await {
    Ok(()) => true,
    Err(why) => {
      warn!(command = %command.name, error = ?why, "dispatch failed");
      false
    }
  }
}
