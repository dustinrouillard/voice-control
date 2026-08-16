//! Running a local program as a command step.
//!
//! Everything else a command can do reaches something over a socket -
//! HTTP, obs-websocket, the HAL - which leaves anything that ships a
//! CLI and no API out of reach. This closes that: `run` names the
//! binary, `args` are handed to it as written, and what it does with
//! them is its own business.
//!
//! There is no shell. The program is executed directly, so a value in
//! `args` is one argument however many spaces are in it, there is
//! nothing to quote, glob or expand - and nothing a misheard phrase
//! could smuggle in either, since the arguments come from the config
//! file and never from the transcript.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use tokio::process::Command;
use tracing::warn;

/// What one `run` step invokes.
#[derive(Debug)]
pub struct Program<'a> {
  /// A path, or a bare name to find on PATH.
  pub bin: &'a str,
  pub args: &'a [String],
  /// Set on top of the environment the daemon itself was started with.
  pub env: &'a HashMap<String, String>,
  /// How long it may take before it is killed.
  pub timeout: Duration,
}

/// Enough of what a program printed to recognise what it did from a
/// log line. Nothing downstream reads it, so a CLI that writes a
/// megabyte is not a reason to hold a megabyte.
const MAX_OUTPUT: usize = 300;

/// Runs a program to completion, failing on anything but exit 0.
///
/// The exit status is the only thing the program tells us, so it has
/// to be believed: a command that plays the success tone over a
/// program that just failed is worse than having no command at all.
///
/// Returns whatever it printed, clipped, for the caller to log.
pub async fn run(program: Program<'_>) -> Result<String> {
  let child = Command::new(program.bin)
    .args(program.args)
    .envs(program.env)
    // There is no terminal here to read from, and a program that
    // stops to ask something should fail rather than wait for an
    // answer that is never coming.
    .stdin(Stdio::null())
    // Captured rather than inherited: the daemon's stdout is the log
    // file, and a CLI drawing a progress bar into it would be
    // unreadable interleaved with everything else.
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    // A timeout that left the process running would be no timeout at
    // all. Dropping the future below is what kills it, so this is the
    // half that makes the timeout mean anything.
    .kill_on_drop(true)
    .spawn()
    .with_context(|| format!("starting {}", program.bin))?;

  // Bluetooth, a network hop, a device that is asleep - the things
  // worth saying out loud to are exactly the ones that can take a
  // while or never answer, and the dispatch is holding the pipeline
  // open while this runs.
  let waited =
    tokio::time::timeout(program.timeout, child.wait_with_output()).await;

  let Ok(output) = waited else {
    bail!(
      "{} was still running after {}ms, so it has been killed - raise \
       timeout_ms if it is only slow",
      program.bin,
      program.timeout.as_millis()
    );
  };

  let output =
    output.with_context(|| format!("waiting for {}", program.bin))?;

  let stdout = clip(&String::from_utf8_lossy(&output.stdout));
  let stderr = clip(&String::from_utf8_lossy(&output.stderr));

  if !output.status.success() {
    // "exit status: 1" says nothing about what went wrong; whatever
    // the program printed on its way out usually does.
    let why = if stderr.is_empty() { &stdout } else { &stderr };

    if why.is_empty() {
      bail!(
        "{} exited with {} and printed nothing",
        program.bin,
        output.status
      );
    }

    bail!("{} exited with {}: {why}", program.bin, output.status);
  }

  // A warning on a successful run is still worth having, and this is
  // the only place it would ever be seen.
  if !stderr.is_empty() {
    warn!(program = program.bin, stderr, "wrote to stderr");
  }

  Ok(stdout)
}

/// Where the program would be found, or `None` if it would not be.
///
/// launchd starts an agent with a PATH of `/usr/bin:/bin:/usr/sbin:
/// /sbin` and nothing else, so a bare name that works in your shell -
/// anything under `/opt/homebrew/bin`, say - is not on the daemon's
/// PATH at all. Checking at load turns that into a warning at startup
/// rather than a command that fails the first time you say it.
pub fn locate(bin: &str) -> Option<PathBuf> {
  if bin.contains('/') {
    let path = Path::new(bin);

    return path.is_file().then(|| path.to_path_buf());
  }

  let paths = std::env::var_os("PATH")?;

  std::env::split_paths(&paths)
    .map(|dir| dir.join(bin))
    .find(|candidate| candidate.is_file())
}

/// Trimmed, flattened onto one line, and cut to what a log line can
/// reasonably hold.
fn clip(text: &str) -> String {
  let text = text.trim();
  let mut out = String::new();

  for (index, ch) in text.chars().enumerate() {
    if index == MAX_OUTPUT {
      out.push('…');
      break;
    }

    out.push(if ch == '\n' || ch == '\r' { ' ' } else { ch });
  }

  out
}

#[cfg(test)]
mod tests {
  use super::*;

  fn program<'a>(
    bin: &'a str,
    args: &'a [String],
    env: &'a HashMap<String, String>,
  ) -> Program<'a> {
    Program {
      bin,
      args,
      env,
      timeout: Duration::from_secs(5),
    }
  }

  #[tokio::test]
  async fn a_program_that_succeeds_gives_back_what_it_printed() {
    let args = ["hello".to_string()];
    let env = HashMap::new();

    let output = run(program("/bin/echo", &args, &env)).await.unwrap();

    assert_eq!(output, "hello");
  }

  /// Arguments go to the program as written - there is no shell in
  /// between to split them on the spaces.
  #[tokio::test]
  async fn an_argument_with_spaces_stays_one_argument() {
    let args = ["one two".to_string()];
    let env = HashMap::new();

    let output = run(program("/bin/echo", &args, &env)).await.unwrap();

    assert_eq!(output, "one two");
  }

  #[tokio::test]
  async fn env_reaches_the_program() {
    let args = ["-c".to_string(), "printf %s \"$WHERE\"".to_string()];
    let env = HashMap::from([("WHERE".to_string(), "desk".to_string())]);

    let output = run(program("/bin/sh", &args, &env)).await.unwrap();

    assert_eq!(output, "desk");
  }

  /// The exit status is the only thing the program tells us, so a
  /// non-zero one has to fail the step rather than chirp success.
  #[tokio::test]
  async fn a_non_zero_exit_fails_with_what_it_printed() {
    let args = [
      "-c".to_string(),
      "echo not connected >&2; exit 3".to_string(),
    ];
    let env = HashMap::new();

    let why = run(program("/bin/sh", &args, &env))
      .await
      .unwrap_err()
      .to_string();

    assert!(why.contains("not connected"), "{why}");
    assert!(why.contains('3'), "{why}");
  }

  #[tokio::test]
  async fn a_program_that_hangs_is_killed() {
    let args = ["30".to_string()];
    let env = HashMap::new();

    let why = run(Program {
      bin: "/bin/sleep",
      args: &args,
      env: &env,
      timeout: Duration::from_millis(100),
    })
    .await
    .unwrap_err()
    .to_string();

    assert!(why.contains("killed"), "{why}");
  }

  #[tokio::test]
  async fn a_program_that_is_not_there_says_so() {
    let env = HashMap::new();

    let why = run(program("/usr/bin/definitely-not-installed", &[], &env))
      .await
      .unwrap_err()
      .to_string();

    assert!(why.contains("definitely-not-installed"), "{why}");
  }

  #[test]
  fn locate_finds_a_path_and_a_bare_name() {
    assert!(locate("/bin/sh").is_some());
    assert!(locate("sh").is_some());
    assert!(locate("/bin/definitely-not-installed").is_none());
    assert!(locate("definitely-not-installed").is_none());
  }

  #[test]
  fn long_output_is_clipped_onto_one_line() {
    assert_eq!(clip("  one\ntwo  "), "one two");
    assert_eq!(
      clip(&"x".repeat(MAX_OUTPUT + 10)).chars().count(),
      MAX_OUTPUT + 1
    );
  }
}
