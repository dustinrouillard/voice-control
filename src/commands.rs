use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;
use strsim::jaro_winkler;

use crate::audio::vad::Timing;
use crate::config::expand_tilde;
use crate::devices::Direction;
use crate::exec::{self, Program};
use crate::hass::{self, HassConfig, ServiceCall};
use crate::media::MediaKey;
use crate::obs::{ObsConfig, SourceAction, Visibility};
use tracing::warn;

/// Minimum similarity for a fuzzy match to count.
const MATCH_THRESHOLD: f64 = 0.85;
/// How far ahead of the runner-up the winner must be. Two commands
/// scoring 0.86 and 0.85 means we genuinely cannot tell them apart,
/// and guessing wrong is worse than doing nothing.
const MATCH_MARGIN: f64 = 0.05;

#[derive(Debug, Deserialize)]
pub struct CommandFile {
  #[serde(default)]
  pub wake: WakeConfig,
  #[serde(default)]
  pub stt: SttConfig,
  /// Where to publish what the daemon is doing.
  #[serde(default)]
  pub status: StatusConfig,
  /// How long a capture runs for.
  #[serde(default)]
  pub listen: Timing,
  #[serde(default)]
  pub targets: HashMap<String, String>,
  /// Audio device aliases, referenced by name from `output` and
  /// `input` below.
  #[serde(default)]
  pub devices: HashMap<String, String>,
  #[serde(default)]
  pub obs: ObsConfig,
  #[serde(default)]
  pub hass: HassConfig,
  #[serde(default)]
  pub commands: Vec<Command>,
}

#[derive(Debug, Deserialize)]
pub struct WakeConfig {
  /// The openWakeWord classifier for the word itself.
  #[serde(default = "default_wake_model")]
  pub model: String,
  /// openWakeWord's shared feature extractors. The same two files for
  /// every wake word, so they are named separately from the model and
  /// almost never need changing.
  #[serde(default = "default_melspectrogram_model")]
  pub melspectrogram: String,
  #[serde(default = "default_embedding_model")]
  pub embedding: String,
  #[serde(default = "default_threshold")]
  pub threshold: f32,
  /// Consecutive 80 ms hops that must clear `threshold` before this
  /// counts as the wake word.
  #[serde(default = "default_patience")]
  pub patience: usize,
}

#[derive(Debug, Deserialize)]
pub struct SttConfig {
  #[serde(default = "default_stt_model")]
  pub model: String,
}

/// Where to post state changes - the wake word landing, a command
/// being dispatched, the microphone going quiet. Empty publishes
/// nothing, which is the default: this exists for the OBS overlay and
/// costs a running HTTP endpoint to use.
#[derive(Debug, Default, Deserialize)]
pub struct StatusConfig {
  #[serde(default)]
  pub url: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Command {
  pub name: String,
  pub phrases: Vec<String>,
  /// A command that does several things in order.
  ///
  /// Most commands do one thing, so the fields of a single step can be
  /// written directly on the command instead - the two forms are the
  /// same thing and a command uses one or the other.
  #[serde(default)]
  pub steps: Vec<Step>,
  #[serde(flatten)]
  pub step: Step,
}

/// One action. A command is a list of these, run in order, stopping at
/// the first failure.
#[derive(Debug, Deserialize, Clone)]
pub struct Step {
  #[serde(default = "default_method")]
  pub method: String,
  /// Set for an HTTP step. Mutually exclusive with `scene`, `source`
  /// and `wait_ms`.
  #[serde(default)]
  pub url: Option<String>,
  /// On its own, an OBS scene switch. Alongside `source`, the scene
  /// that source lives in — otherwise the source is looked for in
  /// whichever scene is on program at the time.
  ///
  /// Scenes are named one step at a time on purpose — the daemon can
  /// only reach scenes you named here, not whatever it thinks it
  /// heard.
  #[serde(default)]
  pub scene: Option<String>,
  /// Set for an OBS source (scene item) whose visibility this step
  /// changes.
  #[serde(default)]
  pub source: Option<String>,
  /// Which way `source` goes: `show`, `hide`, or `toggle` (default).
  #[serde(default)]
  pub visible: Option<Visibility>,
  /// A Move filter on `scene`, enabled once the source is showing, to
  /// animate it in.
  #[serde(default)]
  pub show_filter: Option<String>,
  /// A Move filter on `scene`, enabled before the source is hidden, to
  /// animate it out.
  #[serde(default)]
  pub hide_filter: Option<String>,
  /// How long `hide_filter` needs to finish before the source is
  /// actually hidden. Ignored without one.
  #[serde(default)]
  pub hide_delay_ms: Option<u64>,
  /// A step that only waits, for letting whatever the step before it
  /// started actually land.
  #[serde(default)]
  pub wait_ms: Option<u64>,
  /// A system media key - `play_pause`, `next` or `previous`. It goes
  /// to whichever application macOS considers to be playing, so there
  /// is nothing to name here beyond the key itself.
  #[serde(default)]
  pub media: Option<MediaKey>,
  /// The audio output to make default - a name from the `[devices]`
  /// table. Rewritten in place to the device pattern it stands for,
  /// the way `url` is rewritten from `[targets]`.
  #[serde(default)]
  pub output: Option<String>,
  /// A wav in `SOUNDS_DIR` to play, named without the extension.
  ///
  /// A command whose only step is this does nothing but answer, which
  /// is the point of it: "computa, are you listening?" wants a noise
  /// back and nothing else. The generic success tone is suppressed for
  /// a command that makes its own, so it answers once rather than
  /// twice.
  #[serde(default)]
  pub sound: Option<String>,
  /// The audio input to make default, named the same way.
  ///
  /// This does not move the daemon's own microphone: the capture
  /// stream was opened against a device rather than against "whatever
  /// is default", so it stays where it is until a restart.
  #[serde(default)]
  pub input: Option<String>,
  /// A Home Assistant entity id to call a service on -
  /// `switch.desk_speakers`. Which service is `service`, below.
  ///
  /// Entities are named one step at a time, the way scenes are: the
  /// daemon can only reach the ones you put here.
  #[serde(default)]
  pub hass: Option<String>,
  /// The service for `hass`, defaulting to `toggle`. Bare, and taken
  /// as belonging to the entity's own domain - `turn_on` on a
  /// `switch.` entity is `switch.turn_on` - or with a domain of its
  /// own for the services that are not tied to one, such as
  /// `homeassistant.turn_on`.
  #[serde(default)]
  pub service: Option<String>,
  /// Anything else the service takes, sent alongside the entity:
  /// `data = { brightness = 128 }`. Rarely needed for a switch, which
  /// is either on or off.
  #[serde(default)]
  pub data: HashMap<String, Value>,
  /// A program to run: a path, `~` and all, or a bare name to find on
  /// PATH. For the things that ship a CLI and no API.
  #[serde(default)]
  pub run: Option<String>,
  /// Arguments for `run`, handed over as written - there is no shell
  /// here to split or expand them, so one entry is one argument
  /// however many spaces are in it.
  #[serde(default)]
  pub args: Vec<String>,
  /// Environment for `run`, on top of the daemon's own.
  #[serde(default)]
  pub env: HashMap<String, String>,
  /// How long `run` may take before it is killed. Ignored without
  /// one.
  #[serde(default)]
  pub timeout_ms: Option<u64>,
}

impl Default for Step {
  fn default() -> Self {
    Self {
      method: default_method(),
      url: None,
      scene: None,
      source: None,
      visible: None,
      show_filter: None,
      hide_filter: None,
      hide_delay_ms: None,
      wait_ms: None,
      media: None,
      output: None,
      sound: None,
      input: None,
      hass: None,
      service: None,
      data: HashMap::new(),
      run: None,
      args: Vec::new(),
      env: HashMap::new(),
      timeout_ms: None,
    }
  }
}

/// Long enough for the move animations these are written against, and
/// what the equivalent Companion button waits.
const DEFAULT_HIDE_DELAY: Duration = Duration::from_millis(350);

/// Long enough for a CLI that has to wake something up and connect to
/// it, short enough that a command which is never coming back stops
/// holding the pipeline open. `timeout_ms` overrides it per step.
const DEFAULT_RUN_TIMEOUT: Duration = Duration::from_secs(10);

/// What one step actually does.
pub enum Action<'a> {
  Http {
    method: &'a str,
    url: &'a str,
  },
  Scene(&'a str),
  Source(SourceAction<'a>),
  Media(MediaKey),
  /// Point one of the system's default audio devices somewhere else.
  /// `pattern` is a case-insensitive substring of the device name.
  Device {
    direction: Direction,
    pattern: &'a str,
  },
  /// Play a wav from the sounds directory, named without the
  /// extension.
  Sound(&'a str),
  /// Call a Home Assistant service on one entity.
  Hass(ServiceCall<'a>),
  /// Run a local program and wait for it.
  Run(Program<'a>),
  Wait(Duration),
}

/// The fields a step can name an action with, for the errors about
/// naming none of them or several.
const ACTION_FIELDS: &str = "url, scene, source, media, output, input, sound, hass, run and \
   wait_ms";

impl Command {
  /// The steps to run, in order - the explicit list, or the one the
  /// command's own fields describe.
  pub fn steps(&self) -> &[Step] {
    if self.steps.is_empty() {
      std::slice::from_ref(&self.step)
    } else {
      &self.steps
    }
  }

  /// Whether the command makes a noise of its own, in which case the
  /// generic success tone is left off - answering "computa, ping" with
  /// a ping and then a second unrelated chirp is one tone too many.
  pub fn answers_for_itself(&self) -> bool {
    self
      .steps()
      .iter()
      .any(|step| matches!(step.action(), Action::Sound(_)))
  }

  /// How the command reads in a log line.
  pub fn target(&self) -> String {
    self
      .steps()
      .iter()
      .map(Step::target)
      .collect::<Vec<_>>()
      .join(" -> ")
  }
}

impl Step {
  pub fn action(&self) -> Action<'_> {
    // Checked ahead of the rest rather than as further arms: a media
    // key and a device switch each take none of the other fields, so
    // they have nothing to be ambiguous with.
    if let Some(key) = self.media {
      return Action::Media(key);
    }

    if let Some(pattern) = &self.output {
      return Action::Device {
        direction: Direction::Output,
        pattern,
      };
    }

    if let Some(pattern) = &self.input {
      return Action::Device {
        direction: Direction::Input,
        pattern,
      };
    }

    if let Some(name) = &self.sound {
      return Action::Sound(name);
    }

    if let Some(entity) = &self.hass {
      return Action::Hass(ServiceCall {
        entity,
        service: self.service.as_deref().unwrap_or(hass::DEFAULT_SERVICE),
        data: &self.data,
      });
    }

    if let Some(bin) = &self.run {
      return Action::Run(Program {
        bin,
        args: &self.args,
        env: &self.env,
        timeout: self
          .timeout_ms
          .map_or(DEFAULT_RUN_TIMEOUT, Duration::from_millis),
      });
    }

    match (&self.url, &self.source, &self.scene, self.wait_ms) {
      (Some(url), None, None, None) => Action::Http {
        method: &self.method,
        url,
      },
      (None, Some(source), scene, None) => Action::Source(SourceAction {
        source,
        scene: scene.as_deref(),
        visibility: self.visible.unwrap_or_default(),
        show_filter: self.show_filter.as_deref(),
        hide_filter: self.hide_filter.as_deref(),
        hide_delay: self
          .hide_delay_ms
          .map_or(DEFAULT_HIDE_DELAY, Duration::from_millis),
      }),
      (None, None, Some(scene), None) => Action::Scene(scene),
      (None, None, None, Some(wait)) => {
        Action::Wait(Duration::from_millis(wait))
      }
      // Rejected at load time by `validate`.
      _ => unreachable!("step has no valid action"),
    }
  }

  /// Nothing set at all, which is how the one-step shorthand reads on
  /// a command that uses `[[commands.steps]]` instead.
  fn is_empty(&self) -> bool {
    self.url.is_none()
      && self.scene.is_none()
      && self.source.is_none()
      && self.wait_ms.is_none()
      && self.media.is_none()
      && self.output.is_none()
      && self.input.is_none()
      && self.sound.is_none()
      && self.hass.is_none()
      && self.service.is_none()
      && self.data.is_empty()
      && self.run.is_none()
      && self.args.is_empty()
      && self.env.is_empty()
      && self.timeout_ms.is_none()
      && self.visible.is_none()
      && self.show_filter.is_none()
      && self.hide_filter.is_none()
      && self.hide_delay_ms.is_none()
  }

  /// Checks the step names exactly one thing to do, and that the
  /// fields belonging to one action have that action to act on.
  fn validate(&self) -> Result<()> {
    if self.run.is_none() {
      let stray = [
        (!self.args.is_empty()).then_some("args"),
        (!self.env.is_empty()).then_some("env"),
        self.timeout_ms.is_some().then_some("timeout_ms"),
      ]
      .into_iter()
      .flatten()
      .next();

      if let Some(field) = stray {
        bail!("sets {field} without a run");
      }
    }

    if self.hass.is_none() {
      let stray = [
        self.service.is_some().then_some("service"),
        (!self.data.is_empty()).then_some("data"),
      ]
      .into_iter()
      .flatten()
      .next();

      if let Some(field) = stray {
        bail!("sets {field} without a hass entity");
      }
    }

    if self.source.is_none() {
      let stray = [
        self.visible.is_some().then_some("visible"),
        self.show_filter.is_some().then_some("show_filter"),
        self.hide_filter.is_some().then_some("hide_filter"),
        self.hide_delay_ms.is_some().then_some("hide_delay_ms"),
      ]
      .into_iter()
      .flatten()
      .next();

      if let Some(field) = stray {
        bail!("sets {field} without a source");
      }
    }

    // A move animation has to be undone by the matching move back, or
    // the source ends up parked wherever the last one left it - off
    // screen, and invisible next time it is shown.
    if self.visible.is_none_or(|v| v == Visibility::Toggle)
      && self.show_filter.is_some() != self.hide_filter.is_some()
    {
      bail!(
        "toggles with only one of show_filter and hide_filter - a \
         toggle goes both ways, so it needs both"
      );
    }

    // A scene alongside a source says where the source lives, so the
    // two together are one action, not two.
    let actions = [
      self.url.is_some(),
      self.source.is_some(),
      self.scene.is_some() && self.source.is_none(),
      self.wait_ms.is_some(),
      self.media.is_some(),
      self.output.is_some(),
      self.input.is_some(),
      self.sound.is_some(),
      self.hass.is_some(),
      self.run.is_some(),
    ]
    .into_iter()
    .filter(|set| *set)
    .count();

    match actions {
      1 => Ok(()),
      0 => bail!("sets none of {ACTION_FIELDS}"),
      _ => bail!(
        "sets more than one of {ACTION_FIELDS} - a step does one \
         thing, so split it across steps"
      ),
    }
  }

  /// Rewrites `output = "headphones"` into the device pattern the
  /// `[devices]` table gives that name.
  ///
  /// Aliases only, with no literal fallback: a device name that is
  /// wrong matches nothing, and nothing is exactly what a command
  /// naming a device that is currently unplugged also does - so a typo
  /// would stay invisible until the day you said the words and
  /// wondered why. Requiring the name to exist here moves that to
  /// startup.
  fn resolve_devices(
    &mut self,
    devices: &HashMap<String, String>,
  ) -> Result<()> {
    let fields =
      [("output", &mut self.output), ("input", &mut self.input)];

    for (field, value) in fields {
      let Some(alias) = value else {
        continue;
      };

      let Some(pattern) = devices.get(alias.as_str()) else {
        bail!(
          "names {field} device {alias:?}, which the [devices] table \
           does not have{}",
          known(devices)
        );
      };

      *alias = pattern.clone();
    }

    Ok(())
  }

  /// Checks a `hass` step names an entity id, and that there is a
  /// Home Assistant configured for it to be an entity of.
  ///
  /// At startup rather than at dispatch, for the same reason a device
  /// alias is: Home Assistant answers a call naming an entity it does
  /// not have with a 200 and no state change, so a config that is
  /// wrong here looks exactly like one that works right up until you
  /// say the words.
  fn resolve_hass(&self, hass: &HassConfig) -> Result<()> {
    let Some(entity) = &self.hass else {
      return Ok(());
    };

    hass::validate_entity(entity)?;

    if !hass.configured() {
      bail!(
        "names hass entity {entity:?}, but there is no [hass] table \
         with a url to reach Home Assistant at"
      );
    }

    Ok(())
  }

  /// Expands `~` in `run`, and says at startup when the program is not
  /// where the config says it is.
  ///
  /// A warning rather than an error: unlike a device alias, the path
  /// itself is the whole name, and a binary that is missing today
  /// might be one that is only not installed yet - which is no reason
  /// for every other command in the file to stop working. The warning
  /// is there because the alternative is finding out by saying the
  /// words and getting the failure tone.
  fn resolve_program(&mut self) -> Result<()> {
    let Some(bin) = &mut self.run else {
      return Ok(());
    };

    if bin.trim().is_empty() {
      bail!("has an empty run");
    }

    // launchd does not run a shell, so nothing else would expand it.
    *bin = expand_tilde(bin).to_string_lossy().into_owned();

    if exec::locate(bin).is_none() {
      warn!(
        program = %bin,
        "no such program - launchd gives an agent a PATH of \
         /usr/bin:/bin:/usr/sbin:/sbin and nothing else, so name it in \
         full if it lives anywhere else"
      );
    }

    Ok(())
  }

  fn expand_targets(
    &mut self,
    targets: &HashMap<String, String>,
  ) -> Result<()> {
    // Scene, source and wait take no substitution - a name goes to
    // OBS verbatim.
    let Some(mut url) = self.url.take() else {
      return Ok(());
    };

    for (key, base) in targets {
      url = url.replace(&format!("{{{key}}}"), base.trim_end_matches('/'));
    }

    if url.contains('{') {
      bail!(
        "has an unresolved target in {url:?} - check the [targets] table"
      );
    }

    if !url.starts_with("http") {
      bail!("has a non-http url {url:?}");
    }

    self.url = Some(url);

    Ok(())
  }

  fn target(&self) -> String {
    match self.action() {
      Action::Http { method, url } => format!("{method} {url}"),
      Action::Scene(scene) => format!("obs scene {scene:?}"),
      Action::Media(key) => format!("media key {}", key.as_str()),
      Action::Device { direction, pattern } => {
        format!("{} device {pattern:?}", direction.as_str())
      }
      Action::Sound(name) => format!("play {name}.wav"),
      Action::Hass(call) => {
        format!("hass {} on {:?}", call.service(), call.entity)
      }
      Action::Wait(delay) => format!("wait {}ms", delay.as_millis()),
      Action::Run(program) => {
        // Arguments included: two commands that run the same binary
        // are told apart by nothing else.
        if program.args.is_empty() {
          format!("run {}", program.bin)
        } else {
          format!("run {} {}", program.bin, program.args.join(" "))
        }
      }
      Action::Source(action) => {
        let mut target = match action.scene {
          Some(scene) => format!(
            "obs {} source {:?} in {scene:?}",
            action.visibility.as_str(),
            action.source
          ),
          None => format!(
            "obs {} source {:?}",
            action.visibility.as_str(),
            action.source
          ),
        };

        // Which filter runs depends on which way a toggle goes, so
        // name both rather than guess.
        let filters: Vec<&str> = [action.show_filter, action.hide_filter]
          .into_iter()
          .flatten()
          .collect();

        if !filters.is_empty() {
          target.push_str(&format!(" via {}", filters.join(" / ")));
        }

        target
      }
    }
  }
}

/// The names the `[devices]` table does have, for the error about a
/// name it does not.
fn known(devices: &HashMap<String, String>) -> String {
  if devices.is_empty() {
    return " (there is no [devices] table)".into();
  }

  let mut names: Vec<&str> = devices.keys().map(String::as_str).collect();
  names.sort_unstable();

  format!(" - it has {}", names.join(", "))
}

fn default_wake_model() -> String {
  "~/.config/voice-control/computa.onnx".into()
}

fn default_melspectrogram_model() -> String {
  "~/.config/voice-control/melspectrogram.onnx".into()
}

fn default_embedding_model() -> String {
  "~/.config/voice-control/embedding_model.onnx".into()
}

fn default_stt_model() -> String {
  "~/.config/voice-control/ggml-base.en-q5_1.bin".into()
}

fn default_threshold() -> f32 {
  0.5
}

fn default_patience() -> usize {
  2
}

fn default_method() -> String {
  "POST".into()
}

impl Default for WakeConfig {
  fn default() -> Self {
    Self {
      model: default_wake_model(),
      melspectrogram: default_melspectrogram_model(),
      embedding: default_embedding_model(),
      threshold: default_threshold(),
      patience: default_patience(),
    }
  }
}

impl Default for SttConfig {
  fn default() -> Self {
    Self {
      model: default_stt_model(),
    }
  }
}

impl CommandFile {
  pub fn load(path: &Path) -> Result<Self> {
    let raw = std::fs::read_to_string(path).with_context(|| {
      format!("reading {} (see commands.example.toml)", path.display())
    })?;

    let mut file: CommandFile = toml::from_str(&raw)
      .with_context(|| format!("parsing {}", path.display()))?;

    if file.commands.is_empty() {
      bail!("{} defines no commands", path.display());
    }

    file.resolve()?;

    Ok(file)
  }

  /// Rewrites `{discord}/mute/on` into a full URL using `[targets]`
  /// and `output = "headphones"` into a device pattern using
  /// `[devices]`, and checks every step names exactly one thing to do.
  fn resolve(&mut self) -> Result<()> {
    for command in &mut self.commands {
      let name = command.name.clone();

      match (command.steps.is_empty(), command.step.is_empty()) {
        (false, false) => bail!(
          "command {name:?} sets both steps and a one-step action - \
           put everything in steps, or nothing"
        ),
        (true, true) => bail!("command {name:?} does nothing"),
        _ => {}
      }

      let steps = if command.steps.is_empty() {
        std::slice::from_mut(&mut command.step)
      } else {
        command.steps.as_mut_slice()
      };

      for (index, step) in steps.iter_mut().enumerate() {
        step
          .validate()
          .and_then(|()| step.expand_targets(&self.targets))
          .and_then(|()| step.resolve_devices(&self.devices))
          .and_then(|()| step.resolve_hass(&self.hass))
          .and_then(|()| step.resolve_program())
          .with_context(|| {
            format!("command {name:?}, step {}", index + 1)
          })?;
      }
    }

    Ok(())
  }

  /// Every phrase, for biasing whisper's decoder.
  pub fn vocabulary(&self) -> Vec<String> {
    self
      .commands
      .iter()
      .flat_map(|c| c.phrases.iter().cloned())
      .collect()
  }
}

/// Lowercase, strip punctuation, collapse whitespace.
pub fn normalise(text: &str) -> String {
  let mut out = String::with_capacity(text.len());
  let mut last_was_space = true;

  for ch in text.chars() {
    if ch.is_alphanumeric() {
      out.extend(ch.to_lowercase());
      last_was_space = false;
    } else if !last_was_space {
      out.push(' ');
      last_was_space = true;
    }
  }

  out.trim_end().to_string()
}

/// The transcript with the wake word still attached, plus every
/// word-aligned suffix of it.
///
/// The clip starts before the wake word (the capture buffer is seeded
/// with pre-roll), and whisper has never heard "computa" so it invents
/// a new spelling almost every time - "Computer", "Compute a",
/// "Come puta". Rather than maintain a list of spellings that will
/// always be one surprise behind, drop leading words until something
/// matches. Splitting on word boundaries is what keeps "unmute" from
/// being read as "mute".
pub fn candidates(transcript: &str) -> Vec<String> {
  let normalised = normalise(transcript);
  let words: Vec<&str> = normalised.split_whitespace().collect();

  (0..words.len()).map(|i| words[i..].join(" ")).collect()
}

#[derive(Debug)]
pub struct Match<'a> {
  pub command: &'a Command,
  pub score: f64,
}

/// Exact match first, then best fuzzy match if it is both good enough
/// and clearly ahead of the alternatives.
pub fn match_command<'a>(
  commands: &'a [Command],
  transcript: &str,
) -> Option<Match<'a>> {
  let candidates = candidates(transcript);

  if candidates.is_empty() {
    return None;
  }

  // Candidates run longest first, and the first hit wins. That
  // precedence is what stops "on deafen" - whisper's usual rendering
  // of "undeafen" - from falling through to its own "deafen" suffix
  // and muting when you asked for the opposite.
  for candidate in &candidates {
    for command in commands {
      for phrase in &command.phrases {
        if normalise(phrase) == *candidate {
          return Some(Match {
            command,
            score: 1.0,
          });
        }
      }
    }
  }

  for candidate in &candidates {
    if let Some(hit) = best_for(commands, candidate) {
      return Some(hit);
    }
  }

  None
}

/// Best command for one candidate, if it is both good enough and
/// clearly ahead of the runner-up.
fn best_for<'a>(
  commands: &'a [Command],
  candidate: &str,
) -> Option<Match<'a>> {
  let mut scored: Vec<(f64, &Command)> = commands
    .iter()
    .map(|command| {
      let score = command
        .phrases
        .iter()
        .map(|phrase| jaro_winkler(&normalise(phrase), candidate))
        .fold(0.0_f64, f64::max);

      (score, command)
    })
    .collect();

  scored.sort_by(|a, b| b.0.total_cmp(&a.0));

  let (score, command) = *scored.first()?;
  let runner_up = scored.get(1).map_or(0.0, |&(score, _)| score);

  if score < MATCH_THRESHOLD {
    return None;
  }

  // Two commands scoring within a hair of each other means we
  // genuinely cannot tell them apart, and guessing wrong is worse
  // than doing nothing.
  if score - runner_up < MATCH_MARGIN {
    warn!(candidate, score, runner_up, "ambiguous command, ignoring");
    return None;
  }

  Some(Match { command, score })
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Parsed rather than built by hand, so a new optional field does
  /// not mean editing every fixture.
  fn commands() -> Vec<Command> {
    let raw = r#"
      [[commands]]
      name = "mute"
      phrases = ["mute", "mute me"]
      url = "http://x/mute/on"

      [[commands]]
      name = "unmute"
      phrases = ["unmute", "un mute"]
      url = "http://x/mute/off"

      [[commands]]
      name = "deafen"
      phrases = ["deafen"]
      url = "http://x/deaf/on"
    "#;

    toml::from_str::<CommandFile>(raw).unwrap().commands
  }

  #[test]
  fn normalises_case_and_punctuation() {
    assert_eq!(normalise("  Computa, MUTE! "), "computa mute");
  }

  #[test]
  fn candidates_are_word_aligned_suffixes() {
    assert_eq!(
      candidates("Compute a mute!"),
      vec!["compute a mute", "a mute", "mute"]
    );
  }

  #[test]
  fn matches_exact_phrases() {
    let commands = commands();
    let hit = match_command(&commands, "Computa, mute.").unwrap();

    assert_eq!(hit.command.name, "mute");
  }

  #[test]
  fn matches_close_transcriptions() {
    let commands = commands();
    let hit = match_command(&commands, "computa deafin").unwrap();

    assert_eq!(hit.command.name, "deafen");
  }

  #[test]
  fn rejects_unrelated_speech() {
    let commands = commands();

    assert!(match_command(&commands, "computa banana").is_none());
    assert!(
      match_command(&commands, "Computer, what is the weather?").is_none()
    );
  }

  #[test]
  fn mute_and_unmute_are_not_confused() {
    let commands = commands();

    // Every spelling of the wake word that whisper actually produced
    // for `say`-generated samples during development.
    let unmute = [
      "computa unmute",
      "Compute unmute.",
      "Computer, unmute.",
      "Compute a unmute.",
      "Come puta unmute",
    ];

    for transcript in unmute {
      let hit = match_command(&commands, transcript)
        .unwrap_or_else(|| panic!("no match for {transcript:?}"));
      assert_eq!(hit.command.name, "unmute", "for {transcript:?}");
    }

    for transcript in ["computa mute", "Compute a mute.", "Computer mute"]
    {
      let hit = match_command(&commands, transcript)
        .unwrap_or_else(|| panic!("no match for {transcript:?}"));
      assert_eq!(hit.command.name, "mute", "for {transcript:?}");
    }
  }

  #[test]
  fn a_bare_wake_word_matches_nothing() {
    let commands = commands();

    assert!(match_command(&commands, "Computer.").is_none());
  }

  /// Every phrase in the shipped example must resolve to its own
  /// command. Phrases are easy to add and collisions are invisible
  /// until you say the words and the wrong thing happens - "screen"
  /// for the main scene against "show me" for the camera is exactly
  /// the kind of pair that goes wrong.
  #[test]
  fn every_example_phrase_maps_to_its_own_command() {
    let raw = std::fs::read_to_string("commands.example.toml").unwrap();
    let mut file: CommandFile = toml::from_str(&raw).unwrap();
    file.resolve().unwrap();

    let mut problems = Vec::new();

    for command in &file.commands {
      for phrase in &command.phrases {
        match match_command(&file.commands, phrase) {
          Some(hit) if hit.command.name == command.name => {}
          Some(hit) => problems.push(format!(
            "{:?} ({}) matched {:?}",
            phrase, command.name, hit.command.name
          )),
          None => problems.push(format!(
            "{:?} ({}) matched nothing",
            phrase, command.name
          )),
        }
      }
    }

    assert!(problems.is_empty(), "{}", problems.join("\n"));
  }

  #[test]
  fn a_media_step_names_the_key() {
    let raw = r#"
      [[commands]]
      name = "skip"
      phrases = ["skip"]
      media = "next"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();
    file.resolve().unwrap();

    assert!(matches!(
      file.commands[0].steps()[0].action(),
      Action::Media(MediaKey::Next)
    ));
    assert_eq!(file.commands[0].target(), "media key next");
  }

  /// The keyboard has one play/pause key, so a config that spells it
  /// either way has to reach the same one.
  #[test]
  fn play_and_pause_are_the_same_key() {
    let raw = r#"
      [[commands]]
      name = "play"
      phrases = ["play"]
      media = "play"

      [[commands]]
      name = "pause"
      phrases = ["pause"]
      media = "pause"
    "#;

    let file: CommandFile = toml::from_str(raw).unwrap();

    assert_eq!(file.commands[0].step.media, Some(MediaKey::PlayPause));
    assert_eq!(file.commands[1].step.media, Some(MediaKey::PlayPause));
  }

  /// The alias is what the config says; the device pattern is what
  /// reaches the HAL. Nothing downstream of the load should still be
  /// holding the alias.
  #[test]
  fn a_device_step_resolves_its_alias() {
    let raw = r#"
      [devices]
      headphones = "AirPods"
      microphone = "Wireless microphone"

      [[commands]]
      name = "headphones"
      phrases = ["headphones"]
      output = "headphones"

      [[commands]]
      name = "wireless mic"
      phrases = ["wireless mic"]
      input = "microphone"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();
    file.resolve().unwrap();

    assert!(matches!(
      file.commands[0].steps()[0].action(),
      Action::Device {
        direction: Direction::Output,
        pattern: "AirPods"
      }
    ));
    assert!(matches!(
      file.commands[1].steps()[0].action(),
      Action::Device {
        direction: Direction::Input,
        pattern: "Wireless microphone"
      }
    ));

    assert_eq!(file.commands[0].target(), "output device \"AirPods\"");
    assert_eq!(
      file.commands[1].target(),
      "input device \"Wireless microphone\""
    );
  }

  /// A plug command is said the same way twice, so a step that names
  /// nothing but the entity toggles it.
  #[test]
  fn a_hass_step_defaults_to_toggling() {
    let raw = r#"
      [hass]
      url = "https://hass.lan"
      token = "x"

      [[commands]]
      name = "speakers"
      phrases = ["speakers"]
      hass = "switch.desk_speakers"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();
    file.resolve().unwrap();

    let Action::Hass(call) = file.commands[0].steps()[0].action() else {
      panic!("not a hass action");
    };

    assert_eq!(call.entity, "switch.desk_speakers");
    assert_eq!(call.service(), "switch.toggle");
    assert_eq!(
      file.commands[0].target(),
      "hass switch.toggle on \"switch.desk_speakers\""
    );
  }

  #[test]
  fn a_hass_step_carries_its_service_and_data() {
    let raw = r#"
      [hass]
      url = "https://hass.lan"
      token = "x"

      [[commands]]
      name = "desk lamp"
      phrases = ["desk lamp"]
      hass = "light.desk"
      service = "turn_on"
      data = { brightness = 128 }
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();
    file.resolve().unwrap();

    let Action::Hass(call) = file.commands[0].steps()[0].action() else {
      panic!("not a hass action");
    };

    assert_eq!(call.service(), "light.turn_on");
    assert_eq!(call.data["brightness"], 128);
  }

  /// The one thing a bare service cannot express: the services that
  /// belong to no domain in particular.
  #[test]
  fn a_hass_service_can_name_its_own_domain() {
    let raw = r#"
      [hass]
      url = "https://hass.lan"
      token = "x"

      [[commands]]
      name = "speakers off"
      phrases = ["speakers off"]
      hass = "switch.desk_speakers"
      service = "homeassistant.turn_off"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();
    file.resolve().unwrap();

    let Action::Hass(call) = file.commands[0].steps()[0].action() else {
      panic!("not a hass action");
    };

    assert_eq!(call.service(), "homeassistant.turn_off");
  }

  /// Home Assistant answers a call naming an entity it does not have
  /// with a 200 and no state change, so nothing downstream of the load
  /// would ever notice this.
  #[test]
  fn rejects_a_hass_entity_without_a_domain() {
    let raw = r#"
      [hass]
      url = "https://hass.lan"
      token = "x"

      [[commands]]
      name = "speakers"
      phrases = ["speakers"]
      hass = "desk_speakers"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();
    let why = format!("{:#}", file.resolve().unwrap_err());

    assert!(why.contains("desk_speakers"), "{why}");
    assert!(why.contains("switch.desk_speakers"), "{why}");
  }

  #[test]
  fn rejects_a_hass_step_without_a_hass_table() {
    let raw = r#"
      [[commands]]
      name = "speakers"
      phrases = ["speakers"]
      hass = "switch.desk_speakers"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();
    let why = format!("{:#}", file.resolve().unwrap_err());

    assert!(why.contains("[hass] table"), "{why}");
  }

  /// Silently ignoring them would leave a config that does not do what
  /// it plainly reads as - a `turn_on` that never reaches anything.
  #[test]
  fn rejects_a_service_without_a_hass_entity() {
    let raw = r#"
      [[commands]]
      name = "confused"
      phrases = ["x"]
      url = "http://example/x"
      service = "turn_on"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();
    let why = format!("{:#}", file.resolve().unwrap_err());

    assert!(why.contains("service"), "{why}");
  }

  #[test]
  fn rejects_a_step_that_sets_both_hass_and_output() {
    let raw = r#"
      [devices]
      speakers = "CalDigit"

      [hass]
      url = "https://hass.lan"
      token = "x"

      [[commands]]
      name = "confused"
      phrases = ["x"]
      hass = "switch.desk_speakers"
      output = "speakers"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();

    assert!(file.resolve().is_err());
  }

  #[test]
  fn a_run_step_carries_its_arguments() {
    let raw = r#"
      [[commands]]
      name = "lights"
      phrases = ["lights"]
      run = "/bin/echo"
      args = ["toggle", "desk lamp"]
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();
    file.resolve().unwrap();

    let Action::Run(program) = file.commands[0].steps()[0].action() else {
      panic!("not a run action");
    };

    assert_eq!(program.bin, "/bin/echo");
    assert_eq!(program.args, ["toggle", "desk lamp"]);
    assert_eq!(program.timeout, DEFAULT_RUN_TIMEOUT);
    assert_eq!(
      file.commands[0].target(),
      "run /bin/echo toggle desk lamp"
    );
  }

  #[test]
  fn a_run_step_takes_a_timeout_and_an_environment() {
    let raw = r#"
      [[commands]]
      name = "lights"
      phrases = ["lights"]
      run = "/bin/echo"
      timeout_ms = 2500
      env = { LIGHTS_HOST = "10.0.0.4" }
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();
    file.resolve().unwrap();

    let Action::Run(program) = file.commands[0].steps()[0].action() else {
      panic!("not a run action");
    };

    assert_eq!(program.timeout, Duration::from_millis(2500));
    assert_eq!(program.env["LIGHTS_HOST"], "10.0.0.4");
  }

  /// launchd does not run a shell, so a `~` that nothing expanded
  /// would be looked for as a directory literally called "~".
  #[test]
  fn a_run_step_expands_a_tilde() {
    let raw = r#"
      [[commands]]
      name = "lights"
      phrases = ["lights"]
      run = "~/bin/lights"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();
    file.resolve().unwrap();

    let bin = file.commands[0].steps()[0].run.as_deref().unwrap();

    assert!(!bin.starts_with('~'), "{bin}");
    assert!(bin.ends_with("/bin/lights"), "{bin}");
  }

  /// Silently ignoring them would leave a config that does not do what
  /// it plainly reads as - the arguments would simply never arrive.
  #[test]
  fn rejects_args_without_a_run() {
    let raw = r#"
      [[commands]]
      name = "confused"
      phrases = ["x"]
      url = "http://example/x"
      args = ["toggle"]
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();
    let why = format!("{:#}", file.resolve().unwrap_err());

    assert!(why.contains("args"), "{why}");
  }

  #[test]
  fn rejects_a_step_that_sets_both_run_and_url() {
    let raw = r#"
      [[commands]]
      name = "confused"
      phrases = ["x"]
      run = "/bin/echo"
      url = "http://example/x"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();

    assert!(file.resolve().is_err());
  }

  /// A command that makes its own noise is the one kind that should
  /// not also get the generic success chirp.
  #[test]
  fn a_sound_step_answers_for_itself() {
    let raw = r#"
      [[commands]]
      name = "ping"
      phrases = ["ping", "hello"]
      sound = "ping"

      [[commands]]
      name = "mute"
      phrases = ["mute"]
      url = "http://x/mute/on"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();
    file.resolve().unwrap();

    assert!(matches!(
      file.commands[0].steps()[0].action(),
      Action::Sound("ping")
    ));
    assert_eq!(file.commands[0].target(), "play ping.wav");
    assert!(file.commands[0].answers_for_itself());
    assert!(!file.commands[1].answers_for_itself());
  }

  /// "hi" is two letters, which is short enough for the fuzzy matcher
  /// to find it inside things that are not it. Exact matching runs
  /// first and over suffixes, which is what keeps it in its lane.
  #[test]
  fn a_two_letter_phrase_does_not_swallow_longer_ones() {
    let raw = std::fs::read_to_string("commands.example.toml").unwrap();
    let mut file: CommandFile = toml::from_str(&raw).unwrap();
    file.resolve().unwrap();

    for transcript in ["computa hi", "computa hello", "computa ping"] {
      let hit = match_command(&file.commands, transcript)
        .unwrap_or_else(|| panic!("no match for {transcript:?}"));
      assert_eq!(hit.command.name, "ping", "for {transcript:?}");
    }

    for transcript in [
      "computa hide ps5",
      "computa headphones",
      "computa hide playstation",
    ] {
      let hit = match_command(&file.commands, transcript)
        .unwrap_or_else(|| panic!("no match for {transcript:?}"));
      assert_ne!(hit.command.name, "ping", "for {transcript:?}");
    }
  }

  /// A device that is merely unplugged matches nothing too, so a
  /// misspelt alias would otherwise stay invisible until the day you
  /// said the words.
  #[test]
  fn an_unknown_device_alias_names_the_ones_that_exist() {
    let raw = r#"
      [devices]
      speakers = "CalDigit"
      headphones = "AirPods"

      [[commands]]
      name = "headphones"
      phrases = ["headphones"]
      output = "headfones"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();
    let why = format!("{:#}", file.resolve().unwrap_err());

    assert!(why.contains("headfones"), "{why}");
    assert!(why.contains("headphones, speakers"), "{why}");
  }

  #[test]
  fn rejects_a_device_step_without_a_devices_table() {
    let raw = r#"
      [[commands]]
      name = "headphones"
      phrases = ["headphones"]
      output = "headphones"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();
    let why = format!("{:#}", file.resolve().unwrap_err());

    assert!(why.contains("no [devices] table"), "{why}");
  }

  /// Moving the output and the input are two things, and a config that
  /// asks for both in one step has no say in which order they happen.
  #[test]
  fn rejects_a_step_that_sets_both_output_and_input() {
    let raw = r#"
      [devices]
      headphones = "AirPods"

      [[commands]]
      name = "headphones"
      phrases = ["headphones"]
      output = "headphones"
      input = "headphones"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();

    assert!(file.resolve().is_err());
  }

  #[test]
  fn rejects_a_command_with_both_output_and_scene() {
    let raw = r#"
      [devices]
      speakers = "CalDigit"

      [[commands]]
      name = "confused"
      phrases = ["x"]
      output = "speakers"
      scene = "Main Screen"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();

    assert!(file.resolve().is_err());
  }

  #[test]
  fn rejects_a_command_with_both_media_and_scene() {
    let raw = r#"
      [[commands]]
      name = "confused"
      phrases = ["x"]
      media = "next"
      scene = "Main Screen"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();

    assert!(file.resolve().is_err());
  }

  #[test]
  fn rejects_an_unknown_media_key() {
    let raw = r#"
      [[commands]]
      name = "eject"
      phrases = ["eject"]
      media = "eject"
    "#;

    assert!(toml::from_str::<CommandFile>(raw).is_err());
  }

  /// "playstation" and "play" both start with the same four letters,
  /// and the media commands put them one word apart. Exact matches run
  /// before any fuzzy scoring, which is what keeps saying either one
  /// off the other's command.
  #[test]
  fn playstation_is_not_heard_as_play() {
    let raw = std::fs::read_to_string("commands.example.toml").unwrap();
    let mut file: CommandFile = toml::from_str(&raw).unwrap();
    file.resolve().unwrap();

    for transcript in [
      "computa playstation",
      "Compute a play station.",
      "computer ps5",
    ] {
      let hit = match_command(&file.commands, transcript)
        .unwrap_or_else(|| panic!("no match for {transcript:?}"));
      assert_eq!(hit.command.name, "ps5", "for {transcript:?}");
    }

    for transcript in ["computa play", "Computer, pause.", "computa skip"]
    {
      let hit = match_command(&file.commands, transcript)
        .unwrap_or_else(|| panic!("no match for {transcript:?}"));
      assert!(
        hit.command.name != "ps5",
        "{transcript:?} matched {}",
        hit.command.name
      );
    }
  }

  #[test]
  fn rejects_a_command_with_both_url_and_scene() {
    let raw = r#"
      [[commands]]
      name = "confused"
      phrases = ["x"]
      url = "http://example/x"
      scene = "Main Screen"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();

    assert!(file.resolve().is_err());
  }

  #[test]
  fn rejects_a_command_with_neither_url_nor_scene() {
    let raw = r#"
      [[commands]]
      name = "empty"
      phrases = ["x"]
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();

    assert!(file.resolve().is_err());
  }

  fn source_action(raw: &str) -> SourceAction<'_> {
    // Leaked so the borrow outlives the parse, which is fine for the
    // life of a test.
    let file: &'static mut CommandFile =
      Box::leak(Box::new(toml::from_str(raw).unwrap()));
    file.resolve().unwrap();

    match file.commands[0].steps()[0].action() {
      Action::Source(action) => action,
      _ => panic!("not a source action"),
    }
  }

  #[test]
  fn a_source_defaults_to_toggling_in_the_current_scene() {
    let action = source_action(
      r#"
      [[commands]]
      name = "ps5"
      phrases = ["ps5"]
      source = "Cam Link Screen"
    "#,
    );

    assert_eq!(action.source, "Cam Link Screen");
    assert_eq!(action.scene, None);
    assert_eq!(action.visibility, Visibility::Toggle);
    assert_eq!(action.show_filter, None);
    assert_eq!(action.hide_filter, None);
  }

  /// A scene alongside a source says where the source lives - it is
  /// not a scene switch, and must not be read as one.
  #[test]
  fn a_source_can_name_the_scene_it_lives_in() {
    let action = source_action(
      r#"
      [[commands]]
      name = "show ps5"
      phrases = ["show ps5"]
      source = "Cam Link Screen"
      scene = "Main Screen"
      visible = "show"
    "#,
    );

    assert_eq!(action.scene, Some("Main Screen"));
    assert_eq!(action.visibility, Visibility::Show);
  }

  #[test]
  fn move_filters_carry_through_with_a_default_delay() {
    let action = source_action(
      r#"
      [[commands]]
      name = "ps5"
      phrases = ["ps5"]
      source = "Cam Link Screen"
      scene = "Main Screen"
      show_filter = "Move-In"
      hide_filter = "Move-Out"
    "#,
    );

    assert_eq!(action.show_filter, Some("Move-In"));
    assert_eq!(action.hide_filter, Some("Move-Out"));
    assert_eq!(action.hide_delay, DEFAULT_HIDE_DELAY);
  }

  #[test]
  fn a_hide_delay_is_taken_in_milliseconds() {
    let action = source_action(
      r#"
      [[commands]]
      name = "hide ps5"
      phrases = ["hide ps5"]
      source = "Cam Link Screen"
      visible = "hide"
      hide_filter = "Move-Out"
      hide_delay_ms = 900
    "#,
    );

    assert_eq!(action.hide_delay, Duration::from_millis(900));
  }

  /// A one-way toggle parks the source wherever the move left it -
  /// off screen, and invisible the next time it is shown.
  #[test]
  fn rejects_a_toggle_with_only_one_move_filter() {
    let raw = r#"
      [[commands]]
      name = "ps5"
      phrases = ["ps5"]
      source = "Cam Link Screen"
      show_filter = "Move-In"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();

    assert!(file.resolve().is_err());
  }

  /// One-way commands are the exception: a `show` never has to undo
  /// itself, so it needs no hide filter.
  #[test]
  fn allows_one_filter_when_the_command_only_goes_one_way() {
    let raw = r#"
      [[commands]]
      name = "show ps5"
      phrases = ["show ps5"]
      source = "Cam Link Screen"
      visible = "show"
      show_filter = "Move-In"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();

    assert!(file.resolve().is_ok());
  }

  #[test]
  fn rejects_a_filter_without_a_source() {
    let raw = r#"
      [[commands]]
      name = "camera"
      phrases = ["camera"]
      scene = "Camera Screen"
      show_filter = "Move-In"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();

    assert!(file.resolve().is_err());
  }

  #[test]
  fn on_and_off_read_as_show_and_hide() {
    let raw = r#"
      [[commands]]
      name = "on"
      phrases = ["x"]
      source = "Cam Link Screen"
      visible = "on"

      [[commands]]
      name = "off"
      phrases = ["y"]
      source = "Cam Link Screen"
      visible = "off"
    "#;

    let file: CommandFile = toml::from_str(raw).unwrap();

    assert_eq!(file.commands[0].step.visible, Some(Visibility::Show));
    assert_eq!(file.commands[1].step.visible, Some(Visibility::Hide));
  }

  #[test]
  fn rejects_an_unknown_visibility() {
    let raw = r#"
      [[commands]]
      name = "ps5"
      phrases = ["ps5"]
      source = "Cam Link Screen"
      visible = "flicker"
    "#;

    assert!(toml::from_str::<CommandFile>(raw).is_err());
  }

  /// Silently ignoring it would leave a config that does not do what
  /// it plainly reads as.
  #[test]
  fn rejects_visible_without_a_source() {
    let raw = r#"
      [[commands]]
      name = "camera"
      phrases = ["camera"]
      scene = "Camera Screen"
      visible = "show"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();

    assert!(file.resolve().is_err());
  }

  #[test]
  fn rejects_a_command_with_both_url_and_source() {
    let raw = r#"
      [[commands]]
      name = "confused"
      phrases = ["x"]
      url = "http://example/x"
      source = "Cam Link Screen"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();

    assert!(file.resolve().is_err());
  }

  /// The flow the ps5 commands use: get to the scene, then animate
  /// the source in once you are there.
  #[test]
  fn a_flow_runs_its_steps_in_order() {
    let raw = r#"
      [[commands]]
      name = "show ps5"
      phrases = ["show ps5"]

      [[commands.steps]]
      scene = "Main Screen"

      [[commands.steps]]
      wait_ms = 200

      [[commands.steps]]
      source = "Cam Link Screen"
      scene = "Main Screen"
      visible = "show"
      show_filter = "Move-In"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();
    file.resolve().unwrap();

    let steps = file.commands[0].steps();
    assert_eq!(steps.len(), 3);

    assert!(matches!(steps[0].action(), Action::Scene("Main Screen")));
    assert!(matches!(
      steps[1].action(),
      Action::Wait(delay) if delay == Duration::from_millis(200)
    ));

    let Action::Source(action) = steps[2].action() else {
      panic!("last step is not a source action");
    };
    assert_eq!(action.source, "Cam Link Screen");
    assert_eq!(action.show_filter, Some("Move-In"));

    assert_eq!(
      file.commands[0].target(),
      "obs scene \"Main Screen\" -> wait 200ms -> obs show source \
       \"Cam Link Screen\" in \"Main Screen\" via Move-In"
    );
  }

  /// The one-step shorthand is the same thing as a one-entry list, so
  /// every command that predates flows keeps working untouched.
  #[test]
  fn a_plain_command_is_a_one_step_flow() {
    let commands = commands();

    assert_eq!(commands[0].steps().len(), 1);
    assert!(matches!(
      commands[0].steps()[0].action(),
      Action::Http { .. }
    ));
  }

  #[test]
  fn targets_are_expanded_inside_steps() {
    let raw = r#"
      [targets]
      discord = "http://127.0.0.1:8009/v1/voice/canary/"

      [[commands]]
      name = "brb"
      phrases = ["brb"]

      [[commands.steps]]
      url = "{discord}/mute/on"

      [[commands.steps]]
      scene = "Camera Screen"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();
    file.resolve().unwrap();

    assert_eq!(
      file.commands[0].steps()[0].url.as_deref().unwrap(),
      "http://127.0.0.1:8009/v1/voice/canary/mute/on"
    );
  }

  /// Naming the error's step is the difference between a config you
  /// can fix and one you have to bisect.
  #[test]
  fn a_bad_step_says_which_step_it_is() {
    let raw = r#"
      [[commands]]
      name = "brb"
      phrases = ["brb"]

      [[commands.steps]]
      scene = "Camera Screen"

      [[commands.steps]]
      url = "{typo}/mute/on"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();
    let why = format!("{:#}", file.resolve().unwrap_err());

    assert!(why.contains("step 2"), "{why}");
  }

  #[test]
  fn rejects_a_step_that_does_two_things() {
    let raw = r#"
      [[commands]]
      name = "confused"
      phrases = ["x"]

      [[commands.steps]]
      scene = "Main Screen"
      wait_ms = 200
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();

    assert!(file.resolve().is_err());
  }

  /// Both forms at once reads as if the shorthand were a fourth step,
  /// and there is no order that makes that unambiguous.
  #[test]
  fn rejects_steps_alongside_the_shorthand() {
    let raw = r#"
      [[commands]]
      name = "confused"
      phrases = ["x"]
      scene = "Camera Screen"

      [[commands.steps]]
      scene = "Main Screen"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();

    assert!(file.resolve().is_err());
  }

  #[test]
  fn rejects_a_command_that_does_nothing() {
    let raw = r#"
      [[commands]]
      name = "empty"
      phrases = ["x"]
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();

    assert!(file.resolve().is_err());
  }

  #[test]
  fn expands_targets() {
    let raw = r#"
      [targets]
      discord = "http://127.0.0.1:8009/v1/voice/canary/"

      [[commands]]
      name = "mute"
      phrases = ["mute"]
      url = "{discord}/mute/on"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();
    file.resolve().unwrap();

    assert_eq!(
      file.commands[0].steps()[0].url.as_deref().unwrap(),
      "http://127.0.0.1:8009/v1/voice/canary/mute/on"
    );
  }

  #[test]
  fn rejects_an_unresolved_target() {
    let raw = r#"
      [[commands]]
      name = "mute"
      phrases = ["mute"]
      url = "{typo}/mute/on"
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();

    assert!(file.resolve().is_err());
  }
}
