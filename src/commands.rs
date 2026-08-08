use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use strsim::jaro_winkler;

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
  #[serde(default)]
  pub targets: HashMap<String, String>,
  #[serde(default)]
  pub obs: ObsConfig,
  #[serde(default)]
  pub commands: Vec<Command>,
}

#[derive(Debug, Deserialize)]
pub struct WakeConfig {
  #[serde(default = "default_wake_model")]
  pub model: String,
  #[serde(default = "default_threshold")]
  pub threshold: f32,
  #[serde(default = "default_avg_threshold")]
  pub avg_threshold: f32,
  /// Fire as soon as enough partial scores agree, rather than waiting
  /// for the score to peak. Costs a little accuracy and buys back a
  /// few hundred milliseconds of latency, which matters more here
  /// because the transcription stage re-checks the result anyway.
  #[serde(default = "default_eager")]
  pub eager: bool,
}

#[derive(Debug, Deserialize)]
pub struct SttConfig {
  #[serde(default = "default_stt_model")]
  pub model: String,
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
    }
  }
}

/// Long enough for the move animations these are written against, and
/// what the equivalent Companion button waits.
const DEFAULT_HIDE_DELAY: Duration = Duration::from_millis(350);

/// What one step actually does.
pub enum Action<'a> {
  Http { method: &'a str, url: &'a str },
  Scene(&'a str),
  Source(SourceAction<'a>),
  Media(MediaKey),
  Wait(Duration),
}

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
    // Checked ahead of the rest rather than as a fifth arm: a media key
    // takes none of the other fields, so it has nothing to be
    // ambiguous with.
    if let Some(key) = self.media {
      return Action::Media(key);
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
      && self.visible.is_none()
      && self.show_filter.is_none()
      && self.hide_filter.is_none()
      && self.hide_delay_ms.is_none()
  }

  /// Checks the step names exactly one thing to do, and that the
  /// source-only fields have a source to act on.
  fn validate(&self) -> Result<()> {
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
    ]
    .into_iter()
    .filter(|set| *set)
    .count();

    match actions {
      1 => Ok(()),
      0 => bail!("sets none of url, scene, source, media or wait_ms"),
      _ => bail!(
        "sets more than one of url, scene, source, media and wait_ms - \
         a step does one thing, so split it across steps"
      ),
    }
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
      Action::Wait(delay) => format!("wait {}ms", delay.as_millis()),
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

fn default_wake_model() -> String {
  "~/.config/voice-control/computa.rpw".into()
}

fn default_stt_model() -> String {
  "~/.config/voice-control/ggml-base.en-q5_1.bin".into()
}

fn default_threshold() -> f32 {
  0.5
}

fn default_avg_threshold() -> f32 {
  0.2
}

fn default_eager() -> bool {
  true
}

fn default_method() -> String {
  "POST".into()
}

impl Default for WakeConfig {
  fn default() -> Self {
    Self {
      model: default_wake_model(),
      threshold: default_threshold(),
      avg_threshold: default_avg_threshold(),
      eager: default_eager(),
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

    file.expand_targets()?;

    Ok(file)
  }

  /// Rewrites `{discord}/mute/on` into a full URL using `[targets]`,
  /// and checks every step names exactly one thing to do.
  fn expand_targets(&mut self) -> Result<()> {
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
    file.expand_targets().unwrap();

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
    file.expand_targets().unwrap();

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

    assert!(file.expand_targets().is_err());
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
    file.expand_targets().unwrap();

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

    assert!(file.expand_targets().is_err());
  }

  #[test]
  fn rejects_a_command_with_neither_url_nor_scene() {
    let raw = r#"
      [[commands]]
      name = "empty"
      phrases = ["x"]
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();

    assert!(file.expand_targets().is_err());
  }

  fn source_action(raw: &str) -> SourceAction<'_> {
    // Leaked so the borrow outlives the parse, which is fine for the
    // life of a test.
    let file: &'static mut CommandFile =
      Box::leak(Box::new(toml::from_str(raw).unwrap()));
    file.expand_targets().unwrap();

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

    assert!(file.expand_targets().is_err());
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

    assert!(file.expand_targets().is_ok());
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

    assert!(file.expand_targets().is_err());
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

    assert!(file.expand_targets().is_err());
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

    assert!(file.expand_targets().is_err());
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
    file.expand_targets().unwrap();

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
    file.expand_targets().unwrap();

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
    let why = format!("{:#}", file.expand_targets().unwrap_err());

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

    assert!(file.expand_targets().is_err());
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

    assert!(file.expand_targets().is_err());
  }

  #[test]
  fn rejects_a_command_that_does_nothing() {
    let raw = r#"
      [[commands]]
      name = "empty"
      phrases = ["x"]
    "#;

    let mut file: CommandFile = toml::from_str(raw).unwrap();

    assert!(file.expand_targets().is_err());
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
    file.expand_targets().unwrap();

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

    assert!(file.expand_targets().is_err());
  }
}
