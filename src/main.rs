use std::io::IsTerminal;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use envconfig::Envconfig;
use tracing::{error, info, warn};
use tracing_subscriber::{
  EnvFilter, Registry, fmt, prelude::__tracing_subscriber_SubscriberExt,
};

use crate::audio::{SAMPLE_RATE, capture, ms_to_samples, samples_to_ms};
use crate::commands::{Command, CommandFile, match_command};
use crate::config::Config;
use crate::dispatch::Dispatcher;
use crate::feedback::Feedback;
use crate::pipeline::Pipeline;
use crate::status::Status;
use crate::stt::Transcriber;
use crate::stt::whisper::WhisperTranscriber;
use crate::wake::Detector;
use crate::wake::rustpotter::RustpotterDetector;

pub mod audio;
pub mod commands;
pub mod config;
pub mod devices;
pub mod dispatch;
pub mod feedback;
pub mod media;
pub mod obs;
pub mod pipeline;
pub mod status;
pub mod stt;
pub mod tray;
pub mod wake;

/// Silence appended by `replay`, standing in for the audio a live
/// microphone would keep producing after the command.
const REPLAY_TRAILING_MS: usize = 2000;

const USAGE: &str = "\
usage: voice-control [command]

  (none)              run the daemon
  devices             list audio devices, with the [devices] aliases
  output <name>       switch the default output - an alias or part of
                      a device name
  input <name>        switch the default input, named the same way
  listen              print wake word detections only, no STT or HTTP
  transcribe <file>   transcribe a 16kHz mono wav and match it
  run <phrase>        match a phrase and dispatch it, as if you said it
  replay <file>       run a 16kHz mono wav through the whole pipeline
  obs [scene]         list obs scenes, or switch to one
  obs sources [scene] list the sources in a scene
  obs filters [scene] list the filters on a scene or source
  obs show|hide|toggle <source> [scene]
                      flip one source, without its filters
  media <key>         press a media key: play_pause, next or previous
";

/// Not `#[tokio::main]`: the menu bar needs the main thread for
/// AppKit's event loop, so the runtime is built by hand and the
/// pipeline runs on it while the main thread belongs to the tray.
/// Every subcommand is headless and just blocks on the runtime.
fn main() -> Result<()> {
  let config = Config::init_from_env()?;

  // ort logs the whole ONNX graph optimisation at info, which is a
  // few hundred lines every startup and would bury the log file.
  let env_filter = EnvFilter::try_from_env("DSTN_LOG")
    .unwrap_or(EnvFilter::new("info,ort=warn"));
  // launchd redirects stdout to a file, where colour codes are just
  // noise to grep through.
  let fmt_layer = fmt::layer()
    .with_target(false)
    .with_ansi(std::io::stdout().is_terminal());
  let subscriber = Registry::default().with(env_filter).with(fmt_layer);
  tracing::subscriber::set_global_default(subscriber)
    .expect("Failed to initalize global tracing subscriber");

  let args: Vec<String> = std::env::args().skip(1).collect();

  let result = match args.first().map(String::as_str) {
    None => run(config),
    Some("devices") => devices(&config),
    Some("output") => {
      switch(&config, devices::Direction::Output, args.get(1))
    }
    Some("input") => {
      switch(&config, devices::Direction::Input, args.get(1))
    }
    Some("listen") => blocking(listen(config)),
    Some("transcribe") => blocking(transcribe(config, args.get(1))),
    Some("run") => blocking(run_phrase(config, args.get(1))),
    Some("replay") => blocking(replay(config, args.get(1))),
    Some("obs") => blocking(obs(config, &args[1..])),
    Some("media") => media_key(args.get(1)),
    Some("-h" | "--help" | "help") => {
      print!("{USAGE}");
      Ok(())
    }
    Some(other) => {
      print!("{USAGE}");
      bail!("unknown command {other:?}");
    }
  };

  if let Err(why) = result {
    error!(error = ?why, "fatal");
    return Err(why);
  }

  Ok(())
}

/// Builds the runtime, starts the pipeline on it, and - unless the
/// tray is switched off - hands the main thread to AppKit.
fn run(config: Config) -> Result<()> {
  let runtime = tokio::runtime::Builder::new_multi_thread()
    .enable_all()
    .build()
    .context("building the tokio runtime")?;

  // `capture::start` spawns its resampler, and the pipeline's
  // transcription hop wants a blocking pool, so all of this has to be
  // constructed with the runtime entered rather than merely alive.
  let _guard = runtime.enter();

  let status = Arc::new(Status::new());
  let path = config.resolved_config_path();
  let file = CommandFile::load(&path)?;

  info!(
    config = %path.display(),
    commands = file.commands.len(),
    "loaded configuration"
  );

  let detector = RustpotterDetector::load(
    &file.wake.model,
    file.wake.threshold,
    file.wake.avg_threshold,
    file.wake.eager,
  )?;
  let transcriber =
    WhisperTranscriber::load(&file.stt.model, &file.vocabulary())?;

  let (capture, audio) = capture::start(&config.input_device)?;
  status.set_device(&capture.device);

  let pipeline = Pipeline::new(
    Box::new(detector) as Box<dyn Detector>,
    Box::new(transcriber) as Box<dyn Transcriber>,
    file.commands,
    Feedback::new(&config.sounds_dir),
    file.obs,
    &file.listen,
    Arc::clone(&status),
  )?;

  if !config.tray {
    return runtime.block_on(async move {
      // Dropping the handle closes the stream, so it has to outlive
      // the pipeline rather than the setup above.
      let _capture = capture;

      tokio::select! {
        result = pipeline.run(audio) => result,
        _ = tokio::signal::ctrl_c() => {
          info!("shutting down");
          Ok(())
        }
      }
    });
  }

  // A thread of its own rather than a task: neither the wake word
  // detector nor the VAD is `Sync`, so the pipeline future cannot be
  // handed to the work-stealing scheduler. Driving it with
  // `Handle::block_on` keeps it on one thread - which is where it ran
  // before the tray existed - while everything it spawns (the
  // resampler, transcription, HTTP) still lands on the runtime.
  let handle = runtime.handle().clone();

  std::thread::Builder::new()
    .name("pipeline".into())
    .spawn(move || {
      // Dropping the handle closes the stream, so it has to outlive
      // the pipeline rather than the setup above.
      let _capture = capture;

      if let Err(why) = handle.block_on(pipeline.run(audio)) {
        error!(error = ?why, "pipeline stopped");
      }
    })
    .context("spawning the pipeline thread")?;

  // The event loop never returns. The runtime stays owned by this
  // frame, which is what keeps the task above alive.
  tray::run(tray::Context {
    status,
    log_dir: config::expand_tilde(&config.log_dir),
    launchd: config.launchd_target(),
  })
}

/// Runs a subcommand to completion on a throwaway runtime. The
/// subcommands are all headless, so the main thread is theirs.
fn blocking<F: std::future::Future<Output = Result<()>>>(
  future: F,
) -> Result<()> {
  tokio::runtime::Builder::new_multi_thread()
    .enable_all()
    .build()
    .context("building the tokio runtime")?
    .block_on(future)
}

/// Every audio device, both directions, with the current defaults and
/// the `[devices]` aliases pointing at each - the way `obs sources`
/// shows which sources are wired up, and for the same reason: an alias
/// that resolves to nothing is silent until you say the words.
///
/// The config is read if it parses and ignored if it does not, since
/// this is also what you run before there is one.
fn devices(config: &Config) -> Result<()> {
  let aliases = CommandFile::load(&config.resolved_config_path())
    .map(|file| file.devices)
    .unwrap_or_default();

  let all = devices::list()?;

  for direction in [devices::Direction::Input, devices::Direction::Output]
  {
    let current = devices::current(direction).ok();

    println!("{}s", direction.as_str());

    for device in all.iter().filter(|device| device.does(direction)) {
      let mut tags = Vec::new();

      if current.as_ref().is_some_and(|it| it.id == device.id) {
        tags.push("default".to_string());
      }

      for (alias, pattern) in &aliases {
        if devices::matches(&device.name, pattern) {
          tags.push(alias.clone());
        }
      }

      if tags.is_empty() {
        println!("  {}", device.name);
      } else {
        println!("  {}  ({})", device.name, tags.join(", "));
      }
    }
  }

  // An alias matching nothing at all is either a device that is not
  // plugged in - normal, and the command will start working when it
  // is - or a typo that never will. Nothing here can tell those apart,
  // so say both.
  for (alias, pattern) in &aliases {
    if !all
      .iter()
      .any(|device| devices::matches(&device.name, pattern))
    {
      println!(
        "\n!! nothing matches {alias} = {pattern:?} - fine if it is \
         only unplugged, a typo otherwise"
      );
    }
  }

  Ok(())
}

/// Points one of the system's defaults at a device, without going
/// through a command. `name` is a `[devices]` alias if there is one by
/// that name, and part of a device name otherwise - which is how you
/// find out what to put in the table in the first place.
fn switch(
  config: &Config,
  direction: devices::Direction,
  name: Option<&String>,
) -> Result<()> {
  let Some(name) = name else {
    bail!("usage: voice-control {} <name>", direction.as_str());
  };

  let pattern = CommandFile::load(&config.resolved_config_path())
    .ok()
    .and_then(|file| file.devices.get(name).cloned())
    .unwrap_or_else(|| name.clone());

  let switch = devices::set_default(direction, &pattern)?;

  if switch.changed {
    println!("{} is now {:?}", direction.as_str(), switch.name);
  } else {
    println!("{} was already {:?}", direction.as_str(), switch.name);
  }

  Ok(())
}

/// Wake word only: no model loading beyond rustpotter, no HTTP. Use
/// this to tune `threshold` / `avg_threshold`.
async fn listen(config: Config) -> Result<()> {
  let path = config.resolved_config_path();
  let file = CommandFile::load(&path)?;

  let mut detector = RustpotterDetector::load(
    &file.wake.model,
    file.wake.threshold,
    file.wake.avg_threshold,
    file.wake.eager,
  )?;

  let (_capture, mut audio) = capture::start(&config.input_device)?;
  let frame_size = detector.frame_size();
  let mut pending: Vec<f32> = Vec::new();
  let mut hits = 0_u32;
  let mut peak = 0.0_f32;
  let mut meter = tokio::time::interval(Duration::from_secs(1));
  let mut silent_seconds = 0_u32;

  info!("say the wake word - ctrl-c to stop");

  loop {
    tokio::select! {
      chunk = audio.recv() => {
        let Some(chunk) = chunk else { break };

        for &sample in &chunk {
          peak = peak.max(sample.abs());
        }

        pending.extend_from_slice(&chunk);

        while pending.len() >= frame_size {
          let frame: Vec<f32> =
            pending.drain(..frame_size).collect();

          if let Some(detection) = detector.push(frame) {
            hits += 1;
            println!(
              "{hits:>3}  {}  score {:.3}",
              detection.name, detection.score
            );
          }
        }
      }
      _ = meter.tick() => {
        // Dead silence for long enough almost always means the
        // microphone permission was never granted, which otherwise
        // looks exactly like a wake word model that never fires.
        if peak < 1e-5 {
          silent_seconds += 1;

          if silent_seconds == 5 {
            warn!(
              "no audio at all for 5s - check System Settings -> \
               Privacy & Security -> Microphone"
            );
          }
        } else {
          silent_seconds = 0;
          println!("  level {:.3}{}", peak, bar(peak));
        }

        peak = 0.0;
      }
      _ = tokio::signal::ctrl_c() => break,
    }
  }

  println!("\n{hits} detection(s)");

  Ok(())
}

fn bar(peak: f32) -> String {
  let width = (peak.clamp(0.0, 1.0) * 40.0).round() as usize;
  format!("  {}", "#".repeat(width))
}

/// STT + matcher against a wav file, so phrase lists can be checked
/// without touching the microphone.
async fn transcribe(config: Config, path: Option<&String>) -> Result<()> {
  let Some(path) = path else {
    bail!("usage: voice-control transcribe <file.wav>");
  };

  let file = CommandFile::load(&config.resolved_config_path())?;
  let mut transcriber =
    WhisperTranscriber::load(&file.stt.model, &file.vocabulary())?;

  let audio = read_wav(path)?;
  println!("{} ms of audio", samples_to_ms(audio.len()));

  let transcript = transcriber.transcribe(&audio)?;
  println!("transcript: {transcript:?}");

  match match_command(&file.commands, &transcript) {
    Some(hit) => println!(
      "matched:    {} (score {:.3}) -> {}",
      hit.command.name,
      hit.score,
      hit.command.target()
    ),
    None => println!("matched:    <none>"),
  }

  Ok(())
}

/// Wake word, VAD, STT, matcher and dispatch over a file - everything
/// the daemon does except opening the microphone.
async fn replay(config: Config, path: Option<&String>) -> Result<()> {
  let Some(path) = path else {
    bail!("usage: voice-control replay <file.wav>");
  };

  let file = CommandFile::load(&config.resolved_config_path())?;

  let detector = RustpotterDetector::load(
    &file.wake.model,
    file.wake.threshold,
    file.wake.avg_threshold,
    file.wake.eager,
  )?;
  let transcriber =
    WhisperTranscriber::load(&file.stt.model, &file.vocabulary())?;

  let mut pipeline = Pipeline::new(
    Box::new(detector) as Box<dyn Detector>,
    Box::new(transcriber) as Box<dyn Transcriber>,
    file.commands,
    Feedback::new(&config.sounds_dir),
    file.obs,
    &file.listen,
    Arc::new(Status::new()),
  )?;

  let mut audio = read_wav(path)?;
  info!(ms = samples_to_ms(audio.len()), "replaying");

  // The detector accumulates partial scores and only settles once it
  // has seen frames past the wake word, and the endpointer needs to
  // hear the silence that ends the command. A live microphone
  // supplies both simply by continuing to run; a file that stops
  // dead on the last syllable does not, so make up the difference.
  audio.resize(audio.len() + ms_to_samples(REPLAY_TRAILING_MS), 0.0);

  // Chunked the way cpal would deliver it, so the buffering and
  // pre-roll logic get exercised rather than bypassed.
  for chunk in audio.chunks(512) {
    pipeline.feed(chunk).await;
  }

  pipeline.flush().await;

  Ok(())
}

/// `obs`, and its `sources` / `filters` / `show` / `hide` / `toggle`
/// forms.
///
/// A scene named "sources" or "show" would be shadowed by the
/// subcommand of the same name; if you have one, the daemon still
/// switches to it, only this debugging shortcut cannot.
async fn obs(config: Config, args: &[String]) -> Result<()> {
  let visibility = match args.first().map(String::as_str) {
    Some("show") => Some(obs::Visibility::Show),
    Some("hide") => Some(obs::Visibility::Hide),
    Some("toggle") => Some(obs::Visibility::Toggle),
    _ => None,
  };

  match (args.first().map(String::as_str), visibility) {
    (Some("sources"), _) => obs_sources(config, args.get(1)).await,
    (Some("filters"), _) => obs_filters(config, args.get(1)).await,
    (_, Some(visibility)) => {
      let Some(source) = args.get(1) else {
        bail!("usage: voice-control obs {} <source> [scene]", args[0]);
      };

      obs_source(config, source, args.get(2), visibility).await
    }
    _ => obs_scenes(config, args.first()).await,
  }
}

/// With no argument, lists the scenes obs reports so `scene = "..."`
/// entries can be copied out verbatim - the names are matched exactly.
/// With one, switches to it, which is the quick way to check the
/// wiring without saying anything.
async fn obs_scenes(config: Config, scene: Option<&String>) -> Result<()> {
  let file = CommandFile::load(&config.resolved_config_path())?;

  if let Some(scene) = scene {
    obs::set_scene(&file.obs, scene).await?;
    println!("switched to {scene:?}");

    return Ok(());
  }

  let (scenes, current) = obs::scene_list(&file.obs).await?;

  // Every scene any step names, whether to switch to it or to say
  // where a source lives. Deduplicated, so a scene named by several
  // steps is not warned about once per step.
  let mut wired: Vec<&str> = file
    .commands
    .iter()
    .flat_map(Command::steps)
    .filter_map(|step| step.scene.as_deref())
    .collect();

  wired.sort_unstable();
  wired.dedup();

  for scene in &scenes {
    let mut tags = Vec::new();

    if *scene == current {
      tags.push("current");
    }
    if wired.contains(&scene.as_str()) {
      tags.push("wired up");
    }

    if tags.is_empty() {
      println!("  {scene}");
    } else {
      println!("  {scene}  ({})", tags.join(", "));
    }
  }

  // A typo here is silent until you say the words, so call it out.
  for scene in wired {
    if !scenes.iter().any(|s| s == scene) {
      println!(
        "\n!! commands.toml refers to {scene:?}, which obs does not have"
      );
    }
  }

  Ok(())
}

/// The sources in one scene, so `source = "..."` entries can be copied
/// out verbatim, with the current visibility of each - the two things
/// you want to see when a source command does nothing.
async fn obs_sources(
  config: Config,
  scene: Option<&String>,
) -> Result<()> {
  let file = CommandFile::load(&config.resolved_config_path())?;
  let (scene, items) =
    obs::source_list(&file.obs, scene.map(String::as_str)).await?;

  println!("{scene}");

  // Sources wired to a command that could land in this scene: the
  // ones naming it, plus the ones naming no scene at all, which are
  // resolved against whatever is on program when they are said.
  // Deduplicated, since show / hide / toggle of one source is three
  // commands naming it.
  let mut wired: Vec<&str> = file
    .commands
    .iter()
    .flat_map(Command::steps)
    .filter(|step| {
      step.scene.as_deref().is_none_or(|named| named == scene)
    })
    .filter_map(|step| step.source.as_deref())
    .collect();

  wired.sort_unstable();
  wired.dedup();

  for item in &items {
    let mut tags = Vec::new();

    if item.is_group {
      tags.push("group".to_string());
    }
    if item.scene != scene {
      tags.push(format!("in {}", item.scene));
    }
    if !item.enabled {
      tags.push("hidden".to_string());
    }
    if wired.contains(&item.name.as_str()) {
      tags.push("wired up".to_string());
    }

    if tags.is_empty() {
      println!("  {}", item.name);
    } else {
      println!("  {}  ({})", item.name, tags.join(", "));
    }
  }

  // A typo here is silent until you say the words, so call it out.
  for source in &wired {
    let matches = items.iter().filter(|item| &item.name == source).count();

    if matches == 0 {
      println!(
        "\n!! commands.toml refers to source {source:?}, which \
         {scene:?} does not have"
      );
    }

    // Two items sharing a name is legal in OBS and invisible until
    // one of them moves and the other does not. Which one a command
    // gets is OBS's answer to the name, not ours, so the only honest
    // thing to do is say the question is ambiguous.
    if matches > 1 {
      println!(
        "\n!! {scene:?} has {matches} sources named {source:?} - a \
         command naming it gets whichever obs resolves first; rename \
         one to be sure of which"
      );
    }
  }

  Ok(())
}

/// The filters on a scene or source, so `show_filter = "..."` entries
/// can be copied out verbatim.
async fn obs_filters(
  config: Config,
  target: Option<&String>,
) -> Result<()> {
  let file = CommandFile::load(&config.resolved_config_path())?;
  let (target, filters) =
    obs::filter_list(&file.obs, target.map(String::as_str)).await?;

  println!("{target}");

  if filters.is_empty() {
    println!("  (no filters)");
  }

  for filter in &filters {
    println!(
      "  {}  ({}{})",
      filter.name,
      filter.kind,
      if filter.enabled { ", enabled" } else { "" }
    );
  }

  Ok(())
}

/// Flips one source, without the filters a configured command would
/// play around it - this is the bare visibility change, for checking
/// that the source name and scene are right. `run` is the way to
/// exercise a command's full sequence.
async fn obs_source(
  config: Config,
  source: &str,
  scene: Option<&String>,
  visibility: obs::Visibility,
) -> Result<()> {
  let file = CommandFile::load(&config.resolved_config_path())?;

  let visible = obs::run_source(
    &file.obs,
    obs::SourceAction {
      source,
      scene: scene.map(String::as_str),
      visibility,
      show_filter: None,
      hide_filter: None,
      hide_delay: Duration::ZERO,
    },
  )
  .await?;

  println!(
    "{source:?} is now {}",
    if visible { "visible" } else { "hidden" }
  );

  Ok(())
}

/// Presses one media key without going through a command, which is
/// what tells a missing Accessibility grant apart from a phrase that
/// did not match. No config is read: there is nothing in it to read.
fn media_key(key: Option<&String>) -> Result<()> {
  let Some(key) = key else {
    bail!("usage: voice-control media <play_pause|next|previous>");
  };

  let key: media::MediaKey = key.parse()?;
  media::press(key)?;

  println!("pressed {}", key.as_str());

  Ok(())
}

/// Matches a phrase and dispatches it, exactly as the daemon would -
/// the way to check a command's whole sequence, filters and waits
/// included, without saying anything.
async fn run_phrase(
  config: Config,
  phrase: Option<&String>,
) -> Result<()> {
  let Some(phrase) = phrase else {
    bail!("usage: voice-control run <phrase>");
  };

  let file = CommandFile::load(&config.resolved_config_path())?;

  let Some(hit) = match_command(&file.commands, phrase) else {
    bail!("no command matches {phrase:?}");
  };

  println!(
    "matched: {} (score {:.3}) -> {}",
    hit.command.name,
    hit.score,
    hit.command.target()
  );

  Dispatcher::new(file.obs, Feedback::new(&config.sounds_dir))?
    .run(hit.command)
    .await
}

fn read_wav(path: &str) -> Result<Vec<f32>> {
  let mut reader = hound::WavReader::open(path)
    .with_context(|| format!("opening {path}"))?;
  let spec = reader.spec();

  if spec.sample_rate != SAMPLE_RATE {
    warn!(
      rate = spec.sample_rate,
      "expected {SAMPLE_RATE} Hz; transcription will be off"
    );
  }

  let samples: Vec<f32> = match spec.sample_format {
    hound::SampleFormat::Float => reader
      .samples::<f32>()
      .collect::<Result<Vec<_>, _>>()
      .context("reading float samples")?,
    hound::SampleFormat::Int => {
      let scale = 1.0 / (1_i64 << (spec.bits_per_sample - 1)) as f32;
      reader
        .samples::<i32>()
        .map(|s| s.map(|s| s as f32 * scale))
        .collect::<Result<Vec<_>, _>>()
        .context("reading int samples")?
    }
  };

  if spec.channels <= 1 {
    return Ok(samples);
  }

  let channels = spec.channels as usize;

  Ok(
    samples
      .chunks_exact(channels)
      .map(|frame| frame.iter().sum::<f32>() / channels as f32)
      .collect(),
  )
}
