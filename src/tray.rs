//! Menu bar status item.
//!
//! `NSStatusItem` has to be built on the main thread under a running
//! `NSApplication`, so [`run`] takes the main thread for the rest of
//! the process's life and the pipeline runs on the tokio runtime
//! behind it.
//!
//! The menu is rebuilt from scratch in `menuNeedsUpdate:`, which
//! AppKit calls immediately before display - the standard Cocoa way to
//! avoid keeping a menu in sync with something that changes
//! constantly. A timer covers the two things that has to happen while
//! nothing is being clicked: the icon, always, and the lines of an
//! already-open menu, in place.

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::{DefinedClass, MainThreadOnly, define_class, msg_send, sel};
use objc2_app_kit::{
  NSApplication, NSApplicationActivationPolicy, NSImage, NSMenu,
  NSMenuDelegate, NSMenuItem, NSStatusBar, NSStatusItem,
  NSVariableStatusItemLength,
};
use objc2_foundation::{
  MainThreadMarker, NSObject, NSObjectProtocol, NSRunLoop,
  NSRunLoopCommonModes, NSString, NSTimer,
};
use tracing::{info, warn};

use crate::status::{Outcome, Phase, Snapshot, Status};

/// Icon refresh, and the in-place refresh of an open menu. Fast enough
/// that the level meter reads as live, slow enough to be free.
const TICK: Duration = Duration::from_millis(400);

const TAG_PAUSE: isize = 1;
const TAG_LOGS: isize = 2;
const TAG_RESTART: isize = 3;
const TAG_QUIT: isize = 4;

/// What the tray needs from the rest of the process.
pub struct Context {
  pub status: Arc<Status>,
  pub log_dir: PathBuf,
  /// launchd service target (`gui/501/com.dstn.voice-control`), when
  /// we were started by launchd. `None` when run from a shell, where
  /// restarting and quitting mean something different.
  pub launchd: Option<String>,
}

/// Which lines of an open menu are worth refreshing on the timer.
enum Line {
  Phase,
  Level,
  LastWake,
  History(usize),
}

struct Ivars {
  context: Context,
  item: Retained<NSStatusItem>,
  /// Only refresh menu titles while the menu is actually on screen.
  open: Cell<bool>,
  /// The lines built by the last `menuNeedsUpdate:`, so the timer can
  /// retitle them without rebuilding the menu under the cursor.
  lines: RefCell<Vec<(Retained<NSMenuItem>, Line)>>,
  /// Last symbol and tooltip handed to the button, to skip redundant
  /// work on a timer that fires more often than either changes.
  symbol: RefCell<String>,
  tip: RefCell<String>,
}

define_class!(
  #[unsafe(super(NSObject))]
  #[thread_kind = MainThreadOnly]
  #[name = "VoiceControlTray"]
  #[ivars = Ivars]
  struct Tray;

  unsafe impl NSObjectProtocol for Tray {}

  unsafe impl NSMenuDelegate for Tray {
    #[unsafe(method(menuNeedsUpdate:))]
    fn menu_needs_update(&self, menu: &NSMenu) {
      self.rebuild(menu);
    }

    #[unsafe(method(menuWillOpen:))]
    fn menu_will_open(&self, _menu: &NSMenu) {
      self.ivars().open.set(true);
    }

    #[unsafe(method(menuDidClose:))]
    fn menu_did_close(&self, _menu: &NSMenu) {
      self.ivars().open.set(false);
    }
  }

  impl Tray {
    #[unsafe(method(tick:))]
    fn tick(&self, _timer: *mut AnyObject) {
      let snapshot = self.ivars().context.status.snapshot();

      self.set_icon(&snapshot);

      if self.ivars().open.get() {
        self.retitle(&snapshot);
      }
    }

    #[unsafe(method(action:))]
    fn action(&self, sender: &NSMenuItem) {
      self.dispatch(sender.tag());
    }
  }
);

impl Tray {
  fn new(mtm: MainThreadMarker, context: Context) -> Retained<Self> {
    let bar = NSStatusBar::systemStatusBar();
    let item = bar.statusItemWithLength(NSVariableStatusItemLength);

    let this = mtm.alloc::<Self>().set_ivars(Ivars {
      context,
      item,
      open: Cell::new(false),
      lines: RefCell::new(Vec::new()),
      symbol: RefCell::new(String::new()),
      tip: RefCell::new(String::new()),
    });
    let this: Retained<Self> = unsafe { msg_send![super(this), init] };

    let menu = NSMenu::new(mtm);
    // Informational lines carry no action, and AppKit would grey them
    // out on its own - but it would also decide for the real items,
    // so take the whole question away from it.
    menu.setAutoenablesItems(false);
    menu.setDelegate(Some(ProtocolObject::from_ref(&*this)));

    let ivars = this.ivars();
    ivars.item.setMenu(Some(&menu));

    // The menu is only built when it is about to be shown, so give
    // the button something to display before that first happens.
    this.set_icon(&ivars.context.status.snapshot());

    // Worth a line in the log: a menu bar manager like Ice or
    // Bartender will happily file a brand new item away in its hidden
    // section, and "the icon is missing" then looks identical to "the
    // icon was never created".
    match ivars.item.button(mtm) {
      Some(_) => info!("menu bar item created"),
      None => warn!("menu bar item has no button - it will be invisible"),
    }

    this
  }

  /// Replaces the whole menu. Called just before it is displayed.
  fn rebuild(&self, menu: &NSMenu) {
    let mtm = self.mtm();
    let ivars = self.ivars();
    let snapshot = ivars.context.status.snapshot();
    let mut lines = ivars.lines.borrow_mut();

    menu.removeAllItems();
    lines.clear();

    let mut label = |title: String, line: Option<Line>| {
      let item = NSMenuItem::new(mtm);
      item.setTitle(&NSString::from_str(&title));
      item.setEnabled(false);
      menu.addItem(&item);

      if let Some(line) = line {
        lines.push((item, line));
      }
    };

    label(phase_line(&snapshot), Some(Line::Phase));
    label(level_line(&snapshot), Some(Line::Level));
    label(wake_line(&snapshot), Some(Line::LastWake));

    menu.addItem(&NSMenuItem::separatorItem(mtm));

    if snapshot.history.is_empty() {
      label("No commands yet".into(), None);
    } else {
      label("Recent".into(), None);

      for (i, entry) in snapshot.history.iter().enumerate() {
        let item = NSMenuItem::new(mtm);
        item.setTitle(&NSString::from_str(&history_line(entry)));
        item.setEnabled(false);
        item.setIndentationLevel(1);
        menu.addItem(&item);
        lines.push((item, Line::History(i)));
      }
    }

    drop(lines);
    menu.addItem(&NSMenuItem::separatorItem(mtm));

    let paused = ivars.context.status.paused();
    self.button(
      menu,
      if paused {
        "Resume listening"
      } else {
        "Pause listening"
      },
      TAG_PAUSE,
    );
    self.button(menu, "Open logs", TAG_LOGS);

    // Nothing to kickstart when launchd is not the one running us.
    if ivars.context.launchd.is_some() {
      self.button(menu, "Restart agent", TAG_RESTART);
    }

    menu.addItem(&NSMenuItem::separatorItem(mtm));

    // KeepAlive means a plain exit would be undone within the second,
    // so under launchd the honest label is the stronger one.
    self.button(
      menu,
      if ivars.context.launchd.is_some() {
        "Quit until next login"
      } else {
        "Quit"
      },
      TAG_QUIT,
    );
  }

  fn button(&self, menu: &NSMenu, title: &str, tag: isize) {
    let item = NSMenuItem::new(self.mtm());
    item.setTitle(&NSString::from_str(title));
    item.setTag(tag);
    item.setEnabled(true);
    unsafe {
      item.setTarget(Some(self));
      item.setAction(Some(sel!(action:)));
    }
    menu.addItem(&item);
  }

  /// Refreshes the lines of a menu that is already on screen. Retitles
  /// in place rather than rebuilding: replacing items under the cursor
  /// drops the highlight and flickers.
  fn retitle(&self, snapshot: &Snapshot) {
    for (item, line) in self.ivars().lines.borrow().iter() {
      let title = match line {
        Line::Phase => phase_line(snapshot),
        Line::Level => level_line(snapshot),
        Line::LastWake => wake_line(snapshot),
        Line::History(i) => match snapshot.history.get(*i) {
          Some(entry) => history_line(entry),
          // History only grows while the menu is open, and the new
          // entry has nowhere to go until the next rebuild.
          None => continue,
        },
      };

      item.setTitle(&NSString::from_str(&title));
    }
  }

  fn set_icon(&self, snapshot: &Snapshot) {
    let Some(button) = self.ivars().item.button(self.mtm()) else {
      return;
    };

    // The status line doubles as the hover tooltip and as the icon's
    // accessibility description, so the state is readable without
    // opening the menu at all.
    let tip = phase_line(snapshot);

    if *self.ivars().tip.borrow() != tip {
      button.setToolTip(Some(&NSString::from_str(&tip)));
      *self.ivars().tip.borrow_mut() = tip.clone();
    }

    let symbol = symbol_for(snapshot);

    if *self.ivars().symbol.borrow() == symbol {
      return;
    }

    let image =
      NSImage::imageWithSystemSymbolName_accessibilityDescription(
        &NSString::from_str(symbol),
        Some(&NSString::from_str(&tip)),
      );

    match image {
      Some(image) => {
        // Template images are recoloured by AppKit to suit the menu
        // bar, which is the only way to look right in both
        // appearances and under a tinted wallpaper.
        image.setTemplate(true);
        button.setImage(Some(&image));
      }
      None => {
        // Every symbol used here has shipped since macOS 11, so this
        // is close to unreachable - but an icon-less status item is
        // an invisible one, so leave something clickable behind.
        button.setImage(None);
        button.setTitle(&NSString::from_str(fallback_for(snapshot)));
      }
    }

    *self.ivars().symbol.borrow_mut() = symbol.to_string();
  }

  fn dispatch(&self, tag: isize) {
    let context = &self.ivars().context;

    match tag {
      TAG_PAUSE => {
        let paused = context.status.toggle_pause();
        info!(paused, "wake word scoring toggled from the menu");
      }
      TAG_LOGS => {
        // Only launchd redirects our output to a file. Run from a
        // shell the directory may not exist at all, and handing Finder
        // a missing path gets a dialog rather than a shrug.
        if context.log_dir.is_dir() {
          spawn("open", &[&context.log_dir.to_string_lossy()]);
        } else {
          warn!(dir = %context.log_dir.display(), "no log directory");
        }
      }
      TAG_RESTART => {
        if let Some(target) = &context.launchd {
          info!(%target, "restarting via launchctl");
          spawn("launchctl", &["kickstart", "-k", target]);
        }
      }
      TAG_QUIT => match &context.launchd {
        // A bare exit would be undone by KeepAlive, so ask launchd to
        // stop the job instead. It terminates us as part of that,
        // which is why nothing here exits on its own: if the bootout
        // fails we are better off still running.
        Some(target) => {
          info!(%target, "booting out");
          spawn("launchctl", &["bootout", target]);
        }
        None => {
          info!("quitting");
          std::process::exit(0);
        }
      },
      other => warn!(tag = other, "unknown menu item"),
    }
  }
}

/// Takes over the main thread: sets up the status item and runs the
/// AppKit event loop, which does not return.
pub fn run(context: Context) -> ! {
  let mtm = MainThreadMarker::new()
    .expect("the tray must be started on the main thread");

  let app = NSApplication::sharedApplication(mtm);
  // Accessory keeps us out of the Dock and the app switcher: a menu
  // bar item is the whole interface. This is what LSUIElement does
  // for a bundled app, and it works without a bundle.
  app.setActivationPolicy(NSApplicationActivationPolicy::Accessory);

  let tray = Tray::new(mtm, context);

  // Registered for the common modes rather than scheduled, which is
  // the whole reason the level meter moves. An open menu runs the loop
  // in event tracking mode, and a plainly scheduled timer belongs to
  // the default mode only - so it would stop firing at exactly the
  // moment someone is looking at the menu.
  let timer = unsafe {
    NSTimer::timerWithTimeInterval_target_selector_userInfo_repeats(
      TICK.as_secs_f64(),
      &tray,
      sel!(tick:),
      None,
      true,
    )
  };

  unsafe {
    NSRunLoop::currentRunLoop()
      .addTimer_forMode(&timer, NSRunLoopCommonModes);
  }

  app.run();

  unreachable!("NSApplication::run does not return");
}

fn symbol_for(snapshot: &Snapshot) -> &'static str {
  match snapshot.phase {
    Phase::Hearing => "mic.fill",
    Phase::Thinking => "waveform",
    Phase::Paused => "mic.slash",
    Phase::Starting => "mic.badge.plus",
    Phase::Deaf(_) | Phase::Stalled(_) | Phase::Stopped => {
      "exclamationmark.triangle.fill"
    }
    Phase::Idle => match snapshot.flash {
      Some(true) => "checkmark.circle.fill",
      Some(false) => "exclamationmark.circle.fill",
      None => "mic",
    },
  }
}

fn fallback_for(snapshot: &Snapshot) -> &'static str {
  match snapshot.phase {
    Phase::Hearing | Phase::Thinking => "◉",
    Phase::Deaf(_) | Phase::Stalled(_) | Phase::Stopped => "!",
    Phase::Paused => "◌",
    _ => "◎",
  }
}

fn phase_line(snapshot: &Snapshot) -> String {
  match &snapshot.phase {
    Phase::Starting => "Starting up".into(),
    Phase::Hearing => "Heard you - listening".into(),
    Phase::Thinking => "Working out what you said".into(),
    Phase::Paused => "Paused".into(),
    Phase::Stopped => "Audio stream ended".into(),
    Phase::Deaf(since) => {
      format!("Silent for {} - check microphone access", ago(*since))
    }
    Phase::Stalled(since) => {
      format!("No audio for {} - the input device is gone", ago(*since))
    }
    Phase::Idle => match snapshot.flash {
      Some(true) => "Done".into(),
      Some(false) => "That did not work".into(),
      None => "Listening for \"computa\"".into(),
    },
  }
}

fn level_line(snapshot: &Snapshot) -> String {
  let device = if snapshot.device.is_empty() {
    "input"
  } else {
    &snapshot.device
  };

  format!("{device}  {}", meter(snapshot.level))
}

fn wake_line(snapshot: &Snapshot) -> String {
  match snapshot.last_wake {
    Some(since) => format!("Last woken {} ago", ago(since)),
    None => "Not woken yet this session".into(),
  }
}

fn history_line(entry: &crate::status::Entry) -> String {
  let outcome = match &entry.outcome {
    Outcome::Dispatched(name) => format!("{name} OK"),
    Outcome::Failed(name) => format!("{name} failed"),
    Outcome::NoMatch => "no match".into(),
    Outcome::Unheard => "nothing heard".into(),
  };

  // A failed transcription has no words to show, only a verdict.
  if entry.transcript.is_empty() {
    return format!("{}  {outcome}", ago(entry.ago));
  }

  format!("{}  \"{}\"  {outcome}", ago(entry.ago), entry.transcript)
}

/// Eight blocks of level, on a dB scale - a linear peak spends nearly
/// all of its range looking like silence.
fn meter(peak: f32) -> String {
  const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
  /// Stands in for an unfilled block, so the meter keeps its width.
  const EMPTY: char = '·';
  const FLOOR: f32 = -60.0;

  if peak <= 0.0 {
    return String::from(EMPTY).repeat(BLOCKS.len());
  }

  let db = 20.0 * peak.log10();
  let filled =
    (((db - FLOOR) / -FLOOR) * BLOCKS.len() as f32).round() as isize;

  (0..BLOCKS.len() as isize)
    .map(|i| {
      if i < filled {
        BLOCKS[i as usize]
      } else {
        '\u{00b7}'
      }
    })
    .collect()
}

fn ago(since: Duration) -> String {
  let seconds = since.as_secs();

  match seconds {
    0..=4 => "just now".into(),
    5..=59 => format!("{seconds}s"),
    60..=3599 => format!("{}m", seconds / 60),
    _ => format!("{}h", seconds / 3600),
  }
}

fn spawn(program: &str, args: &[&str]) {
  match std::process::Command::new(program).args(args).spawn() {
    Ok(mut child) => {
      // Reaped on a thread so a menu click never blocks the run loop.
      std::thread::spawn(move || {
        let _ = child.wait();
      });
    }
    Err(why) => warn!(program, error = %why, "failed to spawn"),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn the_meter_spans_silence_to_clipping() {
    assert_eq!(meter(0.0), "·".repeat(8));
    assert!(meter(1.0).ends_with('█'));
    assert!(meter(0.05).contains('·'));
  }

  #[test]
  fn history_lines_name_the_command_and_the_verdict() {
    let status = Status::new();
    status.finished("computa mute", Outcome::Dispatched("mute".into()));
    status.finished("computa muted", Outcome::NoMatch);

    let snapshot = status.snapshot();
    let lines: Vec<String> =
      snapshot.history.iter().map(history_line).collect();

    assert_eq!(lines[0], "just now  \"computa muted\"  no match");
    assert_eq!(lines[1], "just now  \"computa mute\"  mute OK");
  }

  #[test]
  fn a_failed_transcription_has_no_words_to_quote() {
    let status = Status::new();
    status.finished("", Outcome::Unheard);

    let snapshot = status.snapshot();

    assert_eq!(
      history_line(&snapshot.history[0]),
      "just now  nothing heard"
    );
  }

  #[test]
  fn the_fault_states_say_what_to_check() {
    let deaf = Snapshot {
      phase: Phase::Deaf(Duration::from_secs(90)),
      flash: None,
      level: 0.0,
      device: "MV7".into(),
      last_wake: None,
      history: Vec::new(),
      utterances: 0,
    };

    assert_eq!(
      phase_line(&deaf),
      "Silent for 1m - check microphone access"
    );
    assert_eq!(symbol_for(&deaf), "exclamationmark.triangle.fill");
  }

  #[test]
  fn ages_read_as_prose() {
    assert_eq!(ago(Duration::from_secs(2)), "just now");
    assert_eq!(ago(Duration::from_secs(42)), "42s");
    assert_eq!(ago(Duration::from_secs(90)), "1m");
    assert_eq!(ago(Duration::from_secs(7200)), "2h");
  }
}
