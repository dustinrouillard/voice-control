//! System media keys.
//!
//! macOS routes play / next / previous to whichever application it
//! considers to be playing, which is why this controls Spotify without
//! talking to Spotify - and controls Music, or a browser tab, on the
//! days that is what is making noise instead.
//!
//! They are not ordinary key codes. A media key arrives as
//! `NSEventTypeSystemDefined` with subtype 8, and the key itself is
//! packed into `data1` alongside whether it is going down or up - so
//! this builds an `NSEvent` and posts the `CGEvent` behind it rather
//! than synthesising a keystroke.

use anyhow::{Result, anyhow, bail};
use objc2_app_kit::{NSEvent, NSEventModifierFlags, NSEventType};
use objc2_core_graphics::{CGEvent, CGEventTapLocation};
use objc2_foundation::NSPoint;
use serde::Deserialize;

/// `NX_SUBTYPE_AUX_CONTROL_BUTTONS` - the subtype that says the rest of
/// the event describes one of the keyboard's auxiliary keys.
const AUX_CONTROL_BUTTONS: i16 = 8;

/// `NX_KEYSTATE_DOWN` / `NX_KEYSTATE_UP`, which say which way the key is
/// going. They appear twice in each event: in `data1` next to the key,
/// and again in the modifier flags.
const DOWN: isize = 0xa;
const UP: isize = 0xb;

/// One of the keyboard's transport keys.
///
/// The names here are the `NX_KEYTYPE_*` ones; the aliases are what a
/// config is likely to say instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaKey {
  /// `NX_KEYTYPE_PLAY`. One key for both directions - the hardware has
  /// no separate play and pause, and every player treats it as a
  /// toggle - so `pause` while already paused starts playing again.
  #[serde(
    alias = "play",
    alias = "pause",
    alias = "stop",
    alias = "resume",
    alias = "playpause",
    alias = "toggle"
  )]
  PlayPause,
  /// `NX_KEYTYPE_NEXT`.
  #[serde(alias = "skip", alias = "next_track", alias = "forward")]
  Next,
  /// `NX_KEYTYPE_PREVIOUS`. Most players read one press as "back to the
  /// start of this track" and two as "the track before".
  #[serde(alias = "prev", alias = "previous_track", alias = "back")]
  Previous,
}

impl MediaKey {
  fn code(self) -> isize {
    match self {
      MediaKey::PlayPause => 16,
      MediaKey::Next => 17,
      MediaKey::Previous => 18,
    }
  }

  /// How the key reads in a log line.
  pub fn as_str(self) -> &'static str {
    match self {
      MediaKey::PlayPause => "play/pause",
      MediaKey::Next => "next",
      MediaKey::Previous => "previous",
    }
  }
}

impl std::str::FromStr for MediaKey {
  type Err = anyhow::Error;

  /// Parsed through serde, so the spellings the config accepts and the
  /// spellings the `media` subcommand accepts cannot drift apart.
  fn from_str(text: &str) -> Result<Self> {
    use serde::de::IntoDeserializer;

    Self::deserialize(text.into_deserializer()).map_err(
      |_: serde::de::value::Error| {
        anyhow!(
          "unknown media key {text:?} - it is play_pause, next or \
           previous"
        )
      },
    )
  }
}

/// Presses and releases one media key.
pub fn press(key: MediaKey) -> Result<()> {
  // Posting without the grant is not an error: the event is dropped and
  // nothing says so, which looks exactly like a player that ignored it.
  // Checking first is the only way to tell those apart.
  if !trusted() {
    bail!(
      "macOS has not granted this process Accessibility access, so a \
       synthesised key press goes nowhere - add voice-control under \
       System Settings -> Privacy & Security -> Accessibility"
    );
  }

  post(key, DOWN)?;
  post(key, UP)
}

fn post(key: MediaKey, state: isize) -> Result<()> {
  #[allow(non_snake_case)]
  let event = NSEvent::otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2(
    NSEventType::SystemDefined,
    NSPoint::ZERO,
    // Both halves are read, and an event that sets only one of them is
    // ignored - hence the state appearing here as well as in `data1`.
    NSEventModifierFlags((state as usize) << 8),
    0.0,
    0,
    None,
    AUX_CONTROL_BUTTONS,
    (key.code() << 16) | (state << 8),
    -1,
  )
  .ok_or_else(|| anyhow!("appkit would not build the key event"))?;

  let event = event
    .CGEvent()
    .ok_or_else(|| anyhow!("the key event had no CGEvent to post"))?;

  // The HID tap sits above every per-application tap, so the event
  // reaches the system's media handler by the same route the physical
  // key does.
  CGEvent::post(CGEventTapLocation::HIDEventTap, Some(&event));

  Ok(())
}

/// Whether macOS will let this process post events at all.
fn trusted() -> bool {
  // In ApplicationServices, which AppKit already links.
  #[link(name = "ApplicationServices", kind = "framework")]
  unsafe extern "C" {
    fn AXIsProcessTrusted() -> u8;
  }

  // Returns a `Boolean`, which is a byte rather than a Rust `bool`.
  unsafe { AXIsProcessTrusted() != 0 }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn keys_parse_from_every_spelling_the_config_allows() {
    for text in ["play_pause", "play", "pause", "stop", "resume"] {
      assert_eq!(text.parse::<MediaKey>().unwrap(), MediaKey::PlayPause);
    }

    assert_eq!("skip".parse::<MediaKey>().unwrap(), MediaKey::Next);
    assert_eq!("prev".parse::<MediaKey>().unwrap(), MediaKey::Previous);
  }

  #[test]
  fn an_unknown_key_says_what_the_keys_are() {
    let why = "eject".parse::<MediaKey>().unwrap_err().to_string();

    assert!(why.contains("play_pause"), "{why}");
  }
}
