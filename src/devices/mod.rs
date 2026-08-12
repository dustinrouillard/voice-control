//! The system's default input and output devices.
//!
//! This is the same thing the Sound pane does when you pick a device
//! from its list, done through the HAL directly: "computa, headphones"
//! moves everything that is not pinned to a device of its own onto the
//! AirPods, and "computa, speakers" moves it back.
//!
//! Devices are named by a case-insensitive substring, because the HAL
//! spells them out in full - "Dustin's AirPods Pro #3", "CalDigit
//! USB-C Pro Audio" - and a config nobody wants to retype every time
//! the pairing renames itself needs only enough of that to be
//! unambiguous.
//!
//! Switching the input does not move the daemon's own microphone. cpal
//! opens a device, not "whatever is default", so the capture stream
//! stays where it was until the daemon restarts.

use std::ffi::c_void;
use std::mem::{MaybeUninit, size_of};
use std::ptr;

use anyhow::{Result, anyhow, bail};
use tracing::{debug, warn};

pub mod sys;

/// Which end of the machine a command is talking about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
  Input,
  Output,
}

impl Direction {
  /// How the direction reads in a log line or an error.
  pub fn as_str(self) -> &'static str {
    match self {
      Direction::Input => "input",
      Direction::Output => "output",
    }
  }

  /// The scope a device's streams are counted in, which is what says
  /// whether the device can do this direction at all.
  fn scope(self) -> u32 {
    match self {
      Direction::Input => sys::SCOPE_INPUT,
      Direction::Output => sys::SCOPE_OUTPUT,
    }
  }

  /// The HAL property holding the current default for this direction.
  fn selector(self) -> u32 {
    match self {
      Direction::Input => sys::DEFAULT_INPUT_DEVICE,
      Direction::Output => sys::DEFAULT_OUTPUT_DEVICE,
    }
  }
}

#[derive(Clone, Debug)]
pub struct Device {
  pub id: sys::AudioObjectID,
  pub name: String,
  /// A pair of AirPods is both, and an output-only device is still in
  /// the list when you ask for inputs - hence a flag each rather than
  /// a direction.
  pub input: bool,
  pub output: bool,
}

impl Device {
  pub fn does(&self, direction: Direction) -> bool {
    match direction {
      Direction::Input => self.input,
      Direction::Output => self.output,
    }
  }
}

/// What a switch actually did, which is not always a change: asking
/// for the device that is already default is a no-op worth reporting
/// as one rather than dressing up as a switch.
pub struct Switch {
  pub name: String,
  pub changed: bool,
}

/// Every device the HAL knows about, in the order it reports them.
pub fn list() -> Result<Vec<Device>> {
  let ids: Vec<sys::AudioObjectID> = property_vec(
    sys::SYSTEM_OBJECT,
    &address(sys::DEVICES, sys::SCOPE_GLOBAL),
    "listing the audio devices",
  )?;

  let mut devices = Vec::new();

  for id in ids {
    match device(id) {
      Ok(device) => devices.push(device),
      // A device that disconnected between the list and the name is
      // not an error - it is a pair of AirPods going back in the case
      // while we were looking at them.
      Err(why) => debug!(id, error = ?why, "skipped a device"),
    }
  }

  Ok(devices)
}

/// The devices that can do one direction, which is the list a command
/// naming that direction is matched against.
pub fn list_for(direction: Direction) -> Result<Vec<Device>> {
  Ok(
    list()?
      .into_iter()
      .filter(|device| device.does(direction))
      .collect(),
  )
}

/// The current default for one direction - what every app that has not
/// pinned a device of its own uses.
pub fn current(direction: Direction) -> Result<Device> {
  device(current_id(direction)?)
}

/// The first device whose name contains `pattern`.
///
/// The HAL's order is stable enough that "first" is a real answer, and
/// a pattern matching two devices is a pattern that needs to be longer
/// - which the error says, rather than picking one quietly.
pub fn find(direction: Direction, pattern: &str) -> Result<Device> {
  let devices = list_for(direction)?;
  let hits: Vec<&Device> = devices
    .iter()
    .filter(|device| matches(&device.name, pattern))
    .collect();

  if hits.len() > 1 {
    warn!(
      pattern,
      direction = direction.as_str(),
      devices = hits
        .iter()
        .map(|device| device.name.as_str())
        .collect::<Vec<_>>()
        .join(", "),
      "several devices match, taking the first"
    );
  }

  if let Some(device) = hits.first() {
    return Ok((*device).clone());
  }

  let available: Vec<&str> =
    devices.iter().map(|device| device.name.as_str()).collect();

  if available.is_empty() {
    bail!("there are no {} devices at all", direction.as_str());
  }

  Err(anyhow!(
    "no {} device matching {pattern:?} - there is {}",
    direction.as_str(),
    available.join(", ")
  ))
}

/// Points one direction at the first device matching `pattern`.
pub fn set_default(direction: Direction, pattern: &str) -> Result<Switch> {
  let device = find(direction, pattern)?;

  // Setting the default fires the HAL notification it was set from,
  // and there is at least one other agent on this machine listening
  // for that one. Not writing a value that is already there keeps a
  // repeated command from waking everything up for nothing.
  if current_id(direction).ok() == Some(device.id) {
    return Ok(Switch {
      name: device.name,
      changed: false,
    });
  }

  set_device_id(direction.selector(), device.id)?;

  // Alerts and interface sounds are a separate default, and leaving
  // them behind means notifications keep coming out of whatever you
  // just switched away from. Not fatal on its own: the audio you
  // asked about has already moved.
  if direction == Direction::Output
    && let Err(why) =
      set_device_id(sys::DEFAULT_SYSTEM_OUTPUT_DEVICE, device.id)
  {
    warn!(
      device = %device.name,
      error = ?why,
      "switched the output but not the system sound effects"
    );
  }

  Ok(Switch {
    name: device.name,
    changed: true,
  })
}

/// Case-insensitive substring match, which is how every device name in
/// the config is matched: the HAL reports "Dustin's AirPods Pro #3",
/// and nobody wants to type that.
pub fn matches(name: &str, pattern: &str) -> bool {
  name.to_lowercase().contains(&pattern.to_lowercase())
}

pub fn device(id: sys::AudioObjectID) -> Result<Device> {
  Ok(Device {
    id,
    name: property_string(
      id,
      &address(sys::NAME, sys::SCOPE_GLOBAL),
      "reading a device name",
    )?,
    input: has_streams(id, Direction::Input),
    output: has_streams(id, Direction::Output),
  })
}

fn current_id(direction: Direction) -> Result<sys::AudioObjectID> {
  property(
    sys::SYSTEM_OBJECT,
    &address(direction.selector(), sys::SCOPE_GLOBAL),
    "reading a default device",
  )
}

fn set_device_id(selector: u32, id: sys::AudioObjectID) -> Result<()> {
  let address = address(selector, sys::SCOPE_GLOBAL);

  let status = unsafe {
    sys::AudioObjectSetPropertyData(
      sys::SYSTEM_OBJECT,
      &address,
      0,
      ptr::null(),
      size_of::<sys::AudioObjectID>() as u32,
      (&raw const id).cast(),
    )
  };

  check(status, "setting a default device")
}

/// Whether the device does this direction at all. Every device is in
/// the one list whichever way it points, and the stream count in a
/// scope is what tells them apart.
fn has_streams(id: sys::AudioObjectID, direction: Direction) -> bool {
  let address = address(sys::STREAMS, direction.scope());
  let mut size = 0;

  let status = unsafe {
    sys::AudioObjectGetPropertyDataSize(
      id,
      &address,
      0,
      ptr::null(),
      &mut size,
    )
  };

  status == 0 && size > 0
}

const fn address(
  selector: u32,
  scope: u32,
) -> sys::AudioObjectPropertyAddress {
  sys::AudioObjectPropertyAddress {
    selector,
    scope,
    element: sys::ELEMENT_MAIN,
  }
}

/// One fixed-size property: a device id, or the CFStringRef behind
/// [`property_string`].
fn property<T>(
  id: sys::AudioObjectID,
  address: &sys::AudioObjectPropertyAddress,
  what: &str,
) -> Result<T> {
  let mut value = MaybeUninit::<T>::uninit();
  let mut size = size_of::<T>() as u32;

  let status = unsafe {
    sys::AudioObjectGetPropertyData(
      id,
      address,
      0,
      ptr::null(),
      &mut size,
      value.as_mut_ptr().cast(),
    )
  };

  check(status, what)?;

  // A short write would leave part of `value` uninitialised, so treat
  // it as a failure rather than reading whatever was on the stack.
  if size as usize != size_of::<T>() {
    bail!("{what} returned {size} bytes, expected {}", size_of::<T>());
  }

  Ok(unsafe { value.assume_init() })
}

/// A variable-length property, sized first and then read.
fn property_vec<T: Copy>(
  id: sys::AudioObjectID,
  address: &sys::AudioObjectPropertyAddress,
  what: &str,
) -> Result<Vec<T>> {
  let mut size = 0;

  let status = unsafe {
    sys::AudioObjectGetPropertyDataSize(
      id,
      address,
      0,
      ptr::null(),
      &mut size,
    )
  };

  check(status, what)?;

  let count = size as usize / size_of::<T>();
  let mut values: Vec<T> = Vec::with_capacity(count);

  if count == 0 {
    return Ok(values);
  }

  // The size the HAL was asked for, not the one it reported, so a
  // device list that grew between the two calls cannot overrun the
  // allocation.
  let mut size = (count * size_of::<T>()) as u32;

  let status = unsafe {
    sys::AudioObjectGetPropertyData(
      id,
      address,
      0,
      ptr::null(),
      &mut size,
      values.as_mut_ptr().cast(),
    )
  };

  check(status, what)?;

  // The second call reports how much it actually wrote, which is what
  // is initialised - it can be less if a device went away in between.
  unsafe { values.set_len((size as usize / size_of::<T>()).min(count)) };

  Ok(values)
}

fn property_string(
  id: sys::AudioObjectID,
  address: &sys::AudioObjectPropertyAddress,
  what: &str,
) -> Result<String> {
  let string: *const c_void = property(id, address, what)?;

  if string.is_null() {
    bail!("{what} returned no string");
  }

  // Ours to release: the HAL hands back a +1 reference for a property
  // read, whatever happens to the decode below.
  let decoded = unsafe { decode(string) };
  unsafe { sys::CFRelease(string) };

  decoded
}

unsafe fn decode(string: *const c_void) -> Result<String> {
  let length = unsafe { sys::CFStringGetLength(string) };
  let capacity =
    unsafe { sys::CFStringGetMaximumSizeForEncoding(length, sys::UTF8) }
      + 1;

  let mut buffer = vec![0_u8; capacity as usize];

  let ok = unsafe {
    sys::CFStringGetCString(
      string,
      buffer.as_mut_ptr().cast(),
      capacity,
      sys::UTF8,
    )
  };

  if ok == 0 {
    bail!("could not decode a device name as UTF-8");
  }

  let end = buffer.iter().position(|&byte| byte == 0).unwrap_or(0);
  buffer.truncate(end);

  Ok(String::from_utf8(buffer)?)
}

fn check(status: sys::OSStatus, what: &str) -> Result<()> {
  if status == 0 {
    return Ok(());
  }

  bail!("{what} failed: {}", describe(status));
}

/// CoreAudio error codes are four-character codes reinterpreted as a
/// signed integer - 'nope' and the like - and are far easier to look up
/// that way round than as the number they decode to.
fn describe(status: sys::OSStatus) -> String {
  let bytes = status.to_be_bytes();

  if bytes
    .iter()
    .all(|byte| byte.is_ascii_graphic() || *byte == b' ')
  {
    format!("'{}' ({status})", String::from_utf8_lossy(&bytes))
  } else {
    status.to_string()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn names_match_on_a_case_insensitive_substring() {
    assert!(matches("Dustin's AirPods Pro #3", "airpods"));
    assert!(matches("CalDigit USB-C Pro Audio", "CalDigit"));
    assert!(!matches("Mac mini Speakers", "airpods"));
  }
}
