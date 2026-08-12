//! The slice of the CoreAudio HAL and CoreFoundation this needs.
//!
//! Hand-written rather than taken from coreaudio-sys: what follows is
//! the whole of the API surface, and bindgen would pull libclang into
//! the build to generate it.

use std::ffi::{c_char, c_void};

pub type AudioObjectID = u32;
pub type OSStatus = i32;
pub type CFIndex = isize;

/// The HAL itself, the object whose properties are the system-wide
/// settings - the device list and the default devices.
pub const SYSTEM_OBJECT: AudioObjectID = 1;

/// CoreAudio selectors are four-character codes, big endian.
const fn fourcc(code: &[u8; 4]) -> u32 {
  u32::from_be_bytes(*code)
}

pub const DEFAULT_INPUT_DEVICE: u32 = fourcc(b"dIn ");
pub const DEFAULT_OUTPUT_DEVICE: u32 = fourcc(b"dOut");
/// Where alert sounds and interface noises go. The Sound pane keeps it
/// with the output device unless you tell it otherwise.
pub const DEFAULT_SYSTEM_OUTPUT_DEVICE: u32 = fourcc(b"sOut");
pub const DEVICES: u32 = fourcc(b"dev#");
pub const NAME: u32 = fourcc(b"lnam");
pub const STREAMS: u32 = fourcc(b"stm#");

pub const SCOPE_GLOBAL: u32 = fourcc(b"glob");
pub const SCOPE_INPUT: u32 = fourcc(b"inpt");
pub const SCOPE_OUTPUT: u32 = fourcc(b"outp");
pub const ELEMENT_MAIN: u32 = 0;

pub const UTF8: u32 = 0x0800_0100;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AudioObjectPropertyAddress {
  pub selector: u32,
  pub scope: u32,
  pub element: u32,
}

#[link(name = "CoreAudio", kind = "framework")]
unsafe extern "C" {
  pub fn AudioObjectGetPropertyDataSize(
    object: AudioObjectID,
    address: *const AudioObjectPropertyAddress,
    qualifier_size: u32,
    qualifier: *const c_void,
    size: *mut u32,
  ) -> OSStatus;

  pub fn AudioObjectGetPropertyData(
    object: AudioObjectID,
    address: *const AudioObjectPropertyAddress,
    qualifier_size: u32,
    qualifier: *const c_void,
    size: *mut u32,
    data: *mut c_void,
  ) -> OSStatus;

  pub fn AudioObjectSetPropertyData(
    object: AudioObjectID,
    address: *const AudioObjectPropertyAddress,
    qualifier_size: u32,
    qualifier: *const c_void,
    size: u32,
    data: *const c_void,
  ) -> OSStatus;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
  pub fn CFRelease(cf: *const c_void);

  pub fn CFStringGetLength(string: *const c_void) -> CFIndex;

  pub fn CFStringGetMaximumSizeForEncoding(
    length: CFIndex,
    encoding: u32,
  ) -> CFIndex;

  /// Returns a `Boolean`, which is a byte rather than the `bool` it
  /// looks like.
  pub fn CFStringGetCString(
    string: *const c_void,
    buffer: *mut c_char,
    buffer_size: CFIndex,
    encoding: u32,
  ) -> u8;
}
