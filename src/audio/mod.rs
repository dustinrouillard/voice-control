pub mod capture;
pub mod ring;
pub mod vad;

/// Everything downstream of the resampler works at 16 kHz mono f32:
/// what openWakeWord's models were trained on, Silero's only supported
/// rate, and whisper's required rate.
pub const SAMPLE_RATE: u32 = 16_000;

pub fn samples_to_ms(samples: usize) -> usize {
  samples * 1000 / SAMPLE_RATE as usize
}

pub fn ms_to_samples(ms: usize) -> usize {
  ms * SAMPLE_RATE as usize / 1000
}
