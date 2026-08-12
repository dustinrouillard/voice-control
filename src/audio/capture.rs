use anyhow::{Context, Result, anyhow, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample, StreamConfig};
use rubato::{
  Resampler, SincFixedIn, SincInterpolationParameters,
  SincInterpolationType, WindowFunction,
};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::audio::SAMPLE_RATE;

/// Input frames handed to rubato per call.
const RESAMPLE_CHUNK: usize = 1024;

/// ~2.5s of 20ms buffers. Deep enough to absorb a whisper pass
/// without dropping audio, shallow enough that a wedged consumer
/// does not silently accumulate minutes of stale sound.
const CHANNEL_DEPTH: usize = 128;

/// Holds the CoreAudio stream open. Dropping it stops capture.
pub struct Capture {
  /// Which device was actually opened, for the menu bar to name.
  pub device: String,
  _shutdown: mpsc::Sender<()>,
}

/// Opens the input device and returns a stream of 16 kHz mono frames.
///
/// `device_hint` is a case-insensitive substring of the device name;
/// empty selects the system default input.
pub fn start(
  device_hint: &str,
) -> Result<(Capture, mpsc::Receiver<Vec<f32>>)> {
  // Native-rate mono, straight off the audio thread.
  let (raw_tx, raw_rx) = mpsc::channel::<Vec<f32>>(CHANNEL_DEPTH);
  // 16 kHz mono, after resampling.
  let (out_tx, out_rx) = mpsc::channel::<Vec<f32>>(CHANNEL_DEPTH);

  let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
  let (ready_tx, ready_rx) =
    std::sync::mpsc::channel::<Result<(u32, String)>>();
  let hint = device_hint.to_string();

  // cpal::Stream is !Send, so it has to be built and dropped on one
  // dedicated thread rather than moved into a tokio task.
  std::thread::Builder::new()
    .name("audio-capture".into())
    .spawn(move || {
      let stream = match open_stream(&hint, raw_tx) {
        Ok((stream, rate, device)) => {
          let _ = ready_tx.send(Ok((rate, device)));
          stream
        }
        Err(why) => {
          let _ = ready_tx.send(Err(why));
          return;
        }
      };

      if let Err(why) = stream.play() {
        error!(error = %why, "failed to start the input stream");
        return;
      }

      // Park until the Capture handle is dropped.
      shutdown_rx.blocking_recv();
      debug!("capture thread shutting down");
    })
    .context("spawning the capture thread")?;

  let (native_rate, device) = ready_rx
    .recv()
    .context("capture thread died before reporting readiness")??;

  tokio::spawn(resample_task(native_rate, raw_rx, out_tx));

  Ok((
    Capture {
      device,
      _shutdown: shutdown_tx,
    },
    out_rx,
  ))
}

fn open_stream(
  hint: &str,
  tx: mpsc::Sender<Vec<f32>>,
) -> Result<(cpal::Stream, u32, String)> {
  let host = cpal::default_host();

  let device = if hint.is_empty() {
    host
      .default_input_device()
      .ok_or_else(|| anyhow!("no default input device"))?
  } else {
    let needle = hint.to_lowercase();
    host
      .input_devices()
      .context("enumerating input devices")?
      .find(|d| d.to_string().to_lowercase().contains(&needle))
      .ok_or_else(|| anyhow!("no input device matching {hint:?}"))?
  };

  let supported = device
    .default_input_config()
    .context("querying the default input config")?;
  let sample_format = supported.sample_format();
  let config: StreamConfig = supported.into();
  let channels = config.channels as usize;
  let rate = config.sample_rate;
  let name = device.to_string();

  info!(
    device = %device,
    rate,
    channels,
    format = ?sample_format,
    "opened input device"
  );

  let stream = match sample_format {
    SampleFormat::F32 => build::<f32>(&device, config, channels, tx),
    SampleFormat::I16 => build::<i16>(&device, config, channels, tx),
    SampleFormat::I32 => build::<i32>(&device, config, channels, tx),
    SampleFormat::U16 => build::<u16>(&device, config, channels, tx),
    other => bail!("unsupported sample format {other:?}"),
  }?;

  Ok((stream, rate, name))
}

fn build<T>(
  device: &cpal::Device,
  config: StreamConfig,
  channels: usize,
  tx: mpsc::Sender<Vec<f32>>,
) -> Result<cpal::Stream>
where
  T: SizedSample,
  f32: FromSample<T>,
{
  let stream = device
    .build_input_stream::<T, _, _>(
      config,
      move |data: &[T], _| {
        let mut mono = Vec::with_capacity(data.len() / channels);

        for frame in data.chunks_exact(channels) {
          let sum: f32 = frame.iter().map(|&s| f32::from_sample(s)).sum();
          mono.push(sum / channels as f32);
        }

        // Never block the audio thread. A full channel means the
        // pipeline is wedged, and stale audio is worth less than a
        // glitch-free stream.
        if tx.try_send(mono).is_err() {
          warn!("audio buffer full, dropping a frame");
        }
      },
      |why| error!(error = %why, "input stream error"),
      None,
    )
    .context("building the input stream")?;

  Ok(stream)
}

async fn resample_task(
  native_rate: u32,
  mut raw_rx: mpsc::Receiver<Vec<f32>>,
  out_tx: mpsc::Sender<Vec<f32>>,
) {
  if native_rate == SAMPLE_RATE {
    debug!("device is already at 16 kHz, bypassing the resampler");

    while let Some(chunk) = raw_rx.recv().await {
      if out_tx.send(chunk).await.is_err() {
        return;
      }
    }

    return;
  }

  let mut resampler = match make_resampler(native_rate) {
    Ok(resampler) => resampler,
    Err(why) => {
      error!(error = %why, "failed to build the resampler");
      return;
    }
  };

  // rubato wants exactly RESAMPLE_CHUNK frames per call, but cpal
  // hands us whatever size CoreAudio feels like.
  let mut pending: Vec<f32> = Vec::with_capacity(RESAMPLE_CHUNK * 2);

  while let Some(chunk) = raw_rx.recv().await {
    pending.extend_from_slice(&chunk);

    while pending.len() >= RESAMPLE_CHUNK {
      let input: Vec<f32> = pending.drain(..RESAMPLE_CHUNK).collect();

      match resampler.process(&[input], None) {
        Ok(mut out) => {
          let frames = out.remove(0);

          if !frames.is_empty() && out_tx.send(frames).await.is_err() {
            return;
          }
        }
        Err(why) => error!(error = %why, "resampling failed"),
      }
    }
  }
}

fn make_resampler(native_rate: u32) -> Result<SincFixedIn<f32>> {
  let parameters = SincInterpolationParameters {
    sinc_len: 128,
    f_cutoff: 0.95,
    interpolation: SincInterpolationType::Linear,
    oversampling_factor: 128,
    window: WindowFunction::BlackmanHarris2,
  };

  let resampler = SincFixedIn::<f32>::new(
    SAMPLE_RATE as f64 / native_rate as f64,
    1.1,
    parameters,
    RESAMPLE_CHUNK,
    1,
  )
  .context("constructing the sinc resampler")?;

  Ok(resampler)
}
