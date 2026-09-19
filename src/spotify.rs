//! Spotify Web API playback control.
//!
//! Unlike the system media keys, these calls always target Spotify and
//! expose the controls that do not exist as keys at all: volume, seek,
//! repeat, shuffle, queueing, and Spotify Connect device transfer.

use std::fs::OpenOptions;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use reqwest::{Method, Response, StatusCode, Url};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

use crate::config::expand_tilde;

const API_URL: &str = "https://api.spotify.com/v1";
const ACCOUNTS_URL: &str = "https://accounts.spotify.com";
const DEFAULT_REDIRECT_URI: &str = "http://127.0.0.1:8888/callback";
const DEFAULT_TOKEN_FILE: &str =
  "~/.config/voice-control/spotify-refresh-token";
const TIMEOUT: Duration = Duration::from_secs(8);
const SCOPES: &str = "user-modify-playback-state user-read-playback-state";

#[derive(Debug, Clone, Deserialize)]
pub struct SpotifyConfig {
  /// The public client id from a Spotify developer application. Empty
  /// reads `SPOTIFY_CLIENT_ID`.
  #[serde(default)]
  pub client_id: String,
  /// A PKCE refresh token. Empty reads `SPOTIFY_REFRESH_TOKEN`, then
  /// `token_file`.
  #[serde(default)]
  pub refresh_token: String,
  /// Where `voice-control spotify authorize` stores the refresh token.
  #[serde(default = "default_token_file")]
  pub token_file: String,
  /// Must exactly match a redirect URI registered for the Spotify app.
  #[serde(default = "default_redirect_uri")]
  pub redirect_uri: String,
}

impl Default for SpotifyConfig {
  fn default() -> Self {
    Self {
      client_id: String::new(),
      refresh_token: String::new(),
      token_file: default_token_file(),
      redirect_uri: default_redirect_uri(),
    }
  }
}

impl SpotifyConfig {
  pub fn configured(&self) -> bool {
    !self.client_id().is_empty()
  }

  fn client_id(&self) -> String {
    if self.client_id.is_empty() {
      std::env::var("SPOTIFY_CLIENT_ID").unwrap_or_default()
    } else {
      self.client_id.clone()
    }
  }

  fn token_path(&self) -> PathBuf {
    expand_tilde(&self.token_file)
  }

  fn refresh_token(&self) -> Result<String> {
    if !self.refresh_token.is_empty() {
      return Ok(self.refresh_token.clone());
    }

    if let Ok(token) = std::env::var("SPOTIFY_REFRESH_TOKEN")
      && !token.trim().is_empty()
    {
      return Ok(token);
    }

    let path = self.token_path();
    let token = std::fs::read_to_string(&path).with_context(|| {
      format!(
        "reading Spotify refresh token from {} - run `voice-control \
         spotify authorize` first",
        path.display()
      )
    })?;

    if token.trim().is_empty() {
      bail!("Spotify refresh token file {} is empty", path.display());
    }

    Ok(token.trim().to_string())
  }
}

fn default_redirect_uri() -> String {
  DEFAULT_REDIRECT_URI.into()
}

fn default_token_file() -> String {
  DEFAULT_TOKEN_FILE.into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepeatState {
  Off,
  Track,
  Context,
}

impl RepeatState {
  fn as_str(self) -> &'static str {
    match self {
      Self::Off => "off",
      Self::Track => "track",
      Self::Context => "context",
    }
  }
}

impl std::str::FromStr for RepeatState {
  type Err = anyhow::Error;

  fn from_str(value: &str) -> Result<Self> {
    match value {
      "off" => Ok(Self::Off),
      "track" => Ok(Self::Track),
      "context" => Ok(Self::Context),
      _ => bail!("repeat state is off, track or context"),
    }
  }
}

/// One of Spotify's playback-modifying Player API operations.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum SpotifyAction {
  Play {
    #[serde(default)]
    device_id: Option<String>,
    #[serde(default)]
    context_uri: Option<String>,
    #[serde(default)]
    uris: Vec<String>,
    #[serde(default)]
    offset_position: Option<u32>,
    #[serde(default)]
    offset_uri: Option<String>,
    #[serde(default)]
    position_ms: Option<u64>,
  },
  Pause {
    #[serde(default)]
    device_id: Option<String>,
  },
  Next {
    #[serde(default)]
    device_id: Option<String>,
  },
  Previous {
    #[serde(default)]
    device_id: Option<String>,
  },
  Seek {
    position_ms: u64,
    #[serde(default)]
    device_id: Option<String>,
  },
  Repeat {
    state: RepeatState,
    #[serde(default)]
    device_id: Option<String>,
  },
  Shuffle {
    state: bool,
    #[serde(default)]
    device_id: Option<String>,
  },
  Volume {
    percent: u8,
    #[serde(default)]
    device_id: Option<String>,
  },
  /// Read the target device's current volume and add this signed
  /// percentage, clamping the result to 0..=100.
  VolumeChange {
    percent: i8,
    #[serde(default)]
    device_id: Option<String>,
  },
  Queue {
    uri: String,
    #[serde(default)]
    device_id: Option<String>,
  },
  Transfer {
    device_id: String,
    #[serde(default)]
    play: Option<bool>,
  },
}

impl SpotifyAction {
  pub fn validate(&self) -> Result<()> {
    match self {
      Self::Play {
        context_uri,
        uris,
        offset_position,
        offset_uri,
        ..
      } => {
        if context_uri.is_some() && !uris.is_empty() {
          bail!("spotify play sets both context_uri and uris");
        }
        if offset_position.is_some() && offset_uri.is_some() {
          bail!("spotify play sets both offset_position and offset_uri");
        }
        if (offset_position.is_some() || offset_uri.is_some())
          && context_uri.is_none()
        {
          bail!("spotify play sets an offset without context_uri");
        }
      }
      Self::Volume { percent, .. } if *percent > 100 => {
        bail!("spotify volume percent must be from 0 through 100");
      }
      Self::VolumeChange { percent, .. }
        if *percent == 0 || !(-100..=100).contains(percent) =>
      {
        bail!(
          "spotify volume_change percent must be from -100 through \
           100, excluding zero"
        );
      }
      Self::Queue { uri, .. } if uri.trim().is_empty() => {
        bail!("spotify queue uri is empty");
      }
      Self::Transfer { device_id, .. } if device_id.trim().is_empty() => {
        bail!("spotify transfer device_id is empty");
      }
      _ => {}
    }

    if let Some(device_id) = self.device_id()
      && device_id.trim().is_empty()
    {
      bail!("spotify device_id is empty");
    }

    Ok(())
  }

  pub fn target(&self) -> String {
    match self {
      Self::Play { .. } => "spotify play".into(),
      Self::Pause { .. } => "spotify pause".into(),
      Self::Next { .. } => "spotify next".into(),
      Self::Previous { .. } => "spotify previous".into(),
      Self::Seek { position_ms, .. } => {
        format!("spotify seek to {position_ms}ms")
      }
      Self::Repeat { state, .. } => {
        format!("spotify repeat {}", state.as_str())
      }
      Self::Shuffle { state, .. } => {
        format!("spotify shuffle {}", if *state { "on" } else { "off" })
      }
      Self::Volume { percent, .. } => {
        format!("spotify volume {percent}%")
      }
      Self::VolumeChange { percent, .. } => {
        format!("spotify volume {percent:+}%")
      }
      Self::Queue { uri, .. } => format!("spotify queue {uri}"),
      Self::Transfer {
        device_id, play, ..
      } => match play {
        Some(play) => format!(
          "spotify transfer to {device_id:?} and {}",
          if *play { "play" } else { "pause" }
        ),
        None => format!("spotify transfer to {device_id:?}"),
      },
    }
  }

  fn device_id(&self) -> Option<&str> {
    match self {
      Self::Play { device_id, .. }
      | Self::Pause { device_id }
      | Self::Next { device_id }
      | Self::Previous { device_id }
      | Self::Seek { device_id, .. }
      | Self::Repeat { device_id, .. }
      | Self::Shuffle { device_id, .. }
      | Self::Volume { device_id, .. }
      | Self::VolumeChange { device_id, .. }
      | Self::Queue { device_id, .. } => device_id.as_deref(),
      Self::Transfer { device_id, .. } => Some(device_id),
    }
  }

  fn request(&self) -> Option<RequestSpec> {
    let device = |id: &Option<String>| {
      id.as_ref()
        .map(|id| vec![("device_id", id.clone())])
        .unwrap_or_default()
    };

    match self {
      Self::Play {
        device_id,
        context_uri,
        uris,
        offset_position,
        offset_uri,
        position_ms,
      } => {
        let mut body = serde_json::Map::new();
        if let Some(uri) = context_uri {
          body.insert("context_uri".into(), json!(uri));
        }
        if !uris.is_empty() {
          body.insert("uris".into(), json!(uris));
        }
        if let Some(position) = offset_position {
          body.insert("offset".into(), json!({ "position": position }));
        }
        if let Some(uri) = offset_uri {
          body.insert("offset".into(), json!({ "uri": uri }));
        }
        if let Some(position) = position_ms {
          body.insert("position_ms".into(), json!(position));
        }

        Some(RequestSpec {
          method: Method::PUT,
          path: "/me/player/play",
          query: device(device_id),
          body: (!body.is_empty()).then_some(Value::Object(body)),
        })
      }
      Self::Pause { device_id } => Some(RequestSpec::plain(
        Method::PUT,
        "/me/player/pause",
        device(device_id),
      )),
      Self::Next { device_id } => Some(RequestSpec::plain(
        Method::POST,
        "/me/player/next",
        device(device_id),
      )),
      Self::Previous { device_id } => Some(RequestSpec::plain(
        Method::POST,
        "/me/player/previous",
        device(device_id),
      )),
      Self::Seek {
        position_ms,
        device_id,
      } => {
        let mut query = vec![("position_ms", position_ms.to_string())];
        query.extend(device(device_id));
        Some(RequestSpec::plain(Method::PUT, "/me/player/seek", query))
      }
      Self::Repeat {
        state, device_id, ..
      } => {
        let mut query = vec![("state", state.as_str().into())];
        query.extend(device(device_id));
        Some(RequestSpec::plain(Method::PUT, "/me/player/repeat", query))
      }
      Self::Shuffle {
        state, device_id, ..
      } => {
        let mut query = vec![("state", state.to_string())];
        query.extend(device(device_id));
        Some(RequestSpec::plain(Method::PUT, "/me/player/shuffle", query))
      }
      Self::Volume { percent, device_id } => {
        let mut query = vec![("volume_percent", percent.to_string())];
        query.extend(device(device_id));
        Some(RequestSpec::plain(Method::PUT, "/me/player/volume", query))
      }
      Self::VolumeChange { .. } => None,
      Self::Queue { uri, device_id } => {
        let mut query = vec![("uri", uri.clone())];
        query.extend(device(device_id));
        Some(RequestSpec::plain(Method::POST, "/me/player/queue", query))
      }
      Self::Transfer {
        device_id, play, ..
      } => {
        let mut body = json!({ "device_ids": [device_id] });
        if let Some(play) = play {
          body["play"] = json!(play);
        }
        Some(RequestSpec {
          method: Method::PUT,
          path: "/me/player",
          query: Vec::new(),
          body: Some(body),
        })
      }
    }
  }
}

#[derive(Debug)]
struct RequestSpec {
  method: Method,
  path: &'static str,
  query: Vec<(&'static str, String)>,
  body: Option<Value>,
}

impl RequestSpec {
  fn plain(
    method: Method,
    path: &'static str,
    query: Vec<(&'static str, String)>,
  ) -> Self {
    Self {
      method,
      path,
      query,
      body: None,
    }
  }
}

struct CachedToken {
  value: String,
  expires: Instant,
}

struct AuthState {
  refresh_token: String,
  access_token: Option<CachedToken>,
}

pub struct Spotify {
  client: reqwest::Client,
  client_id: String,
  token_path: Option<PathBuf>,
  auth: Mutex<AuthState>,
}

impl Spotify {
  pub fn connect(config: &SpotifyConfig) -> Result<Option<Self>> {
    if !config.configured() {
      return Ok(None);
    }

    let from_file = config.refresh_token.is_empty()
      && std::env::var("SPOTIFY_REFRESH_TOKEN").is_err();
    let refresh_token = config.refresh_token()?;
    let client = reqwest::Client::builder()
      .timeout(TIMEOUT)
      .build()
      .context("building the Spotify HTTP client")?;

    Ok(Some(Self {
      client,
      client_id: config.client_id(),
      token_path: from_file.then(|| config.token_path()),
      auth: Mutex::new(AuthState {
        refresh_token,
        access_token: None,
      }),
    }))
  }

  pub async fn run(&self, action: &SpotifyAction) -> Result<()> {
    action.validate()?;

    if let SpotifyAction::VolumeChange { percent, device_id } = action {
      let current = self.current_volume(device_id.as_deref()).await?;
      let percent = changed_volume(current, *percent);
      let action = SpotifyAction::Volume {
        percent,
        device_id: device_id.clone(),
      };
      return self.execute(action.request().unwrap()).await;
    }

    self.execute(action.request().unwrap()).await
  }

  pub async fn warm(&self) -> Result<()> {
    self.access_token().await.map(|_| ())
  }

  pub async fn devices(&self) -> Result<Vec<Device>> {
    let response = self
      .send(RequestSpec::plain(
        Method::GET,
        "/me/player/devices",
        vec![],
      ))
      .await?;
    let devices: Devices = response
      .json()
      .await
      .context("reading Spotify's device list")?;
    Ok(devices.devices)
  }

  async fn current_volume(&self, device_id: Option<&str>) -> Result<u8> {
    let devices = self.devices().await?;
    let device = match device_id {
      Some(id) => devices
        .iter()
        .find(|device| device.id.as_deref() == Some(id)),
      None => devices.iter().find(|device| device.is_active),
    };

    let Some(device) = device else {
      match device_id {
        Some(id) => {
          bail!("Spotify has no available device with id {id:?}")
        }
        None => bail!("Spotify has no active playback device"),
      }
    };

    if !device.supports_volume {
      bail!("Spotify device {:?} does not support volume", device.name);
    }

    device.volume_percent.ok_or_else(|| {
      anyhow!(
        "Spotify did not report a volume for device {:?}",
        device.name
      )
    })
  }

  async fn execute(&self, spec: RequestSpec) -> Result<()> {
    self.send(spec).await.map(|_| ())
  }

  async fn send(&self, spec: RequestSpec) -> Result<Response> {
    let token = self.access_token().await?;
    let response = self.send_once(&spec, &token).await?;

    // A token may be revoked before its nominal expiry. Refresh once;
    // anything after that is a real authorization failure.
    let response = if response.status() == StatusCode::UNAUTHORIZED {
      self.auth.lock().await.access_token = None;
      let token = self.access_token().await?;
      self.send_once(&spec, &token).await?
    } else {
      response
    };

    if !response.status().is_success() {
      return Err(response_error(response).await);
    }

    Ok(response)
  }

  async fn send_once(
    &self,
    spec: &RequestSpec,
    token: &str,
  ) -> Result<Response> {
    let url = format!("{API_URL}{}", spec.path);
    let mut request = self
      .client
      .request(spec.method.clone(), &url)
      .bearer_auth(token)
      .query(&spec.query);
    if let Some(body) = &spec.body {
      request = request.json(body);
    } else {
      // Spotify's edge rejects body-less POST/PUT mutations with 411
      // Length Required. An explicit empty body makes reqwest emit
      // Content-Length: 0 while preserving the documented query
      // parameter form used by volume, seek, repeat, and shuffle.
      request = request
        .header(reqwest::header::CONTENT_LENGTH, "0")
        .body(Vec::<u8>::new());
    }
    request
      .send()
      .await
      .with_context(|| format!("calling {url}"))
  }

  async fn access_token(&self) -> Result<String> {
    let mut auth = self.auth.lock().await;

    if let Some(token) = &auth.access_token
      && token.expires > Instant::now()
    {
      return Ok(token.value.clone());
    }

    let url = format!("{ACCOUNTS_URL}/api/token");
    let response = self
      .client
      .post(&url)
      .form(&[
        ("grant_type", "refresh_token"),
        ("refresh_token", auth.refresh_token.as_str()),
        ("client_id", self.client_id.as_str()),
      ])
      .send()
      .await
      .context("refreshing the Spotify access token")?;

    if !response.status().is_success() {
      return Err(response_error(response).await);
    }

    let token: TokenResponse = response
      .json()
      .await
      .context("reading Spotify's token response")?;

    if let Some(refresh_token) = token.refresh_token
      && refresh_token != auth.refresh_token
    {
      if let Some(path) = &self.token_path {
        write_token(path, &refresh_token)?;
      }
      auth.refresh_token = refresh_token;
    }

    let expires = Instant::now()
      + Duration::from_secs(token.expires_in.saturating_sub(30));
    auth.access_token = Some(CachedToken {
      value: token.access_token.clone(),
      expires,
    });

    Ok(token.access_token)
  }
}

fn changed_volume(current: u8, change: i8) -> u8 {
  (current as i16 + change as i16).clamp(0, 100) as u8
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
  access_token: String,
  expires_in: u64,
  #[serde(default)]
  refresh_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Devices {
  devices: Vec<Device>,
}

#[derive(Debug, Deserialize)]
pub struct Device {
  pub id: Option<String>,
  pub is_active: bool,
  pub is_restricted: bool,
  pub name: String,
  #[serde(rename = "type")]
  pub kind: String,
  pub volume_percent: Option<u8>,
  pub supports_volume: bool,
}

async fn response_error(response: Response) -> anyhow::Error {
  let status = response.status();
  let retry_after = response
    .headers()
    .get(reqwest::header::RETRY_AFTER)
    .and_then(|value| value.to_str().ok())
    .map(str::to_owned);
  let body = response.text().await.unwrap_or_default();
  let detail = serde_json::from_str::<Value>(&body)
    .ok()
    .and_then(|value| {
      value
        .pointer("/error/message")
        .or_else(|| value.get("error_description"))
        .or_else(|| value.get("error"))
        .and_then(Value::as_str)
        .map(str::to_owned)
    })
    .filter(|message| !message.is_empty())
    .unwrap_or(body);

  if status == StatusCode::TOO_MANY_REQUESTS {
    return anyhow!(
      "Spotify returned {status}: rate limited{}{}",
      retry_after
        .as_deref()
        .map(|seconds| format!("; retry after {seconds}s"))
        .unwrap_or_default(),
      if detail.is_empty() {
        String::new()
      } else {
        format!(": {detail}")
      }
    );
  }

  anyhow!(
    "Spotify returned {status}{}",
    if detail.is_empty() {
      String::new()
    } else {
      format!(": {detail}")
    }
  )
}

/// Runs the one-time PKCE authorization flow and stores its refresh
/// token in the configured file with owner-only permissions.
pub async fn authorize(config: &SpotifyConfig) -> Result<PathBuf> {
  let client_id = config.client_id();
  if client_id.is_empty() {
    bail!(
      "[spotify] needs client_id (or SPOTIFY_CLIENT_ID) before it can \
       authorize"
    );
  }

  let redirect = Url::parse(&config.redirect_uri)
    .context("parsing spotify redirect_uri")?;
  if redirect.scheme() != "http"
    || !matches!(redirect.host_str(), Some("127.0.0.1") | Some("::1"))
  {
    bail!(
      "spotify redirect_uri must use an explicit loopback address, for \
       example {DEFAULT_REDIRECT_URI:?}"
    );
  }
  let port = redirect
    .port_or_known_default()
    .ok_or_else(|| anyhow!("spotify redirect_uri has no port"))?;
  let host = redirect.host_str().unwrap();
  let listener = TcpListener::bind((host, port))
    .await
    .with_context(|| format!("listening for Spotify at {host}:{port}"))?;

  let verifier = random_urlsafe(64)?;
  let challenge =
    URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
  let state = random_urlsafe(24)?;
  let mut auth_url = Url::parse(&format!("{ACCOUNTS_URL}/authorize"))?;
  auth_url.query_pairs_mut().extend_pairs([
    ("client_id", client_id.as_str()),
    ("response_type", "code"),
    ("redirect_uri", config.redirect_uri.as_str()),
    ("scope", SCOPES),
    ("state", state.as_str()),
    ("code_challenge_method", "S256"),
    ("code_challenge", challenge.as_str()),
  ]);

  println!("Open this URL if the browser does not open:\n\n{auth_url}\n");
  let _ = std::process::Command::new("/usr/bin/open")
    .arg(auth_url.as_str())
    .status();

  let (mut stream, _) = listener.accept().await?;
  let mut request = vec![0_u8; 16 * 1024];
  let read = stream.read(&mut request).await?;
  let request = String::from_utf8_lossy(&request[..read]);
  let target = request
    .lines()
    .next()
    .and_then(|line| line.split_whitespace().nth(1))
    .ok_or_else(|| anyhow!("browser sent an invalid Spotify callback"))?;
  let callback = Url::parse(&format!("http://127.0.0.1{target}"))?;
  let query: std::collections::HashMap<_, _> =
    callback.query_pairs().into_owned().collect();

  let received_state = query.get("state").map(String::as_str);
  if received_state != Some(state.as_str()) {
    send_browser_response(&mut stream, false).await?;
    bail!("Spotify authorization callback had the wrong state");
  }
  if let Some(error) = query.get("error") {
    send_browser_response(&mut stream, false).await?;
    bail!("Spotify authorization failed: {error}");
  }
  let code = query.get("code").ok_or_else(|| {
    anyhow!("Spotify callback had no authorization code")
  })?;

  let client = reqwest::Client::builder().timeout(TIMEOUT).build()?;
  let token_url = format!("{ACCOUNTS_URL}/api/token");
  let response = client
    .post(token_url)
    .form(&[
      ("client_id", client_id.as_str()),
      ("grant_type", "authorization_code"),
      ("code", code.as_str()),
      ("redirect_uri", config.redirect_uri.as_str()),
      ("code_verifier", verifier.as_str()),
    ])
    .send()
    .await
    .context("exchanging Spotify's authorization code")?;
  if !response.status().is_success() {
    send_browser_response(&mut stream, false).await?;
    return Err(response_error(response).await);
  }
  let token: TokenResponse = response.json().await?;
  let refresh_token = token
    .refresh_token
    .ok_or_else(|| anyhow!("Spotify did not return a refresh token"))?;
  let path = config.token_path();
  write_token(&path, &refresh_token)?;
  send_browser_response(&mut stream, true).await?;

  Ok(path)
}

async fn send_browser_response(
  stream: &mut tokio::net::TcpStream,
  success: bool,
) -> Result<()> {
  let message = if success {
    "Spotify authorization complete. You can close this tab."
  } else {
    "Spotify authorization failed. Return to the terminal for details."
  };
  let body = format!(
    "<!doctype html><meta charset=utf-8><title>voice-control</title>\
     <body style=\"font:20px system-ui;margin:4rem\">{message}</body>"
  );
  let response = format!(
    "HTTP/1.1 {}\r\nContent-Type: text/html; charset=utf-8\r\n\
     Content-Length: {}\r\nConnection: close\r\n\r\n{}",
    if success { "200 OK" } else { "400 Bad Request" },
    body.len(),
    body
  );
  stream.write_all(response.as_bytes()).await?;
  Ok(())
}

fn random_urlsafe(bytes: usize) -> Result<String> {
  let mut random = vec![0_u8; bytes];
  std::fs::File::open("/dev/urandom")
    .and_then(|mut file| std::io::Read::read_exact(&mut file, &mut random))
    .context("reading random bytes for Spotify authorization")?;
  Ok(URL_SAFE_NO_PAD.encode(random))
}

fn write_token(path: &Path, token: &str) -> Result<()> {
  if let Some(parent) = path.parent() {
    std::fs::create_dir_all(parent)
      .with_context(|| format!("creating {}", parent.display()))?;
  }
  let mut file = OpenOptions::new()
    .create(true)
    .truncate(true)
    .write(true)
    .mode(0o600)
    .open(path)
    .with_context(|| format!("writing {}", path.display()))?;
  std::io::Write::write_all(&mut file, token.as_bytes())?;
  file.sync_all()?;
  std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  fn parse(action: &str) -> SpotifyAction {
    #[derive(Debug, Deserialize)]
    struct Wrapper {
      spotify: SpotifyAction,
    }

    toml::from_str::<Wrapper>(&format!("spotify = {action}"))
      .unwrap()
      .spotify
  }

  #[test]
  fn volume_maps_to_the_player_volume_endpoint() {
    let action = parse(r#"{ action = "volume", percent = 35 }"#);
    action.validate().unwrap();
    let request = action.request().unwrap();

    assert_eq!(request.method, Method::PUT);
    assert_eq!(request.path, "/me/player/volume");
    assert_eq!(request.query, [("volume_percent", "35".into())]);
  }

  #[test]
  fn every_native_control_maps_to_the_documented_endpoint() {
    let cases = [
      (r#"{ action = "play" }"#, Method::PUT, "/me/player/play"),
      (r#"{ action = "pause" }"#, Method::PUT, "/me/player/pause"),
      (r#"{ action = "next" }"#, Method::POST, "/me/player/next"),
      (
        r#"{ action = "previous" }"#,
        Method::POST,
        "/me/player/previous",
      ),
      (
        r#"{ action = "seek", position_ms = 5 }"#,
        Method::PUT,
        "/me/player/seek",
      ),
      (
        r#"{ action = "repeat", state = "track" }"#,
        Method::PUT,
        "/me/player/repeat",
      ),
      (
        r#"{ action = "shuffle", state = true }"#,
        Method::PUT,
        "/me/player/shuffle",
      ),
      (
        r#"{ action = "queue", uri = "spotify:track:x" }"#,
        Method::POST,
        "/me/player/queue",
      ),
      (
        r#"{ action = "transfer", device_id = "x" }"#,
        Method::PUT,
        "/me/player",
      ),
    ];

    for (raw, method, path) in cases {
      let action = parse(raw);
      action.validate().unwrap();
      let request = action.request().unwrap();
      assert_eq!(request.method, method, "{raw}");
      assert_eq!(request.path, path, "{raw}");
    }
  }

  #[test]
  fn play_carries_context_offset_and_position() {
    let action = parse(
      r#"{ action = "play", context_uri = "spotify:album:x", offset_position = 2, position_ms = 500 }"#,
    );
    action.validate().unwrap();
    let body = action.request().unwrap().body.unwrap();

    assert_eq!(body["context_uri"], "spotify:album:x");
    assert_eq!(body["offset"]["position"], 2);
    assert_eq!(body["position_ms"], 500);
  }

  #[test]
  fn rejects_invalid_volume_and_ambiguous_play() {
    assert!(
      parse(r#"{ action = "volume", percent = 101 }"#)
        .validate()
        .is_err()
    );
    assert!(
      parse(
        r#"{ action = "play", context_uri = "spotify:album:x", uris = ["spotify:track:y"] }"#,
      )
      .validate()
      .is_err()
    );
  }

  #[test]
  fn relative_volume_is_clamped_to_spotifys_range() {
    assert_eq!(changed_volume(50, 10), 60);
    assert_eq!(changed_volume(95, 10), 100);
    assert_eq!(changed_volume(5, -10), 0);
  }

  #[test]
  fn rejects_fields_that_belong_to_a_different_action() {
    let why = toml::from_str::<SpotifyAction>(
      "action = \"volume\"\npercent = 30\nstate = true",
    )
    .unwrap_err()
    .to_string();

    assert!(why.contains("unknown field"), "{why}");
  }
}
