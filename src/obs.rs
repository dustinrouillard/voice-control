use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio_tungstenite::tungstenite::Message;
use tracing::debug;

/// obs-websocket is on localhost or the LAN; if it has not completed
/// the handshake by now it is not going to.
const TIMEOUT: Duration = Duration::from_secs(4);

/// The RPC version this client speaks. obs-websocket 5.x.
const RPC_VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize)]
pub struct ObsConfig {
  /// `ws://127.0.0.1:4455`, or `wss://` for a remote instance.
  #[serde(default = "default_url")]
  pub url: String,
  /// Leave empty to take it from `OBS_PASSWORD` instead, which keeps
  /// the secret out of a config file.
  #[serde(default)]
  pub password: String,
}

fn default_url() -> String {
  "ws://127.0.0.1:4455".into()
}

impl Default for ObsConfig {
  fn default() -> Self {
    Self {
      url: default_url(),
      password: String::new(),
    }
  }
}

impl ObsConfig {
  fn resolved_password(&self) -> String {
    if self.password.is_empty() {
      std::env::var("OBS_PASSWORD").unwrap_or_default()
    } else {
      self.password.clone()
    }
  }
}

/// Switches the active OBS scene.
///
/// Connects per call rather than holding the socket open. Commands are
/// seconds apart at best, and a fresh connection sidesteps every way a
/// long-lived one can rot — OBS restarting, the machine sleeping, the
/// server being toggled off mid-session. The handshake costs a few
/// milliseconds against a pipeline that already spent a second
/// listening.
pub async fn set_scene(config: &ObsConfig, scene: &str) -> Result<()> {
  let scene = scene.to_string();

  tokio::time::timeout(TIMEOUT, async move {
    let mut session = Session::connect(config).await?;
    session
      .request("SetCurrentProgramScene", json!({ "sceneName": scene }))
      .await?;
    Ok::<_, anyhow::Error>(())
  })
  .await
  .map_err(|_| anyhow!("obs did not respond within {TIMEOUT:?}"))?
}

/// What a source command does to a scene item.
///
/// `toggle` is the default because that is how a source command is
/// usually said - "computa, ps5" to put it up and the same words again
/// to take it down - while `show` / `hide` exist for when you want to
/// say which way it goes and not have to remember where it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Visibility {
  #[serde(alias = "on")]
  Show,
  #[serde(alias = "off")]
  Hide,
  #[default]
  Toggle,
}

impl Visibility {
  pub fn as_str(self) -> &'static str {
    match self {
      Visibility::Show => "show",
      Visibility::Hide => "hide",
      Visibility::Toggle => "toggle",
    }
  }
}

/// A source to show or hide, and the Move filters that animate it.
///
/// Without the filters this is a bare visibility flip. With them it is
/// the sequence a Companion button runs, which is the order the
/// animation needs and not one you can get from a single request.
pub struct SourceAction<'a> {
  pub source: &'a str,
  /// The scene the source lives in, and where the filters live.
  /// `None` means whichever scene is on program right now.
  pub scene: Option<&'a str>,
  pub visibility: Visibility,
  /// Enabled after the source is shown, so the animation plays into a
  /// source that is already on screen.
  pub show_filter: Option<&'a str>,
  /// Enabled before the source is hidden, with `hide_delay` to let it
  /// finish - hiding first would cut the animation off at frame one.
  pub hide_filter: Option<&'a str>,
  pub hide_delay: Duration,
}

/// Shows or hides one source, playing its Move filters around the
/// visibility change.
///
/// Showing is source-then-filter and hiding is filter-then-source,
/// because a Move filter animates something that is on screen: on the
/// way in the source has to exist before it can move, and on the way
/// out it has to survive until the move ends. That asymmetry - and the
/// wait in the middle of the hide - is the whole reason this is a
/// sequence rather than one request.
///
/// Returns the visibility the source ended up with.
pub async fn run_source(
  config: &ObsConfig,
  action: SourceAction<'_>,
) -> Result<bool> {
  let scene = action.scene.map(str::to_string);
  let source = action.source.to_string();
  let show_filter = action.show_filter.map(str::to_string);
  let hide_filter = action.hide_filter.map(str::to_string);
  let SourceAction {
    visibility,
    hide_delay,
    ..
  } = action;

  // The wait is dead time in the middle of the sequence, so it has to
  // be on top of the budget for the round trips rather than eat into
  // it.
  let budget = TIMEOUT + hide_delay;

  tokio::time::timeout(budget, async move {
    let mut session = Session::connect(config).await?;

    let scene = match scene {
      Some(scene) => scene,
      None => session.current_scene().await?,
    };

    let item = session.find_item(&scene, &source).await?;

    let show = match visibility {
      Visibility::Show => true,
      Visibility::Hide => false,
      Visibility::Toggle => !item.enabled,
    };

    if show {
      session.set_item_enabled(&item, true).await?;

      if let Some(filter) = &show_filter {
        session.set_filter_enabled(&scene, filter, true).await?;
      }
    } else {
      if let Some(filter) = &hide_filter {
        session.set_filter_enabled(&scene, filter, true).await?;
        tokio::time::sleep(hide_delay).await;
      }

      session.set_item_enabled(&item, false).await?;
    }

    Ok::<_, anyhow::Error>(show)
  })
  .await
  .map_err(|_| anyhow!("obs did not respond within {budget:?}"))?
}

/// The filters on a scene or source, for `voice-control obs filters`.
pub async fn filter_list(
  config: &ObsConfig,
  target: Option<&str>,
) -> Result<(String, Vec<Filter>)> {
  let target = target.map(str::to_string);

  tokio::time::timeout(TIMEOUT, async move {
    let mut session = Session::connect(config).await?;

    let target = match target {
      Some(target) => target,
      None => session.current_scene().await?,
    };

    let filters = session.filters(&target).await?;

    Ok::<_, anyhow::Error>((target, filters))
  })
  .await
  .map_err(|_| anyhow!("obs did not respond within {TIMEOUT:?}"))?
}

/// One filter on a scene or source.
#[derive(Debug, Clone)]
pub struct Filter {
  pub name: String,
  pub enabled: bool,
  pub kind: String,
}

/// The sources in one scene, groups flattened, for
/// `voice-control obs sources`.
pub async fn source_list(
  config: &ObsConfig,
  scene: Option<&str>,
) -> Result<(String, Vec<SceneItem>)> {
  let scene = scene.map(str::to_string);

  tokio::time::timeout(TIMEOUT, async move {
    let mut session = Session::connect(config).await?;

    let scene = match scene {
      Some(scene) => scene,
      None => session.current_scene().await?,
    };

    let items = session.items_including_groups(&scene).await?;

    Ok::<_, anyhow::Error>((scene, items))
  })
  .await
  .map_err(|_| anyhow!("obs did not respond within {TIMEOUT:?}"))?
}

/// One entry in a scene's source list.
#[derive(Debug, Clone)]
pub struct SceneItem {
  /// The scene the item belongs to - or the group, when it is nested
  /// in one. obs-websocket takes a group name wherever it takes a
  /// scene name for scene item requests, so this is what the enable
  /// call is addressed to.
  pub scene: String,
  pub id: i64,
  pub name: String,
  pub enabled: bool,
  pub is_group: bool,
}

/// Scene names as OBS reports them, for `voice-control obs`.
pub async fn scene_list(
  config: &ObsConfig,
) -> Result<(Vec<String>, String)> {
  tokio::time::timeout(TIMEOUT, async move {
    let mut session = Session::connect(config).await?;
    let data = session.request("GetSceneList", json!({})).await?;

    let scenes = data
      .get("scenes")
      .and_then(Value::as_array)
      .ok_or_else(|| anyhow!("GetSceneList returned no scenes"))?
      .iter()
      .filter_map(|s| {
        s.get("sceneName").and_then(Value::as_str).map(String::from)
      })
      .collect();

    let current = data
      .get("currentProgramSceneName")
      .and_then(Value::as_str)
      .unwrap_or_default()
      .to_string();

    Ok::<_, anyhow::Error>((scenes, current))
  })
  .await
  .map_err(|_| anyhow!("obs did not respond within {TIMEOUT:?}"))?
}

type Socket = tokio_tungstenite::WebSocketStream<
  tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
>;

struct Session {
  socket: Socket,
  next_id: u64,
}

impl Session {
  async fn connect(config: &ObsConfig) -> Result<Self> {
    if !config.url.starts_with("ws://")
      && !config.url.starts_with("wss://")
    {
      bail!("obs url {:?} must start with ws:// or wss://", config.url);
    }

    let (socket, _) = tokio_tungstenite::connect_async(&config.url)
      .await
      .with_context(|| format!("connecting to {}", config.url))?;

    let mut session = Session { socket, next_id: 1 };
    session.identify(&config.resolved_password()).await?;

    Ok(session)
  }

  /// Hello -> Identify -> Identified.
  async fn identify(&mut self, password: &str) -> Result<()> {
    let hello = self.recv_op(0).await.context("waiting for obs Hello")?;

    let authentication = match hello.get("authentication") {
      Some(auth) => {
        if password.is_empty() {
          bail!(
            "obs requires a password - set it in [obs] password or \
             the OBS_PASSWORD environment variable"
          );
        }

        let challenge = auth
          .get("challenge")
          .and_then(Value::as_str)
          .ok_or_else(|| anyhow!("obs Hello had no challenge"))?;
        let salt = auth
          .get("salt")
          .and_then(Value::as_str)
          .ok_or_else(|| anyhow!("obs Hello had no salt"))?;

        Some(auth_string(password, salt, challenge))
      }
      // Authentication disabled in OBS. Sending one anyway is an
      // error, so only include it when asked.
      None => {
        debug!("obs has authentication disabled");
        None
      }
    };

    let mut data = json!({
      "rpcVersion": RPC_VERSION,
      // We only issue requests; subscribing would mean draining a
      // stream of events we never read.
      "eventSubscriptions": 0,
    });

    if let Some(authentication) = authentication {
      data["authentication"] = json!(authentication);
    }

    self.send(json!({ "op": 1, "d": data })).await?;
    self
      .recv_op(2)
      .await
      .context("obs rejected the identify handshake")?;

    Ok(())
  }

  async fn current_scene(&mut self) -> Result<String> {
    let data = self.request("GetCurrentProgramScene", json!({})).await?;

    // obs-websocket 5.5 renamed this to `sceneName` and kept the old
    // key alongside it; older builds only have the old one.
    data
      .get("sceneName")
      .or_else(|| data.get("currentProgramSceneName"))
      .and_then(Value::as_str)
      .map(String::from)
      .ok_or_else(|| anyhow!("obs did not say which scene is current"))
  }

  async fn item_enabled(&mut self, scene: &str, id: i64) -> Result<bool> {
    let data = self
      .request(
        "GetSceneItemEnabled",
        json!({ "sceneName": scene, "sceneItemId": id }),
      )
      .await?;

    data
      .get("sceneItemEnabled")
      .and_then(Value::as_bool)
      .ok_or_else(|| anyhow!("obs did not say if the item is enabled"))
  }

  async fn set_item_enabled(
    &mut self,
    item: &SceneItem,
    enabled: bool,
  ) -> Result<()> {
    self
      .request(
        "SetSceneItemEnabled",
        json!({
          "sceneName": item.scene,
          "sceneItemId": item.id,
          "sceneItemEnabled": enabled,
        }),
      )
      .await?;

    Ok(())
  }

  /// Enabling a Move filter is what starts the animation - the plugin
  /// treats the enable itself as the trigger, which is why this is a
  /// plain enable and never an off-then-on.
  async fn set_filter_enabled(
    &mut self,
    target: &str,
    filter: &str,
    enabled: bool,
  ) -> Result<()> {
    let known = self.filters(target).await.unwrap_or_default();

    // Same reasoning as a missing source: a filter name typed by hand
    // against a name OBS matches exactly, and the bare obs-websocket
    // error does not say what is there instead.
    if !known.iter().any(|candidate| candidate.name == filter) {
      let names: Vec<&str> =
        known.iter().map(|f| f.name.as_str()).collect();

      bail!(
        "{target:?} has no filter named {filter:?} - it has: {}",
        if names.is_empty() {
          "nothing".to_string()
        } else {
          names.join(", ")
        }
      );
    }

    self
      .request(
        "SetSourceFilterEnabled",
        json!({
          "sourceName": target,
          "filterName": filter,
          "filterEnabled": enabled,
        }),
      )
      .await?;

    Ok(())
  }

  async fn filters(&mut self, target: &str) -> Result<Vec<Filter>> {
    let data = self
      .request("GetSourceFilterList", json!({ "sourceName": target }))
      .await?;

    let filters = data
      .get("filters")
      .and_then(Value::as_array)
      .ok_or_else(|| anyhow!("GetSourceFilterList returned no filters"))?;

    Ok(
      filters
        .iter()
        .filter_map(|filter| {
          Some(Filter {
            name: filter
              .get("filterName")
              .and_then(Value::as_str)?
              .to_string(),
            enabled: filter
              .get("filterEnabled")
              .and_then(Value::as_bool)
              .unwrap_or(false),
            kind: filter
              .get("filterKind")
              .and_then(Value::as_str)
              .unwrap_or_default()
              .to_string(),
          })
        })
        .collect(),
    )
  }

  /// The named source's scene item, resolved by OBS itself and then,
  /// failing that, searched for in the scene's groups.
  ///
  /// The name is handed to `GetSceneItemId` rather than matched
  /// against our own listing because a scene can hold two items with
  /// the same name, and then which one you get comes down to search
  /// order. Letting OBS pick means we pick whatever every other
  /// obs-websocket client picks - the same item a Companion button
  /// would move - instead of inventing a second answer.
  async fn find_item(
    &mut self,
    scene: &str,
    source: &str,
  ) -> Result<SceneItem> {
    let found = self
      .request(
        "GetSceneItemId",
        json!({ "sceneName": scene, "sourceName": source }),
      )
      .await
      .ok()
      .and_then(|data| data.get("sceneItemId").and_then(Value::as_i64));

    if let Some(id) = found {
      return Ok(SceneItem {
        scene: scene.to_string(),
        id,
        name: source.to_string(),
        enabled: self.item_enabled(scene, id).await?,
        is_group: false,
      });
    }

    // GetSceneItemId does not look inside groups, so anything nested
    // lands here.
    let items = self.items_including_groups(scene).await?;

    if let Some(item) = items.iter().find(|item| item.name == source) {
      return Ok(item.clone());
    }

    // A source that is not there is the one failure worth spelling
    // out - it is almost always a name typed by hand against a name
    // OBS matches exactly, or a source that lives in a scene other
    // than this one. A busy scene runs to dozens of items though, and
    // this lands in the daemon log every time the command is said, so
    // name a few and point at the subcommand for the rest.
    const SHOWN: usize = 8;

    let known: Vec<&str> = items
      .iter()
      .map(|item| item.name.as_str())
      .take(SHOWN)
      .collect();

    let rest = items.len().saturating_sub(SHOWN);

    bail!(
      "scene {scene:?} has no source named {source:?} - it has {}{} \
       (voice-control obs sources {scene:?} for the rest)",
      if known.is_empty() {
        "nothing".to_string()
      } else {
        known.join(", ")
      },
      if rest > 0 {
        format!(" and {rest} more")
      } else {
        String::new()
      }
    )
  }

  /// A scene's items, with the contents of any group spliced in after
  /// the group itself.
  async fn items_including_groups(
    &mut self,
    scene: &str,
  ) -> Result<Vec<SceneItem>> {
    let mut out = Vec::new();

    for item in self.scene_items("GetSceneItemList", scene).await? {
      let group = item.is_group.then(|| item.name.clone());
      out.push(item);

      // One level down only. OBS does not nest groups, so this is the
      // whole tree, not an arbitrary cut-off.
      if let Some(group) = group {
        out.extend(
          self.scene_items("GetGroupSceneItemList", &group).await?,
        );
      }
    }

    Ok(out)
  }

  async fn scene_items(
    &mut self,
    request_type: &str,
    scene: &str,
  ) -> Result<Vec<SceneItem>> {
    let data = self
      .request(request_type, json!({ "sceneName": scene }))
      .await?;

    let items = data
      .get("sceneItems")
      .and_then(Value::as_array)
      .ok_or_else(|| anyhow!("{request_type} returned no scene items"))?;

    Ok(
      items
        .iter()
        .filter_map(|item| {
          Some(SceneItem {
            scene: scene.to_string(),
            id: item.get("sceneItemId").and_then(Value::as_i64)?,
            name: item
              .get("sourceName")
              .and_then(Value::as_str)?
              .to_string(),
            enabled: item
              .get("sceneItemEnabled")
              .and_then(Value::as_bool)
              .unwrap_or(true),
            // Null for anything that is not a group.
            is_group: item
              .get("isGroup")
              .and_then(Value::as_bool)
              .unwrap_or(false),
          })
        })
        .collect(),
    )
  }

  async fn request(
    &mut self,
    request_type: &str,
    request_data: Value,
  ) -> Result<Value> {
    let request_id = self.next_id.to_string();
    self.next_id += 1;

    self
      .send(json!({
        "op": 6,
        "d": {
          "requestType": request_type,
          "requestId": request_id,
          "requestData": request_data,
        }
      }))
      .await?;

    let response = self.recv_op(7).await?;

    let status = response
      .get("requestStatus")
      .ok_or_else(|| anyhow!("{request_type} response had no status"))?;

    if status.get("result").and_then(Value::as_bool) != Some(true) {
      let code = status.get("code").and_then(Value::as_u64).unwrap_or(0);
      let comment = status
        .get("comment")
        .and_then(Value::as_str)
        .unwrap_or("no detail");

      bail!("{request_type} failed ({code}): {comment}");
    }

    Ok(response.get("responseData").cloned().unwrap_or(json!({})))
  }

  async fn send(&mut self, message: Value) -> Result<()> {
    self
      .socket
      .send(Message::Text(message.to_string().into()))
      .await
      .context("sending to obs")
  }

  /// Reads until a message with the wanted opcode arrives, skipping
  /// anything else obs volunteers.
  async fn recv_op(&mut self, op: u64) -> Result<Value> {
    while let Some(message) = self.socket.next().await {
      let message = message.context("reading from obs")?;

      let text = match message {
        Message::Text(text) => text.to_string(),
        Message::Close(frame) => {
          bail!("obs closed the connection: {frame:?}")
        }
        _ => continue,
      };

      let value: Value =
        serde_json::from_str(&text).context("obs sent invalid json")?;

      match value.get("op").and_then(Value::as_u64) {
        Some(found) if found == op => {
          return Ok(value.get("d").cloned().unwrap_or(json!({})));
        }
        Some(other) => debug!(op = other, "ignoring obs message"),
        None => debug!("obs message had no op"),
      }
    }

    bail!("obs connection ended while waiting for op {op}")
  }
}

/// base64(sha256(base64(sha256(password + salt)) + challenge))
fn auth_string(password: &str, salt: &str, challenge: &str) -> String {
  let secret =
    BASE64.encode(Sha256::digest(format!("{password}{salt}").as_bytes()));

  BASE64.encode(Sha256::digest(format!("{secret}{challenge}").as_bytes()))
}

#[cfg(test)]
mod tests {
  use super::auth_string;

  /// Regression guard on the handshake.
  ///
  /// The password, salt and challenge are the example values from the
  /// obs-websocket 5.x protocol doc, but the expected output is not
  /// official — the doc publishes the four algorithm steps and no
  /// worked result. This constant was produced by an independent
  /// implementation of those steps and agrees with ours, so it pins
  /// the behaviour against silent drift; it is not a vendor test
  /// vector. The real proof is that OBS accepts the handshake.
  #[test]
  fn auth_string_follows_the_obs_websocket_algorithm() {
    let auth = auth_string(
      "supersecretpassword",
      "lM1GncleQOaCu9lT1yeUZhFYnqhsLLP1G5lAGo3ixaI=",
      "+IxH4CnCiqpX1rM9scsNynZzbOe4KhDeYcTNS3PDaeY=",
    );

    assert_eq!(auth, "1Ct943GAT+6YQUUX47Ia/ncufilbe6+oD6lY+5kaCu4=");
  }
}
