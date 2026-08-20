use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};

/// Home Assistant is on the LAN; if it has not answered by now it is
/// down, and blocking the pipeline helps nobody.
const TIMEOUT: Duration = Duration::from_secs(3);

/// The service a step gets when it names an entity and nothing else.
///
/// `toggle` for the same reason an OBS source defaults to it: a plug
/// command is usually said the same way twice - "computa, speakers" to
/// put them on and the same words again to take them off.
pub const DEFAULT_SERVICE: &str = "toggle";

/// Where Home Assistant is and how to get in.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct HassConfig {
  /// The base URL, without `/api` - `https://hass.lan`. Empty means
  /// there is no Home Assistant, which is what a config that uses none
  /// of it looks like.
  #[serde(default)]
  pub url: String,
  /// A long-lived access token, from the bottom of your profile page.
  ///
  /// Leave empty to take it from `HASS_TOKEN` instead, which keeps the
  /// secret out of a config file - the same arrangement `[obs]` has
  /// with `OBS_PASSWORD`.
  #[serde(default)]
  pub token: String,
  /// Accept the certificate whatever it is - expired, self signed, or
  /// issued by a CA only this network knows about.
  ///
  /// Which is the point of it: an ingress with a private CA is a
  /// perfectly ordinary way to run Home Assistant at home, and the
  /// alternative is teaching every machine about the CA. It does mean
  /// nothing is checking who answered, so leave it off for anything
  /// reached over the internet.
  #[serde(default)]
  pub insecure: bool,
}

impl HassConfig {
  /// Whether there is a Home Assistant to talk to at all.
  pub fn configured(&self) -> bool {
    !self.url.is_empty()
  }

  fn resolved_token(&self) -> String {
    if self.token.is_empty() {
      std::env::var("HASS_TOKEN").unwrap_or_default()
    } else {
      self.token.clone()
    }
  }
}

/// One service call: `turn_on`, against `switch.desk_speakers`.
pub struct ServiceCall<'a> {
  /// An entity id, `domain.name` the way Home Assistant writes it.
  pub entity: &'a str,
  /// The service, resolved against the entity's own domain - `turn_on`
  /// on a `switch.` entity is `switch.turn_on`. A service that names a
  /// domain of its own (`homeassistant.turn_on`) is taken as written,
  /// which is how the domain-agnostic services are reached.
  pub service: &'a str,
  /// Anything else the service takes, sent alongside the entity.
  pub data: &'a HashMap<String, Value>,
}

impl ServiceCall<'_> {
  /// The service as Home Assistant names it, domain and all.
  pub fn service(&self) -> String {
    if self.service.contains('.') {
      self.service.to_string()
    } else {
      format!("{}.{}", domain_of(self.entity), self.service)
    }
  }

  /// Where that service lives in the REST API. Service names have no
  /// dots of their own, so the only one here is the domain separator.
  fn path(&self) -> String {
    self.service().replace('.', "/")
  }
}

/// The domain half of an entity id - `switch` for
/// `switch.desk_speakers`.
fn domain_of(entity: &str) -> &str {
  entity.split_once('.').map_or(entity, |(domain, _)| domain)
}

/// Checks an entity id is shaped like one Home Assistant could have.
///
/// Called at load rather than at dispatch: Home Assistant answers a
/// call naming an entity it has never heard of with a perfectly happy
/// 200 and no state change, so a typo would otherwise be silent until
/// the day you said the words and nothing happened.
pub fn validate_entity(entity: &str) -> Result<()> {
  let (domain, name) = entity.split_once('.').unwrap_or_default();

  if domain.is_empty() || name.is_empty() {
    bail!(
      "names hass entity {entity:?}, which is not an entity id - they \
       read as `switch.desk_speakers`, a domain and a name"
    );
  }

  Ok(())
}

/// One entity, as `/api/states` reports it.
#[derive(Debug, Deserialize)]
pub struct Entity {
  pub entity_id: String,
  #[serde(default)]
  pub state: String,
  #[serde(default)]
  pub attributes: Attributes,
}

#[derive(Debug, Default, Deserialize)]
pub struct Attributes {
  #[serde(default)]
  pub friendly_name: String,
}

pub struct Hass {
  client: reqwest::Client,
  /// The base URL, with any trailing slash already taken off.
  url: String,
  token: String,
}

impl Hass {
  /// `None` when there is no `[hass]` table, which is what a config
  /// using none of it looks like - there is no client to build against
  /// a URL that is nowhere.
  pub fn connect(config: &HassConfig) -> Result<Option<Self>> {
    if !config.configured() {
      return Ok(None);
    }

    let token = config.resolved_token();

    if token.is_empty() {
      bail!(
        "[hass] has a url but no token, and HASS_TOKEN is unset - Home \
         Assistant refuses every call without one"
      );
    }

    let client = reqwest::Client::builder()
      .timeout(TIMEOUT)
      .danger_accept_invalid_certs(config.insecure)
      .build()
      .context("building the home assistant client")?;

    Ok(Some(Self {
      client,
      url: config.url.trim_end_matches('/').to_string(),
      token,
    }))
  }

  /// Calls a service, answering with how many entities it changed.
  ///
  /// Zero is not an error: a switch that was already on changed
  /// nothing, and so did a call naming an entity that does not exist.
  /// Home Assistant returns 200 for both, so the count is all the
  /// caller has to tell them apart with - which is why it is logged.
  pub async fn call(&self, call: ServiceCall<'_>) -> Result<usize> {
    let url = format!("{}/api/services/{}", self.url, call.path());

    let mut body = serde_json::Map::new();
    body.insert("entity_id".into(), json!(call.entity));

    for (key, value) in call.data {
      body.insert(key.clone(), value.clone());
    }

    let response = self
      .client
      .post(&url)
      .bearer_auth(&self.token)
      .json(&body)
      .send()
      .await
      .with_context(|| format!("calling {url}"))?;

    let status = response.status();

    if !status.is_success() {
      let body = response.text().await.unwrap_or_default();
      bail!("{url} returned {status}: {body}");
    }

    // The body is the list of states the call changed. Anything else
    // is a Home Assistant that answered in a shape this does not know,
    // which is not worth failing a call that already returned 200.
    let changed: Vec<Value> = response.json().await.unwrap_or_default();

    Ok(changed.len())
  }

  /// Every entity Home Assistant knows about, for finding the ids to
  /// put in the config.
  pub async fn states(&self) -> Result<Vec<Entity>> {
    let url = format!("{}/api/states", self.url);

    let response = self
      .client
      .get(&url)
      .bearer_auth(&self.token)
      .send()
      .await
      .with_context(|| format!("calling {url}"))?;

    let status = response.status();

    if !status.is_success() {
      let body = response.text().await.unwrap_or_default();
      bail!("{url} returned {status}: {body}");
    }

    let mut entities: Vec<Entity> =
      response.json().await.context("reading the entity list")?;

    entities.sort_by(|a, b| a.entity_id.cmp(&b.entity_id));

    Ok(entities)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn call<'a>(
    entity: &'a str,
    service: &'a str,
    data: &'a HashMap<String, Value>,
  ) -> ServiceCall<'a> {
    ServiceCall {
      entity,
      service,
      data,
    }
  }

  /// The whole point of naming an entity: `turn_on` means the one that
  /// belongs to whatever kind of thing this is.
  #[test]
  fn a_service_takes_the_domain_of_its_entity() {
    let data = HashMap::new();
    let call = call("switch.desk_speakers", "turn_on", &data);

    assert_eq!(call.service(), "switch.turn_on");
    assert_eq!(call.path(), "switch/turn_on");
  }

  /// `homeassistant.turn_on` works on anything, and there is no way to
  /// reach it if the entity's domain is always imposed.
  #[test]
  fn a_service_can_name_its_own_domain() {
    let data = HashMap::new();
    let call =
      call("switch.desk_speakers", "homeassistant.turn_on", &data);

    assert_eq!(call.service(), "homeassistant.turn_on");
    assert_eq!(call.path(), "homeassistant/turn_on");
  }

  #[test]
  fn rejects_an_entity_without_a_domain() {
    assert!(validate_entity("desk_speakers").is_err());
    assert!(validate_entity("switch.").is_err());
    assert!(validate_entity(".desk_speakers").is_err());
    assert!(validate_entity("switch.desk_speakers").is_ok());
  }

  /// A URL written with a trailing slash is the easiest thing in the
  /// world to paste, and `{url}/api/...` would double the separator.
  #[test]
  fn a_trailing_slash_is_taken_off_the_base_url() {
    let config = HassConfig {
      url: "https://hass.lan/".into(),
      token: "x".into(),
      insecure: true,
    };

    let hass = Hass::connect(&config).unwrap().unwrap();

    assert_eq!(hass.url, "https://hass.lan");
  }

  #[test]
  fn no_url_means_no_client() {
    assert!(Hass::connect(&HassConfig::default()).unwrap().is_none());
  }

  #[test]
  fn a_url_without_a_token_is_an_error() {
    // Only when the environment has not got one either, which it will
    // not have here unless the machine running the tests uses one.
    if std::env::var("HASS_TOKEN").is_ok() {
      return;
    }

    let config = HassConfig {
      url: "https://hass.lan".into(),
      ..HassConfig::default()
    };

    assert!(Hass::connect(&config).is_err());
  }
}
