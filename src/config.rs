use envconfig::Envconfig;

#[derive(Envconfig, Clone)]
pub struct Config {
  /// Path to commands.toml. Empty means
  /// `~/.config/voice-control/commands.toml`.
  #[envconfig(from = "CONFIG_PATH", default = "")]
  pub config_path: String,

  /// Case-insensitive substring of the input device name. Empty uses
  /// the system default input.
  #[envconfig(from = "INPUT_DEVICE", default = "")]
  pub input_device: String,

  /// Directory holding wake.wav / ok.wav / fail.wav. Empty disables
  /// audio feedback.
  #[envconfig(from = "SOUNDS_DIR", default = "")]
  pub sounds_dir: String,

  /// Show the menu bar status item. Off gives you the old headless
  /// daemon, which is what you want when running under a debugger or
  /// over ssh, where there is no window server to talk to.
  #[envconfig(from = "TRAY", default = "true")]
  pub tray: bool,

  /// Where the LaunchAgent's logs land, for the menu's "Open logs".
  #[envconfig(
    from = "LOG_DIR",
    default = "~/Library/Application Support/voice-control/logs"
  )]
  pub log_dir: String,

  /// LaunchAgent label, for the menu's restart and quit items.
  #[envconfig(from = "LAUNCHD_LABEL", default = "com.dstn.voice-control")]
  pub launchd_label: String,
}

impl Config {
  pub fn resolved_config_path(&self) -> std::path::PathBuf {
    if self.config_path.is_empty() {
      default_config_dir().join("commands.toml")
    } else {
      expand_tilde(&self.config_path)
    }
  }

  /// The launchd service target to kickstart or boot out, or `None`
  /// when nothing started us but a shell.
  ///
  /// launchd sets `XPC_SERVICE_NAME` to the job's label; a plain
  /// terminal either leaves it unset or, on older systems, sets it to
  /// "0". Without this the menu would offer to restart an agent that
  /// does not exist.
  pub fn launchd_target(&self) -> Option<String> {
    match std::env::var("XPC_SERVICE_NAME") {
      Ok(name) if name != "0" => {
        let uid = unsafe { libc::getuid() };
        Some(format!("gui/{uid}/{}", self.launchd_label))
      }
      _ => None,
    }
  }
}

pub fn default_config_dir() -> std::path::PathBuf {
  expand_tilde("~/.config/voice-control")
}

/// `~` expansion, since paths come from a hand-edited TOML file and
/// launchd does not run a shell to do it for us.
pub fn expand_tilde(path: &str) -> std::path::PathBuf {
  let Some(rest) = path.strip_prefix("~/") else {
    return std::path::PathBuf::from(path);
  };

  match std::env::var_os("HOME") {
    Some(home) => std::path::PathBuf::from(home).join(rest),
    None => std::path::PathBuf::from(path),
  }
}
