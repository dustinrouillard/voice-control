#!/usr/bin/env bash
#
# Build voice-control and (re)install it as a per-user LaunchAgent.
#
# An agent rather than a daemon, for two reasons: microphone access is
# granted per login session by TCC, and afplay needs a session to route
# audio into. A root LaunchDaemon gets neither.
#
# Configuration lives in ~/.config/voice-control/commands.toml, which
# the agent reads at startup - only the env vars below are baked into
# the plist.

set -euo pipefail

LABEL="com.dstn.voice-control"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUPPORT="$HOME/Library/Application Support/voice-control"
CONFIG="$HOME/.config/voice-control"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
TARGET="gui/$(id -u)/$LABEL"

if [ -f "$REPO/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO/.env"
  set +a
fi

CONFIG_PATH="${CONFIG_PATH:-$CONFIG/commands.toml}"
INPUT_DEVICE="${INPUT_DEVICE:-}"
SOUNDS_DIR="${SOUNDS_DIR:-$SUPPORT/sounds}"
OBS_PASSWORD="${OBS_PASSWORD:-}"
DSTN_LOG="${DSTN_LOG:-info}"
TRAY="${TRAY:-true}"
LOG_DIR="${LOG_DIR:-$SUPPORT/logs}"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "no config at $CONFIG_PATH" >&2
  echo "  mkdir -p '$CONFIG'" >&2
  echo "  cp '$REPO/commands.example.toml' '$CONFIG_PATH'" >&2
  echo "then edit it and re-run this script." >&2
  exit 1
fi

xml_escape() {
  printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'
}

echo "==> building release binary"
cargo build --release --manifest-path "$REPO/Cargo.toml"

echo "==> stopping any running agent"
launchctl bootout "$TARGET" 2>/dev/null || true

# bootout returns before launchd has finished unregistering, and
# bootstrapping into a half-removed service fails with EIO.
for _ in $(seq 20); do
  launchctl print "$TARGET" >/dev/null 2>&1 || break
  sleep 0.25
done

echo "==> installing to $SUPPORT"
mkdir -p "$SUPPORT/logs" "$SUPPORT/sounds"
cp "$REPO/target/release/voice-control" "$SUPPORT/voice-control"
chmod 755 "$SUPPORT/voice-control"
cp "$REPO"/sounds/*.wav "$SUPPORT/sounds/"

# TCC keys the microphone grant to the binary's code signature. Without
# a stable one, every rebuild looks like a brand new app and the
# permission is dropped - so ad-hoc sign it before launchd sees it.
echo "==> ad-hoc signing"
codesign --force --sign - "$SUPPORT/voice-control"

echo "==> writing $PLIST"
mkdir -p "$(dirname "$PLIST")"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$SUPPORT/voice-control</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>CONFIG_PATH</key>
    <string>$(xml_escape "$CONFIG_PATH")</string>
    <key>INPUT_DEVICE</key>
    <string>$(xml_escape "$INPUT_DEVICE")</string>
    <key>SOUNDS_DIR</key>
    <string>$(xml_escape "$SOUNDS_DIR")</string>
    <key>OBS_PASSWORD</key>
    <string>$(xml_escape "$OBS_PASSWORD")</string>
    <key>DSTN_LOG</key>
    <string>$(xml_escape "$DSTN_LOG")</string>
    <key>TRAY</key>
    <string>$(xml_escape "$TRAY")</string>
    <key>LOG_DIR</key>
    <string>$(xml_escape "$LOG_DIR")</string>
    <key>LAUNCHD_LABEL</key>
    <string>$(xml_escape "$LABEL")</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <!-- Adaptive, not Background: the menu bar item makes this a process
       with a user interface, and Background invites App Nap to throttle
       a timer that is meant to keep an icon current. -->
  <key>ProcessType</key>
  <string>Adaptive</string>
  <key>StandardOutPath</key>
  <string>$SUPPORT/logs/stdout.log</string>
  <key>StandardErrorPath</key>
  <string>$SUPPORT/logs/stderr.log</string>
</dict>
</plist>
EOF

# The plist carries OBS_PASSWORD when one is set.
chmod 600 "$PLIST"

echo "==> starting agent"
for attempt in $(seq 5); do
  if launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null; then
    break
  fi
  if [ "$attempt" = 5 ]; then
    echo "bootstrap failed after 5 attempts" >&2
    launchctl bootstrap "gui/$(id -u)" "$PLIST"
    exit 1
  fi
  sleep 1
done

sleep 1
launchctl print "$TARGET" | grep -E "state|pid" | head -3 || true

echo
echo "config:    $CONFIG_PATH"
echo "logs:      $SUPPORT/logs/"
echo "restart:   launchctl kickstart -k $TARGET"
echo "uninstall: launchctl bootout $TARGET && rm '$PLIST'"
echo
echo "If it never hears you, check System Settings -> Privacy &"
echo "Security -> Microphone for an entry named voice-control."
