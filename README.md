# voice-control

A background macOS agent that listens for **"computa, …"** and turns the
command that follows into an HTTP request, an OBS change, or a media
key.

```
computa, mute            ->  POST :8009/v1/voice/canary/mute/on
computa, unmute          ->  POST :8009/v1/voice/canary/mute/off
computa, deafen          ->  POST :8009/v1/voice/canary/deaf/on
computa, push to talk    ->  POST :8009/v1/voice/canary/mode/PUSH_TO_TALK
computa, desk mic        ->  POST :8080/v1/input/set/desk
computa, camera          ->  OBS scene "Camera Screen"
computa, main            ->  OBS scene "Main Screen"
computa, ps5             ->  OBS scene "Main Screen", then animate the
                             "Cam Link Screen" source in (or out)
computa, hide ps5        ->  animate "Cam Link Screen" out
computa, skip            ->  media key: next track
computa, play            ->  media key: play/pause
```

Targets are [discord-rpc-control][drc] on localhost and [mic-api][mic] on
the Linux box. Nothing is hardcoded — commands are a TOML table, so
adding one is a config edit and a restart.

Discord RPC mutes *inside Discord* and leaves the OS input device open,
which is what makes "computa, unmute" work while muted. Do not swap the
dispatch for an OS-level mute; the daemon would go deaf to its own
un-mute.

[drc]: https://github.com/dustinrouillard/discord-rpc-control
[mic]: https://github.com/dustinrouillard/mic-api

## How it works

```
cpal capture ──► resample 48k→16k mono ──► 3s pre-roll ring
                          │
                    ┌─────┴─────┐
                    │  IDLE     │  rustpotter scores every frame
                    └─────┬─────┘
                    "computa" ✔ → chirp
                          │
                    ┌─────┴─────┐
                    │ LISTENING │  Silero VAD, ends on 700ms silence
                    └─────┬─────┘     (seeded with 900ms of pre-roll)
                          │
                  whisper.cpp base.en  →  "computa mute"
                          │
                  normalise → suffix candidates → fuzzy match
                          │
                  commands.toml → POST … → ok/fail tone
```

Two stages rather than one because a wake-word model is cheap enough to
run continuously and whisper is not.

The pre-roll is the part that is easy to get wrong. "computa, mute" is
one breath, so the command is already partly spoken when the wake word
starts scoring — and rustpotter then needs several more frames before
it commits. Between them the whole utterance can be in the past by the
time the detector says anything, so the window reaches back far enough
to recover the wake word too. That the wake word ends up in the clip is
fine: the matcher works over suffixes and drops it.

Both stages sit behind traits (`wake::Detector`, `stt::Transcriber`), so
swapping in [openWakeWord][oww] later touches only `src/wake/`.

[oww]: https://github.com/dscripka/openWakeWord

## Setup

Needs `cmake` (`brew install cmake`) to build whisper.cpp. The first
build takes a few minutes; that is whisper.cpp compiling, not a hang.

### 1. Whisper model

```bash
mkdir -p ~/.config/voice-control
curl -L -o ~/.config/voice-control/ggml-base.en-q5_1.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-q5_1.bin
```

`base.en` is noticeably better than `tiny.en` on one- and two-word
commands and still runs in well under 200 ms on Apple Silicon. Drop to
`ggml-tiny.en-q5_1.bin` if you want the latency back.

### 2. Wake word model

Nothing ships a "computa" model, so train one. Record on **the mic you
actually use** — the model does not transfer well between microphones.

```bash
# --locked is required: rustpotter pulls an old candle-core that only
# builds against the versions in its own published lockfile.
cargo install rustpotter-cli --locked

mkdir -p samples
for i in $(seq 1 12); do
  rustpotter-cli record "samples/computa-$i.wav"   # say "computa", ctrl-c
done

# Crop each take to the word itself. Do not skip this.
python3 scripts/trim-samples.py samples samples-trimmed

rustpotter-cli build \
  --name computa \
  --path ~/.config/voice-control/computa.rpw \
  samples-trimmed/*.wav
```

`build` makes a reference model from a handful of recordings, which is
what you want here. (`train` builds a neural model and wants a corpus.)

**Trimming is not optional.** `record` runs until you hit ctrl-c, so a
take is two or three seconds of room tone with half a second of speech
somewhere inside it — and rustpotter builds its template from the whole
file. Untrimmed takes compare mostly-silence against mostly-silence at
different offsets and score near zero; a model built from raw takes
failed to detect *its own training samples*. After trimming, the same
takes detect 11/11 at the default threshold. Aim for templates of
roughly 600–800 ms, which is what the script produces.

A take that is all silence — ctrl-c pressed too early — makes
`rustpotter-cli build` panic with `index out of bounds: the len is 0`
rather than naming the file. `trim-samples.py` reports and skips those;
otherwise check for a suspiciously small wav.

Vary the delivery across takes: normal, quiet, fast, and mid-sentence.
Ten to fifteen is plenty. `say -v Samantha -r 180 computa` and friends
make useful extra samples, but recordings of your own voice on your own
mic do most of the work.

Sanity-check the model before moving on — every take should detect:

```bash
for f in samples/*.wav; do
  echo -n "$f "
  rustpotter-cli test ~/.config/voice-control/computa.rpw "$f" \
    | grep -oE 'score: [0-9.]+' | tail -1
done
```

### 3. Commands

```bash
cp commands.example.toml ~/.config/voice-control/commands.toml
```

Set `mic` under `[targets]` to the Linux host, and check the Discord
branch in the `discord` URL matches `DISCORD_BRANCH` in
discord-rpc-control (its installer defaults to `canary`).

### 4. Tune the threshold

```bash
cargo run --release -- listen
```

Prints an input level once a second and a line per detection:

```
  level 0.067  ###
  1  computa  score 0.612
```

If the level stays at zero, the microphone permission is the problem,
not the model — it says so after five seconds of silence.

Say the wake word twenty times and count the hits, then leave it
running through a normal conversation and count the false ones. Raise
`threshold` / `avg_threshold` if it fires while you are just talking;
lower them if it misses you.

`eager = true` (the default) makes rustpotter commit as soon as enough
partial scores agree instead of waiting for the score to peak. That
costs a little accuracy and saves a few hundred milliseconds, which is
the better trade here because whisper re-checks the result anyway. It
also keeps the captured clip tight — with `eager = false` the detection
can land so late that the command has already scrolled out of the
pre-roll window.

### 5. Install

```bash
./scripts/install-agent.sh
```

Builds release, installs to `~/Library/Application Support/voice-control`,
ad-hoc signs the binary (so the microphone grant survives rebuilds), and
bootstraps the LaunchAgent.

macOS will ask for microphone access the first time it runs. If no
prompt appears and it never hears anything, check System Settings →
Privacy & Security → Microphone.

Media key commands need a second grant, **Accessibility**, which is
never prompted for — add
`~/Library/Application Support/voice-control/voice-control` under
System Settings → Privacy & Security → Accessibility by hand. Skip it
if you have no `media` commands. Both grants are keyed to the code
signature, which is why the script ad-hoc signs the binary before
launchd sees it: without that, every rebuild would look like a brand
new app and drop them.

## Usage

```
voice-control                    run the daemon
voice-control devices            list input devices
voice-control listen             wake word detections only
voice-control transcribe f.wav   STT + matcher against a file
voice-control run "computa ps5"  match a phrase and dispatch it
voice-control replay f.wav       whole pipeline against a file
voice-control obs                list obs scenes
voice-control obs "Camera Screen"  switch to a scene
voice-control obs sources        list the sources in the current scene
voice-control obs filters "Main Screen"      list a scene's filters
voice-control obs toggle "Cam Link Screen"   flip a source, bare
voice-control media next         press a media key, bare
```

`transcribe` is the fast way to grow phrase lists without talking to a
microphone. `run` takes the phrase directly and dispatches it exactly
as the daemon would, sequences and waits included — the way to check
what a command *does* once you know it matches. `replay` runs a file
through wake word, VAD, STT, matching and dispatch — everything except
opening the microphone — so you can check a change end to end. Generate
test clips with `say -v Samantha -o x.aiff "computa mute"` piped
through `afconvert -f WAVE -d LEI16@16000 -c 1 x.aiff x.wav`.

### OBS scenes

Scene switching goes over obs-websocket 5.x. Enable the server in OBS
under Tools -> WebSocket Server Settings, then:

```toml
[obs]
url = "ws://127.0.0.1:4455"   # wss:// for an OBS on another machine
password = ""                 # empty reads OBS_PASSWORD instead

[[commands]]
name = "camera scene"
phrases = ["camera", "cam", "show me"]
scene = "Camera Screen"
```

Scenes are enumerated one command at a time on purpose: the daemon can
only switch to scenes you named, never to whatever it thought it heard.
`scene` is matched by OBS exactly, so copy the names from:

```bash
voice-control obs                    # list them
voice-control obs "Camera Screen"    # switch, to check the wiring
```

The bare form lists OBS's scenes, marks which are wired up, and warns about any
`scene =` in your config that OBS does not have — a typo there is
otherwise silent until you say the words.

### OBS sources

A `source` command changes one source's visibility inside a scene
rather than switching scenes:

```toml
[[commands]]
name = "ps5"
phrases = ["ps5", "ps 5", "playstation"]
source = "Cam Link Screen"
scene = "Main Screen"
visible = "toggle"          # show | hide | toggle (the default)
show_filter = "Move-In"     # Move filters on the scene
hide_filter = "Move-Out"
hide_delay_ms = 350
```

`toggle` is the useful default — "computa, ps5" puts it up and the same
words take it down, so you never have to remember which way it was.
Pair it with explicit `visible = "show"` / `"hide"` commands when you do
want to say which way it goes.

Without a `scene`, the source is looked for in whichever scene is on
program when you say it — right for a source that appears in several,
and it means the same command works from wherever you are. Add
`scene = "Main Screen"` to pin it to one scene instead; alongside
`source`, `scene` says where the source lives and is *not* a scene
switch. Sources inside a group are found without naming the group.

#### Animating it in and out

`show_filter` / `hide_filter` name [Move][move] filters on the scene,
and turn the visibility flip into an animation. The order is the whole
point, and it is not symmetric:

```
show:  make the source visible  ->  enable show_filter
hide:  enable hide_filter  ->  wait hide_delay_ms  ->  hide the source
```

A Move filter animates something that is on screen, so on the way in
the source has to exist before it can move, and on the way out it has
to survive until the move ends — hiding first would cut the animation
off at frame one. That asymmetry, and the wait in the middle of the
hide, is why this is a sequence rather than one request. It is the same
sequence a Companion button runs for the same effect.

Enabling the filter is what triggers it; the plugin turns the filter
back off when the move finishes, which is what lets the next command
trigger it again. `hide_delay_ms` should be at least as long as the
move or the source is yanked mid-animation — it defaults to 350, and
shows up as about 350 ms of extra latency on a hide and none on a show.

A `toggle` needs both filters or neither: a move that is never undone
parks the source wherever it left it — off screen, and invisible the
next time it is shown. That is rejected at load. One-way `show` /
`hide` commands only need their own filter.

[move]: https://obsproject.com/forum/resources/move.913/

```bash
voice-control obs sources                     # names, exactly as OBS has them
voice-control obs sources "Camera Screen"     # a scene other than the current one
voice-control obs filters "Main Screen"       # filter names, same deal
voice-control run "computa ps5"               # the whole sequence, to check the wiring
voice-control obs toggle "Cam Link Screen"    # bare flip, no filters
```

Copy names from `sources` and `filters` rather than from another tool's
dropdown — OBS matches exactly, and it does not always spell a name the
way something else displays it (`Move-In`, not `Move-in`).

`sources` marks which entries are currently hidden, which live in a
group, and which are wired to a command — and warns about a `source =`
the scene does not have, the same way the scene list does. A source
that is in a different scene than the one you are on reads as "no
source named …", with the names it *does* have listed; that message is
the usual answer to a source command that does nothing.

It also warns when a scene holds **two sources with the same name**,
which OBS allows. Which one a command gets is then OBS's answer to the
name rather than ours — `GetSceneItemId` resolves it, so it is the same
item any other obs-websocket client would move — but if the two are
meant to behave differently, rename one.

### Media keys

A `media` command presses one of the keyboard's transport keys:

```toml
[[commands]]
name = "next track"
phrases = ["next", "skip", "next song", "skip this song"]
media = "next"              # play_pause | next | previous
```

```
computa, skip        ->  media key: next track
computa, play        ->  media key: play/pause
computa, last song   ->  media key: previous track
```

macOS routes these to whichever app it considers to be playing, so this
controls Spotify **without talking to Spotify** — and controls Music, or
a browser tab, on the days one of those is making the noise instead.
There is nothing about Spotify configured anywhere, and nothing to keep
in sync when you switch players.

Spellings are interchangeable: `play`, `pause`, `stop`, `resume` and
`toggle` all mean `play_pause`, `skip` means `next`, and `prev` / `back`
mean `previous`.

**There is one play/pause key, not a play and a pause.** The hardware
has only the one and every player treats it as a toggle, so "computa,
pause" while already paused starts it playing again. Said once, either
word does the obvious thing.

`previous` is the player's business, not ours: most read one press as
"back to the start of this song" and only two as "the song before".

#### Accessibility

Synthesising a key press needs the **Accessibility** grant, which is
separate from the microphone one and is never prompted for. Add
`~/Library/Application Support/voice-control/voice-control` under
System Settings → Privacy & Security → Accessibility.

Without it macOS drops the event and says nothing, which looks exactly
like a player that ignored it — so the grant is checked first and the
command fails with a real message instead. The bare form is the quick
way to tell that apart from a phrase that did not match:

```bash
voice-control media next          # no config read, no matching
voice-control run "computa skip"  # the command, matched and dispatched
```

A grant belongs to the *responsible* process, so run from a terminal
the check reflects that terminal's Accessibility rather than the
binary's. Only the launchd copy answers for itself.

### Flows

A command that does more than one thing is a list of steps, run in
order, stopping at the first failure. A step takes the same fields a
one-step command does, plus `wait_ms` for a step that only waits.

The reason the ps5 commands are flows: a Move filter animates the
source *into the scene it lives in*, so saying "show ps5" from the
camera scene has to get you there first.

```toml
[[commands]]
name = "show ps5"
phrases = ["show ps5", "show the ps5", "show playstation"]

  [[commands.steps]]
  scene = "Main Screen"

  [[commands.steps]]
  source = "Cam Link Screen"
  scene = "Main Screen"
  visible = "show"
  show_filter = "Move-In"
```

```
computa, show ps5   ->  obs scene "Main Screen"
                    ->  obs show source "Cam Link Screen" via Move-In
```

Everything above is the one-step shorthand for exactly this, so
single-action commands need no `[[commands.steps]]` at all — and a
command uses one form or the other, never both. Failures stop the flow
rather than pressing on: if the scene switch did not happen there is no
point enabling the move that was meant to play on it, and continuing
would leave the source shown somewhere you cannot see it.

Steps are not limited to OBS — HTTP and media keys are steps like any
other, so one phrase can pause the music, mute Discord, switch scene
and bring a source in:

```toml
[[commands]]
name = "brb"
phrases = ["brb", "be right back"]

  [[commands.steps]]
  media = "play_pause"

  [[commands.steps]]
  url = "{discord}/mute/on"

  [[commands.steps]]
  scene = "Main Screen"

  [[commands.steps]]
  wait_ms = 300          # let the scene transition land

  [[commands.steps]]
  source = "Showering text"
  scene = "Main Screen"
  visible = "show"
```

Errors name the step they came from (`command "brb", step 5: …`), which
is the difference between a config you can fix and one you have to
bisect. Check a whole flow without saying anything:

```bash
voice-control run "computa show ps5"
```

It prints the flow it matched before running it:

```
matched: show ps5 (score 1.000) -> obs scene "Main Screen" ->
  obs show source "Cam Link Screen" in "Main Screen" via Move-In
```

If you would rather not keep the password in the config file, leave it
empty and set `OBS_PASSWORD` in the environment (the LaunchAgent reads
it from `.env` via `scripts/install-agent.sh`). If you do put it in the
file, `chmod 600 ~/.config/voice-control/commands.toml`.

The daemon opens a fresh connection per switch rather than holding one
open. Commands are seconds apart at best, and a short-lived connection
cannot rot when OBS restarts or the machine sleeps.

### Menu bar

The daemon puts an icon in the menu bar. It is the answer to the two
questions the logs used to be the only way to answer: is it hearing me
at all, and what did it think I said?

```
mic          idle, waiting for the wake word
mic.fill     heard you, capturing the command
waveform     transcribing and dispatching
checkmark    the last command went through   (1.5s, then back to idle)
mic.slash    paused from the menu
!            not hearing anything - see below
```

They are SF Symbols rendered as template images, so they follow the
menu bar in light, dark and tinted appearances. Hovering shows the
status line without opening anything.

The menu holds the current state, the input device and a live level
meter, how long ago it was last woken, and the last ten utterances with
what became of each:

```
Listening for "computa"
Wireless microphone  ▃▅▂·····
Last woken 4m ago
────────────────────
Recent
   2m  "computa mute"    mute OK
   4m  "computa muted"   no match
────────────────────
Pause listening
Open logs
Restart agent
────────────────────
Quit until next login
```

The no-match lines are the point of the list. Growing the phrase lists
in commands.toml used to mean grepping `stdout.log`; now the last ten
are one click away, and the log is still there for the rest.

**Pause listening** stops the wake word from scoring without stopping
the daemon - for a screen share, or a meeting where "computa" is going
to come up. Audio keeps flowing, so the pre-roll stays warm and
resuming is instant. A capture already in flight is allowed to finish.

**Quit** has to boot the job out of launchd, because `KeepAlive` would
undo a plain exit within the second. It comes back at next login, or
immediately with `./scripts/install-agent.sh`.

Two distinct ways of hearing nothing get distinct warnings, because
they have different causes:

| Menu says | Means |
| --- | --- |
| `Silent for 30s - check microphone access` | audio is arriving, all of it empty: the TCC grant was revoked, or the mic is muted in hardware |
| `No audio for 12s - the input device is gone` | no buffers at all: the device was unplugged, or CoreAudio dropped the stream |

The second used to be invisible - the stream ending would exit the
process, launchd would restart it, and nothing said so.

If no icon appears, check the log for `menu bar item created`. If it is
there, the item exists and something is hiding it: Ice, Bartender and
friends file new items into their hidden section by default, and it has
to be dragged out (cmd-drag along the menu bar) once.

`TRAY=false` gives back the old headless daemon, which is what you want
over ssh or under a debugger, where there is no window server to talk
to. Every subcommand is headless regardless.

### Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `CONFIG_PATH` | `~/.config/voice-control/commands.toml` | command table |
| `INPUT_DEVICE` | system default | case-insensitive substring of the device name |
| `SOUNDS_DIR` | unset (silent) | directory holding `wake.wav`, `ok.wav`, `fail.wav` |
| `OBS_PASSWORD` | unset | obs-websocket password, if not in the config |
| `DSTN_LOG` | `info` | tracing filter |
| `TRAY` | `true` | menu bar status item |
| `LOG_DIR` | `~/Library/Application Support/voice-control/logs` | what the menu's "Open logs" opens |
| `LAUNCHD_LABEL` | `com.dstn.voice-control` | what the menu's restart and quit act on |

Unmatched transcripts are logged at `info` on purpose — grepping them is
how the phrase lists get grown:

```bash
grep "no matching command" ~/Library/Application\ Support/voice-control/logs/stdout.log
```

## Notes

- **The main thread belongs to AppKit.** `NSStatusItem` has to be
  created on it under a running `NSApplication`, so `main` is not
  `#[tokio::main]`: the runtime is built by hand and the pipeline is
  driven on a thread of its own. It gets a thread rather than a task
  because neither the wake word detector nor the VAD is `Sync`, so the
  pipeline future cannot go to a work-stealing scheduler. Everything it
  spawns - the resampler, whisper, HTTP - still lands on the runtime.
- **Speaker loopback.** On speakers, someone in voice chat saying
  "computa mute" will trigger it. Headphones avoid this.
- Commands are a literal phrase → a fixed action. No parameters ("set
  volume to 30") without extending the matcher.
- `half` is pinned to `=2.4.1` in `Cargo.toml`. It is not used directly;
  rustpotter pulls candle-core 0.2.2, which calls the rand 0.8 API, and
  half ≥ 2.5 moved its impls to rand 0.9.
