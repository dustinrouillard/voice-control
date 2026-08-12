#!/usr/bin/env python3
"""Generate the feedback cues into sounds/.

Kept as a script rather than committed-only binaries so the tones can
be retuned without hunting for an audio editor. Run from the repo root:

    python3 scripts/make-sounds.py
"""

import math
import os
import struct
import wave

RATE = 44100
AMPLITUDE = 0.22


def tone(freq, ms, fade_ms=8):
    """One note with short fades, so it clicks at neither end."""
    total = int(RATE * ms / 1000)
    fade = max(1, int(RATE * fade_ms / 1000))
    out = []

    for i in range(total):
        envelope = min(1.0, i / fade, (total - i) / fade)
        out.append(math.sin(2 * math.pi * freq * i / RATE) * envelope)

    return out


def gap(ms):
    """A real pause, so a repeated note reads as two and not as one."""
    return [0.0] * int(RATE * ms / 1000)


def write(name, samples):
    path = os.path.join("sounds", name)

    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(
            b"".join(
                struct.pack("<h", int(max(-1.0, min(1.0, s)) * AMPLITUDE * 32767))
                for s in samples
            )
        )

    print(f"{path}: {len(samples) / RATE * 1000:.0f} ms")


os.makedirs("sounds", exist_ok=True)

# Rising blip: "listening".
write("wake.wav", tone(880, 70) + tone(1320, 70))

# Two ascending notes: "done".
write("ok.wav", tone(660, 80) + tone(990, 110))

# Descending, lower: "did not understand".
write("fail.wav", tone(440, 110) + tone(311, 150))

# Two taps at one pitch: "yes, I am here". Every other cue is a pair of
# notes going somewhere, so a flat double-tap is the one shape left
# that cannot be mistaken for any of them at a glance.
write("ping.wav", tone(1318, 55) + gap(45) + tone(1318, 55))
