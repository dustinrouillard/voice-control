#!/usr/bin/env python3
"""Trim wake word recordings down to the word itself.

A take recorded by hand is typically two or three seconds of room tone
with half a second of speech somewhere inside it. Nothing in the daemon
needs these cropped any more -- openWakeWord scores a rolling window
and does not care what surrounds the word -- but a trainer being handed
samples of a new wake word generally does, and so does anyone reading
through a directory of takes.

This crops each take to the speech, converts to the 16 kHz mono the
detector runs at, and writes to an output directory. Run from the repo
root:

    python3 scripts/trim-samples.py samples samples-trimmed
"""

import array
import os
import struct
import sys
import wave

TARGET_RATE = 16000
# Speech starts where energy first crosses this fraction of the take's
# peak. Low enough to catch a soft leading consonant.
ONSET = 0.10
# Kept either side of the detected speech.
PAD_MS = 120
WINDOW_MS = 10


def read_wav(path):
    """Return (samples as floats, rate). Handles float32 and int wavs."""
    raw = open(path, "rb").read()

    if raw[:4] != b"RIFF":
        raise ValueError(f"{path}: not a RIFF file")

    fmt = None
    pos = 12

    while pos + 8 <= len(raw):
        cid = raw[pos : pos + 4]
        size = struct.unpack("<I", raw[pos + 4 : pos + 8])[0]
        body = raw[pos + 8 : pos + 8 + size]

        if cid == b"fmt ":
            tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])

            # WAVE_FORMAT_EXTENSIBLE, which CoreAudio emits: the real
            # tag is the first two bytes of the SubFormat GUID.
            if tag == 0xFFFE and len(body) >= 26:
                tag = struct.unpack("<H", body[24:26])[0]

            fmt = (tag, channels, rate, bits)
        elif cid == b"data" and fmt:
            tag, channels, rate, bits = fmt

            if tag == 3 and bits == 32:
                n = len(body) // 4
                vals = list(struct.unpack("<%df" % n, body[: n * 4]))
            elif tag == 1 and bits == 16:
                a = array.array("h")
                a.frombytes(body[: len(body) // 2 * 2])
                vals = [v / 32768.0 for v in a]
            elif tag == 1 and bits == 32:
                a = array.array("i")
                a.frombytes(body[: len(body) // 4 * 4])
                vals = [v / 2147483648.0 for v in a]
            else:
                raise ValueError(f"{path}: unsupported format tag={tag} bits={bits}")

            if channels > 1:
                vals = [
                    sum(vals[i : i + channels]) / channels
                    for i in range(0, len(vals) - channels + 1, channels)
                ]

            return vals, rate

        pos += 8 + size + (size & 1)

    raise ValueError(f"{path}: no data chunk")


def resample(samples, src_rate, dst_rate):
    """Linear resample. Good enough: the detector's own front end is a
    mel filterbank, not something that cares about a perfect sinc."""
    if src_rate == dst_rate:
        return samples

    ratio = src_rate / dst_rate
    out = []

    for i in range(int(len(samples) / ratio)):
        pos = i * ratio
        lo = int(pos)
        hi = min(lo + 1, len(samples) - 1)
        frac = pos - lo
        out.append(samples[lo] * (1 - frac) + samples[hi] * frac)

    return out


def trim(samples, rate):
    window = max(1, rate * WINDOW_MS // 1000)
    energies = [
        max(abs(v) for v in samples[i : i + window])
        for i in range(0, len(samples) - window + 1, window)
    ]

    if not energies:
        return samples

    threshold = max(energies) * ONSET
    loud = [i for i, e in enumerate(energies) if e >= threshold]

    if not loud:
        return samples

    pad = rate * PAD_MS // 1000
    start = max(0, loud[0] * window - pad)
    end = min(len(samples), (loud[-1] + 1) * window + pad)

    return samples[start:end]


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "samples"
    dst = sys.argv[2] if len(sys.argv) > 2 else "samples-trimmed"
    os.makedirs(dst, exist_ok=True)

    names = sorted(n for n in os.listdir(src) if n.lower().endswith(".wav"))

    if not names:
        sys.exit(f"no wav files in {src}/")

    kept = []

    for name in names:
        samples, rate = read_wav(os.path.join(src, name))
        before = len(samples) / rate * 1000

        cropped = resample(trim(samples, rate), rate, TARGET_RATE)
        after = len(cropped) / TARGET_RATE * 1000

        if after < 200:
            print(f"{name}: {after:.0f} ms after trimming - SKIPPED, re-record it")
            continue

        kept.append((name, before, after, cropped))

    # A take that trims to far longer than the rest means the energy
    # gate never closed — background noise, a cough, a second word.
    # Whatever is fed these, a sample that is mostly not the wake word
    # is teaching it the wrong thing, so they are worth flagging.
    lengths = sorted(a for _, _, a, _ in kept)
    median = lengths[len(lengths) // 2] if lengths else 0
    limit = median * 1.8

    for name, before, after, cropped in kept:
        if after > limit:
            print(
                f"{name}: {after:.0f} ms vs {median:.0f} ms median "
                f"- SKIPPED as an outlier, the word was not isolated"
            )
            continue

        out = os.path.join(dst, name)

        with wave.open(out, "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(TARGET_RATE)
            f.writeframes(
                b"".join(
                    struct.pack("<h", int(max(-1.0, min(1.0, v)) * 32767))
                    for v in cropped
                )
            )

        print(f"{name}: {before:.0f} ms -> {after:.0f} ms")


main()
