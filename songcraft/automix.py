#!/usr/bin/env python3
"""
SongCraft Phase-0 auto-mix / auto-master engine.

Takes a dry vocal + an instrumental and returns a finished, release-ready mix
WITHOUT the user touching EQ/compression/mixing. The "intelligence":
  * loudness analysis (pyloudnorm) auto-balances the vocal above the instrumental
  * a fixed pro-style processing chain handles cleanup/EQ/compression/space
  * master bus glue + limiter + normalize to -14 LUFS (streaming standard)

Outputs both a naive equal-sum "before" and the processed "after" for A/B.

Usage:
  python automix.py --vocal v.wav --instrumental i.wav [--out-dir out]
"""
import argparse, os
import numpy as np
import pyloudnorm as pyln
from pedalboard import (Pedalboard, Compressor, Gain, Reverb, HighpassFilter,
                        HighShelfFilter, PeakFilter, Limiter, NoiseGate)
from pedalboard.io import AudioFile

SR = 44100
TARGET_LUFS = -14.0      # Spotify/Apple/YouTube streaming target
VOCAL_LEAD_DB = 3.0      # how far the vocal sits ABOVE the instrumental (in LU)


# ---------- io helpers (pedalboard uses shape (channels, samples)) ----------
def read_audio(path):
    with AudioFile(path).resampled_to(SR) as f:
        audio = f.read(f.frames)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    if audio.shape[0] == 1:                      # mono -> stereo
        audio = np.repeat(audio, 2, axis=0)
    return audio.astype(np.float32)


def write_audio(path, audio):
    audio = np.clip(audio, -1.0, 1.0)
    with AudioFile(path, "w", SR, num_channels=audio.shape[0]) as f:
        f.write(audio)


def match_len(a, b):
    n = max(a.shape[1], b.shape[1])
    pad = lambda x: np.pad(x, ((0, 0), (0, n - x.shape[1])))
    return pad(a), pad(b)


def lufs(audio):
    """measure integrated loudness; audio is (channels, samples)."""
    data = audio.T                               # -> (samples, channels)
    meter = pyln.Meter(SR)
    if data.shape[0] < int(0.4 * SR):            # too short to measure
        return -70.0
    val = meter.integrated_loudness(data)
    return -70.0 if np.isneginf(val) else val


def gain_db(audio, db):
    return audio * (10 ** (db / 20.0))


# ---------- processing chains ----------
VOCAL_CHAIN = Pedalboard([
    NoiseGate(threshold_db=-45, ratio=2.5, attack_ms=2, release_ms=120),
    HighpassFilter(cutoff_frequency_hz=90),          # kill rumble
    PeakFilter(cutoff_frequency_hz=300, gain_db=-2.5, q=1.0),   # de-mud
    PeakFilter(cutoff_frequency_hz=3000, gain_db=2.0, q=0.8),   # presence
    PeakFilter(cutoff_frequency_hz=7000, gain_db=-3.0, q=2.5),  # tame sibilance
    Compressor(threshold_db=-20, ratio=3.5, attack_ms=5, release_ms=120),
    HighShelfFilter(cutoff_frequency_hz=9000, gain_db=3.0),     # air
    Compressor(threshold_db=-12, ratio=2.0, attack_ms=10, release_ms=150),
    Reverb(room_size=0.22, damping=0.5, wet_level=0.10, dry_level=0.92, width=0.9),
])

# instrumental: carve a pocket so the vocal has room to sit
INSTRUMENT_CHAIN = Pedalboard([
    PeakFilter(cutoff_frequency_hz=300, gain_db=-1.5, q=1.0),
    PeakFilter(cutoff_frequency_hz=3000, gain_db=-2.0, q=1.2),  # vocal pocket
])

MASTER_CHAIN = Pedalboard([
    HighpassFilter(cutoff_frequency_hz=28),
    Compressor(threshold_db=-12, ratio=2.0, attack_ms=30, release_ms=200),  # glue
    HighShelfFilter(cutoff_frequency_hz=11000, gain_db=1.5),    # sheen
    Limiter(threshold_db=-1.0, release_ms=120),                 # ceiling
])


def normalize_lufs(audio, target):
    data = audio.T
    cur = lufs(audio)
    if cur <= -70.0:
        return audio
    out = pyln.normalize.loudness(data, cur, target)
    return out.T.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocal", required=True)
    ap.add_argument("--instrumental", required=True)
    ap.add_argument("--out-dir", default=os.path.expanduser("~/songcraft/out"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    voc = read_audio(args.vocal)
    inst = read_audio(args.instrumental)
    voc, inst = match_len(voc, inst)

    print(f"  raw vocal:        {lufs(voc):6.1f} LUFS")
    print(f"  raw instrumental: {lufs(inst):6.1f} LUFS")

    # --- BEFORE: naive equal sum (what a beginner gets) ---
    before = voc + inst
    peak = np.max(np.abs(before)) or 1.0
    before_n = before / peak * 0.89
    write_audio(os.path.join(args.out_dir, "before_naive_mix.wav"), before_n)
    print(f"  -> before_naive_mix.wav  ({lufs(before_n):.1f} LUFS)")

    # --- AFTER: auto-mix + auto-master ---
    voc_p = VOCAL_CHAIN(voc, SR)
    inst_p = INSTRUMENT_CHAIN(inst, SR)

    # auto-balance: push instrumental to (vocal_LUFS - lead) so vocal sits on top
    target_inst = lufs(voc_p) - VOCAL_LEAD_DB
    inst_p = gain_db(inst_p, target_inst - lufs(inst_p))

    mix = voc_p + inst_p
    mastered = MASTER_CHAIN(mix, SR)                  # glue, EQ, limiter (maximizes)
    mastered = normalize_lufs(mastered, TARGET_LUFS)  # FINAL: pull down to target

    out_path = os.path.join(args.out_dir, "after_master.wav")
    write_audio(out_path, mastered)
    print(f"  -> after_master.wav      ({lufs(mastered):.1f} LUFS)  [target {TARGET_LUFS}]")
    print("\nDone. A/B these two files in", args.out_dir)


if __name__ == "__main__":
    main()
