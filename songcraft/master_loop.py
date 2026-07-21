#!/usr/bin/env python3
"""
SongCraft auto-mastering, written as an AGENT LOOP.

This is the example from the article "Stop prompting, start building loops".
Instead of one fixed processing chain, the engine iterates:

    OBSERVE  measure the track (integrated loudness, true peak, tonal balance, crest)
    GATE     within tolerance? out of passes? -> stop
    DECIDE   pick ONE corrective move
    ACT      apply it with a real DSP processor
    ...and re-measure. Repeat until it converges or hits max_passes.

The VERIFIER is a set of unforgeable measurements (a loudness meter does not
negotiate). The DECIDE step here is a deterministic policy so the demo runs
offline and reproducibly; in the real product that single call is where a model
chooses the move. Everything else -- observation, the terminal gate, the budget,
memory -- is ordinary code we own. That is the whole point.

    python master_loop.py --in samples/vocal_mix.wav                 # -> converges
    python master_loop.py --in samples/guitar_mix.wav                # -> converges
    python master_loop.py --in samples/guitar_mix.wav --target -9    # impossible -> halts

Deps: numpy, scipy, pyloudnorm, pedalboard, soundfile   (see requirements.txt)
"""
import argparse, os, sys
import numpy as np
import pyloudnorm as pyln
from scipy.signal import resample_poly
from pedalboard import (Pedalboard, Gain, Limiter, Compressor,
                        LowShelfFilter, HighShelfFilter, PeakFilter)
from pedalboard.io import AudioFile

SR = 44100
BALANCE_TARGET = np.array([0.46, 0.40, 0.14])   # target energy split: low / mid / high
BANDS = [(20, 250), (250, 4000), (4000, 18000)]

# ---------- terminal colours (only when writing to a real tty) ----------
_tty = sys.stdout.isatty()
def _c(code): return (lambda s: f"\033[{code}m{s}\033[0m") if _tty else (lambda s: s)
DIM, CYN, YEL, BLU, GRN, MAG, RED, BOLD = map(_c, ["2;37", "36", "33", "34", "32", "35", "31", "1"])
TAGCOL = {"OBSERVE": CYN, "DECIDE": YEL, "ACT": BLU, "VERIFY": GRN,
          "GATE": MAG, "DONE": GRN, "HALT": RED}

def emit(p, tag, msg):
    print(f"{DIM(f'pass {p:>2}')}  {TAGCOL[tag](BOLD(f'{tag:<7}'))} {msg}")

# ---------- io (pedalboard uses shape (channels, samples)) ----------
def read_audio(path):
    with AudioFile(path).resampled_to(SR) as f:
        a = f.read(f.frames)
    if a.ndim == 1:
        a = a[np.newaxis, :]
    if a.shape[0] == 1:
        a = np.repeat(a, 2, axis=0)
    return a.astype(np.float32)

def write_audio(path, a):
    with AudioFile(path, "w", SR, num_channels=a.shape[0]) as f:
        f.write(np.clip(a, -1.0, 1.0))

# ---------- the verifier: ground truth, not self-report ----------
def lufs(a):
    data = a.T
    if data.shape[0] < int(0.4 * SR):
        return -70.0
    v = pyln.Meter(SR).integrated_loudness(data)
    return -70.0 if np.isneginf(v) else float(v)

def true_peak_dbtp(a):
    peak = max(np.max(np.abs(resample_poly(a[c], 4, 1))) for c in range(a.shape[0]))
    return 20.0 * np.log10(peak + 1e-12)

def band_fractions(a):
    mono = a.mean(axis=0)
    mag = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2
    freqs = np.fft.rfftfreq(len(mono), 1.0 / SR)
    fr = np.array([mag[(freqs >= lo) & (freqs < hi)].sum() for lo, hi in BANDS])
    return fr / (fr.sum() + 1e-12)

def crest_db(a):
    peak = float(np.max(np.abs(a))) + 1e-12
    rms = float(np.sqrt(np.mean(a ** 2))) + 1e-12
    return 20.0 * np.log10(peak / rms)

def observe(a):
    fr = band_fractions(a)
    return {
        "lufs": lufs(a),
        "dbtp": true_peak_dbtp(a),
        "fr": fr,
        "balance": 100.0 * (1.0 - min(1.0, np.abs(fr - BALANCE_TARGET).sum())),
        "crest": crest_db(a),
    }

def obs_line(o):
    return (f"{o['lufs']:5.1f} LUFS {DIM('·')} peak {o['dbtp']:4.1f} dBTP {DIM('·')} "
            f"balance {o['balance']:.0f}% {DIM('·')} crest {o['crest']:.1f} dB")

def apply(a, *plugins):
    return Pedalboard(list(plugins))(a, SR)

# ---------- the DECIDE step (a policy; swap for a model in production) ----------
def decide(o, st, target, ceiling, min_crest):
    lufs_v, dbtp, dev, crest = o["lufs"], o["dbtp"], o["fr"] - BALANCE_TARGET, o["crest"]
    tol = 0.6

    if dbtp > ceiling + 0.3:                                   # 1. peaks over the ceiling
        return f"trim {-(dbtp - ceiling) - 0.3:.1f} dB, peaks over ceiling", (Gain(gain_db=-(dbtp - ceiling) - 0.3),)

    if lufs_v >= target - tol and crest < min_crest:           # 2. loud but crushed -> back off
        return "back off 1.0 dB, crest too low", (Gain(gain_db=-1.0),)

    worst = int(np.argmax(np.abs(dev)))                        # 3. one corrective EQ move (bounded)
    if abs(dev[worst]) > 0.06 and st["eq"][worst] < 2:
        st["eq"][worst] += 1
        table = {
            (0, 1): ("low shelf -3.0 dB @ 120 Hz, tame boom", LowShelfFilter(cutoff_frequency_hz=120, gain_db=-3.0, q=0.7)),
            (0, -1): ("low shelf +3.0 dB @ 90 Hz, add weight", LowShelfFilter(cutoff_frequency_hz=90, gain_db=3.0, q=0.7)),
            (1, 1): ("bell -3.0 dB @ 350 Hz, clear the mud", PeakFilter(cutoff_frequency_hz=350, gain_db=-3.0, q=1.0)),
            (1, -1): ("bell +2.5 dB @ 2.5 kHz, add presence", PeakFilter(cutoff_frequency_hz=2500, gain_db=2.5, q=0.9)),
            (2, 1): ("de-ess -3.0 dB @ 6.5 kHz, tame highs", PeakFilter(cutoff_frequency_hz=6500, gain_db=-3.0, q=2.2)),
            (2, -1): ("high shelf +3.5 dB @ 9 kHz, add air", HighShelfFilter(cutoff_frequency_hz=9000, gain_db=3.5, q=0.7)),
        }
        desc, plug = table[(worst, 1 if dev[worst] > 0 else -1)]
        return desc, (plug,)

    if lufs_v < target - tol:                                  # 4. reach the loudness target
        if not st["glued"]:
            st["glued"] = True
            return "glue compressor 2:1, add density", (Compressor(threshold_db=-18, ratio=2.0, attack_ms=25, release_ms=200),)
        step = min(target - lufs_v, 3.0)
        return f"limiter, push +{step:.1f} dB toward {target:g} LUFS", (
            Gain(gain_db=step), Limiter(threshold_db=ceiling - 1.0, release_ms=120))

    if lufs_v > target + tol:                                  # 5. too loud -> trim
        return f"trim {target - lufs_v:.1f} dB to target", (Gain(gain_db=target - lufs_v),)

    return None, None

def satisfied(o, target, ceiling, min_crest):
    # hard spec = loudness + true peak + a quality floor (crest factor).
    # tonal balance is guidance the policy improves, not a release gate.
    return (abs(o["lufs"] - target) <= 0.6 and o["dbtp"] <= ceiling + 0.3
            and o["crest"] >= min_crest)

# ---------- the loop ----------
def run_loop(path, target, ceiling, max_passes, min_crest, out_path):
    a = read_audio(path)
    print(f"{DIM('loop')}  goal: {BOLD(f'master to {target:g} LUFS')}   "
          f"tools: eq, comp, limiter, deess   verify: {CYN('loudness meter')}   "
          f"file: {os.path.basename(path)}")
    print(DIM("─" * 78))

    st = {"eq": [0, 0, 0], "glued": False}
    o = observe(a)
    moves = 0
    for p in range(max_passes):
        emit(p, "OBSERVE", obs_line(o))
        if satisfied(o, target, ceiling, min_crest):
            emit(p, "GATE", "all targets within tolerance " + GRN("✓"))
            break
        desc, plugins = decide(o, st, target, ceiling, min_crest)
        if plugins is None:
            emit(p, "GATE", "no move improves the goal")
            break
        emit(p, "DECIDE", desc)
        a = apply(a, *plugins)
        moves += 1
        o = observe(a)
        emit(p, "VERIFY", obs_line(o))
    print(DIM("─" * 78))

    if out_path:
        write_audio(out_path, a)
    if satisfied(o, target, ceiling, min_crest):
        print(f"{GRN(BOLD('✓ MASTERED'))} in {moves} moves {DIM('·')} {o['lufs']:.1f} LUFS "
              f"{DIM('·')} {o['dbtp']:.1f} dBTP {DIM('·')} balance {o['balance']:.0f}% "
              f"{DIM('·')} crest {o['crest']:.1f} dB"
              + (f"   {DIM('-> out/' + os.path.basename(out_path))}" if out_path else ""))
        return 0
    print(f"{RED(BOLD('✗ HALTED'))} at max_passes ({max_passes}) {DIM('·')} "
          f"{target:g} LUFS unreachable without audible distortion")
    print(DIM(f"  stuck at {o['lufs']:.1f} LUFS, crest {o['crest']:.1f} dB (< {min_crest:g} dB floor); "
              f"constraints conflict, the loop refused to trade quality forever."))
    return 1


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--in", dest="inp", default=os.path.join(here, "samples", "vocal_mix.wav"))
    ap.add_argument("--target", type=float, default=-14.0)
    ap.add_argument("--ceiling", type=float, default=-1.0)
    ap.add_argument("--max-passes", type=int, default=16)
    ap.add_argument("--min-crest", type=float, default=11.0, help="quality floor: crest factor in dB")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        outdir = os.path.join(here, "out")
        os.makedirs(outdir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.inp))[0]
        args.out = os.path.join(outdir, f"{base}_mastered_{int(abs(args.target))}.wav")
    return run_loop(args.inp, args.target, args.ceiling, args.max_passes, args.min_crest, args.out)


if __name__ == "__main__":
    sys.exit(main())
