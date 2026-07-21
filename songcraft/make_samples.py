#!/usr/bin/env python3
"""
Synthesize two intentionally-imperfect sample mixes for the auto-mastering loop:

  vocal_mix.wav   a quiet, dynamically uneven, slightly harsh/sibilant vocal
  guitar_mix.wav  a strummed guitar that is boomy in the low-mids and a touch dull

Both are deliberately NOT at release spec (too quiet, unbalanced) so master_loop.py
has real work to do: measure, correct, re-measure, repeat.

    python make_samples.py
"""
import os
import numpy as np
import soundfile as sf

SR = 44100
DUR = 5.0
t = np.linspace(0, DUR, int(SR * DUR), endpoint=False)
HERE = os.path.dirname(os.path.abspath(__file__))
SAMP = os.path.join(HERE, "samples")
os.makedirs(SAMP, exist_ok=True)
rng = np.random.default_rng(7)


def tone(freq, harmonics=(1, 0.5, 0.25, 0.12), vibrato=0.0):
    f = freq * (1 + vibrato * np.sin(2 * np.pi * 5.5 * t))
    phase = 2 * np.pi * np.cumsum(f) / SR
    sig = sum(amp * np.sin(i * phase) for i, amp in enumerate(harmonics, start=1))
    return sig / max(np.max(np.abs(sig)), 1e-9)


# ---------------- VOCAL: quiet, uneven, harsh sibilance ----------------
voc = tone(220.0, harmonics=(1, 0.6, 0.4, 0.3, 0.2), vibrato=0.012)
env = np.ones_like(t) * 0.3
env[(t > 0.8) & (t < 1.8)] = 0.9        # loud phrase
env[(t > 2.2) & (t < 2.9)] = 0.12       # nearly inaudible phrase
env[(t > 3.4) & (t < 4.6)] = 0.7
voc *= env
for c in (1.2, 3.6, 4.2):               # harsh "s" bursts (filtered noise)
    m = (t > c) & (t < c + 0.11)
    voc[m] += 0.5 * rng.standard_normal(m.sum())
voc /= np.max(np.abs(voc))
voc *= 0.30                              # deliberately quiet, needs auto-leveling
sf.write(os.path.join(SAMP, "vocal_mix.wav"), voc.astype(np.float32), SR)


# ---------------- GUITAR: strummed E major, boomy low-mids ----------------
def pluck(freq, start, dur=1.6, amp=1.0):
    phase = 2 * np.pi * freq * t
    harm = [1, 0.7, 0.55, 0.42, 0.32, 0.25, 0.19, 0.14, 0.10, 0.07, 0.05]
    body = sum(a * np.sin((i + 1) * phase) for i, a in enumerate(harm))
    out = np.zeros_like(t)
    seg = (t >= start) & (t < start + dur)
    out[seg] = body[seg] * np.exp(-(t[seg] - start) * 4.2) * amp
    # short broadband pick-attack transient (adds high-frequency content)
    atk = (t >= start) & (t < start + 0.02)
    out[atk] += 0.25 * amp * rng.standard_normal(atk.sum()) * np.exp(-(t[atk] - start) * 180)
    return out


# E2 B2 E3 G#3 B3 E4 ; low strings a touch hot => boomy
notes = [(82.41, 1.15), (123.47, 1.05), (164.81, 1.0),
         (207.65, 0.95), (246.94, 0.9), (329.63, 0.9)]
guitar = np.zeros_like(t)
for strum in (0.0, 1.7, 3.4):                       # three downstrokes
    for k, (f, a) in enumerate(notes):
        guitar += pluck(f, strum + k * 0.018, amp=a)
guitar = np.tanh(1.3 * guitar)                      # mild amp-style saturation
guitar /= np.max(np.abs(guitar))
# widen to stereo with a short Haas delay, keep it quiet
left = guitar
right = np.roll(guitar, 40)
guitar_st = np.stack([left, right], axis=1) * 0.34
sf.write(os.path.join(SAMP, "guitar_mix.wav"), guitar_st.astype(np.float32), SR)

print("wrote:")
print("  samples/vocal_mix.wav   (mono,  ~5s, quiet + harsh)")
print("  samples/guitar_mix.wav  (stereo,~5s, boomy low-mids)")
