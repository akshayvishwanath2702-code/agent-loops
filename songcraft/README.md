# The auto-mastering loop (runnable)

This is the worked example from the article, as real, runnable code. It masters an
audio file to release spec **as an agent loop**: measure, decide one move, apply it,
re-measure, and repeat until it converges or gives up.

It is deliberately small and dependency-honest. Real DSP (via
[`pedalboard`](https://github.com/spotify/pedalboard)), real loudness measurement
(via [`pyloudnorm`](https://github.com/csteinmetz1/pyloudnorm), an ITU-R BS.1770
meter), real true-peak and crest-factor analysis. No mock numbers.

## The loop

```
OBSERVE   integrated loudness (LUFS), true peak (dBTP), tonal balance, crest factor
GATE      within tolerance? out of passes? -> stop
DECIDE    pick ONE corrective move
ACT       apply it with a real processor (EQ / compressor / limiter / de-ess)
          ...re-measure. repeat.
```

The **verifier** is the set of measurements. A loudness meter does not negotiate.

The **DECIDE** step here is a deterministic policy (`decide()` in `master_loop.py`)
so the demo runs offline and reproducibly. In the real product, that single call is
where a model chooses the move. That is the article's whole point: the model is one
swappable call inside the loop. Observation, the terminal gate, the budget, and
memory are ordinary code you own.

The hard spec is **loudness + true peak + a crest-factor floor** (the quality guard
that stops the loop from crushing the track to hit a number). Tonal balance is
guidance the policy improves, not a release gate.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python make_samples.py                                  # writes the two sample mixes

python master_loop.py --in samples/vocal_mix.wav        # converges to -14 LUFS
python master_loop.py --in samples/guitar_mix.wav       # converges to -14 LUFS
python master_loop.py --in samples/guitar_mix.wav --target -9   # impossible -> HALTS
```

## What is here

| File | What it is |
|---|---|
| `master_loop.py` | the mastering engine written as an agent loop (the example) |
| `make_samples.py` | synthesizes `vocal_mix.wav` (quiet, harsh) and `guitar_mix.wav` (boomy) |
| `automix.py` | the original one-shot chain it grew out of: a fixed pipeline, no loop |
| `samples/` | the two input mixes |
| `out/*.run.txt` | captured console output of the two converging runs |
| `out/guitar_mix.halt.txt` | the `-9 LUFS` run that correctly halts at `max_passes` |
| `out/*_mastered_14.wav` | the mastered results (the "after"), for A/B against the `samples/` inputs |
| `out/guitar_mix_mastered_9_halted.wav` | the over-limited `-9 LUFS` attempt, for reference |

Before / after loudness (measured with `pyloudnorm`):

| Track | Before (`samples/`) | After (`out/`) |
|---|---|---|
| vocal  | -27.2 LUFS (too quiet) | **-14.2 LUFS** |
| guitar | -12.7 LUFS | **-14.0 LUFS** |
| guitar @ -9 target | -12.7 LUFS | -10.3 LUFS (halted, never reached -9) |

## Why `-9 LUFS` halts (and should)

`-14 LUFS` is the streaming standard and these mixes reach it with headroom to spare.
Chasing `-9 LUFS` (roughly loudness-war territory) forces so much limiting that the
crest factor collapses below the quality floor. Watch `out/guitar_mix.halt.txt`: the
loop pushes to -8.5 LUFS, blows past the ceiling, trims back, pushes again, and the
crest factor grinds down pass after pass without ever landing cleanly. `max_passes`
ends it. A loop that cannot give up is not autonomous, it is unsupervised.

## Note

`automix.py` is included as the original one-shot version (it also does the vocal +
instrumental *mix*, not just the master). `master_loop.py` is the loop rewrite used
in the article. Both use the same stack.
