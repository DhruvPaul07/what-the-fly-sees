# What the Fly Sees: executed run

This folder is the guide ("What the Fly Sees: Step-by-Step Build Guide") carried out end to end: the flyvis visual
model (Lappalainen et al., Nature 2024) was run on video clips, decoders were trained on its simulated neurons, and
the demo video was rendered.

## How this run differs from the guide

The guide uses the official flyvis package in Colab. That wasn't possible here: flyvis needs PyTorch, and its
pretrained networks download from Google Drive, which this environment couldn't reach. The same model was rebuilt
from published sources instead, and checked at each step:

| Piece | Source | Check |
| --- | --- | --- |
| The 721-column lattice | flyvis 1.2.0's own connectome file (`fib25-fib19_v2.2.json`), compiled with flyvis's own rules | 45,669 neurons and 1,513,231 connections, exactly flyvis's published counts |
| The 50 trained networks | flyvis's checkpoint numbers as shipped in the `flycns` package (PyPI, MIT) | Every connection maps to a trained synapse-count group; every sign matches the connectome file |
| The dynamics | flyvis's `PPNeuronIGRSynapses` equation, in numpy | Matches flycns's reference engine to 1.9e-6 over 40 steps (flycns reports its engine matches flyvis itself to 2.4e-6) |
| The eye | flyvis's `BoxEye` (13 x 13 pixel box mean at 721 centres) in numpy | Same receptor layout and 391 x 391 pixel footprint |

There was no phone footage, so clips were built from photos bundled with scikit-image and matplotlib: camera pans
(the whole scene moves), an object gliding over a still photo, plus random-walk blocks, gratings and still scenes.
Training used 14 photos; tests used 3 photos never seen in training (astronaut, chelsea, motorcycle_right).
Time step 10 ms, 1-second fade-in, decoder history 5 steps (50 ms), ridge regression on standardised features.

## Results

Sanity checks (guide Step 5), networks 000 to 004:

| Network | Grey stays flat | R1 tracks input | ON edge: T4 > T5 | OFF edge: T5 > T4 | Settles |
| --- | --- | --- | --- | --- | --- |
| 000 | pass | pass | pass | pass | pass |
| 001 | fail (slow drift) | pass | pass | pass | pass |
| 002 | fail (slow drift) | pass | pass | pass | fail |
| 003 | pass | pass | pass | pass | pass |
| 004 | pass | pass | pass | fail | fail |

The failures in 002 and 004 come from slow cell types (time constants up to about 0.5 s) and, in 004, a T2 cell
whose activity climbs to about 99 under plain grey. They persist at flyvis's own 5 ms evaluation step, so they are
properties of those trained networks, not of this run.

Decoder scores (guide Steps 7 and 8): correlation between decoded and true frames, per frame, averaged; test clips
use unseen photos. Network 000 shown, with the range over networks 000 to 004.

| Decoder group | Train | Test, unseen photos | Test, synthetic | Range over 5 networks (unseen photos) |
| --- | --- | --- | --- | --- |
| Photoreceptors (R1 to R6) | 1.000 | 1.000 | 1.000 | 0.999 to 1.000 |
| Lamina (L1 to L5) | 0.977 | 0.989 | 0.971 | 0.948 to 0.992 |
| Medulla (Mi, Tm) | 0.992 | 0.996 | 0.984 | 0.989 to 0.997 |
| Motion detectors (T4, T5): brightness | 0.293 | 0.385 | 0.275 | 0.385 to 0.902 |
| Motion detectors (T4, T5): brightness change | 0.677 | 0.683 | 0.712 | 0.633 to 0.756 |
| Shuffled control | 0.074 | -0.066 | -0.065 | -0.112 to 0.285 |
| Mean-brightness baseline | 0.307 | 0.218 | 0.280 | same for all |

Main findings:

- The picture survives almost intact through the lamina and medulla: linear decoders recover it at 0.95 or better.
- The motion detectors carry brightness change far more reliably than brightness itself. "What changed" decodes at
  0.63 to 0.76 in every network; how much plain brightness leaks into T4/T5 varies widely (0.39 to 0.90).
- Still scenes did not go fully quiet: brightness decoded from T4/T5 on still photos still correlates 0.49 to 0.91
  with the truth. The decoder is linear with no memory of scenes, so this is static brightness carried by T4/T5
  activity in these models, not painting from memory.
- On synthetic clips, network 000's T4/T5 brightness decoder (0.275) only ties the mean-brightness baseline (0.280).

## Files

- `flyeye.py`: lattice builder, trained-parameter loader, numpy eye, batched simulator
- `check_parity.py`: the simulator against flycns's reference engine
- `clips.py`, `make_clips.py`: training and test clips
- `sanity.py`: Step 5 checks (`sanity_results.json`)
- `decode.py`: Steps 6 to 8 for one network (`decode_000.json` to `decode_004.json`)
- `summarize.py`: the score table (`score_table.json`) and chart
- `demo.py`: Step 9, the demo video

## Rerun

You need Python 3.11 or newer (flycns requires it) and ffmpeg for the video (`apt install ffmpeg` on Linux,
`brew install ffmpeg` on a Mac; Colab already has it). Pin the versions this run used, because the training photos
ship inside scikit-image 0.26 and matplotlib 3.11 and older or newer releases may not include all of them.

```bash
pip install -r requirements.txt
pip download flyvis==1.2.0 --no-deps -d flyvis_pkg      # only for its connectome JSON, about 0.4 MB
python3 -c "import flyeye, numpy as np; lat = flyeye.build_lattice(flyeye.load_connectome_spec('flyvis_pkg/flyvis-1.2.0-py3-none-any.whl')); np.savez_compressed('lattice.npz', **{k: v for k, v in lat.items() if k != 'input_types'})"
python3 check_parity.py
python3 sanity.py
python3 make_clips.py              # about 3 minutes on one CPU core
for m in 0 1 2 3 4; do python3 decode.py $m; done   # about 2.5 minutes per network
python3 summarize.py
python3 demo.py                    # needs decoders_000.npz from decode.py 0
```

Nothing is downloaded from Google Drive: the trained networks come inside the flycns package. With the official
flyvis package and a GPU, the guide's own code does the same job faster.
