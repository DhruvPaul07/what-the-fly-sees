# What the Fly Sees: Plan B (footage, fly's-eye rendering, demo video)

All scripts take `--root` (default: current dir); `clips/`, `results/`, `demo/` live under it.

## 1. Laptop: preprocess footage (needs only numpy + opencv)

```bash
python prep_clips.py --test-main main.mov --test-still still.mov \
    --train t1.mov t2.mov t3.mov t4.mov t5.mov t6.mov --out clips/
```

Upload `clips/` to the Drive folder `fly_sees/clips/`.

## 2. Colab (or local with flyvis): see `colab_cells.md`

```bash
python orientation_test.py     # results/orientation.json + orientation_check.png
python coarseness.py           # demo/coarseness.png
python make_fake_payload.py    # results/demo_payload_FAKE.npz
python make_fake_payload.py --still
python make_demo.py --payload results/demo_payload_FAKE.npz --out demo/FAKE_what_the_fly_sees.mp4
```

## 3. Real payload from teammate A

```bash
python make_demo.py --payload results/demo_payload.npz --clip clips/test_main.npz \
    --orientation results/orientation.json --out demo/what_the_fly_sees.mp4
python make_demo.py --still    # results/demo_payload_still.npz -> demo/still_scene.mp4
```

Each run also writes a half-resolution `.gif`. Slow render: add `--dpi 80`.

Tests: `python -m pytest -q test_logic.py`

## Optional add-on: AI-upscaled panel (Plan B2)

```bash
pip install spandrel scipy scikit-image
python enhance_flyeye.py --root <fly_sees folder>
```

Needs `clips/test_main.npz` and `results/orientation.json`. Downloads the
Real-ESRGAN weights to `weights/` on first run. Writes `demo/ai_enhanced.mp4`,
`.gif`, `ai_enhanced_still.png`, `demo/ai_frames.npy`, `results/fly_lowres.npy`
and `results/ai_enhance_metrics.json`, and prints a one-line verdict.

## Plan C: sanity checks, write-up, submission

```bash
flyvis download-pretrained --skip_large_files      # once; set FLYVIS_ROOT_DIR first if you use one
python sanity_checks.py --root <fly_sees folder>   # 5 checks -> results/sanity/*.png + sanity.json
python looming_bonus.py --root <fly_sees folder>   # optional bonus figure
python build_writeup.py --root <fly_sees folder> --team "A=..., B=..., C=..."
```

`build_writeup.py` reads A's `results/scores.csv` and `results/checks.json`, the
sanity results, and B's `demo/` files and payloads, writes `WRITEUP.md` into the
fly_sees folder, and prints the submission checklist. Anything missing is marked
"not available yet"; table cells without a score stay blank. Colab version:
`colab_cells_C.md`.
