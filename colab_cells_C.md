# Colab cells for Plan C (sanity checks, write-up, submission)

Paste each block into its own cell, in order. Use a **T4 GPU** runtime.
These use the same Drive folder and the same `hex_render.py` as Plan B, so the
checks exercise exactly the input path the demo uses.

## Cell 1: mount Drive, install flyvis, download the pretrained networks

```python
import os
os.environ["FLYVIS_ROOT_DIR"] = "/content/flyvis_data"   # must be set before importing flyvis
os.environ["MPLBACKEND"] = "Agg"

from google.colab import drive
drive.mount("/content/drive")

ROOT = "/content/drive/MyDrive/fly_sees"
assert os.path.isdir(ROOT), (
    f"{ROOT} not found. Create fly_sees/ in My Drive (C owns it), or add a shortcut "
    "to the shared folder: right-click -> Organize -> Add shortcut -> My Drive."
)
for sub in ("clips", "results", "demo"):
    os.makedirs(f"{ROOT}/{sub}", exist_ok=True)

!pip install -q flyvis
!flyvis download-pretrained --skip_large_files
# if pip asks for a restart: restart, then re-run this cell
```

## Cell 2: get the code

```python
!git clone -q https://github.com/DhruvPaul07/what-the-fly-sees /content/fly_sees_code || git -C /content/fly_sees_code pull -q
%cd /content/fly_sees_code
import sys; sys.path.insert(0, "/content/fly_sees_code")
from pathlib import Path
from IPython.display import Image, display
import flysim as fs, sanity_checks as sc

ROOT = Path("/content/drive/MyDrive/fly_sees")
network = fs.load_network(fs.MAIN_NETWORK)        # flow/0000/000
```

## Cells 3-7: one sanity check per cell

Each saves `results/sanity/<name>.png` and records `{check, passed, note}` in
`results/sanity/sanity.json`. Re-running a cell replaces that check's entry.

```python
sc.check_uniform_gray(ROOT, network)               # 1. uniform gray stays flat
display(Image(str(ROOT / "results/sanity/uniform_gray.png")))
```

```python
sc.check_photoreceptors_track(ROOT, network)       # 2. uses clips/test_main.npz, or a random-walk clip until B uploads it
display(Image(str(ROOT / "results/sanity/photoreceptors_track.png")))
```

```python
sc.check_on_off_edges(ROOT, network)               # 3. ON edges drive T4, OFF edges drive T5
display(Image(str(ROOT / "results/sanity/on_off_edges.png")))
```

```python
sc.check_activity_settles(ROOT, network)           # 4. activity settles
display(Image(str(ROOT / "results/sanity/activity_settles.png")))
```

```python
sc.check_second_network(ROOT)                      # 5. repeats check 3 with flow/0000/001
display(Image(str(ROOT / "results/sanity/second_network.png")))
```

Summary for the team chat (re-run check 2 once `clips/test_main.npz` is up):

```python
import json
results = json.loads((ROOT / "results/sanity/sanity.json").read_text())
print(f"{sum(r['passed'] for r in results)}/{len(results)} sanity checks pass")
```

If one fails, tell A immediately: it likely means an input scaling or orientation bug.

## Cell 8 (bonus, only if the checks are done): looming vs sliding

```python
!python looming_bonus.py --root $ROOT
display(Image(str(ROOT / "results/sanity/looming_vs_sliding.png")))
```

## Cell 9: build the write-up and print the submission checklist

Run this any time; it uses whatever is in the Drive folder and marks the rest
as not available. Re-run after A's `results/scores.csv` and B's `demo/` files land.

```python
!python build_writeup.py --root $ROOT \
    --team "A=<name>, B=<name>, C=<name>" \
    --notebook "A=<Colab link>" "B=<Colab link>" "C=<Colab link>"
print((ROOT / "WRITEUP.md").read_text())
```

Add `--preliminary` if A's final scores are not in by 1:45; the numbers are then
labelled "preliminary (synthetic-trained)". This also happens automatically when
`results/demo_payload.npz` carries `is_placeholder: True`.
