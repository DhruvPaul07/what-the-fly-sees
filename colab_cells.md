# Colab cells for Plan B (Parts 3-5)

Paste each block into its own cell, in order. No GPU needed.

## Cell 1: mount Drive, install flyvis

```python
import os
os.environ["FLYVIS_ROOT_DIR"] = "/content/flyvis_data"   # must be set before importing flyvis
os.environ["MPLBACKEND"] = "Agg"

from google.colab import drive
drive.mount("/content/drive")

ROOT = "/content/drive/MyDrive/fly_sees"
# Stop here rather than silently creating a private fly_sees/ the team cannot see.
assert os.path.isdir(ROOT), (
    f"{ROOT} not found. In Drive, right-click the shared fly_sees folder -> "
    "Organize -> Add shortcut -> My Drive, then rerun this cell."
)
CODE = "/content/fly_sees_code"
os.makedirs(CODE, exist_ok=True)
for sub in ("clips", "results", "demo"):
    os.makedirs(f"{ROOT}/{sub}", exist_ok=True)
if not os.listdir(f"{ROOT}/clips"):
    print("WARNING: clips/ is empty. Run prep_clips.py on the laptop and upload the .npz files first.")

!pip install -q flyvis
```

Do NOT download the pretrained networks; only the eye is needed here.

## Cell 2: upload the scripts

Select `hex_render.py`, `orientation_test.py`, `coarseness.py`,
`make_fake_payload.py`, `make_demo.py` in the file picker.

```python
from google.colab import files
%cd $CODE
uploaded = files.upload()
!ls -la $CODE
```

(Alternative: put the scripts in `MyDrive/fly_sees/code/` and set `CODE = f"{ROOT}/code"`.)

## Cell 3: check the clips are on Drive

```python
!ls -la $ROOT/clips
```

Expect `test_main.npz`, `test_still.npz`, `train_00.npz`, ... (made on the laptop by `prep_clips.py`).

## Cell 4: orientation test (Part 3.1), then tell the team

```python
%cd $CODE
!python orientation_test.py --root $ROOT
from IPython.display import Image, display
display(Image(f"{ROOT}/results/orientation_check.png"))
!cat $ROOT/results/orientation.json
```

## Cell 5: coarseness hero image (Part 3.2)

```python
!python coarseness.py --root $ROOT
display(Image(f"{ROOT}/demo/coarseness.png"))
```

## Cell 6: fake payloads + demo renderer test (Part 4)

```python
!python make_fake_payload.py --root $ROOT
!python make_fake_payload.py --root $ROOT --still
!python make_demo.py --root $ROOT --payload results/demo_payload_FAKE.npz --clip clips/test_main.npz --orientation results/orientation.json --out demo/FAKE_what_the_fly_sees.mp4
!python make_demo.py --root $ROOT --still --payload results/demo_payload_still_FAKE.npz --out demo/FAKE_still_scene.mp4
display(Image(f"{ROOT}/demo/FAKE_what_the_fly_sees.gif"))
```

## Cell 7: real payload from A (Part 5); rerun whenever a new payload lands

```python
!python make_demo.py --root $ROOT --payload results/demo_payload.npz --clip clips/test_main.npz --orientation results/orientation.json --out demo/what_the_fly_sees.mp4
!python make_demo.py --root $ROOT --still
display(Image(f"{ROOT}/demo/what_the_fly_sees.gif"))
!ls -la $ROOT/demo
```

`--still` reads `results/demo_payload_still.npz` + `clips/test_still.npz` and writes `demo/still_scene.mp4` (+ `.gif`).

If rendering is slow add `--dpi 80`. If ffmpeg fails the script prints a warning and saves the GIF only.
If the payload has `is_placeholder: True`, the title automatically says the decoder was trained on synthetic clips.
