"""Build a contract-format FAKE demo payload so the renderer can be tested
before teammate A's real payload exists. NOT a model output.

    python make_fake_payload.py [--root .]           # test_main  -> results/demo_payload_FAKE.npz
    python make_fake_payload.py [--root .] --still   # test_still -> results/demo_payload_still_FAKE.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import hex_render as hr

PHOTO_NOISE_STD = 0.03
# Fake "motion" is |input[t] - input[t-(K-1)]| divided by max(its peak, this floor),
# so a still scene stays flat instead of having sensor noise blown up to 0-1.
MOTION_SCALE_FLOOR = 0.1
MOTION_PEAK_PERCENTILE = 99.9
NOISE_SEED = 0


def n_model_steps(n_frames: int, fps: float, dt: float = hr.DT) -> int:
    return int(round(n_frames / fps / dt))


def build_step_to_frame(n_steps: int, fps: float, n_frames: int, dt: float = hr.DT) -> np.ndarray:
    """Frame index shown at each model step, clamped to the last frame."""
    raw = (np.arange(n_steps) * dt * fps).astype(np.int64)
    return np.minimum(raw, n_frames - 1)


def fake_recon_photo(eye_input: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    noisy = eye_input[k - 1 :] + rng.normal(0.0, PHOTO_NOISE_STD, eye_input[k - 1 :].shape)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def fake_recon_motion(eye_input: np.ndarray, k: int) -> np.ndarray:
    """|temporal difference over the K-step history window|, rescaled to 0-1."""
    difference = np.abs(eye_input[k - 1 :] - eye_input[: len(eye_input) - k + 1])
    scale = max(float(np.percentile(difference, MOTION_PEAK_PERCENTILE)), MOTION_SCALE_FLOOR)
    return np.clip(difference / scale, 0.0, 1.0).astype(np.float32)


def build_fake_payload(frames: np.ndarray, fps: float, k: int = hr.K, dt: float = hr.DT) -> dict:
    n_steps = n_model_steps(len(frames), fps, dt)
    if n_steps < k:
        raise ValueError(f"clip gives only {n_steps} model steps, need at least K={k}")
    step_to_frame = build_step_to_frame(n_steps, fps, len(frames), dt)
    rendered_frames = hr.render_eye(frames)  # (n_frames, 721)
    eye_input = rendered_frames[step_to_frame].astype(np.float32)  # (T, 721)
    return {
        "input": eye_input,
        "recon_motion": fake_recon_motion(eye_input, k),
        "recon_photo": fake_recon_photo(eye_input, k, np.random.default_rng(NOISE_SEED)),
        "step_to_frame": step_to_frame,
        "K": np.int64(k),
        "dt": np.float64(dt),
        "is_fake": np.bool_(True),  # extra key: make_demo labels the video as a layout test
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--still", action="store_true", help="use test_still instead of test_main")
    parser.add_argument("--clip", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    default_clip = "clips/test_still.npz" if args.still else "clips/test_main.npz"
    default_out = "results/demo_payload_still_FAKE.npz" if args.still else "results/demo_payload_FAKE.npz"
    clip_path = hr.resolve(args.root, args.clip or default_clip)
    out_path = hr.resolve(args.root, args.out or default_out)

    frames, fps = hr.load_clip(clip_path)
    payload = build_fake_payload(frames, fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    for key, value in payload.items():
        print(f"{key}: shape={np.shape(value)} dtype={np.asarray(value).dtype}")
    print(f"Wrote FAKE payload {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
