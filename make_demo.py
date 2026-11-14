"""Render the 4-panel demo video (mp4 + GIF) from a contract demo payload.

    python make_demo.py --payload results/demo_payload.npz --clip clips/test_main.npz \
        --orientation results/orientation.json --out demo/what_the_fly_sees.mp4
    python make_demo.py --still      # demo_payload_still.npz + test_still -> demo/still_scene.mp4

Relative paths are resolved against --root (default: current dir).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import hex_render as hr

PANEL_TITLES = (
    "Original",
    "Fly's-eye input (721 columns)",
    "Decoded from photoreceptors (R1–R6)",
    "Decoded from motion detectors (T4/T5)",
)
CAPTION = (
    "Reconstruction decoded from simulated neurons (flyvis connectome model), "
    "not what the fly experiences."
)
MAIN_TITLE = "What the Fly Sees"
STILL_TITLE = "Still scene (no motion): the motion-detector panel should stay flat"
PLACEHOLDER_NOTE = "decoder trained on synthetic clips (placeholder)"
FAKE_NOTE = "FAKE payload, layout test only (not model output)"

REQUIRED_KEYS = ("input", "recon_motion", "recon_photo", "step_to_frame", "K", "dt")
STEP_STRIDE = 4  # every 4th 10 ms step
VIDEO_FPS = 25
DEFAULT_DPI = 100
GIF_DPI_FRACTION = 0.5
FIGSIZE_INCHES = (16.0, 5.2)
TITLE_FONTSIZE = 11
CAPTION_FONTSIZE = 12
SUPTITLE_FONTSIZE = 15
# h264 needs even pixel dimensions and yuv420p for players/slides to accept it.
FFMPEG_ARGS = ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p"]


class PayloadError(ValueError):
    """The demo payload does not follow the team contract."""


def _flag(payload: dict, key: str) -> bool:
    return key in payload and bool(np.asarray(payload[key]).item())


def validate_payload(payload: dict, n_clip_frames: int) -> dict:
    """Check keys/shapes against the contract; return a normalized copy."""
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise PayloadError(f"payload is missing keys {missing}; has {sorted(payload)}")
    try:
        k = int(np.asarray(payload["K"]).item())
        dt = float(np.asarray(payload["dt"]).item())
    except (TypeError, ValueError) as error:
        raise PayloadError(f"K and dt must be scalars: {error}") from error
    if k < 1:
        raise PayloadError(f"K must be >= 1, got {k}")

    eye_input = np.asarray(payload["input"])
    if eye_input.ndim != 2 or eye_input.shape[1] != hr.N_COLUMNS:
        raise PayloadError(f"input must be (T, {hr.N_COLUMNS}), got {eye_input.shape}")
    n_steps = eye_input.shape[0]
    expected = (n_steps - k + 1, hr.N_COLUMNS)
    if expected[0] < 1:
        raise PayloadError(f"T={n_steps} is shorter than K={k}")
    for key in ("recon_motion", "recon_photo"):
        shape = np.asarray(payload[key]).shape
        if shape != expected:
            raise PayloadError(f"{key} must be (T-K+1, 721) = {expected}, got {shape}")
        if not np.all(np.isfinite(payload[key])):
            raise PayloadError(f"{key} contains NaN/inf")

    step_to_frame = np.asarray(payload["step_to_frame"])
    if step_to_frame.shape != (n_steps,):
        raise PayloadError(f"step_to_frame must be (T,) = ({n_steps},), got {step_to_frame.shape}")
    if not np.issubdtype(step_to_frame.dtype, np.integer):
        raise PayloadError(f"step_to_frame must be integer, got dtype {step_to_frame.dtype}")
    low, high = int(step_to_frame.min()), int(step_to_frame.max())
    if low < 0 or high >= n_clip_frames:
        raise PayloadError(
            f"step_to_frame spans {low}..{high} but the clip has {n_clip_frames} frames "
            "(wrong --clip for this payload?)"
        )
    return {
        "input": np.clip(eye_input, 0, 1).astype(np.float32),
        "recon_motion": np.clip(payload["recon_motion"], 0, 1).astype(np.float32),
        "recon_photo": np.clip(payload["recon_photo"], 0, 1).astype(np.float32),
        "step_to_frame": step_to_frame.astype(np.int64),
        "K": k,
        "dt": dt,
        "is_placeholder": _flag(payload, "is_placeholder"),
        "is_fake": _flag(payload, "is_fake"),
    }


def load_payload(path: Path, n_clip_frames: int) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Payload not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        raw = {key: data[key] for key in data.files}
    return validate_payload(raw, n_clip_frames)


def decoder_step(i: int, k: int) -> int:
    """Decoder output index i belongs to model step t = i + K - 1."""
    return i + k - 1


def animation_indices(n_recon: int, stride: int = STEP_STRIDE) -> list[int]:
    return list(range(0, n_recon, stride))


def figure_title(payload: dict, still: bool) -> str:
    notes = [note for flag, note in (("is_placeholder", PLACEHOLDER_NOTE), ("is_fake", FAKE_NOTE)) if payload[flag]]
    title = STILL_TITLE if still else MAIN_TITLE
    return f"{title}  [{'; '.join(notes)}]" if notes else title


def build_figure(frames: np.ndarray, payload: dict, orientation: dict, title: str):
    """Draw all panels once; returns (fig, update) where update(i) only swaps data."""
    k = payload["K"]
    fig, axes = plt.subplots(1, len(PANEL_TITLES), figsize=FIGSIZE_INCHES, facecolor="black")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.12, wspace=0.04)
    x, y = hr.oriented_hex_xy(orientation)
    first_step = decoder_step(0, k)

    image = hr.show_image(axes[0], frames[payload["step_to_frame"][first_step]])
    eye_dots = hr.hex_scatter(axes[1], x, y, payload["input"][first_step])
    photo_dots = hr.hex_scatter(axes[2], x, y, payload["recon_photo"][0])
    motion_dots = hr.hex_scatter(axes[3], x, y, payload["recon_motion"][0])
    for ax, panel_title in zip(axes, PANEL_TITLES):
        ax.set_title(panel_title, color="white", fontsize=TITLE_FONTSIZE)
    fig.suptitle(title, color="white", fontsize=SUPTITLE_FONTSIZE, y=0.965)
    fig.text(0.5, 0.05, CAPTION, color="white", ha="center", va="center", fontsize=CAPTION_FONTSIZE)

    def update(i: int):
        step = decoder_step(i, k)
        image.set_data(frames[payload["step_to_frame"][step]])
        eye_dots.set_array(payload["input"][step])
        photo_dots.set_array(payload["recon_photo"][i])
        motion_dots.set_array(payload["recon_motion"][i])
        return image, eye_dots, photo_dots, motion_dots

    return fig, update


def save_outputs(anim: animation.FuncAnimation, out_path: Path, dpi: int) -> list[Path]:
    """Save mp4 (if ffmpeg works) and a ~half-resolution GIF next to it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gif_path = out_path.with_suffix(".gif")
    saved = []
    # Render to local disk, then copy: ffmpeg's mp4 muxer seeks back to patch the
    # header, which is unreliable on network mounts such as Colab's Google Drive.
    with tempfile.TemporaryDirectory() as tmp:
        local_mp4, local_gif = Path(tmp) / out_path.name, Path(tmp) / gif_path.name
        if animation.writers.is_available("ffmpeg"):
            try:
                writer = animation.FFMpegWriter(fps=VIDEO_FPS, extra_args=FFMPEG_ARGS)
                anim.save(str(local_mp4), writer=writer, dpi=dpi, savefig_kwargs={"facecolor": "black"})
            except Exception as error:  # ffmpeg failures surface as many exception types
                print(f"WARNING: mp4 export failed ({error}); saving GIF only.", file=sys.stderr)
        else:
            print("WARNING: ffmpeg not available; saving GIF only.", file=sys.stderr)
        out_path.unlink(missing_ok=True)  # never leave an mp4 from an earlier run beside a new GIF
        if local_mp4.is_file():
            shutil.copyfile(local_mp4, out_path)
            saved.append(out_path)
        anim.save(
            str(local_gif), writer=animation.PillowWriter(fps=VIDEO_FPS),
            dpi=max(1, int(dpi * GIF_DPI_FRACTION)), savefig_kwargs={"facecolor": "black"},
        )
        shutil.copyfile(local_gif, gif_path)
        saved.append(gif_path)
    return saved


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--payload", type=Path, default=None)
    parser.add_argument("--clip", type=Path, default=None)
    parser.add_argument("--orientation", type=Path, default=Path("results/orientation.json"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--still", action="store_true", help="still-scene defaults and title")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="default 100; use 80 if rendering is slow")
    args = parser.parse_args(argv)
    still = args.still
    args.payload = args.payload or Path("results/demo_payload_still.npz" if still else "results/demo_payload.npz")
    args.clip = args.clip or Path("clips/test_still.npz" if still else "clips/test_main.npz")
    args.out = args.out or Path("demo/still_scene.mp4" if still else "demo/what_the_fly_sees.mp4")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        frames, _ = hr.load_clip(hr.resolve(args.root, args.clip))
        payload = load_payload(hr.resolve(args.root, args.payload), len(frames))
        orientation = hr.load_orientation(hr.resolve(args.root, args.orientation))
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    title = figure_title(payload, args.still)
    fig, update = build_figure(frames, payload, orientation, title)
    indices = animation_indices(len(payload["recon_motion"]))
    anim = animation.FuncAnimation(fig, update, frames=indices, interval=1000 / VIDEO_FPS, blit=False)
    saved = save_outputs(anim, hr.resolve(args.root, args.out), args.dpi)
    plt.close(fig)
    print(f"title: {title}")
    print(f"{len(indices)} video frames at {VIDEO_FPS} fps ({len(indices) / VIDEO_FPS:.2f} s)")
    for path in saved:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
