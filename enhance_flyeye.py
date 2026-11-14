"""Plan B2: AI-upscaled fly's-eye panel (Real-ESRGAN) with an honest metric.

    python enhance_flyeye.py --root /content/drive/MyDrive/fly_sees

Reads   clips/test_main.npz, results/orientation.json
Writes  results/fly_lowres.npy            (n, 128, 128) rasterized 721-sample frames
        results/ai_enhance_metrics.json   AI vs bicubic similarity to the real scene
        demo/ai_frames.npy                (n, 512, 512) float16 AI-upscaled frames
        demo/ai_enhanced.mp4 / .gif / ai_enhanced_still.png

The AI panel shows detail invented by a pretrained super-resolution model. It is
never a reconstruction, a decoding, or "what the fly sees".
"""
from __future__ import annotations

import argparse
import http.client
import json
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.interpolate import griddata
from scipy.spatial import Delaunay, cKDTree
from skimage.metrics import structural_similarity

import hex_render as hr

LOWRES_SIZE = 128
UPSCALE = 4
HIGHRES_SIZE = LOWRES_SIZE * UPSCALE
VIDEO_FPS = 25
SLOW_VIDEO_FPS = 15
MAX_SECONDS = 3.0
SLOW_SECONDS_PER_FRAME = 2.0
GPU_BATCH = 16
# A pixel belongs to the eye if it is within this many hex spacings of a sample.
MASK_RADIUS_SPACINGS = 0.5
# Metrics ignore a rim this wide (in hex spacings) so the black border does not count.
METRIC_ERODE_SPACINGS = 1.0
SSIM_WINDOW = 7

# Self-check: a bright square in the top-left quadrant of a black frame.
CHECK_SQUARE = (slice(100, 170), slice(60, 130))  # rows, cols in the 391x391 frame
CHECK_MIN_CONTRAST = 0.3

WEIGHTS_NAME = "realesr-general-x4v3.pth"
WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
DOWNLOAD_TIMEOUT_SECONDS = 120

PANEL_TITLES = ("Original", "Fly's eye (721 samples)", "AI-upscaled (Real-ESRGAN)")
AI_LABEL = "AI-upscaled (Real-ESRGAN): detail is invented by the AI, not present in the fly's 721 samples."
CAPTION = "AI upscaling adds invented detail. It is not information the fly's eye or neurons carry."
FIGSIZE_INCHES = (12.0, 5.0)
DEFAULT_DPI = 100
GIF_DPI_FRACTION = 0.5
FFMPEG_ARGS = ["-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]


class EnhanceError(Exception):
    """The AI-enhanced panel could not be produced."""


# ---------------------------------------------------------------- step 1: frames

def subsample_indices(n_frames: int, fps: float, out_fps: float, max_seconds: float = MAX_SECONDS) -> np.ndarray:
    """Clip-frame index shown at each output frame (`out_fps`, at most `max_seconds`)."""
    seconds = min(n_frames / fps, max_seconds)
    n_out = max(1, int(np.floor(seconds * out_fps + 1e-9)))
    indices = (np.arange(n_out) / out_fps * fps).astype(int)
    return np.clip(indices, 0, n_frames - 1)


# ----------------------------------------------------------- step 2: rasterizing

def sample_positions(orientation: dict, size: int) -> tuple[np.ndarray, np.ndarray]:
    """(row, col) of the 721 samples in a size x size image, row 0 at the top.

    BoxEye's outermost receptors sit on the frame border on both axes, so each
    oriented axis is stretched to the full image; rows come from -y (y points up).
    """
    x, y = hr.oriented_hex_xy(orientation)
    cols = (x - x.min()) / np.ptp(x) * (size - 1)
    rows = (y.max() - y) / np.ptp(y) * (size - 1)
    return rows, cols


def sample_spacing(rows: np.ndarray, cols: np.ndarray) -> float:
    distances, _ = cKDTree(np.column_stack([rows, cols])).query(np.column_stack([rows, cols]), k=2)
    return float(np.median(distances[:, 1]))


def eye_mask(rows: np.ndarray, cols: np.ndarray, size: int, radius_spacings: float = MASK_RADIUS_SPACINGS) -> np.ndarray:
    """True for pixels inside the hexagon covered by the samples: their convex
    hull, grown by `radius_spacings` hex spacings so the outer cells are whole."""
    grid_rows, grid_cols = np.mgrid[0:size, 0:size]
    pixels = np.column_stack([grid_rows.ravel(), grid_cols.ravel()])
    hull = Delaunay(np.column_stack([rows, cols])).find_simplex(pixels).reshape(size, size) >= 0
    distance_to_hull = cv2.distanceTransform((~hull).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return distance_to_hull <= radius_spacings * sample_spacing(rows, cols)


def rasterize(values: np.ndarray, rows: np.ndarray, cols: np.ndarray, size: int, method: str = "linear") -> np.ndarray:
    """(n, 721) samples -> (n, size, size) float32: `method` inside the hull,
    nearest outside it, black outside the hexagon."""
    points = np.column_stack([rows, cols])
    grid = tuple(np.mgrid[0:size, 0:size])
    samples = np.asarray(values, dtype=np.float32).T  # (721, n): all frames at once
    nearest = griddata(points, samples, grid, method="nearest")
    filled = nearest if method == "nearest" else griddata(points, samples, grid, method=method)
    filled = np.where(np.isnan(filled), nearest, filled)
    filled = filled * eye_mask(rows, cols, size)[..., None]
    return np.clip(np.moveaxis(filled, -1, 0), 0.0, 1.0).astype(np.float32)


def self_check_orientation(orientation: dict) -> None:
    """A square drawn top-left must come out top-left after eye + rasterize."""
    frame = np.zeros((1, hr.FRAME_SIZE, hr.FRAME_SIZE), dtype=np.float32)
    frame[0][CHECK_SQUARE] = 1.0
    rows, cols = sample_positions(orientation, LOWRES_SIZE)
    image = rasterize(hr.render_eye(frame), rows, cols, LOWRES_SIZE)[0]
    scale = LOWRES_SIZE / hr.FRAME_SIZE
    row_slice, col_slice = (slice(int(s.start * scale) + 2, int(s.stop * scale) - 2) for s in CHECK_SQUARE)
    inside = float(image[row_slice, col_slice].mean())
    variants = {
        "left-right mirrored": float(image[:, ::-1][row_slice, col_slice].mean()),
        "upside down": float(image[::-1][row_slice, col_slice].mean()),
        "transposed": float(image.T[row_slice, col_slice].mean()),
    }
    worst_name, worst = max(variants.items(), key=lambda item: item[1])
    if inside - worst < CHECK_MIN_CONTRAST:
        raise EnhanceError(
            f"Orientation self-check FAILED: the top-left square is not top-left in the raster "
            f"(expected spot {inside:.2f}, {worst_name} {worst:.2f}). Check results/orientation.json."
        )
    print(f"self-check OK: top-left square lands top-left ({inside:.2f} vs {worst:.2f} {worst_name})")


# ------------------------------------------------------------ step 3: upscaling

def ensure_weights(path: Path) -> Path:
    if path.is_file():
        return path
    print(f"Downloading Real-ESRGAN weights to {path} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(WEIGHTS_URL, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            data = response.read()
    except (OSError, http.client.HTTPException) as error:
        raise EnhanceError(f"Could not download weights ({error}). Run: curl -L -o {path} {WEIGHTS_URL}") from error
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(data)
    partial.replace(path)  # atomic: an interrupted run never leaves a truncated .pth
    return path


def load_upscaler(weights: Path):
    """-> (callable (b, 3, H, W) tensor -> (b, 3, 4H, 4W) tensor, torch device)."""
    import torch
    from spandrel import ModelLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    descriptor = ModelLoader(device=device).load_from_file(str(weights))
    if getattr(descriptor, "scale", UPSCALE) != UPSCALE:
        raise EnhanceError(f"{weights.name} is a x{descriptor.scale} model, expected x{UPSCALE}")
    descriptor.eval().to(device)
    upscale = descriptor if callable(descriptor) else descriptor.model
    return upscale, device


def ai_upscale(lowres: np.ndarray, weights: Path) -> tuple[np.ndarray, float]:
    """(n, 128, 128) gray -> ((n, 512, 512) gray float32, seconds per frame)."""
    import torch

    upscale, device = load_upscaler(weights)
    batch = GPU_BATCH if device.type == "cuda" else 1
    outputs = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(lowres), batch):
            gray = torch.tensor(lowres[start : start + batch], dtype=torch.float32, device=device)
            rgb = gray[:, None].repeat(1, 3, 1, 1)
            outputs.append(upscale(rgb).mean(dim=1).clamp(0, 1).cpu().numpy())
    seconds_per_frame = (time.perf_counter() - started) / len(lowres)
    upscaled = np.concatenate(outputs).astype(np.float32)
    expected = (len(lowres), HIGHRES_SIZE, HIGHRES_SIZE)
    if upscaled.shape != expected:
        raise EnhanceError(f"AI model returned shape {upscaled.shape}, expected {expected}")
    print(f"AI upscaling on {device.type}: {seconds_per_frame:.2f} s/frame")
    return upscaled, seconds_per_frame


def resize_stack(frames: np.ndarray, size: int, interpolation: int) -> np.ndarray:
    resized = [cv2.resize(frame, (size, size), interpolation=interpolation) for frame in frames]
    return np.clip(np.stack(resized), 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------- step 4: metric

def similarity(candidates: np.ndarray, references: np.ndarray, mask: np.ndarray) -> dict:
    """Mean per-frame Pearson and SSIM against the real scene, inside `mask` only."""
    pearsons, ssims = [], []
    for candidate, reference in zip(candidates, references):
        a, b = candidate[mask], reference[mask]
        # Pearson is undefined for a flat frame: skip it rather than count it as 0.
        pearsons.append(float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan)
        _, ssim_map = structural_similarity(reference, candidate, data_range=1.0, win_size=SSIM_WINDOW, full=True)
        ssims.append(float(ssim_map[mask].mean()))
    pearson = float(np.nanmean(pearsons)) if not np.all(np.isnan(pearsons)) else 0.0
    return {"pearson": pearson, "ssim": float(np.mean(ssims))}


def verdict(ai: dict, bicubic: dict) -> str:
    delta_ssim, delta_pearson = ai["ssim"] - bicubic["ssim"], ai["pearson"] - bicubic["pearson"]
    if delta_ssim < -0.005:
        finding = "LESS similar to the real scene than plain bicubic (the AI is inventing texture)"
    elif delta_ssim <= 0.02:
        finding = "sharper-looking, little added similarity to the real scene"
    else:
        finding = "somewhat more similar to the real scene than bicubic, but the added detail is still invented"
    return (
        f"AI upscaling: SSIM {ai['ssim']:.3f} vs bicubic {bicubic['ssim']:.3f} ({delta_ssim:+.3f}), "
        f"Pearson {ai['pearson']:.3f} vs {bicubic['pearson']:.3f} ({delta_pearson:+.3f}) -> {finding}"
    )


def write_metrics(path: Path, ai: dict, bicubic: dict, n_frames: int, seconds_per_frame: float) -> str:
    line = verdict(ai, bicubic)
    metrics = {
        "ai_realesrgan": ai,
        "bicubic": bicubic,
        "difference_ai_minus_bicubic": {key: ai[key] - bicubic[key] for key in ai},
        "n_frames": n_frames,
        "reference": "original frame resized to 512x512, pixels inside the eye hexagon (rim excluded)",
        "model": WEIGHTS_NAME,
        "ai_seconds_per_frame": seconds_per_frame,
        "verdict": line,
        "label": AI_LABEL,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    return line


# ---------------------------------------------------------------- step 5: video

def build_figure(panels: tuple[np.ndarray, ...], first: int = 0):
    fig, axes = plt.subplots(1, len(PANEL_TITLES), figsize=FIGSIZE_INCHES, facecolor="black")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.14, wspace=0.03)
    images = [hr.show_image(ax, panel[first]) for ax, panel in zip(axes, panels)]
    for ax, title in zip(axes, PANEL_TITLES):
        ax.set_title(title, color="white", fontsize=13)
    fig.suptitle("AI upscaling test: the fly's eye vs Real-ESRGAN", color="white", fontsize=15, y=0.965)
    fig.text(0.5, 0.075, CAPTION, color="white", ha="center", va="center", fontsize=12)
    fig.text(0.5, 0.03, AI_LABEL, color="0.7", ha="center", va="center", fontsize=9)

    def update(i: int):
        for image, panel in zip(images, panels):
            image.set_data(panel[i])
        return images

    return fig, update


def save_video(panels: tuple[np.ndarray, ...], out_path: Path, fps: int, dpi: int) -> list[Path]:
    """mp4 (if ffmpeg works) + half-resolution GIF, rendered locally then copied."""
    fig, update = build_figure(panels)
    anim = animation.FuncAnimation(fig, update, frames=len(panels[0]), interval=1000 / fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gif_path = out_path.with_suffix(".gif")
    black = {"facecolor": "black"}
    saved = []
    with tempfile.TemporaryDirectory() as tmp:
        local_mp4, local_gif = Path(tmp) / out_path.name, Path(tmp) / gif_path.name
        if animation.writers.is_available("ffmpeg"):
            try:
                writer = animation.FFMpegWriter(fps=fps, extra_args=FFMPEG_ARGS)
                anim.save(str(local_mp4), writer=writer, dpi=dpi, savefig_kwargs=black)
            except Exception as error:  # ffmpeg failures surface as many exception types
                print(f"WARNING: mp4 export failed ({error}); saving GIF only.", file=sys.stderr)
        else:
            print("WARNING: ffmpeg not available; saving GIF only.", file=sys.stderr)
        out_path.unlink(missing_ok=True)  # never leave an mp4 from an earlier run beside a new GIF
        if local_mp4.is_file():
            shutil.copyfile(local_mp4, out_path)
            saved.append(out_path)
        gif_dpi = max(1, int(dpi * GIF_DPI_FRACTION))
        anim.save(str(local_gif), writer=animation.PillowWriter(fps=fps), dpi=gif_dpi, savefig_kwargs=black)
        shutil.copyfile(local_gif, gif_path)
        saved.append(gif_path)
    plt.close(fig)
    return saved


def save_still(panels: tuple[np.ndarray, ...], out_path: Path, dpi: int) -> Path:
    fig, _ = build_figure(panels, first=len(panels[0]) // 2)
    fig.savefig(out_path, dpi=dpi, facecolor="black")
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------------ main

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."), help="fly_sees folder (default: current dir)")
    parser.add_argument("--clip", type=Path, default=Path("clips/test_main.npz"))
    parser.add_argument("--orientation", type=Path, default=Path("results/orientation.json"))
    parser.add_argument("--weights", type=Path, default=Path(__file__).resolve().parent / "weights" / WEIGHTS_NAME)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    root = args.root
    frames, fps = hr.load_clip(hr.resolve(root, args.clip))
    orientation = hr.load_orientation(hr.resolve(root, args.orientation))
    weights = ensure_weights(args.weights.expanduser())
    self_check_orientation(orientation)

    def compute(out_fps: int):
        clip = frames[subsample_indices(len(frames), fps, out_fps)]
        samples = hr.render_eye(clip)
        rows, cols = sample_positions(orientation, LOWRES_SIZE)
        lowres = rasterize(samples, rows, cols, LOWRES_SIZE)
        ai, seconds_per_frame = ai_upscale(lowres, weights)
        return clip, samples, lowres, ai, seconds_per_frame

    out_fps = VIDEO_FPS
    clip, samples, lowres, ai, seconds_per_frame = compute(out_fps)
    if seconds_per_frame > SLOW_SECONDS_PER_FRAME:
        out_fps = SLOW_VIDEO_FPS
        print(f"Slower than {SLOW_SECONDS_PER_FRAME:.0f} s/frame: dropping to {out_fps} fps")
        keep = np.unique((np.arange(int(len(clip) * out_fps / VIDEO_FPS)) * VIDEO_FPS / out_fps).astype(int))
        clip, samples, lowres, ai = clip[keep], samples[keep], lowres[keep], ai[keep]

    rows_hi, cols_hi = sample_positions(orientation, HIGHRES_SIZE)
    mask_hi = eye_mask(rows_hi, cols_hi, HIGHRES_SIZE)
    ai = ai * mask_hi
    bicubic = resize_stack(lowres, HIGHRES_SIZE, cv2.INTER_CUBIC) * mask_hi
    original = resize_stack(clip, HIGHRES_SIZE, cv2.INTER_AREA)
    cells = rasterize(samples, rows_hi, cols_hi, HIGHRES_SIZE, method="nearest")  # the 721 raw samples

    spacing_hi = sample_spacing(rows_hi, cols_hi)
    metric_mask = _eroded(mask_hi, spacing_hi * METRIC_ERODE_SPACINGS)
    if not metric_mask.any():
        raise EnhanceError("Metric mask is empty; cannot score the upscaled frames")
    # The video is the main deliverable: write it before scoring.
    results_dir, demo_dir = hr.resolve(root, "results"), hr.resolve(root, "demo")
    results_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / "fly_lowres.npy", lowres)
    np.save(demo_dir / "ai_frames.npy", ai.astype(np.float16))
    panels = (original, cells, ai)
    saved = save_video(panels, demo_dir / "ai_enhanced.mp4", out_fps, args.dpi)
    saved.append(save_still(panels, demo_dir / "ai_enhanced_still.png", args.dpi))

    line = write_metrics(
        results_dir / "ai_enhance_metrics.json",
        similarity(ai, original, metric_mask), similarity(bicubic, original, metric_mask),
        len(clip), seconds_per_frame,
    )

    print(f"{len(clip)} frames at {out_fps} fps ({len(clip) / out_fps:.2f} s)")
    for path in [results_dir / "fly_lowres.npy", results_dir / "ai_enhance_metrics.json", demo_dir / "ai_frames.npy", *saved]:
        print(f"Wrote {path}")
    print(f"VERDICT: {line}")


def _eroded(mask: np.ndarray, pixels: float) -> np.ndarray:
    """Pixels of `mask` further than `pixels` from its edge (image border counts as edge)."""
    padded = np.pad(mask.astype(np.uint8), 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return distance[1:-1, 1:-1] > pixels


def main(argv=None) -> int:
    try:
        run(parse_args(argv))
    except (FileNotFoundError, ValueError, EnhanceError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
