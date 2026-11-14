"""Shared fly's-eye helpers: team constants, eye rendering, hex coordinates,
orientation handling and gap-free hex scatter plots.

flyvis/torch are imported lazily so the pure helpers (orientation, paths,
marker size) work on machines without flyvis.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# --- Shared team contract constants (do not change without telling the team) ---
DT = 0.01
K = 5
EXTENT = 15
KERNEL_SIZE = 13
N_COLUMNS = 721
FRAME_SIZE = 391

ORIENTATION_KEYS = ("flip_x", "flip_y", "swap_xy")
IDENTITY_ORIENTATION = {"flip_x": False, "flip_y": False, "swap_xy": False}

RENDER_CHUNK_FRAMES = 64
HEX_PAD_FRACTION = 0.03  # blank margin around the hex lattice, fraction of its span
MARKER_OVERLAP = 1.04  # slight overlap so anti-aliasing leaves no hairline gaps
POINTS_PER_INCH = 72.0
# Marker path circumradius is 0.5*sqrt(s) points; hexagons tile when the
# circumradius is spacing/sqrt(3), hence s = (2*spacing/sqrt(3))**2.
TILING_AREA_FACTOR = 4.0 / 3.0
FLAT_TOP_MARKER = "H"
POINTY_TOP_MARKER = "h"


def resolve(root: str | Path, path: str | Path) -> Path:
    """Return `path` unchanged if absolute, otherwise relative to `root`."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(root).expanduser() / path


def load_clip(path: str | Path) -> tuple[np.ndarray, float]:
    """Load a contract clip npz -> (frames float32 (n, 391, 391), fps)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Clip not found: {path} (run prep_clips.py first)")
    with np.load(path) as data:
        missing = {"frames", "fps"} - set(data.files)
        if missing:
            raise ValueError(f"{path} is missing keys {sorted(missing)}")
        frames = data["frames"].astype(np.float32)
        fps = float(data["fps"])
    expected = (FRAME_SIZE, FRAME_SIZE)
    if frames.ndim != 3 or frames.shape[1:] != expected:
        raise ValueError(f"{path}: frames shape {frames.shape}, expected (n, 391, 391)")
    if fps <= 0:
        raise ValueError(f"{path}: invalid fps {fps}")
    return frames, fps


def render_eye(frames: np.ndarray) -> np.ndarray:
    """Render (n, 391, 391) frames in [0, 1] through BoxEye -> float32 (n, 721)."""
    import torch
    import flyvis  # sets the default torch device (CUDA when available)
    from flyvis.datasets.rendering import BoxEye

    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError(f"frames must be (n, H, W), got shape {frames.shape}")
    eye = BoxEye(extent=EXTENT, kernel_size=KERNEL_SIZE)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(frames), RENDER_CHUNK_FRAMES):
            chunk = torch.tensor(
                frames[start : start + RENDER_CHUNK_FRAMES], device=flyvis.device
            )
            # BoxEye takes (samples, frames, H, W) -> (samples, frames, 1, hexals)
            chunks.append(eye(chunk[None])[0, :, 0].cpu().numpy())
    rendered = np.concatenate(chunks).astype(np.float32)
    if rendered.shape[1] != N_COLUMNS:
        raise RuntimeError(f"Eye gave {rendered.shape[1]} columns, expected {N_COLUMNS}")
    return rendered


def hex_xy() -> tuple[np.ndarray, np.ndarray]:
    """Raw (uncorrected) plot coordinates of the 721 columns, in eye output order."""
    from flyvis.utils.hex_utils import get_hex_coords, hex_to_pixel

    u, v = get_hex_coords(EXTENT)
    x, y = hex_to_pixel(u, v)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def validate_orientation(orientation: dict) -> dict:
    missing = [key for key in ORIENTATION_KEYS if key not in orientation]
    if missing:
        raise ValueError(f"orientation is missing keys {missing}; got {orientation}")
    return {key: bool(orientation[key]) for key in ORIENTATION_KEYS}


def load_orientation(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Orientation file not found: {path} (run orientation_test.py first)"
        )
    with open(path) as handle:
        return validate_orientation(json.load(handle))


def save_orientation(path: str | Path, orientation: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(validate_orientation(orientation), handle)


def apply_orientation(x, y, orientation: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply the contract orientation to (x, y): swap first, then flips. Returns copies."""
    orientation = validate_orientation(orientation)
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if orientation["swap_xy"]:
        x, y = y, x
    if orientation["flip_x"]:
        x = -x
    if orientation["flip_y"]:
        y = -y
    return x, y


def oriented_hex_xy(orientation: dict) -> tuple[np.ndarray, np.ndarray]:
    return apply_orientation(*hex_xy(), orientation)


def hex_spacing(x: np.ndarray, y: np.ndarray) -> float:
    """Centre-to-centre distance between neighbouring hexagons (data units)."""
    distances = np.hypot(x[1:] - x[0], y[1:] - y[0])
    return float(distances.min())


def tiling_marker(x: np.ndarray, y: np.ndarray) -> str:
    """Hexagon marker whose flat sides face the lattice neighbours.

    flyvis' default layout has neighbours straight above/below (flat-top, "H");
    after swap_xy they sit left/right (pointy-top, "h").
    """
    spacing = hex_spacing(x, y)
    dx, dy = np.abs(x[1:] - x[0]), np.abs(y[1:] - y[0])
    is_neighbour = np.isclose(np.hypot(dx, dy), spacing)
    has_vertical_neighbour = bool(np.any(is_neighbour & np.isclose(dx, 0.0)))
    return FLAT_TOP_MARKER if has_vertical_neighbour else POINTY_TOP_MARKER


def tiling_marker_size(spacing_points: float, overlap: float = MARKER_OVERLAP) -> float:
    """Scatter `s` (points^2) at which hexagons `spacing_points` apart just touch."""
    return TILING_AREA_FACTOR * (spacing_points * overlap) ** 2


def _points_per_data_unit(ax) -> float:
    ax.apply_aspect()
    width_points = ax.get_window_extent().width / ax.figure.dpi * POINTS_PER_INCH
    x_low, x_high = ax.get_xlim()
    return width_points / abs(x_high - x_low)


def hex_scatter(ax, x, y, values, marker_size: float | None = None, marker: str | None = None):
    """Draw a gap-free hexagon scatter (gray, 0-1) on `ax`; returns the collection.

    The marker size is derived from the final axis size in points and the hex
    spacing unless `marker_size` is given. Create the figure layout (subplots,
    margins) BEFORE calling this, since the size depends on the axis extent.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    spacing = hex_spacing(x, y)
    pad_x = np.ptp(x) * HEX_PAD_FRACTION + spacing / 2
    pad_y = np.ptp(y) * HEX_PAD_FRACTION + spacing / 2
    ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
    ax.set_ylim(y.min() - pad_y, y.max() + pad_y)  # y points up; no invert_yaxis
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if marker_size is None:
        marker_size = tiling_marker_size(spacing * _points_per_data_unit(ax))
    return ax.scatter(
        x, y, c=values, cmap="gray", vmin=0, vmax=1, s=marker_size,
        marker=marker or tiling_marker(x, y), linewidths=0, edgecolors="none",
    )


def show_image(ax, frame):
    """Show a grayscale frame (0-1) with the same styling as the hex panels."""
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax.imshow(frame, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
