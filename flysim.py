"""Shared helpers for Plan C: load a pretrained flyvis network, run clips
through it with the same eye and time step the pipeline uses, and build the
synthetic test stimuli.

Uses hex_render (Plan B) for the constants and the eye, so the checks exercise
exactly the input path the demo uses.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import hex_render as hr

LOCAL_FLYVIS_DATA = Path(__file__).resolve().parent / ".flyvis_data"
if "FLYVIS_ROOT_DIR" not in os.environ and LOCAL_FLYVIS_DATA.is_dir():
    os.environ["FLYVIS_ROOT_DIR"] = str(LOCAL_FLYVIS_DATA)  # must be set before importing flyvis

MAIN_NETWORK = "flow/0000/000"
SECOND_NETWORK = "flow/0000/001"
FADE_IN_SECONDS = 1.0
STIMULUS_FPS = 60.0
GRAY = 0.5
T4_TYPES = ("T4a", "T4b", "T4c", "T4d")
T5_TYPES = ("T5a", "T5b", "T5c", "T5d")


class NetworkUnavailable(Exception):
    """The pretrained network could not be loaded."""


def load_network(name: str = MAIN_NETWORK):
    """Pretrained network, e.g. "flow/0000/000" (run `flyvis download-pretrained` first)."""
    import flyvis

    path = flyvis.results_dir / name
    if not path.exists():
        raise NetworkUnavailable(
            f"Pretrained network not found at {path}. Run: flyvis download-pretrained --skip_large_files "
            f"(with FLYVIS_ROOT_DIR set the same way as for this script)"
        )
    return flyvis.NetworkView(path).init_network()


def step_to_frame(n_frames: int, fps: float, dt: float = hr.DT) -> np.ndarray:
    """Frame index shown at each model step (same rule as the team contract)."""
    n_steps = int(n_frames / fps / dt)
    return np.minimum((np.arange(n_steps) * dt * fps).astype(int), n_frames - 1)


def run(frames: np.ndarray, fps: float, network, dt: float = hr.DT):
    """Simulate a clip -> (movie (n_steps, 721) numpy, LayerActivity).

    `activity.central["T4c"]` is the central column, shape (1, n_steps);
    `activity["T4c"]` is every column, shape (1, n_steps, n_columns).
    """
    import flyvis
    import torch
    from flyvis.utils.activity_utils import LayerActivity

    movie = hr.render_eye(frames)[step_to_frame(len(frames), fps, dt)]
    with torch.no_grad():
        stimulus = torch.tensor(movie, dtype=torch.float32, device=flyvis.device)[:, None]  # (n_steps, 1, 721)
        state = network.fade_in_state(FADE_IN_SECONDS, dt, stimulus[[0]])
        responses = network.simulate(stimulus[None], dt, initial_state=state).cpu()
    return movie, LayerActivity(responses, network.connectome, keepref=True)


def central(activity, cell_type: str) -> np.ndarray:
    """Central-column voltage trace of one cell type, shape (n_steps,)."""
    return np.asarray(activity.central[cell_type].cpu()).reshape(-1)


def all_columns(activity, cell_type: str) -> np.ndarray:
    """Voltage of one cell type in every column, shape (n_steps, n_columns)."""
    values = np.asarray(activity[cell_type].cpu())
    return values.reshape(-1, values.shape[-1])


def voltage_change(activity, cell_type: str) -> np.ndarray:
    """Voltage relative to the first step (the settled response to the first frame)."""
    values = all_columns(activity, cell_type)
    return values - values[0]


def mean_depolarization(activity, cell_types: tuple[str, ...]) -> float:
    """Mean voltage rise above the first step (falls count as 0) over time, columns and types.

    "Drives" means excites: a cell that is pushed below its resting level by a
    stimulus is being suppressed, so only rises are counted.
    """
    return float(np.mean([np.clip(voltage_change(activity, t), 0.0, None).mean() for t in cell_types]))


def mean_abs_change(activity, cell_types: tuple[str, ...]) -> float:
    """Mean |voltage change from the first step| (rises and falls) over time, columns and types."""
    return float(np.mean([np.abs(voltage_change(activity, t)).mean() for t in cell_types]))


# ------------------------------------------------------------------- stimuli

def gray_clip(seconds: float, fps: float = STIMULUS_FPS, value: float = GRAY) -> np.ndarray:
    return np.full((int(round(seconds * fps)), hr.FRAME_SIZE, hr.FRAME_SIZE), value, np.float32)


def edge_clip(on: bool = True, n: int = 60, size: int = hr.FRAME_SIZE) -> np.ndarray:
    """An edge sweeping left to right: bright on dark (on=True) or dark on bright."""
    clip = np.full((n, size, size), 0.0 if on else 1.0, np.float32)
    for i in range(n):
        clip[i, :, : int(size * i / n)] = 1.0 if on else 0.0
    return clip


def random_walk_clip(seconds: float = 2.0, fps: float = 30.0, seed: int = 0) -> np.ndarray:
    """A smooth random texture drifting on a random walk (stand-in until footage exists)."""
    import cv2

    rng = np.random.default_rng(seed)
    size, margin, blur = hr.FRAME_SIZE, 120, 12.0
    texture = cv2.GaussianBlur(rng.random((size + 2 * margin, size + 2 * margin)).astype(np.float32), (0, 0), blur)
    texture = (texture - texture.min()) / np.ptp(texture)
    steps = np.cumsum(rng.normal(0, 4.0, size=(int(round(seconds * fps)), 2)), axis=0)
    offsets = np.clip(np.round(steps).astype(int) + margin, 0, 2 * margin)
    return np.stack([texture[r : r + size, c : c + size] for r, c in offsets])


def disc_clip(radii: np.ndarray, centre_cols: np.ndarray, size: int = hr.FRAME_SIZE) -> np.ndarray:
    """Dark disc on a bright background; one frame per (radius, centre column)."""
    rows, cols = np.mgrid[0:size, 0:size]
    centre_row = size // 2
    frames = [
        np.where((rows - centre_row) ** 2 + (cols - centre_col) ** 2 <= radius**2, 0.0, 1.0)
        for radius, centre_col in zip(radii, centre_cols)
    ]
    return np.stack(frames).astype(np.float32)
