"""Plan C Part 3 (bonus): a dark disc looming toward the eye vs the same disc
sliding sideways. Plots summed T4/T5 activity over time for each.

    python looming_bonus.py --root /content/drive/MyDrive/fly_sees

Motion-pathway responses only; escape behavior is not modeled (no whole-brain
coupling). Writes results/sanity/looming_vs_sliding.png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import flysim as fs
import hex_render as hr
from sanity_checks import save_figure, time_axis

SECONDS = 2.0
START_RADIUS = 10.0
END_RADIUS = 150.0
SLIDE_RADIUS = 40.0  # the sliding disc keeps one size
SLIDE_MARGIN = 60.0
LABEL = "Motion-pathway responses only; escape behavior not modeled."
FIGURE_NAME = "looming_vs_sliding"


def looming_clip(seconds: float = SECONDS, fps: float = fs.STIMULUS_FPS) -> np.ndarray:
    n = int(round(seconds * fps))
    centre = np.full(n, hr.FRAME_SIZE // 2, dtype=float)
    return fs.disc_clip(np.linspace(START_RADIUS, END_RADIUS, n), centre)


def sliding_clip(seconds: float = SECONDS, fps: float = fs.STIMULUS_FPS) -> np.ndarray:
    n = int(round(seconds * fps))
    centres = np.linspace(SLIDE_MARGIN, hr.FRAME_SIZE - SLIDE_MARGIN, n)
    return fs.disc_clip(np.full(n, SLIDE_RADIUS), centres)


def summed_motion_activity(activity) -> dict[str, np.ndarray]:
    """Depolarization summed over columns and subtypes, per time step, for T4 and T5."""
    return {
        name: sum(np.clip(fs.voltage_change(activity, t), 0.0, None).sum(axis=1) for t in types)
        for name, types in (("T4", fs.T4_TYPES), ("T5", fs.T5_TYPES))
    }


def make_figure(responses: dict[str, dict[str, np.ndarray]]):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, (title, traces) in zip(axes, responses.items()):
        for name, trace in traces.items():
            ax.plot(time_axis(len(trace)), trace, label=f"{name}a-d")
        ax.set(xlabel="time (s)", title=title)
        ax.legend(frameon=False)
    axes[0].set_ylabel("summed depolarization (a.u.)")
    fig.suptitle(LABEL, fontsize=10)
    return fig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."), help="fly_sees folder (default: current dir)")
    args = parser.parse_args(argv)
    try:
        network = fs.load_network(fs.MAIN_NETWORK)
    except fs.NetworkUnavailable as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    responses = {}
    for title, clip in (("Dark disc looming", looming_clip()), ("Same disc sliding sideways", sliding_clip())):
        _, activity = fs.run(clip, fs.STIMULUS_FPS, network)
        responses[title] = summed_motion_activity(activity)
    path = save_figure(make_figure(responses), args.root, FIGURE_NAME)
    print(f"Wrote {path}")
    print(LABEL)
    print("Note: the looming disc also grows in area, so its larger summed response partly reflects stimulus size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
