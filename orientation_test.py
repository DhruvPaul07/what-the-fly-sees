"""Find the (flip_x, flip_y, swap_xy) that makes hex plots match image orientation.

Renders an asymmetric test image (big square top-left, small square top-right)
through the eye, scores all 8 orientations against the image, and writes
results/orientation.json plus results/orientation_check.png.

    python orientation_test.py [--root .]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import hex_render as hr

# (row_start, row_stop, col_start, col_stop, brightness) in image pixels.
# Both sit inside the hexagonal field of view; different size + brightness and
# no mirror/transpose symmetry, so all 8 orientations are distinguishable.
BIG_SQUARE_TOP_LEFT = (75, 165, 55, 145, 1.0)
SMALL_SQUARE_TOP_RIGHT = (95, 125, 285, 315, 0.6)
MIN_SCORE_MARGIN = 0.05
CHECK_FIGSIZE = (9, 4.8)
CHECK_DPI = 120
TITLE_FONTSIZE = 11


def make_test_image() -> np.ndarray:
    image = np.zeros((hr.FRAME_SIZE, hr.FRAME_SIZE), dtype=np.float32)
    for r0, r1, c0, c1, value in (BIG_SQUARE_TOP_LEFT, SMALL_SQUARE_TOP_RIGHT):
        image[r0:r1, c0:c1] = value
    return image


def all_orientations() -> list[dict]:
    return [
        dict(zip(hr.ORIENTATION_KEYS, flags))
        for flags in itertools.product((False, True), repeat=len(hr.ORIENTATION_KEYS))
    ]


def expected_values(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Image brightness where each plotted point would sit (plot y up = image row up)."""
    height, width = image.shape
    cols = (x - x.min()) / np.ptp(x) * (width - 1)
    rows = (y.max() - y) / np.ptp(y) * (height - 1)
    return image[np.round(rows).astype(int), np.round(cols).astype(int)]


def score_orientation(image, rendered, x, y, orientation: dict) -> float:
    """Correlation between rendered hex values and the image seen at the plotted spots."""
    ox, oy = hr.apply_orientation(x, y, orientation)
    return float(np.corrcoef(expected_values(image, ox, oy), rendered)[0, 1])


def find_orientation(image, rendered, x, y) -> tuple[dict, list[tuple[float, dict]]]:
    scored = sorted(
        ((score_orientation(image, rendered, x, y, o), o) for o in all_orientations()),
        key=lambda item: item[0],
        reverse=True,
    )
    (best_score, best), (second_score, _) = scored[0], scored[1]
    if best_score - second_score < MIN_SCORE_MARGIN:
        raise RuntimeError(
            f"Orientation is ambiguous (best {best_score:.3f} vs next {second_score:.3f}); "
            "inspect the test image / eye output by hand."
        )
    return best, scored


def save_check_figure(path: Path, image, rendered, x, y, orientation: dict) -> None:
    fig, (ax_image, ax_hex) = plt.subplots(1, 2, figsize=CHECK_FIGSIZE, facecolor="black")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.04, wspace=0.05)
    hr.show_image(ax_image, image)
    ax_image.set_title("Test image (big square top-left)", color="white", fontsize=TITLE_FONTSIZE)
    hr.hex_scatter(ax_hex, *hr.apply_orientation(x, y, orientation), rendered)
    ax_hex.set_title("Corrected hex plot (must match the left)", color="white", fontsize=TITLE_FONTSIZE)
    fig.suptitle(str(orientation), color="white", fontsize=TITLE_FONTSIZE)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=CHECK_DPI, facecolor="black")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("results/orientation.json"))
    parser.add_argument("--check", type=Path, default=Path("results/orientation_check.png"))
    args = parser.parse_args(argv)

    image = make_test_image()
    rendered = hr.render_eye(image[None])[0]
    x, y = hr.hex_xy()
    orientation, scored = find_orientation(image, rendered, x, y)
    for score, candidate in scored:
        print(f"score={score:+.3f}  {candidate}")

    out_path, check_path = hr.resolve(args.root, args.out), hr.resolve(args.root, args.check)
    hr.save_orientation(out_path, orientation)
    save_check_figure(check_path, image, rendered, x, y, orientation)
    print(f"ORIENTATION: {orientation}")
    print(f"Wrote {out_path} and {check_path} -- look at the PNG, then tell the team.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
