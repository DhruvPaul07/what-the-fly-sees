"""'How coarse is it' hero image: one test_main frame vs its 721-hexagon version.

    python coarseness.py [--root .] [--frame N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hex_render as hr

CAPTION = "721 samples vs 152,881 pixels (~212× fewer)"
FIGSIZE = (10, 5.6)
DPI = 150


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--clip", type=Path, default=Path("clips/test_main.npz"))
    parser.add_argument("--orientation", type=Path, default=Path("results/orientation.json"))
    parser.add_argument("--out", type=Path, default=Path("demo/coarseness.png"))
    parser.add_argument("--frame", type=int, default=None, help="frame index (default: middle)")
    args = parser.parse_args(argv)

    frames, _ = hr.load_clip(hr.resolve(args.root, args.clip))
    index = len(frames) // 2 if args.frame is None else args.frame
    if not 0 <= index < len(frames):
        print(f"ERROR: --frame {index} outside 0..{len(frames) - 1}", file=sys.stderr)
        return 1
    orientation = hr.load_orientation(hr.resolve(args.root, args.orientation))
    frame = frames[index]
    rendered = hr.render_eye(frame[None])[0]

    fig, (ax_image, ax_hex) = plt.subplots(1, 2, figsize=FIGSIZE, facecolor="black")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.12, wspace=0.05)
    hr.show_image(ax_image, frame)
    ax_image.set_title("Camera: 391 × 391 pixels", color="white", fontsize=14)
    hr.hex_scatter(ax_hex, *hr.oriented_hex_xy(orientation), rendered)
    ax_hex.set_title("Fly's eye: 721 columns", color="white", fontsize=14)
    fig.text(0.5, 0.04, CAPTION, color="white", ha="center", va="center", fontsize=15)

    out_path = hr.resolve(args.root, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor="black")
    plt.close(fig)
    print(f"Wrote {out_path} (frame {index})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
