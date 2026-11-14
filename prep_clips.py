"""Preprocess phone videos into contract clips (laptop-side; needs only opencv + numpy).

    python prep_clips.py --test-main main.mov --test-still still.mov \
        --train a.mov b.mov c.mov --out clips/

Writes clips/test_main.npz, clips/test_still.npz (3 s each) and
clips/train_00.npz, train_01.npz, ... (consecutive 2 s pieces), each with
`frames` float16 (n_frames, 391, 391) in 0-1 and `fps` float.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

FRAME_SIZE = 391
TEST_SECONDS = 3.0
TRAIN_PIECE_SECONDS = 2.0
MAX_PIXEL_VALUE = 255.0
MAX_PLAUSIBLE_FPS = 1000.0


class ClipError(Exception):
    """A video could not be turned into a contract clip."""


def frames_for(seconds: float, fps: float) -> int:
    return int(round(seconds * fps))


def piece_bounds(n_frames: int, fps: float, seconds: float) -> list[tuple[int, int]]:
    """(start, stop) of consecutive full-length pieces; a short tail is dropped."""
    piece = frames_for(seconds, fps)
    if piece <= 0:
        raise ClipError(f"fps {fps} too low for {seconds} s pieces")
    return [(i * piece, (i + 1) * piece) for i in range(n_frames // piece)]


def trim(frames: np.ndarray, fps: float, seconds: float, source: str = "clip") -> np.ndarray:
    """First `seconds` of the clip; errors if the clip is shorter than that."""
    needed = frames_for(seconds, fps)
    if len(frames) < needed:
        raise ClipError(
            f"{source} is too short: {len(frames) / fps:.2f} s "
            f"({len(frames)} frames at {fps:.2f} fps), need {seconds:.0f} s"
        )
    return frames[:needed]


def to_square_gray(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR frame -> centered square crop, 391x391 grayscale float32 in 0-1."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    square = gray[top : top + side, left : left + side]
    resized = cv2.resize(square, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / MAX_PIXEL_VALUE


def read_video(path: Path, max_seconds: float | None = None) -> tuple[np.ndarray, float]:
    """Read a video -> (frames float32 (n, 391, 391) in 0-1, fps).

    OpenCV applies phone rotation metadata by default; we leave that on.
    """
    if not path.is_file():
        raise ClipError(f"Video not found: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ClipError(f"OpenCV cannot open {path} (unsupported codec or corrupt file?)")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0 or fps > MAX_PLAUSIBLE_FPS:
            raise ClipError(
                f"{path}: video reports fps={fps}. Re-export it "
                f"(e.g. ffmpeg -i in.mov -r 30 out.mp4) and try again."
            )
        max_frames = None if max_seconds is None else frames_for(max_seconds, fps)
        frames = []
        while max_frames is None or len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(to_square_gray(frame))
    finally:
        capture.release()
    if not frames:
        raise ClipError(f"{path}: no frames could be decoded")
    return np.stack(frames), fps


def save_clip(out_dir: Path, name: str, frames: np.ndarray, fps: float) -> Path:
    path = out_dir / f"{name}.npz"
    stored = np.clip(frames, 0.0, 1.0).astype(np.float16)
    np.savez_compressed(path, frames=stored, fps=float(fps))
    print(
        f"{path.name}: shape={stored.shape} dtype={stored.dtype} fps={fps:.3f} "
        f"min={float(stored.min()):.3f} max={float(stored.max()):.3f}"
    )
    return path


def prep_test_clip(video: Path, name: str, out_dir: Path) -> Path:
    frames, fps = read_video(video, max_seconds=TEST_SECONDS)
    return save_clip(out_dir, name, trim(frames, fps, TEST_SECONDS, str(video)), fps)


def read_train_pieces(video: Path) -> tuple[list[np.ndarray], float]:
    """One training video -> (its full 2 s pieces, fps); errors if it has none."""
    frames, fps = read_video(video)
    bounds = piece_bounds(len(frames), fps, TRAIN_PIECE_SECONDS)
    if not bounds:
        raise ClipError(
            f"{video} is too short: {len(frames) / fps:.2f} s, "
            f"need at least {TRAIN_PIECE_SECONDS:.0f} s"
        )
    return [frames[start:stop] for start, stop in bounds], fps


def report_error(error: ClipError) -> str:
    message = str(error)
    print(f"ERROR: {message}", file=sys.stderr)
    return message


def prep_train_clips(videos: list[Path], out_dir: Path) -> tuple[list[Path], list[str]]:
    """Cut every training video into 2 s pieces numbered continuously.

    A video that fails is reported and skipped so the rest still get processed.
    Returns (saved paths, error messages).
    """
    saved: list[Path] = []
    errors: list[str] = []
    stale = sorted(out_dir.glob("train_*.npz"))
    for path in stale:
        path.unlink()
    if stale:
        print(f"Removed {len(stale)} train_*.npz from an earlier run so old and new pieces do not mix")
    for video in videos:
        try:
            pieces, fps = read_train_pieces(video)
        except ClipError as error:
            errors.append(report_error(error))
            continue
        for piece in pieces:
            saved.append(save_clip(out_dir, f"train_{len(saved):02d}", piece, fps))
    return saved, errors


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--test-main", type=Path, help="held-out demo video with clear motion (3 s used)")
    parser.add_argument("--test-still", type=Path, help="held-out video with no motion (3 s used)")
    parser.add_argument("--train", type=Path, nargs="+", default=[], help="training videos (cut into 2 s pieces)")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (default: current dir)")
    parser.add_argument("--out", type=Path, default=Path("clips"), help="output dir, relative to --root")
    args = parser.parse_args(argv)
    if not (args.test_main or args.test_still or args.train):
        parser.error("give at least one of --test-main, --test-still, --train")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = args.out if args.out.is_absolute() else args.root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for video, name in ((args.test_main, "test_main"), (args.test_still, "test_still")):
        if not video:
            continue
        try:
            prep_test_clip(video, name, out_dir)
        except ClipError as error:
            errors.append(report_error(error))
    if args.train:
        pieces, train_errors = prep_train_clips(args.train, out_dir)
        errors.extend(train_errors)
        print(f"{len(pieces)} training pieces from {len(args.train) - len(train_errors)} of {len(args.train)} videos")
    if errors:
        print(f"FAILED for {len(errors)} video(s) (see ERROR lines above); the others were written to {out_dir}/", file=sys.stderr)
        return 1
    print(f"Done. Upload {out_dir}/ to the shared Drive folder fly_sees/clips/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
