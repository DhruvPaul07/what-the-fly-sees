"""Generate small SYNTHETIC test videos (not real footage) to exercise the pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

FPS = 30
WIDTH, HEIGHT = 640, 480  # non-square on purpose, to exercise the centre crop
BLOB_RADIUS = 40


def background(rng: np.random.Generator) -> np.ndarray:
    coarse = rng.uniform(40, 120, (12, 16)).astype(np.float32)
    return cv2.resize(coarse, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)


def write(path: Path, frames: list[np.ndarray]) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer for {path}")
    for frame in frames:
        gray = np.clip(frame, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    writer.release()


def blob_video(seconds: float, seed: int, y_fraction: float) -> list[np.ndarray]:
    """Bright blob moving left -> right over a textured background."""
    base = background(np.random.default_rng(seed))
    n = int(seconds * FPS)
    frames = []
    for i in range(n):
        frame = base.copy()
        centre_x = int(120 + (WIDTH - 240) * i / (n - 1))
        cv2.circle(frame, (centre_x, int(HEIGHT * y_fraction)), BLOB_RADIUS, 255.0, -1)
        frames.append(frame)
    return frames


def still_video(seconds: float, seed: int) -> list[np.ndarray]:
    base = background(np.random.default_rng(seed))
    cv2.rectangle(base, (200, 150), (330, 260), 230.0, -1)
    return [base.copy() for _ in range(int(seconds * FPS))]


def grating_video(seconds: float, wavelength: int) -> list[np.ndarray]:
    xs = np.arange(WIDTH)[None, :].repeat(HEIGHT, axis=0)
    return [
        127 + 100 * np.sin(2 * np.pi * (xs - 4 * i) / wavelength)
        for i in range(int(seconds * FPS))
    ]


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write(out_dir / "main.mp4", blob_video(3.5, seed=1, y_fraction=0.35))
    write(out_dir / "still.mp4", still_video(3.5, seed=2))
    write(out_dir / "train_a.mp4", grating_video(5.0, wavelength=80))
    write(out_dir / "train_b.mp4", blob_video(5.0, seed=3, y_fraction=0.6))
    write(out_dir / "too_short.mp4", still_video(1.0, seed=4))
    print(f"synthetic videos in {out_dir}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent)
