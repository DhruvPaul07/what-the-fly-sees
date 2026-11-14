"""Training and test clips for the decoder (guide Steps 2, 3 and 6).

No phone footage was available, so the natural clips are built from photographs bundled with scikit-image and
matplotlib: camera pans across a photo (self-motion) and an object gliding over a still photo (object motion).
Test clips use photos never seen in training. Every clip is 1 s at the model's 10 ms step (100 frames), 391 x 391
pixels, grey levels 0 to 1, rendered straight onto the 721-column eye.
"""
from __future__ import annotations

import os

import matplotlib
import numpy as np
import skimage
from skimage import color, io, transform

import flyeye

S = flyeye.MIN_FRAME          # 391
T = 100                       # frames per clip = steps at dt = 1/100 s
SKDATA = os.path.join(os.path.dirname(skimage.__file__), "data")
TRAIN_IMAGES = ["camera.png", "coffee.png", "coins.png", "grass.png", "gravel.png", "brick.png", "rocket.jpg",
                "moon.png", "hubble_deep_field.jpg", "retina.jpg", "clock_motion.png", "page.png", "horse.png",
                "grace_hopper.jpg"]
TEST_IMAGES = ["astronaut.png", "chelsea.png", "motorcycle_right.png"]


def load_gray(name: str, short_side: int = 800) -> np.ndarray:
    if name == "grace_hopper.jpg":
        path = os.path.join(matplotlib.get_data_path(), "sample_data", name)
    else:
        path = os.path.join(SKDATA, name)
    im = io.imread(path)
    if im.ndim == 3:
        im = color.rgb2gray(im[..., :3])
    im = im.astype(np.float64)
    if im.max() > 1:
        im = im / 255.0
    scale = short_side / min(im.shape)
    im = transform.resize(im, (round(im.shape[0] * scale), round(im.shape[1] * scale)), anti_aliasing=True)
    return np.clip(im, 0, 1).astype(np.float32)


def _crop(canvas, y, x):
    return canvas[y:y + S, x:x + S]


def pan(canvas, rng, speed=None, angle=None):
    """The camera slides across a still photo: the whole scene moves (optic flow from self-motion)."""
    H, W = canvas.shape
    speed = rng.uniform(0.5, 3.0) if speed is None else speed          # pixels per 10 ms step
    angle = rng.uniform(0, 2 * np.pi) if angle is None else angle
    dx, dy = np.cos(angle) * speed * (T - 1), np.sin(angle) * speed * (T - 1)
    x0 = rng.uniform(max(0, -dx), W - S - max(0, dx))
    y0 = rng.uniform(max(0, -dy), H - S - max(0, dy))
    for t in range(T):
        yield _crop(canvas, int(round(y0 + dy * t / (T - 1))), int(round(x0 + dx * t / (T - 1))))


def glide(bg_canvas, obj_canvas, rng, kind=None, radius=None, speed=None):
    """An object glides across a still photo: only the object moves."""
    H, W = bg_canvas.shape
    bg = _crop(bg_canvas, rng.integers(0, H - S), rng.integers(0, W - S)).copy()
    r = rng.uniform(35, 90) if radius is None else radius
    kind = rng.choice(["patch", "dark", "bright"], p=[0.4, 0.3, 0.3]) if kind is None else kind
    speed = rng.uniform(1.5, 5.0) if speed is None else speed
    a = rng.uniform(0, 2 * np.pi)
    start = np.array([S / 2, S / 2]) + 0.45 * S * np.array([np.sin(a), np.cos(a)])
    heading = a + np.pi + rng.uniform(-0.4, 0.4)                       # roughly through the middle
    vel = speed * np.array([np.sin(heading), np.cos(heading)])
    ri = int(np.ceil(r))
    yy, xx = np.mgrid[-ri:ri + 1, -ri:ri + 1]
    mask = (yy ** 2 + xx ** 2) <= r ** 2
    if kind == "patch":
        oh, ow = obj_canvas.shape
        py, px = rng.integers(ri, oh - ri), rng.integers(ri, ow - ri)
        tex = obj_canvas[py - ri:py + ri + 1, px - ri:px + ri + 1]
    else:
        tex = np.full(mask.shape, 0.05 if kind == "dark" else 0.95, np.float32)
    for t in range(T):
        f = bg.copy()
        cy, cx = np.round(start + vel * t).astype(int)
        y0, y1, x0, x1 = cy - ri, cy + ri + 1, cx - ri, cx + ri + 1
        sy0, sx0 = max(0, -y0), max(0, -x0)
        sy1, sx1 = mask.shape[0] - max(0, y1 - S), mask.shape[1] - max(0, x1 - S)
        if sy1 > sy0 and sx1 > sx0:
            region = f[max(0, y0):min(S, y1), max(0, x0):min(S, x1)]
            m = mask[sy0:sy1, sx0:sx1]
            region[m] = tex[sy0:sy1, sx0:sx1][m]
        yield f


def blocks(rng):
    """Random-walk blocks over still noise, in the spirit of flyvis's random_walk_of_blocks."""
    bg = np.clip(rng.normal(0.5, 0.1, (S // 4 + 1, S // 4 + 1)), 0, 1).repeat(4, 0).repeat(4, 1)[:S, :S]
    n = rng.integers(5, 16)
    size = rng.integers(20, 70, n)
    lum = rng.choice([0.0, 1.0], n)
    pos = rng.uniform(0, S, (n, 2))
    vel = rng.normal(0, 3, (n, 2))
    for t in range(T):
        f = bg.astype(np.float32).copy()
        for i in range(n):
            y, x = pos[i].astype(int)
            f[max(0, y):max(0, y + size[i]), max(0, x):max(0, x + size[i])] = lum[i]
        vel = 0.9 * vel + rng.normal(0, 1.0, (n, 2))
        pos = np.mod(pos + vel, S)
        yield f


def grating(rng):
    """A drifting sine grating."""
    theta, lam = rng.uniform(0, np.pi), rng.uniform(30, 150)
    speed, contrast = rng.uniform(0.5, 4.0), rng.uniform(0.3, 1.0)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    proj = xx * np.cos(theta) + yy * np.sin(theta)
    for t in range(T):
        yield (0.5 + 0.5 * contrast * np.sin(2 * np.pi * (proj - speed * t) / lam)).astype(np.float32)


def still(canvas, rng):
    H, W = canvas.shape
    f = _crop(canvas, rng.integers(0, H - S), rng.integers(0, W - S))
    for _ in range(T):
        yield f


def render(frames, chunk=25) -> np.ndarray:
    """Frames (generator) -> (T, 721) eye input."""
    out, buf = [], []
    for f in frames:
        buf.append(f)
        if len(buf) == chunk:
            out.append(flyeye.box_eye(np.stack(buf))); buf = []
    if buf:
        out.append(flyeye.box_eye(np.stack(buf)))
    return np.concatenate(out)


def make_set(kinds_p: dict, n: int, images: list[str], seed: int, canvases: dict):
    rng = np.random.default_rng(seed)
    kinds = rng.choice(list(kinds_p), size=n, p=list(kinds_p.values()))
    movies = np.zeros((n, T, flyeye.N_COLUMNS), np.float32)
    for i, k in enumerate(kinds):
        a, b = rng.choice(images, 2, replace=len(images) < 2)
        if k == "pan":
            fr = pan(canvases[a], rng)
        elif k == "glide":
            fr = glide(canvases[a], canvases[b], rng)
        elif k == "blocks":
            fr = blocks(rng)
        elif k == "grating":
            fr = grating(rng)
        else:
            fr = still(canvases[a], rng)
        movies[i] = render(fr)
    return movies, kinds
