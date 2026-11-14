"""Step 9 of the guide: the demo video, on photos the decoders never saw.

Scene: 2 s of a dark ball rolling across a still photo, then 2 s of the camera panning across another photo,
then 1 s of a still photo. Panels: the original, the fly's-eye input, and what can be decoded from the motion
detectors (T4/T5) of flyvis network 000: brightness, and brightness change over the last 40 ms beside the truth.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

import clips
import flyeye
import flycns.flyvis as fv

MODEL, DT, K = 0, 1 / 100, 5
S = flyeye.MIN_FRAME
rng = np.random.default_rng(7)

astro = clips.load_gray("astronaut.png")
moto = clips.load_gray("motorcycle_right.png")
chel = clips.load_gray("chelsea.png")

# segment 1: a dark ball rolls left to right across a still photo
bg = astro[60:60 + S, 200:200 + S].copy()
yy, xx = np.mgrid[0:S, 0:S]
seg1 = []
for t in range(200):
    f = bg.copy()
    cx, cy = -60 + t * (S + 120) / 199, 250
    f[(yy - cy) ** 2 + (xx - cx) ** 2 <= 55 ** 2] = 0.05
    seg1.append(f)
# segment 2: the camera pans across another photo
H, W = moto.shape
seg2 = [moto[200:200 + S, int(100 + 2.5 * t): int(100 + 2.5 * t) + S] for t in range(200)]
# segment 3: a still photo
still = chel[250:250 + S, 350:350 + S]
seg3 = [still] * 100
frames = np.stack(seg1 + seg2 + seg3).astype(np.float32)          # (500, 391, 391), one frame per 10 ms step
eye = flyeye.box_eye(frames)                                       # (500, 721)

d = np.load("lattice.npz", allow_pickle=True)
lat = {k: d[k] for k in d.files}
ens = fv.load_ensemble()
net = flyeye.FlyVis(lat, ens, MODEL, DT)
types = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
acts = net.run(eye[None], types)
dec = np.load(f"decoders_{MODEL:03d}.npz")
A = np.stack([acts[t][0] for t in types], -1)
X = np.concatenate([A[K - 1 - k: len(A) - k] for k in range(K)], -1).reshape(-1, len(types) * K)
P = (X @ dec["Motion detectors_B"] + dec["Motion detectors_b0"]).reshape(-1, 721, 2)
bright, change = np.clip(P[..., 0], 0, 1), P[..., 1]
true_change = eye[K - 1:] - eye[: len(eye) - K + 1]
steps = np.arange(K - 1, len(eye))                                 # step of each decoded frame

res = json.load(open(f"decode_{MODEL:03d}.json"))
x, y = flyeye.CENTRES[:, 1], -flyeye.CENTRES[:, 0]                 # receptor positions, image-up = plot-up

plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(2, 3, figsize=(12, 8.2), facecolor="#111111")
for ax in axes.flat:
    ax.set_facecolor("#111111"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
titles = ["Original video", "Fly's-eye input (721 columns)", "Brightness decoded from T4/T5",
          "", "True brightness change (40 ms)", "Change decoded from T4/T5"]
for ax, ttl in zip(axes.flat, titles):
    ax.set_title(ttl, color="white", fontsize=11, pad=6)
img = axes[0, 0].imshow(frames[steps[0]], cmap="gray", vmin=0, vmax=1)
kw = dict(marker="h", s=50, linewidths=0)
p_eye = axes[0, 1].scatter(x, y, c=eye[steps[0]], cmap="gray", vmin=0, vmax=1, **kw)
p_br = axes[0, 2].scatter(x, y, c=bright[0], cmap="gray", vmin=0, vmax=1, **kw)
DIV = LinearSegmentedColormap.from_list("dark_div", ["#4aa3ff", "#1a1a1a", "#ff6a3d"])
lim = float(2.5 * true_change.std())
print(f"change colour scale +-{lim:.3f}; true change std {true_change.std():.3f}, decoded {change.std():.3f}")
p_tc = axes[1, 1].scatter(x, y, c=true_change[0], cmap=DIV, vmin=-lim, vmax=lim, **kw)
p_dc = axes[1, 2].scatter(x, y, c=change[0], cmap=DIV, vmin=-lim, vmax=lim, **kw)
for ax in [axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2]]:
    ax.set_aspect("equal"); ax.set_xlim(-205, 205); ax.set_ylim(-205, 205)
nat = res["natural"]["Motion detectors"]
info = (f"flyvis network {MODEL:03d}\n(Lappalainen et al., 2024)\n\n"
        f"Held-out photos, 48 test clips:\n"
        f"  brightness from T4/T5   r = {nat['brightness']:.2f}\n"
        f"  change from T4/T5       r = {nat['change']:.2f}\n"
        f"  brightness from medulla r = {res['natural']['Medulla']['brightness']:.2f}\n\n"
        f"Orange = got brighter, blue = got darker,\nblack = no change (same scale in both).\n\n"
        f"Decoded from simulated neurons,\nnot what the fly experiences.")
axes[1, 0].text(0.02, 0.95, info, color="#dddddd", fontsize=9.5, va="top", family="DejaVu Sans Mono",
                transform=axes[1, 0].transAxes)
label = fig.text(0.5, 0.015, "", color="white", ha="center", fontsize=12)
fig.tight_layout(rect=(0, 0.04, 1, 1))


def update(i):
    k = steps[i]
    img.set_data(frames[k]); p_eye.set_array(eye[k]); p_br.set_array(bright[i])
    p_tc.set_array(true_change[i]); p_dc.set_array(change[i])
    scene = "A dark ball rolls across a still photo" if k < 200 else (
        "The camera pans across a photo" if k < 400 else "A still photo")
    label.set_text(f"{scene}    t = {k * DT:.2f} s")
    return img, p_eye, p_br, p_tc, p_dc, label


anim = FuncAnimation(fig, update, frames=range(0, len(steps), 4), blit=False)
anim.save("what_the_fly_sees.mp4", writer=FFMpegWriter(fps=25, bitrate=2400), dpi=100,
          savefig_kwargs={"facecolor": "#111111"})
for i, name in [(96, "frame_ball"), (296, "frame_pan"), (len(steps) - 1, "frame_still")]:
    update(i); fig.savefig(f"{name}.png", dpi=90, facecolor="#111111")
print("saved what_the_fly_sees.mp4")
