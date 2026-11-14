"""Steps 6 to 8 of the guide: train per-group decoders and test them on unfamiliar clips.

The decoder is one linear map shared by all 721 columns: inputs are a column's activity for the group's cell types
over the last K steps, the target is the brightness that column received (and, as an extra, how much that
brightness changed over the last 40 ms). Ridge regression on standardised features, accumulated clip by clip.
"""
import json
import sys
import time

import numpy as np

import flyeye
import flycns.flyvis as fv

MODEL = int(sys.argv[1]) if len(sys.argv) > 1 else 0
DT, K, LAM, BATCH = 1 / 100, 5, 1e-3, 16

d = np.load("lattice.npz", allow_pickle=True)
lat = {k: d[k] for k in d.files}
ens = fv.load_ensemble()
C = np.load("clips.npz", allow_pickle=True)
train = C["train"].astype(np.float32)
tests = {"natural": C["test_nat"].astype(np.float32), "synthetic": C["test_syn"].astype(np.float32),
         "still": C["test_still"].astype(np.float32)}

GROUPS = {
    "Photoreceptors": ["R1", "R2", "R3", "R4", "R5", "R6"],
    "Lamina": ["L1", "L2", "L3", "L4", "L5"],
    "Medulla": [t for t in ens.types if t.startswith(("Mi", "Tm"))],
    "Motion detectors": ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"],
}
RECORD = sorted({t for g in GROUPS.values() for t in g})
net = flyeye.FlyVis(lat, ens, MODEL, DT)


def features(acts, b, types):
    A = np.stack([acts[t][b] for t in types], -1)                       # (T, 721, n)
    X = np.concatenate([A[K - 1 - k: A.shape[0] - k] for k in range(K)], -1)
    return X.reshape(-1, X.shape[-1])                                     # ((T-K+1)*721, n*K)


def targets(movie):
    y = movie[K - 1:]                                                     # brightness now
    dy = movie[K - 1:] - movie[: movie.shape[0] - K + 1]                  # change over the last 40 ms
    return np.stack([y.reshape(-1), dy.reshape(-1)], -1)


class Ridge:
    def __init__(self, F, nt=2):
        self.n, self.Sx, self.Sxx = 0, np.zeros(F), np.zeros((F, F))
        self.Sy, self.Sxy = np.zeros(nt), np.zeros((F, nt))

    def add(self, X, Y):
        X = X.astype(np.float64); Y = Y.astype(np.float64)
        self.n += len(X); self.Sx += X.sum(0); self.Sxx += X.T @ X
        self.Sy += Y.sum(0); self.Sxy += X.T @ Y

    def solve(self, lam=LAM):
        mu = self.Sx / self.n
        cov = self.Sxx / self.n - np.outer(mu, mu)
        s = np.sqrt(np.clip(np.diag(cov), 0, None)) + 1e-8
        ybar = self.Sy / self.n
        cxy = self.Sxy / self.n - np.outer(mu, ybar)
        B = np.linalg.solve(cov / np.outer(s, s) + lam * np.eye(len(s)), cxy / s[:, None]) / s[:, None]
        self.B, self.b0 = B.astype(np.float32), (ybar - mu @ B).astype(np.float32)
        return self

    def predict(self, X):
        return X @ self.B + self.b0


def frame_corr(P, Y):
    """Per-frame spatial correlation across the 721 columns; frames with no spatial variation are skipped."""
    Pc, Yc = P - P.mean(1, keepdims=True), Y - Y.mean(1, keepdims=True)
    den = np.sqrt((Pc ** 2).sum(1) * (Yc ** 2).sum(1))
    ok = den > 1e-8
    return (Pc * Yc).sum(1)[ok] / den[ok]


t0 = time.time()
dec = {g: Ridge(len(ts) * K) for g, ts in GROUPS.items()}
shuffled = Ridge(len(GROUPS["Motion detectors"]) * K)
col_sum, col_n = np.zeros(flyeye.N_COLUMNS), 0
for start in range(0, len(train), BATCH):
    mov = train[start:start + BATCH]
    acts = net.run(mov, RECORD)
    for b in range(len(mov)):
        Y = targets(mov[b])
        for g, ts in GROUPS.items():
            dec[g].add(features(acts, b, ts), Y)
        other = targets(mov[(b + 1) % len(mov)])
        shuffled.add(features(acts, b, GROUPS["Motion detectors"]), other)
        col_sum += mov[b, K - 1:].sum(0); col_n += mov.shape[1] - K + 1
for r in list(dec.values()) + [shuffled]:
    r.solve()
mean_image = (col_sum / col_n).astype(np.float32)
print(f"network {MODEL:03d}: trained on {len(train)} clips in {time.time() - t0:.0f}s")


def score(movies, keep=None):
    out = {g: {"brightness": [], "change": []} for g in list(GROUPS) + ["Shuffled control", "Mean-brightness baseline"]}
    extra = {"pred_std_ratio": []}
    kept = {}
    for start in range(0, len(movies), BATCH):
        mov = movies[start:start + BATCH]
        acts = net.run(mov, RECORD)
        for b in range(len(mov)):
            Y = targets(mov[b]).reshape(-1, flyeye.N_COLUMNS, 2)
            for g, ts in list(GROUPS.items()) + [("Shuffled control", GROUPS["Motion detectors"])]:
                r = shuffled if g == "Shuffled control" else dec[g]
                P = r.predict(features(acts, b, ts)).reshape(-1, flyeye.N_COLUMNS, 2)
                out[g]["brightness"] += list(frame_corr(P[..., 0], Y[..., 0]))
                out[g]["change"] += list(frame_corr(P[..., 1], Y[..., 1]))
                if g == "Motion detectors":
                    extra["pred_std_ratio"].append(float(P[..., 0].std(1).mean() / (Y[..., 0].std(1).mean() + 1e-8)))
            M = np.repeat(mean_image[None], Y.shape[0], 0)
            out["Mean-brightness baseline"]["brightness"] += list(frame_corr(M, Y[..., 0]))
    summary = {g: {k: (float(np.mean(v)) if v else None) for k, v in m.items()} for g, m in out.items()}
    summary["_motion_pred_std_ratio"] = float(np.mean(extra["pred_std_ratio"]))
    return summary


res = {"model": MODEL, "train": score(train[:32])}
for name, mv in tests.items():
    res[name] = score(mv)
res["seconds"] = round(time.time() - t0)
np.savez(f"decoders_{MODEL:03d}.npz", mean_image=mean_image,
         **{f"{g}_B": r.B for g, r in dec.items()}, **{f"{g}_b0": r.b0 for g, r in dec.items()})
json.dump(res, open(f"decode_{MODEL:03d}.json", "w"), indent=1)
rows = list(GROUPS) + ["Shuffled control", "Mean-brightness baseline"]
print(f"{'group':26s} {'train':>7s} {'natural':>8s} {'synthetic':>9s}   change(natural)")
for g in rows:
    f = lambda s, k="brightness": "   -  " if res[s][g][k] is None else f"{res[s][g][k]:6.3f}"
    print(f"{g:26s} {f('train'):>7s} {f('natural'):>8s} {f('synthetic'):>9s}   {f('natural', 'change')}")
print(f"still scenes: motion-detector decoder output spread / true spread = {res['still']['_motion_pred_std_ratio']:.2f}; "
      f"its brightness corr = {res['still']['Motion detectors']['brightness']:.3f}")
print(f"total {res['seconds']}s")
