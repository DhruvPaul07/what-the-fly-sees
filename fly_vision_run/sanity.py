"""Step 5 of the guide: sanity checks before building anything."""
import json
import numpy as np
import flyeye
import flycns.flyvis as fv

d = np.load("lattice.npz", allow_pickle=True)
lat = {k: d[k] for k in d.files}
ens = fv.load_ensemble()
DT = 1 / 100
CENTRAL = {t: np.flatnonzero((lat["type_name"] == t) & (lat["node_u"] == 0) & (lat["node_v"] == 0))[0]
           for t in ens.types}
T4 = ["T4a", "T4b", "T4c", "T4d"]
T5 = ["T5a", "T5b", "T5c", "T5d"]


def edge_clip(on=True, n=60, size=391):
    clip = np.full((n, size, size), 0.0 if on else 1.0, np.float32)
    for i in range(n):
        clip[i, :, : int(size * i / n)] = 1.0 if on else 0.0
    return clip


def to_steps(hex_frames, fps):
    n_steps = int(len(hex_frames) / fps / DT)
    return hex_frames[(np.arange(n_steps) * DT * fps).astype(int)]


def run_central(net, movie, fade=True):
    """movie (T, 721) -> (T, n_types) activity of each type's central cell."""
    v = net.fade_in(movie[None, 0]) if fade else net.rest(1)
    idx = np.array([CENTRAL[t] for t in ens.types])
    out = np.zeros((len(movie), len(idx)), np.float32)
    for k in range(len(movie)):
        v = net.step(v, movie[k][:, None]); out[k] = v[idx, 0]
    return out, v


def peak_depolarization(trace, cols):
    ti = [ens.types.index(t) for t in cols]
    return float(np.max(trace[:, ti] - trace[0, ti]))


results = {}
on_steps = to_steps(flyeye.box_eye(edge_clip(True)), 60)
off_steps = to_steps(flyeye.box_eye(edge_clip(False)), 60)
for model in range(5):
    net = flyeye.FlyVis(lat, ens, model, DT)
    r = {}
    # 1. uniform grey stays flat
    grey = np.full((100, 721), 0.5, np.float32)
    tr, _ = run_central(net, grey)
    r["grey_max_drift_last_half"] = float(np.max(tr[50:].max(0) - tr[50:].min(0)))
    r["grey_activity_range"] = float(tr.max() - tr.min())
    # 2. photoreceptors track the input (flicker, a new random brightness per column every 50 ms)
    rng = np.random.default_rng(model)
    flick = np.repeat(rng.uniform(0, 1, (20, 721)), 5, axis=0).astype(np.float32)
    v = net.fade_in(flick[None, 0]); R1 = net.layer["R1"]; acts = []
    for k in range(len(flick)):
        v = net.step(v, flick[k][:, None]); acts.append(v[R1, 0])
    acts = np.array(acts)
    cors = [np.corrcoef(acts[:, c], flick[:, c])[0, 1] for c in range(721)]
    r["R1_vs_input_median_corr"] = float(np.median(cors))
    # 3. bright edges drive T4, dark edges drive T5 (peak depolarization of the central cells)
    tr_on, _ = run_central(net, on_steps)
    tr_off, _ = run_central(net, off_steps)
    r["ON_edge_T4"], r["ON_edge_T5"] = peak_depolarization(tr_on, T4), peak_depolarization(tr_on, T5)
    r["OFF_edge_T4"], r["OFF_edge_T5"] = peak_depolarization(tr_off, T4), peak_depolarization(tr_off, T5)
    # 4. activity settles when the stimulus stops: ON edge, then 1.5 s of grey, compared with a grey steady state
    after = np.concatenate([on_steps, np.full((150, 721), 0.5, np.float32)])
    tr_after, _ = run_central(net, after)
    tr_ref, _ = run_central(net, np.full((250, 721), 0.5, np.float32))
    r["settle_max_diff_from_grey_steady"] = float(np.max(np.abs(tr_after[-1] - tr_ref[-1])))
    r["checks"] = {
        "grey_flat": r["grey_max_drift_last_half"] < 0.01 * r["grey_activity_range"] + 1e-3,
        "R1_tracks_input": r["R1_vs_input_median_corr"] > 0.8,
        "ON_edge_T4_gt_T5": r["ON_edge_T4"] > r["ON_edge_T5"],
        "OFF_edge_T5_gt_T4": r["OFF_edge_T5"] > r["OFF_edge_T4"],
        "settles": r["settle_max_diff_from_grey_steady"] < 0.05,
    }
    results[f"{model:03d}"] = r
    print(f"network {model:03d}: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in r["checks"].items()))
    print(f"   grey drift {r['grey_max_drift_last_half']:.4f} (range {r['grey_activity_range']:.2f}); "
          f"R1 corr {r['R1_vs_input_median_corr']:.3f}; ON edge T4 {r['ON_edge_T4']:.2f} vs T5 {r['ON_edge_T5']:.2f}; "
          f"OFF edge T4 {r['OFF_edge_T4']:.2f} vs T5 {r['OFF_edge_T5']:.2f}; settle diff "
          f"{r['settle_max_diff_from_grey_steady']:.4f}")
json.dump(results, open("sanity_results.json", "w"), indent=1)
