"""What the Fly Sees: a numpy run of the flyvis visual model (Lappalainen et al., Nature 2024).

flyvis itself needs PyTorch and downloads its pretrained networks from Google Drive. Neither was reachable here,
so this module rebuilds the same pieces from their published sources:

- The lattice: flyvis 1.2.0's own connectome file (fib25-fib19_v2.2.json, from the flyvis wheel on PyPI),
  compiled exactly as flyvis's ConnectomeFromAvgFilters does (nodes on a 15-column-radius hexagon, convolutional
  edges, the convex-hull fill of data gaps).
- The trained numbers: the 50 pretrained networks (flow/0000) as extracted from flyvis's checkpoints and shipped
  in the flycns package (resting potential and time constant per type, strength per pair of types).
- The dynamics: flyvis's PPNeuronIGRSynapses, tau_eff dV/dt = -V + b + sum_j w_ij relu(V_j) + x, forward Euler,
  tau_eff = max(tau, dt), starting at the resting potentials.
- The eye: flyvis's BoxEye (13 x 13 pixel box mean at 721 receptor centres, 13 pixels apart).
"""
from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib.path as mpath
import numpy as np
import scipy.sparse as sp
import scipy.spatial as ss

EXTENT = 15
KERNEL = 13
N_COLUMNS = 721


# ---------------------------------------------------------------- lattice (flyvis ConnectomeFromAvgFilters)

def load_connectome_spec(wheel: str | Path) -> dict:
    with zipfile.ZipFile(wheel) as z:
        return json.loads(z.read("flyvis/connectome/fib25-fib19_v2.2.json"))


def _strided(extent, su, sv):
    for u in range(-extent, extent + 1):
        for v in range(max(-extent, -extent - u), min(extent, extent - u) + 1):
            if u % su == 0 and v % sv == 0:
                yield u, v


def _fill_hull(offsets, n_syn_fill):
    known = np.array([o[0] for o in offsets])
    known_set = set(map(tuple, known))
    hull = ss.ConvexHull((1 + 1e-6) * known, False, "QJ")
    verts = known[hull.vertices]
    grid = np.concatenate(np.dstack(np.mgrid[known[:, 0].min(): known[:, 0].max() + 1,
                                             known[:, 1].min(): known[:, 1].max() + 1]))
    inside = grid[mpath.Path(verts).contains_points(grid)]
    return offsets + [[[u, v], n_syn_fill] for u, v in inside if (u, v) not in known_set]


def build_lattice(spec: dict, extent: int = EXTENT, n_syn_fill: float = 1) -> dict:
    node_type, node_u, node_v = [], [], []
    for n in spec["nodes"]:
        typ, (pattern, args) = n["name"], n["pattern"]
        if pattern == "stride":
            pts = _strided(extent, *args)
        elif pattern == "tile":
            pts = _strided(extent, args, args)
        elif pattern == "single":
            pts = _strided(0, 1, 1)
        else:
            raise ValueError(pattern)
        for u, v in pts:
            node_type.append(typ); node_u.append(u); node_v.append(v)
    index = {(t, u, v): i for i, (t, u, v) in enumerate(zip(node_type, node_u, node_v))}
    by_type = defaultdict(list)
    for i, t in enumerate(node_type):
        by_type[t].append(i)
    src, tar, sign, nsyn, e_src_t, e_tar_t = [], [], [], [], [], []
    for e in spec["edges"]:
        offs = e["offsets"]
        if n_syn_fill > 0 and len(offs) >= 3:
            offs = _fill_hull(offs, n_syn_fill)
        for (du, dv), n_syn in offs:
            for s in by_type.get(e["src"], []):
                t = index.get((e["tar"], node_u[s] + du, node_v[s] + dv))
                if t is not None:
                    src.append(s); tar.append(t); sign.append(e["alpha"]); nsyn.append(n_syn)
    node_type = np.array(node_type)
    return dict(
        type_name=node_type,
        node_u=np.array(node_u, dtype=np.int64), node_v=np.array(node_v, dtype=np.int64),
        edge_source=np.array(src, dtype=np.int64), edge_target=np.array(tar, dtype=np.int64),
        edge_sign=np.array(sign, dtype=np.float32), edge_n_syn=np.array(nsyn, dtype=np.float32),
        input_types=list(spec["input_units"]),
        input_index=np.stack([np.flatnonzero(node_type == t) for t in spec["input_units"]]),
    )


def lattice_for_flycns(lat: dict, ensemble) -> dict:
    """The lattice in flycns's keys, node types indexed in the ensemble's (flyvis parameter) order."""
    tindex = {t: i for i, t in enumerate(ensemble.types)}
    ntype = np.array([tindex[t] for t in lat["type_name"]])
    pidx = ensemble.pair_index()
    names = lat["type_name"]
    edge_pair = np.array([pidx[(a, b)] for a, b in zip(names[lat["edge_source"]], names[lat["edge_target"]])])
    return dict(node_u=lat["node_u"], node_v=lat["node_v"], node_type=ntype,
                edge_source=lat["edge_source"], edge_target=lat["edge_target"], edge_pair=edge_pair,
                input_index=lat["input_index"])


# ---------------------------------------------------------------- eye (flyvis BoxEye, numpy)

def receptor_centres(extent=EXTENT, d=KERNEL):
    """(721, 2) integer (y, x) offsets, in flyvis's sorted (u, v) order."""
    out = []
    for u in range(-extent, extent + 1):
        for v in range(max(-extent, -extent - u), min(extent, extent - u) + 1):
            out.append((int(d * (u + v / 2)), d * v))   # torch.long truncation == int()
    return np.array(out, dtype=np.int64)


CENTRES = receptor_centres()
MIN_FRAME = int(CENTRES[:, 0].max() - CENTRES[:, 0].min() + 1)   # 391


def box_eye(frames: np.ndarray) -> np.ndarray:
    """(n, H, W) frames in 0..1 with H, W >= 391 -> (n, 721): the mean of the 13 x 13 box (zero padded) around
    each receptor centre, the centres placed around the frame centre, as flyvis's BoxEye does."""
    n, h, w = frames.shape
    assert h >= MIN_FRAME and w >= MIN_FRAME, "resize frames to at least 391 x 391 first"
    pad = (KERNEL - 1) // 2
    f = np.pad(frames.astype(np.float64), ((0, 0), (pad, pad), (pad, pad)))
    ii = np.zeros((n, h + 2 * pad + 1, w + 2 * pad + 1))
    ii[:, 1:, 1:] = f.cumsum(1).cumsum(2)
    cy = CENTRES[:, 0] + h // 2 + pad
    cx = CENTRES[:, 1] + w // 2 + pad
    y0, y1, x0, x1 = cy - pad, cy + pad + 1, cx - pad, cx + pad + 1
    s = ii[:, y1, x1] - ii[:, y0, x1] - ii[:, y1, x0] + ii[:, y0, x0]
    return (s / KERNEL ** 2).astype(np.float32)


# ---------------------------------------------------------------- simulator (flyvis PPNeuronIGRSynapses)

class FlyVis:
    """One pretrained network on the lattice, batched: state is (neurons, clips)."""

    def __init__(self, lat: dict, ensemble, model: int, dt: float = 1 / 100):
        types = ensemble.types
        tindex = {t: i for i, t in enumerate(types)}
        ntype = np.array([tindex[t] for t in lat["type_name"]])
        pair_index = ensemble.pair_index()
        # every edge's pair of types and its synapse-count group (pair, du, dv), as flyvis groups them
        src_t = ntype[lat["edge_source"]]; tar_t = ntype[lat["edge_target"]]
        pair_lut = -np.ones((len(types), len(types)), dtype=np.int64)
        for (a, b), i in pair_index.items():
            pair_lut[tindex[a], tindex[b]] = i
        pair = pair_lut[src_t, tar_t]
        assert (pair >= 0).all(), "an edge joins a pair of types the ensemble does not know"
        W = 64
        gkey = ((ensemble.group_pair * W + (ensemble.group_du + W // 2)) * W + (ensemble.group_dv + W // 2))
        du = lat["node_u"][lat["edge_target"]] - lat["node_u"][lat["edge_source"]]
        dv = lat["node_v"][lat["edge_target"]] - lat["node_v"][lat["edge_source"]]
        ekey = (pair * W + (du + W // 2)) * W + (dv + W // 2)
        order = np.argsort(gkey)
        group = order[np.searchsorted(gkey[order], ekey)]
        assert np.array_equal(gkey[group], ekey), "an edge has no synapse-count group"
        # the JSON's own sign must agree with the trained ensemble's fixed sign
        assert np.array_equal(ensemble.pair_sign[pair].astype(np.float32), lat["edge_sign"])
        weight = ensemble.pair_sign[pair] * ensemble.group_n_syn[group] * ensemble.strength[model][pair]
        n = len(ntype)
        self.n, self.dt, self.model = n, dt, model
        self.W = sp.csr_matrix((weight.astype(np.float32), (lat["edge_target"], lat["edge_source"])), shape=(n, n))
        self.bias = ensemble.bias[model][ntype].astype(np.float32)[:, None]
        tau = ensemble.time_const_s[model][ntype].astype(np.float32)
        self.rate = (1.0 / np.maximum(tau, dt)).astype(np.float32)[:, None]
        self.input_index = lat["input_index"]            # (8, 721)
        self.layer = {t: np.flatnonzero(lat["type_name"] == t) for t in types}
        self.types = types
        self.weight, self.group = weight, group

    def step(self, v, x):
        """v: (n, B); x: (721, B) intensities -> next v."""
        cur = np.zeros_like(v)
        cur[self.input_index.reshape(-1)] = np.tile(x, (self.input_index.shape[0], 1))
        return v + self.dt * self.rate * (-v + self.bias + self.W @ np.maximum(v, 0) + cur)

    def rest(self, batch: int):
        return np.repeat(self.bias, batch, axis=1).copy()

    def fade_in(self, first: np.ndarray, t: float = 1.0):
        """flyvis's fade_in_state: from rest, ramp the first frame's contrast about 0.5 up over t seconds.
        first: (B, 721)."""
        nstep = int(t / self.dt)
        v = self.rest(first.shape[0])
        for a in np.linspace(0, 1, nstep):
            v = self.step(v, (a * (first - 0.5) + 0.5).T)
        return v

    def run(self, movie: np.ndarray, record: list[str], fade: bool = True):
        """movie: (B, T, 721). Returns {type: (B, T, cells)} recorded after each step, float32."""
        B, T, _ = movie.shape
        v = self.fade_in(movie[:, 0]) if fade else self.rest(B)
        idx = {t: self.layer[t] for t in record}
        out = {t: np.zeros((B, T, len(i)), np.float32) for t, i in idx.items()}
        for k in range(T):
            v = self.step(v, movie[:, k].T)
            for t, i in idx.items():
                out[t][:, k] = v[i].T
        return out
