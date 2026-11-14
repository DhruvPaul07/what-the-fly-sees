"""Plan C Part 2: model sanity checks (guide Step 5).

    python sanity_checks.py --root /content/drive/MyDrive/fly_sees            # all five
    python sanity_checks.py --root ... --check on_off_edges                   # one

Each check saves one figure to results/sanity/<name>.png and records
{check, passed, note} in results/sanity/sanity.json. These test the input
pipeline: if one fails, the pipeline is wrong, not the idea.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np

import flysim as fs
import hex_render as hr

SANITY_DIR = Path("results/sanity")
FIGURE_FONTSIZE = 10
TRACE_TYPES = ("R1", "L1", "Mi1", "T4c", "T5c")
SETTLE_STEPS = int(round(0.1 / hr.DT))  # ignore the first / use the last 100 ms
FLAT_TOLERANCE = 0.05  # gray std allowed, as a fraction of the edge-response range
SETTLED_TOLERANCE = 0.10  # distance from baseline allowed after 1 s of gray, same units
MAX_LAG_STEPS = int(round(0.05 / hr.DT))  # search 0-50 ms
TRACKING_MIN_ABS_R = 0.7
TRACKING_COLUMNS = 5
TRACKING_SECONDS = 2.0
SETTLE_GRAY_SECONDS = 1.0


# ------------------------------------------------------------------ records

def record(root: Path, check: str, passed: bool, note: str) -> dict:
    """Add or replace this check's entry in results/sanity/sanity.json."""
    path = hr.resolve(root, SANITY_DIR / "sanity.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else []
    entry = {"check": check, "passed": bool(passed), "note": note}
    entries = [item for item in existing if item.get("check") != check] + [entry]
    path.write_text(json.dumps(entries, indent=2))
    print(f"[{'PASS' if passed else 'FAIL'}] {check}: {note}")
    return entry


def save_figure(fig, root: Path, name: str) -> Path:
    path = hr.resolve(root, SANITY_DIR / f"{name}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    # flyvis shrinks matplotlib's default fonts on import; set readable sizes explicitly.
    for text in fig.findobj(matplotlib.text.Text):
        text.set_fontsize(FIGURE_FONTSIZE)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def time_axis(n_steps: int) -> np.ndarray:
    return np.arange(n_steps) * hr.DT


# ------------------------------------------------------------- pure measures

def best_lagged_correlation(signal: np.ndarray, response: np.ndarray, max_lag: int = MAX_LAG_STEPS) -> tuple[float, int]:
    """(r, lag in steps) with the largest |r| when the response trails by 0..max_lag."""
    best_r, best_lag = 0.0, 0
    for lag in range(max_lag + 1):
        a, b = signal[: len(signal) - lag], response[lag:]
        if a.std() == 0 or b.std() == 0:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if abs(r) > abs(best_r):
            best_r, best_lag = r, lag
    return best_r, best_lag


def edge_range(activity, cell_type: str) -> float:
    """Peak-to-peak central response of a cell type to the ON edge."""
    return float(np.ptp(fs.central(activity, cell_type)))


def edge_selectivity(network) -> dict:
    """Mean depolarization (and unsigned change) of T4a-d and T5a-d to ON and OFF edges."""
    result = {}
    for name, on in (("ON", True), ("OFF", False)):
        _, activity = fs.run(fs.edge_clip(on=on), fs.STIMULUS_FPS, network)
        result[name] = {
            "T4": fs.mean_depolarization(activity, fs.T4_TYPES),
            "T5": fs.mean_depolarization(activity, fs.T5_TYPES),
            "T4_abs": fs.mean_abs_change(activity, fs.T4_TYPES),
            "T5_abs": fs.mean_abs_change(activity, fs.T5_TYPES),
        }
    return result


def selectivity_passed(result: dict) -> bool:
    return result["ON"]["T4"] > result["ON"]["T5"] and result["OFF"]["T5"] > result["OFF"]["T4"]


def selectivity_note(result: dict, network_name: str) -> str:
    on, off = result["ON"], result["OFF"]
    unsigned_agrees = on["T4_abs"] > on["T5_abs"] and off["T5_abs"] > off["T4_abs"]
    return (
        f"network {network_name}: ON edge T4 {on['T4']:.3f} vs T5 {on['T5']:.3f}; "
        f"OFF edge T5 {off['T5']:.3f} vs T4 {off['T4']:.3f} (mean depolarization, arbitrary units). "
        f"Counting falls as well as rises (mean |change|), the same ordering "
        f"{'holds' if unsigned_agrees else 'does NOT hold'}: ON T4 {on['T4_abs']:.3f} vs T5 {on['T5_abs']:.3f}, "
        f"OFF T5 {off['T5_abs']:.3f} vs T4 {off['T4_abs']:.3f}"
    )


def plot_selectivity(result: dict, title: str):
    fig, ax = plt.subplots(figsize=(5, 3.6))
    positions = np.arange(2)
    ax.bar(positions - 0.2, [result["ON"]["T4"], result["OFF"]["T4"]], width=0.4, label="T4a-d")
    ax.bar(positions + 0.2, [result["ON"]["T5"], result["OFF"]["T5"]], width=0.4, label="T5a-d")
    ax.set_xticks(positions, ["ON edge (bright on dark)", "OFF edge (dark on bright)"])
    ax.set_ylabel("mean depolarization (a.u.)")
    ax.set_title(title)
    ax.legend(frameon=False)
    return fig


# -------------------------------------------------------------------- checks

def check_uniform_gray(root: Path, network) -> dict:
    """1. Uniform gray stays flat."""
    _, gray = fs.run(fs.gray_clip(1.0), fs.STIMULUS_FPS, network)
    _, edge = fs.run(fs.edge_clip(on=True), fs.STIMULUS_FPS, network)
    ratios = {t: float(fs.central(gray, t)[SETTLE_STEPS:].std()) / max(edge_range(edge, t), 1e-12) for t in TRACE_TYPES}
    worst_type, worst = max(ratios.items(), key=lambda item: item[1])

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for cell_type in TRACE_TYPES:
        trace = fs.central(gray, cell_type)
        ax.plot(time_axis(len(trace)), trace, label=cell_type)
    ax.set(xlabel="time (s)", ylabel="central-column voltage (a.u.)", title="Uniform gray (0.5): activity stays flat")
    ax.legend(frameon=False, ncol=len(TRACE_TYPES), fontsize=8)
    save_figure(fig, root, "uniform_gray")
    note = f"largest std after the first 100 ms is {worst:.2%} of that cell's edge-response range ({worst_type}); limit {FLAT_TOLERANCE:.0%}"
    return record(root, "uniform_gray", worst < FLAT_TOLERANCE, note)


def tracking_clip(root: Path) -> tuple[np.ndarray, float, str]:
    """First seconds of clips/test_main.npz, or a random-walk clip until it exists."""
    path = hr.resolve(root, "clips/test_main.npz")
    if not path.exists():
        return fs.random_walk_clip(TRACKING_SECONDS), 30.0, "random-walk clip (test_main.npz not uploaded yet)"
    frames, fps = hr.load_clip(path)
    return frames[: int(round(TRACKING_SECONDS * fps))], fps, "clips/test_main.npz"


def check_photoreceptors_track(root: Path, network) -> dict:
    """2. Photoreceptors track the input."""
    frames, fps, source = tracking_clip(root)
    movie, activity = fs.run(frames, fps, network)
    r1 = fs.all_columns(activity, "R1")
    columns = np.argsort(movie.std(axis=0))[-TRACKING_COLUMNS:]  # columns whose brightness changes most
    fits = [best_lagged_correlation(movie[:, c], r1[:, c]) for c in columns]
    correlations = np.array([r for r, _ in fits])
    lags_ms = [lag * hr.DT * 1000 for _, lag in fits]
    median_abs = float(np.median(np.abs(correlations)))
    signs = set(np.sign(correlations))
    sign = "inverted (R1 falls when brightness rises)" if signs == {-1.0} else "same sign as brightness" if signs == {1.0} else "mixed sign"

    fig, ax = plt.subplots(figsize=(6, 3.6))
    column = columns[-1]
    normalise = lambda trace: (trace - trace.mean()) / max(trace.std(), 1e-12)
    ax.plot(time_axis(len(movie)), normalise(movie[:, column]), label="rendered brightness")
    ax.plot(time_axis(len(movie)), normalise(r1[:, column]) * np.sign(correlations[-1]), label="R1 voltage (sign-aligned)")
    ax.set(xlabel="time (s)", ylabel="z-score", title=f"R1 tracks the input (column {column}, r = {correlations[-1]:+.2f})")
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, root, "photoreceptors_track")
    note = (
        f"{source}: median |r| = {median_abs:.2f} over {TRACKING_COLUMNS} columns, {sign}, "
        f"lag {min(lags_ms):.0f}-{max(lags_ms):.0f} ms; limit |r| >= {TRACKING_MIN_ABS_R}"
    )
    return record(root, "photoreceptors_track", median_abs >= TRACKING_MIN_ABS_R, note)


def check_on_off_edges(root: Path, network, network_name: str = fs.MAIN_NETWORK, check: str = "on_off_edges") -> dict:
    """3. ON edges drive T4, OFF edges drive T5 (direction preference is NOT tested)."""
    result = edge_selectivity(network)
    save_figure(plot_selectivity(result, f"ON edges drive T4, OFF edges drive T5\n({network_name})"), root, check)
    return record(root, check, selectivity_passed(result), selectivity_note(result, network_name))


def check_activity_settles(root: Path, network) -> dict:
    """4. Activity returns to baseline after 1 s of gray."""
    clip = np.concatenate([fs.edge_clip(on=True), fs.gray_clip(SETTLE_GRAY_SECONDS)])
    _, activity = fs.run(clip, fs.STIMULUS_FPS, network)
    _, gray = fs.run(fs.gray_clip(1.0), fs.STIMULUS_FPS, network)
    edge_steps = int(len(fs.edge_clip()) / fs.STIMULUS_FPS / hr.DT)
    ratios = {}
    for cell_type in TRACE_TYPES:
        trace = fs.central(activity, cell_type)
        baseline = fs.central(gray, cell_type)[-SETTLE_STEPS:].mean()
        ratios[cell_type] = abs(trace[-SETTLE_STEPS:].mean() - baseline) / max(np.ptp(trace[:edge_steps]), 1e-12)
    worst_type, worst = max(ratios.items(), key=lambda item: item[1])

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for cell_type in TRACE_TYPES:
        trace = fs.central(activity, cell_type)
        ax.plot(time_axis(len(trace)), trace - fs.central(gray, cell_type)[-SETTLE_STEPS:].mean(), label=cell_type)
    ax.axvline(edge_steps * hr.DT, color="0.5", linestyle="--", linewidth=1)
    ax.set(xlabel="time (s)", ylabel="voltage minus gray baseline (a.u.)", title="Edge, then 1 s of gray: activity settles")
    ax.legend(frameon=False, ncol=len(TRACE_TYPES), fontsize=8)
    save_figure(fig, root, "activity_settles")
    note = f"last 100 ms is within {worst:.2%} of the gray baseline, relative to the edge response ({worst_type} is furthest); limit {SETTLED_TOLERANCE:.0%}"
    return record(root, "activity_settles", worst < SETTLED_TOLERANCE, note)


def check_second_network(root: Path, network=None) -> dict:
    """5. A second network agrees (check 3 repeated with flow/0000/001)."""
    network = network if network is not None else fs.load_network(fs.SECOND_NETWORK)
    return check_on_off_edges(root, network, fs.SECOND_NETWORK, check="second_network")


CHECKS = {
    "uniform_gray": check_uniform_gray,
    "photoreceptors_track": check_photoreceptors_track,
    "on_off_edges": check_on_off_edges,
    "activity_settles": check_activity_settles,
    "second_network": check_second_network,
}


def run_checks(root: Path, names: list[str]) -> list[dict]:
    needs_main = any(name != "second_network" for name in names)
    network = fs.load_network(fs.MAIN_NETWORK) if needs_main else None
    return [CHECKS[name](root) if name == "second_network" else CHECKS[name](root, network) for name in names]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."), help="fly_sees folder (default: current dir)")
    parser.add_argument("--check", choices=["all", *CHECKS], default="all")
    args = parser.parse_args(argv)
    names = list(CHECKS) if args.check == "all" else [args.check]
    try:
        results = run_checks(args.root, names)
    except (fs.NetworkUnavailable, FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    passed = sum(result["passed"] for result in results)
    print(f"{passed}/{len(results)} sanity checks pass" + ("" if passed == len(results) else " -- tell A: likely an input scaling or orientation bug"))
    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
