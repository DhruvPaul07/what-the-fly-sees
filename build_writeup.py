"""Plan C Parts 4-5: build the write-up from the team's result files and print
the submission checklist.

    python build_writeup.py --root /content/drive/MyDrive/fly_sees \
        --team "A=Name, B=Name, C=Name" --notebook "A=https://..." "B=https://..."

Reads (all optional; whatever is missing is marked as not available, never guessed)
    results/scores.csv, results/checks.json                 from A
    results/sanity/sanity.json, results/sanity/*.png        from sanity_checks.py
    results/demo_payload.npz, demo_payload_still.npz        from A (via B's contract)
    results/ai_enhance_metrics.json, demo/*                 from B
Writes <root>/WRITEUP.md with links relative to the fly_sees folder.

No number is invented: table cells come straight from scores.csv and stay
blank when the file has no value.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

import hex_render as hr

PITCH = (
    "What the fly sees: what visual information survives in a connectome-constrained "
    "model of the fruit fly's motion-vision pathway."
)
SCORE_COLUMNS = ("train_score", "test_footage_score", "test_synthetic_score")
SCORE_NOTE = "Score = per-frame correlation across the 721 columns, averaged over frames."
PRELIMINARY_LABEL = "preliminary (synthetic-trained)"
FAKE_LABEL = "layout test only (fake payload, not model output)"
NOT_AVAILABLE = "_not available yet_"
PRIMARY_NETWORK = "000"
BANNED_PHRASES = ("we reconstructed what the fly sees", "first ever")
# Still scene counts as quiet if its motion reconstruction changes this much less than the moving clip's.
STILL_QUIET_RATIO = 0.5

# Predicates on the normalised (lower-case, letters and digits only) group name.
CONTROL_MATCHERS = (
    ("shuffled", lambda k: "shuf" in k),
    ("baseline", lambda k: k.startswith("mean") or "baseline" in k),
)
DECODER_MATCHERS = (
    ("photoreceptors", lambda k: "photo" in k or k.startswith("r1")),
    ("lamina", lambda k: "lamina" in k or k.startswith("l1")),
    ("medulla", lambda k: "medulla" in k or k.startswith(("mi", "tm"))),
    ("motion", lambda k: "t4" in k or "t5" in k or "motion" in k),
)
LABELS = {
    "photoreceptors": "Photoreceptors (R1–R6)",
    "lamina": "Lamina (L1–L5)",
    "medulla": "Medulla (Mi, Tm)",
    "motion": "Motion detectors (T4/T5)",
    "shuffled": "Shuffled control",
    "baseline": "Mean-brightness baseline",
}
TABLE_ORDER = ("photoreceptors", "lamina", "medulla", "motion", "shuffled", "baseline")
CONTROL_KEYS = ("shuffled", "baseline")

SANITY_ROWS = (
    ("uniform_gray", "Uniform gray stays flat"),
    ("photoreceptors_track", "Photoreceptors track the input"),
    ("on_off_edges", "ON edges drive T4, OFF edges drive T5"),
    ("activity_settles", "Activity settles after the stimulus"),
    ("second_network", "A second network agrees"),
)
SANITY_FIGURES = ("results/sanity/on_off_edges.png", "results/sanity/photoreceptors_track.png")
LOOMING_FIGURE = "results/sanity/looming_vs_sliding.png"
LIMITS = (
    "It isn't the fly's experience: the reconstruction is what *our decoder* can recover from "
    "*simulated* neurons; a weak decoder can miss information that is there.",
    "The model was tuned for motion, so fading still detail is expected, not a bug.",
    "Every column is identical (a local reconstruction repeated on a regular lattice); real eyes vary.",
    "One flat, grayscale eye: no color, curvature, or second eye.",
    "Outputs are model voltages in arbitrary units, not spikes or real recordings.",
)
METHOD = (
    "Phone video → 391 px grayscale, centered square crop.",
    "→ 721-column hexagonal fly eye (flyvis `BoxEye`).",
    "→ pretrained flyvis network (45,669 neurons, 64 cell types; parameters fit to an optic-flow task, "
    "Lappalainen et al., Nature 2024).",
    "→ linear ridge decoder shared across columns, using 50 ms of history per cell-type group.",
    "→ reconstruction of the eye's input, decoded from simulated neurons.",
)
ROLES = (("A", "Pipeline"), ("B", "Video + Demo"), ("C", "Validation + Submission"))
CHECKLIST = (
    ("demo mp4", "demo/what_the_fly_sees.mp4"),
    ("demo gif", "demo/what_the_fly_sees.gif"),
    ("coarseness image", "demo/coarseness.png"),
    ("still-scene video", "demo/still_scene.mp4"),
    ("scores", "results/scores.csv"),
    ("sanity results", "results/sanity/sanity.json"),
    ("sanity figure", SANITY_FIGURES[0]),
)


# ------------------------------------------------------------------- scores

def normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def group_key(name: str) -> str | None:
    """Table key for a group name from scores.csv, or None if it is not clearly one known group.

    Controls win ("shuffled_T4T5" is a control). A name that fits several decoder
    groups ("L1-L5+Mi+Tm") is left unknown rather than guessed.
    """
    key = normalise(name)
    control = next((group for group, matches in CONTROL_MATCHERS if matches(key)), None)
    if control:
        return control
    fits = [group for group, matches in DECODER_MATCHERS if matches(key)]
    return fits[0] if len(fits) == 1 else None


def network_id(value) -> str:
    """"flow/0000/001", "network_001", "001", 1, 1.0 -> "001"; blank or NaN -> "000" (the main network)."""
    text = str(value if value is not None else "").strip().rstrip("/")
    if text.lower() in ("", "nan", "none"):
        return PRIMARY_NETWORK
    digits = re.search(r"(\d+)(?:\.0+)?\D*$", text)
    return str(int(digits.group(1))).zfill(3) if digits else text


def parse_score(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mark_ambiguous(rows: list[dict]) -> list[dict]:
    """Rows that share a table row within one network keep their own names instead of overwriting each other."""
    counts: dict[tuple, int] = {}
    for row in rows:
        counts[(row["network"], row["key"])] = counts.get((row["network"], row["key"]), 0) + 1
    marked = []
    for row in rows:
        clash = row["key"] is not None and counts[(row["network"], row["key"])] > 1
        if clash:
            print(f"WARNING: scores.csv has several rows for {LABELS[row['key']]} in network {row['network']}; showing '{row['group']}' under its own name", file=sys.stderr)
        marked.append({**row, "key": None} if clash else row)
    return marked


def read_scores(path: Path) -> list[dict]:
    """scores.csv -> rows {group, key, network, train_score, test_footage_score, test_synthetic_score}."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as handle:  # utf-8-sig drops an Excel/Sheets BOM
        raw_rows = list(csv.DictReader(handle))
    cleaned_rows = [{(k or "").strip().lower(): v for k, v in raw.items()} for raw in raw_rows]
    if cleaned_rows and "group" not in cleaned_rows[0]:
        raise ValueError(f"{path} has no 'group' column; found {list(cleaned_rows[0])}")
    rows = []
    for cleaned in cleaned_rows:
        group = (cleaned["group"] or "").strip()
        if not group:
            continue
        row = {"group": group, "key": group_key(group), "network": network_id(cleaned.get("network"))}
        row.update({column: parse_score(cleaned.get(column)) for column in SCORE_COLUMNS})
        rows.append(row)
    return mark_ambiguous(rows)


def primary_network(rows: list[dict]) -> str:
    """The main network (000), or the first network listed if 000 is absent."""
    networks = [row["network"] for row in rows]
    return PRIMARY_NETWORK if PRIMARY_NETWORK in networks or not networks else networks[0]


def primary_rows(rows: list[dict]) -> list[dict]:
    chosen = primary_network(rows)
    return [row for row in rows if row["network"] == chosen]


def row_label(row: dict) -> str:
    return LABELS[row["key"]] if row["key"] else row["group"]


def decoder_rows(rows: list[dict]) -> list[dict]:
    """Every row that is not a control, including groups we do not recognise."""
    return [row for row in rows if row["key"] not in CONTROL_KEYS]


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def spread(rows: list[dict], row: dict) -> str:
    """min–max of the footage test score across networks for the group shown in `row` (one value per network)."""
    same_group = lambda other: other["key"] == row["key"] if row["key"] else other["key"] is None and other["group"] == row["group"]
    per_network: dict[str, float] = {}
    for other in rows:
        if same_group(other) and other["test_footage_score"] is not None:
            per_network.setdefault(other["network"], other["test_footage_score"])
    values = list(per_network.values())
    return f"{min(values):.3f}–{max(values):.3f}" if len(values) > 1 else ""


def results_table(rows: list[dict]) -> str:
    main = primary_rows(rows)
    known = {row["key"]: row for row in main if row["key"]}
    several_networks = len({row["network"] for row in rows}) > 1
    header = ["Decoder group", "Train", "Test (our footage)", "Test (synthetic)"] + (["Networks min–max (our footage)"] if several_networks else [])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    listed = [(LABELS[key], known.get(key)) for key in TABLE_ORDER]
    listed += [(row["group"].replace("|", "\\|"), row) for row in main if row["key"] is None]  # unrecognised groups are still shown
    for label, row in listed:
        cells = [fmt(row[column]) if row else "" for column in SCORE_COLUMNS]
        if several_networks:
            cells.append(spread(rows, row) if row else "")
        lines.append("| " + " | ".join([label, *cells]) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------- honesty checks

REPORTED = "reported"  # a check that states numbers rather than passing or failing


def verdict_line(name: str, passed: bool | str | None, note: str) -> str:
    status = "not available yet" if passed is None else passed if isinstance(passed, str) else "PASS" if passed else "FAIL"
    return f"- **{name}: {status}.** {note}".rstrip()


def check_train_test_gap(main: list[dict]) -> tuple[str | None, str]:
    pairs = [
        (row_label(row), row["train_score"], row["test_footage_score"])
        for row in decoder_rows(main)
        if row["train_score"] is not None and row["test_footage_score"] is not None
    ]
    if not pairs:
        return None, "Needs train and footage-test scores in `results/scores.csv`."
    label, train, test = max(pairs, key=lambda item: item[1] - item[2])
    return REPORTED, f"Train and test scores are in the table for every group reported; the largest gap is {label} (train {fmt(train)}, test {fmt(test)})."


def check_beats_controls(main: list[dict]) -> tuple[bool | None, str]:
    controls = [(LABELS[row["key"]].lower(), row["test_footage_score"]) for row in main if row["key"] in CONTROL_KEYS and row["test_footage_score"] is not None]
    groups = [(row_label(row), row["test_footage_score"]) for row in decoder_rows(main) if row["test_footage_score"] is not None]
    if not controls or not groups:
        return None, "Needs footage-test scores for the groups and at least one control."
    best_control = max(score for _, score in controls)
    losers = [label for label, score in groups if score <= best_control]
    if losers:
        return False, f"These groups do not beat the best control on our footage: {', '.join(losers)}."
    names = " and the ".join(name for name, _ in controls)
    only = " (the only control reported)" if len(controls) == 1 else ""
    return True, f"Every decoder group reported scores above the {names}{only} on our footage."


def check_none_beats_photoreceptors(main: list[dict]) -> tuple[bool | None, str]:
    reference = next((row["test_footage_score"] for row in main if row["key"] == "photoreceptors"), None)
    groups = [(row_label(row), row["test_footage_score"]) for row in decoder_rows(main) if row["key"] != "photoreceptors" and row["test_footage_score"] is not None]
    if reference is None or not groups:
        return None, "Needs footage-test scores for photoreceptors and at least one other group."
    winners = [label for label, score in groups if score > reference]
    if winners:
        return False, f"These groups score above the photoreceptors, which should not happen: {', '.join(winners)}."
    return True, "No other group reported scores above the photoreceptors on our footage."


def npz_flag(data, key: str) -> bool:
    return key in data.files and bool(np.any(data[key]))


def payload_status(root: Path) -> str | None:
    """"fake" (B's layout-test payload), "placeholder" (A's synthetic-trained v1) or None (final / absent)."""
    path = hr.resolve(root, "results/demo_payload.npz")
    if not path.exists():
        return None
    with np.load(path) as data:
        return "fake" if npz_flag(data, "is_fake") else "placeholder" if npz_flag(data, "is_placeholder") else None


def temporal_change(recon: np.ndarray) -> float:
    """How much a reconstruction changes from step to step, on average."""
    return float(np.abs(np.diff(np.asarray(recon, dtype=np.float32), axis=0)).mean())


def check_still_scene(root: Path) -> tuple[bool | None, str]:
    """Motion reconstruction of the still scene must change less than half as much over time as the moving clip's."""
    paths = [hr.resolve(root, name) for name in ("results/demo_payload.npz", "results/demo_payload_still.npz")]
    if not all(path.exists() for path in paths):
        return None, "Needs `results/demo_payload.npz` and `results/demo_payload_still.npz`."
    changes = []
    for path in paths:
        with np.load(path) as data:
            if npz_flag(data, "is_fake"):
                return None, f"`{path.name}` is a layout-test fake, not model output."
            if "recon_motion" not in data.files or len(data["recon_motion"]) < 2:
                return None, f"`{path.name}` has no usable `recon_motion`."
            changes.append(temporal_change(data["recon_motion"]))
    moving, still = changes
    print(f"still-scene check: motion reconstruction changes {still:.5f} per step (still) vs {moving:.5f} (moving)")
    if not moving > 0:
        return None, "The moving clip's motion reconstruction does not change over time, so there is nothing to compare."
    if still < STILL_QUIET_RATIO * moving:
        return True, "The motion-decoded reconstruction of the still scene changes less than half as much over time as that of the moving clip."
    return False, "The motion-decoded reconstruction of the still scene does NOT stay quiet compared with the moving clip."


def check_networks_agree(rows: list[dict], sanity: dict) -> tuple[bool | str | None, str]:
    """PASS/FAIL comes only from the second-network sanity check; decoder scores are reported, not judged."""
    second = sanity.get("second_network")
    networks = sorted({row["network"] for row in rows})
    notes = []
    if second is not None:
        notes.append("A second network shows the same ON/OFF edge preference." if second["passed"] else "A second network does NOT show the same ON/OFF edge preference.")
    if len(networks) > 1:
        notes.append(f"Decoder scores for networks {', '.join(networks)} are in the table (min–max column); their agreement is reported there, not judged here.")
    if second is not None:
        return bool(second["passed"]), " ".join(notes)
    if notes:
        return REPORTED, " ".join(notes)
    return None, "Needs the second-network sanity check or scores from more than one network."


def honesty_section(root: Path, rows: list[dict], sanity: dict) -> str:
    main = primary_rows(rows)
    checks = (
        ("Train/test gap", check_train_test_gap(main)),
        ("Every group beats the controls", check_beats_controls(main)),
        ("No group beats photoreceptors", check_none_beats_photoreceptors(main)),
        ("Still scene stays quiet", check_still_scene(root)),
        ("Networks agree", check_networks_agree(rows, sanity)),
    )
    return "\n".join(verdict_line(name, passed, note) for name, (passed, note) in checks)


def as_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return True if text in ("true", "pass", "passed", "1") else False if text in ("false", "fail", "failed", "0") else None


def parse_check(name, value) -> tuple[str, bool | None, str]:
    if isinstance(value, dict):
        passed = as_bool(next((value[k] for k in ("passed", "pass", "ok") if k in value), None))
        note = next((value[k] for k in ("note", "detail", "message") if k in value), "")
    else:
        passed = None if value is None else as_bool(value)
        note = "" if value is None or passed is not None else value
    return str(name), passed, " ".join(str(note).split())  # one line: a newline would break the bullet


def pipeline_checks(path: Path) -> list[tuple[str, bool | None, str]]:
    """A's results/checks.json in whatever reasonable shape -> (name, passed, note)."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        print(f"WARNING: could not read {path} ({error}); leaving the pipeline checks out", file=sys.stderr)
        return []
    if isinstance(data, dict) and len(data) == 1 and isinstance(next(iter(data.values())), list):
        data = next(iter(data.values()))  # {"checks": [...]}
    if isinstance(data, dict) and any(k in data for k in ("passed", "pass")) and not any(isinstance(v, dict) for v in data.values()):
        data = [data]  # one flat check
    if isinstance(data, dict):
        items = list(data.items())
    elif isinstance(data, list):
        items = [(item.get("check", item.get("name", f"check {i + 1}")), item) if isinstance(item, dict) else (item, None) for i, item in enumerate(data)]
    else:
        items = []
    if not items:
        print(f"WARNING: {path} holds no checks in a shape this script understands; leaving them out", file=sys.stderr)
    return [parse_check(name, value) for name, value in items]


# ------------------------------------------------------------------ sections

def read_sanity(path: Path) -> dict:
    if not path.exists():
        return {}
    return {item["check"]: item for item in json.loads(path.read_text()) if "check" in item}


def sanity_section(root: Path, sanity: dict) -> str:
    lines = ["| Check | Result | Note |", "|---|---|---|"]
    for key, label in SANITY_ROWS:
        item = sanity.get(key)
        result = "" if item is None else "pass" if item["passed"] else "FAIL"
        note = str(item["note"]).replace("|", "\\|") if item else ""  # a bare | would split the table cell
        lines.append(f"| {label} | {result} | {note} |")
    if sanity:
        passed = sum(sanity[key]["passed"] for key, _ in SANITY_ROWS if key in sanity)
        lines.append(f"\n{passed} of {len(SANITY_ROWS)} checks pass. These test the input pipeline, not the idea.")
    else:
        lines.append(f"\n{NOT_AVAILABLE} (run `sanity_checks.py`).")
    lines += [f"\n![sanity figure]({figure})" for figure in SANITY_FIGURES if hr.resolve(root, figure).exists()]
    if hr.resolve(root, LOOMING_FIGURE).exists():
        lines.append(f"\n![looming vs sliding]({LOOMING_FIGURE})\n\nMotion-pathway responses only; escape behavior not modeled. The looming disc also grows in area, so its larger summed response partly reflects stimulus size.")
    return "\n".join(lines)


def demo_section(root: Path, status_label: str | None) -> str:
    exists = lambda name: hr.resolve(root, name).exists()
    lines = []
    if status_label:
        lines.append(f"**Demo status: {status_label}.**\n")
    if exists("demo/what_the_fly_sees.gif"):
        lines.append("![What the fly sees](demo/what_the_fly_sees.gif)\n")
    lines.append("[Full-resolution video (mp4)](demo/what_the_fly_sees.mp4)" if exists("demo/what_the_fly_sees.mp4") else f"Demo video: {NOT_AVAILABLE}")
    lines.append("\nPanels: original clip | fly's-eye input (721 columns) | decoded from photoreceptors (R1–R6) | decoded from motion detectors (T4/T5). The reconstructions are decoded from simulated neurons (flyvis connectome model), not what the fly experiences.")
    if exists("demo/coarseness.png"):
        lines.append("\n![721 samples vs 152,881 pixels](demo/coarseness.png)\n\n721 samples vs 152,881 pixels.")
    if exists("demo/still_scene.mp4"):
        lines.append("\n[Still-scene control video](demo/still_scene.mp4): the same layout for a scene with no motion.")
    return "\n".join(lines)


def ai_section(root: Path) -> str:
    """The two lines Plan B2 asks C to include, filled from B's metrics file."""
    path = hr.resolve(root, "results/ai_enhance_metrics.json")
    if not path.exists():
        return ""
    delta = json.loads(path.read_text())["difference_ai_minus_bicubic"]
    lines = [
        "## AI upscaling test",
        "",
        f"- We also tested a pretrained AI super-resolution model (Real-ESRGAN) on the fly's-eye image. Compared with plain bicubic upscaling, it changed similarity to the real scene by {delta['ssim']:+.3f} SSIM and {delta['pearson']:+.3f} correlation.",
        "- Any sharper detail in the AI panel comes from the AI's training on natural photos, not from the fly's 721 samples or the simulated neurons.",
    ]
    if hr.resolve(root, "demo/ai_enhanced.mp4").exists():
        lines.append("\n[AI upscaling video](demo/ai_enhanced.mp4)")
    return "\n".join(lines) + "\n"


def parse_pairs(text_items: list[str]) -> dict[str, str]:
    """["A=Ann, B=Bo"] or ["A=Ann", "B=Bo"] -> {"A": "Ann", "B": "Bo"}."""
    pairs = {}
    for item in text_items:
        for part in re.split(r",(?=\s*[A-Za-z]\s*=)", item):
            key, _, value = part.partition("=")
            value = value.strip()
            if value and not (value.startswith("<") and value.endswith(">")):  # skip unedited <name> placeholders
                pairs[key.strip().upper()] = value
    return pairs


def credits_section(team: dict[str, str], notebooks: dict[str, str]) -> str:
    lines = [
        "- flyvis (TuragaLab, github.com/TuragaLab/flyvis); Lappalainen et al., Nature 2024.",
        "- Team: " + "; ".join(f"{team[role]} ({title})" if role in team else f"{role} ({title})" for role, title in ROLES) + ".",
    ]
    lines += [f"- Colab notebook {role}: {notebooks[role]}" for role, _ in ROLES if role in notebooks]
    return "\n".join(lines)


def build_writeup(root: Path, team: dict[str, str], notebooks: dict[str, str], preliminary: bool) -> str:
    rows = read_scores(hr.resolve(root, "results/scores.csv"))
    sanity = read_sanity(hr.resolve(root, "results/sanity/sanity.json"))
    status = payload_status(root)
    preliminary = preliminary or status == "placeholder"
    status_label = FAKE_LABEL if status == "fake" else PRELIMINARY_LABEL if preliminary else None
    reported = pipeline_checks(hr.resolve(root, "results/checks.json"))
    table_note = SCORE_NOTE if rows else f"{SCORE_NOTE} Scores: {NOT_AVAILABLE} (cells stay blank until `results/scores.csv` arrives)."
    if rows and preliminary:
        table_note += f" These numbers are **{PRELIMINARY_LABEL}**."
    if rows and primary_network(rows) != PRIMARY_NETWORK:
        table_note += f" The rows show network {primary_network(rows)}; network {PRIMARY_NETWORK} is not in `results/scores.csv`."
    parts = [
        "# What the Fly Sees", "", PITCH, "",
        "## Demo", "", demo_section(root, status_label), "",
        "## Method", "", "\n".join(f"- {step}" for step in METHOD), "",
        "## Results", "", results_table(rows), "", table_note, "",
        "## Honesty checks", "", honesty_section(root, rows, sanity), "",
    ]
    if reported:
        parts += ["Reported by the pipeline (`results/checks.json`):", "", "\n".join(verdict_line(*check) for check in reported), ""]
    parts += [
        "## Sanity checks", "", sanity_section(root, sanity), "",
        ai_section(root),
        "## Limits", "", "\n".join(f"- {limit}" for limit in LIMITS), "",
        "## Credits", "", credits_section(team, notebooks), "",
    ]
    return "\n".join(parts)


def banned_phrases_in(text: str) -> list[str]:
    lowered = text.lower()
    return [phrase for phrase in BANNED_PHRASES if phrase in lowered]


def checklist(root: Path, rows: list[dict], notebooks: dict[str, str], out_path: Path) -> list[tuple[str, bool]]:
    items = [(f"{label} ({name})", hr.resolve(root, name).exists()) for label, name in CHECKLIST]
    filled = any(row["test_footage_score"] is not None for row in primary_rows(rows))
    items.append(("results table filled from scores.csv", filled))
    items.append((f"write-up ({out_path.name})", out_path.exists()))
    items += [(f"Colab notebook link for {role}", role in notebooks) for role, _ in ROLES]
    return items


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."), help="fly_sees folder (default: current dir)")
    parser.add_argument("--out", type=Path, default=Path("WRITEUP.md"), help="output file, relative to --root")
    parser.add_argument("--team", nargs="*", default=[], help='names per role, e.g. "A=Ann, B=Bo, C=Cy"')
    parser.add_argument("--notebook", nargs="*", default=[], help='Colab share links per role, e.g. "A=https://..."')
    parser.add_argument("--preliminary", action="store_true", help=f"label the numbers as {PRELIMINARY_LABEL}")
    args = parser.parse_args(argv)
    team, notebooks = parse_pairs(args.team), parse_pairs(args.notebook)
    try:
        text = build_writeup(args.root, team, notebooks, args.preliminary)
        rows = read_scores(hr.resolve(args.root, "results/scores.csv"))
    except Exception as error:  # A's files can be malformed in many ways; report it rather than a traceback
        print(f"ERROR: could not build the write-up: {error!r}", file=sys.stderr)
        return 1
    banned = banned_phrases_in(text)
    if banned:
        print(f"ERROR: the write-up contains wording the team ruled out: {banned}", file=sys.stderr)
        return 1
    out_path = hr.resolve(args.root, args.out)
    out_path.write_text(text)
    print(f"Wrote {out_path}")
    items = checklist(args.root, rows, notebooks, out_path)
    print("\nSubmission checklist:")
    for label, done in items:
        print(f"  [{'x' if done else ' '}] {label}")
    missing = sum(not done for _, done in items)
    print("Ready to submit." if not missing else f"{missing} item(s) still missing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
