"""Tests for the pure logic (no flyvis, no video files needed)."""
import json
import numpy as np
import pytest

import hex_render as hr
import make_demo
import make_fake_payload as fake
import prep_clips


def make_payload(n_steps=20, k=5, n_frames=6):
    rng = np.random.default_rng(0)
    return {
        "input": rng.random((n_steps, hr.N_COLUMNS), dtype=np.float32),
        "recon_motion": rng.random((n_steps - k + 1, hr.N_COLUMNS), dtype=np.float32),
        "recon_photo": rng.random((n_steps - k + 1, hr.N_COLUMNS), dtype=np.float32),
        "step_to_frame": np.minimum(np.arange(n_steps) // 3, n_frames - 1),
        "K": np.int64(k),
        "dt": np.float64(0.01),
    }


def test_piece_bounds_drops_short_tail():
    assert prep_clips.piece_bounds(150, 30.0, 2.0) == [(0, 60), (60, 120)]


def test_piece_bounds_empty_when_video_shorter_than_piece():
    assert prep_clips.piece_bounds(59, 30.0, 2.0) == []


def test_trim_keeps_first_three_seconds():
    frames = np.arange(100)
    assert len(prep_clips.trim(frames, 29.97, 3.0)) == 90
    assert len(frames) == 100


def test_trim_raises_on_short_clip():
    with pytest.raises(prep_clips.ClipError, match="too short"):
        prep_clips.trim(np.zeros(80), 30.0, 3.0)


def test_to_square_gray_crops_centre_and_scales():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, 80:560] = 255  # exactly the centred square
    out = prep_clips.to_square_gray(frame)
    assert out.shape == (391, 391) and out.dtype == np.float32
    assert out.min() == pytest.approx(1.0)


def test_training_pieces_numbered_continuously(tmp_path, monkeypatch):
    clip = np.zeros((150, 391, 391), dtype=np.float32)
    monkeypatch.setattr(prep_clips, "read_video", lambda path, max_seconds=None: (clip, 30.0))
    saved, errors = prep_clips.prep_train_clips([tmp_path / "a.mov", tmp_path / "b.mov"], tmp_path)
    assert errors == []
    assert [p.name for p in saved] == [f"train_{i:02d}.npz" for i in range(4)]
    with np.load(saved[-1]) as data:
        assert data["frames"].shape == (60, 391, 391) and data["frames"].dtype == np.float16


def test_bad_training_video_is_skipped_and_numbering_stays_continuous(tmp_path, monkeypatch):
    good = np.zeros((150, 391, 391), dtype=np.float32)
    short = np.zeros((30, 391, 391), dtype=np.float32)
    clips = {"a.mov": good, "short.mov": short, "b.mov": good}
    monkeypatch.setattr(prep_clips, "read_video", lambda path, max_seconds=None: (clips[path.name], 30.0))
    videos = [tmp_path / name for name in clips]
    saved, errors = prep_clips.prep_train_clips(videos, tmp_path)
    assert [p.name for p in saved] == [f"train_{i:02d}.npz" for i in range(4)]
    assert len(errors) == 1 and "short.mov" in errors[0]


def test_main_continues_after_a_short_test_clip(tmp_path, monkeypatch):
    clips = {"main.mov": np.zeros((80, 391, 391), dtype=np.float32), "still.mov": np.zeros((90, 391, 391), dtype=np.float32)}
    monkeypatch.setattr(prep_clips, "read_video", lambda path, max_seconds=None: (clips[path.name], 30.0))
    exit_code = prep_clips.main(["--test-main", "main.mov", "--test-still", "still.mov", "--root", str(tmp_path)])
    assert exit_code == 1
    assert not (tmp_path / "clips" / "test_main.npz").exists()
    assert (tmp_path / "clips" / "test_still.npz").exists()


def test_step_to_frame_matches_contract_and_is_clamped():
    steps = fake.build_step_to_frame(300, 30.0, 90)
    assert steps.shape == (300,) and np.issubdtype(steps.dtype, np.integer)
    assert steps[0] == 0 and steps[4] == 1 and steps.max() == 89
    assert fake.build_step_to_frame(300, 30.0, 50).max() == 49


def test_n_model_steps_three_seconds():
    assert fake.n_model_steps(90, 30.0) == 300


def test_fake_recons_have_contract_shapes_and_range():
    eye_input = np.random.default_rng(1).random((20, hr.N_COLUMNS)).astype(np.float32)
    motion = fake.fake_recon_motion(eye_input, 5)
    photo = fake.fake_recon_photo(eye_input, 5, np.random.default_rng(0))
    for recon in (motion, photo):
        assert recon.shape == (16, hr.N_COLUMNS) and recon.dtype == np.float32
        assert recon.min() >= 0 and recon.max() <= 1


def test_fake_motion_is_flat_for_still_input():
    eye_input = np.full((20, hr.N_COLUMNS), 0.5, dtype=np.float32)
    assert fake.fake_recon_motion(eye_input, 5).max() == 0


@pytest.mark.parametrize(
    "orientation, expected",
    [
        ({"flip_x": False, "flip_y": False, "swap_xy": False}, (1, 2)),
        ({"flip_x": True, "flip_y": False, "swap_xy": False}, (-1, 2)),
        ({"flip_x": False, "flip_y": True, "swap_xy": False}, (1, -2)),
        ({"flip_x": False, "flip_y": False, "swap_xy": True}, (2, 1)),
        ({"flip_x": True, "flip_y": False, "swap_xy": True}, (-2, 1)),
    ],
)
def test_apply_orientation(orientation, expected):
    x, y = np.array([1.0]), np.array([2.0])
    out_x, out_y = hr.apply_orientation(x, y, orientation)
    assert (out_x[0], out_y[0]) == expected
    assert (x[0], y[0]) == (1.0, 2.0)  # inputs untouched


def test_orientation_requires_all_keys():
    with pytest.raises(ValueError, match="swap_xy"):
        hr.apply_orientation([0], [0], {"flip_x": True, "flip_y": False})


def test_orientation_roundtrip(tmp_path):
    orientation = {"flip_x": True, "flip_y": False, "swap_xy": True}
    hr.save_orientation(tmp_path / "o.json", orientation)
    assert hr.load_orientation(tmp_path / "o.json") == orientation


def test_tiling_marker_size_makes_hexagons_touch():
    spacing = 10.0
    size = hr.tiling_marker_size(spacing, overlap=1.0)
    circumradius = 0.5 * np.sqrt(size)
    assert circumradius * np.sqrt(3) / 2 == pytest.approx(spacing / 2)  # inradius


def test_tiling_marker_follows_lattice_direction():
    x = np.array([0.0, 0.0, 1.5])
    y = np.array([0.0, np.sqrt(3), np.sqrt(3) / 2])
    assert hr.tiling_marker(x, y) == "H"
    assert hr.tiling_marker(y, x) == "h"


def test_validate_payload_accepts_contract_payload():
    out = make_demo.validate_payload(make_payload(), n_clip_frames=6)
    assert out["K"] == 5 and not out["is_placeholder"] and not out["is_fake"]


def test_validate_payload_detects_placeholder():
    payload = {**make_payload(), "is_placeholder": np.bool_(True)}
    out = make_demo.validate_payload(payload, 6)
    assert "synthetic clips" in make_demo.figure_title(out, still=False)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda p: p.pop("recon_photo"), "missing keys"),
        (lambda p: p.update(input=p["input"][:, :700]), r"input must be"),
        (lambda p: p.update(recon_motion=p["recon_motion"][:-1]), "recon_motion must be"),
        (lambda p: p.update(step_to_frame=p["step_to_frame"].astype(float)), "integer"),
        (lambda p: p.update(step_to_frame=p["step_to_frame"] + 100), "wrong --clip"),
        (lambda p: p.update(recon_photo=p["recon_photo"] * np.nan), "NaN"),
    ],
)
def test_validate_payload_rejects_bad_payloads(change, message):
    payload = make_payload()
    change(payload)
    with pytest.raises(make_demo.PayloadError, match=message):
        make_demo.validate_payload(payload, 6)


def test_decoder_index_maps_to_model_step():
    assert make_demo.decoder_step(0, 5) == 4
    assert make_demo.animation_indices(296) == list(range(0, 296, 4))


# --- enhance_flyeye (Plan B2) ---

def test_subsample_is_25fps_and_at_most_three_seconds():
    import enhance_flyeye

    indices = enhance_flyeye.subsample_indices(1061, 30.0, 25)
    assert len(indices) == 75 and indices[0] == 0 and indices[-1] == int(74 / 25 * 30)
    short = enhance_flyeye.subsample_indices(30, 30.0, 25)
    assert len(short) == 25 and short.max() <= 29


def test_similarity_is_perfect_for_identical_frames_and_lower_for_noise():
    import enhance_flyeye

    rng = np.random.default_rng(0)
    reference = rng.random((2, 64, 64)).astype(np.float32)
    mask = np.ones((64, 64), dtype=bool)
    same = enhance_flyeye.similarity(reference, reference, mask)
    noisy = enhance_flyeye.similarity(rng.random((2, 64, 64)).astype(np.float32), reference, mask)
    assert same["pearson"] == pytest.approx(1.0) and same["ssim"] == pytest.approx(1.0)
    assert noisy["pearson"] < 0.2 and noisy["ssim"] < 0.2


def test_verdict_says_so_when_ai_scores_lower_than_bicubic():
    import enhance_flyeye

    line = enhance_flyeye.verdict({"ssim": 0.30, "pearson": 0.8}, {"ssim": 0.40, "pearson": 0.9})
    assert "LESS similar" in line and "-0.100" in line


def test_eroded_mask_shrinks_but_keeps_the_centre():
    import enhance_flyeye

    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 5:35] = True
    eroded = enhance_flyeye._eroded(mask, 4)
    assert eroded[20, 20] and not eroded[6, 6] and 0 < eroded.sum() < mask.sum()


def test_flat_frames_do_not_drag_pearson_to_zero():
    import enhance_flyeye

    rng = np.random.default_rng(1)
    reference = rng.random((2, 64, 64)).astype(np.float32)
    candidates = reference.copy()
    candidates[1] = 0.0  # e.g. a black frame in a fade
    scores = enhance_flyeye.similarity(candidates, reference, np.ones((64, 64), dtype=bool))
    assert scores["pearson"] == pytest.approx(1.0)


def test_rerun_removes_stale_training_pieces(tmp_path, monkeypatch):
    clip = np.zeros((60, 391, 391), dtype=np.float32)
    monkeypatch.setattr(prep_clips, "read_video", lambda path, max_seconds=None: (clip, 30.0))
    (tmp_path / "train_07.npz").write_bytes(b"old")
    saved, _ = prep_clips.prep_train_clips([tmp_path / "a.mov"], tmp_path)
    assert [p.name for p in saved] == ["train_00.npz"]
    assert sorted(p.name for p in tmp_path.glob("train_*.npz")) == ["train_00.npz"]


# --- build_writeup (Plan C) ---

SCORES_CSV = """group,train_score,test_footage_score,test_synthetic_score,network
R1-R6,0.95,0.93,0.94,flow/0000/000
L1-L5,0.90,0.85,,flow/0000/000
Mi_Tm,0.80,0.70,0.75,flow/0000/000
T4_T5,0.60,0.45,0.50,flow/0000/000
shuffled,0.05,0.02,0.03,flow/0000/000
mean_brightness,0.10,0.08,0.09,flow/0000/000
T4_T5,0.58,0.41,0.47,flow/0000/001
"""


def write_scores(tmp_path, text=SCORES_CSV):
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results" / "scores.csv").write_text(text)
    import build_writeup

    return build_writeup, build_writeup.read_scores(tmp_path / "results" / "scores.csv")


@pytest.mark.parametrize("name,key", [
    ("R1-R6", "photoreceptors"), ("photoreceptors", "photoreceptors"), ("Lamina (L1-L5)", "lamina"),
    ("lamina", "lamina"), ("Mi_Tm", "medulla"), ("medulla", "medulla"), ("T4/T5", "motion"),
    ("motion", "motion"), ("shuffled_control", "shuffled"), ("mean_brightness", "baseline"), ("LPTC", None),
])
def test_group_names_from_scores_csv_map_to_table_rows(name, key):
    import build_writeup

    assert build_writeup.group_key(name) == key


def test_results_table_leaves_missing_scores_blank_and_adds_network_spread(tmp_path):
    bw, rows = write_scores(tmp_path)
    table = bw.results_table(rows)
    assert "| Lamina (L1–L5) | 0.900 | 0.850 |  |  |" in table
    assert "| Motion detectors (T4/T5) | 0.600 | 0.450 | 0.500 | 0.410–0.450 |" in table


def test_results_table_is_all_blank_without_scores(tmp_path):
    import build_writeup

    table = build_writeup.results_table(build_writeup.read_scores(tmp_path / "missing.csv"))
    assert "| Photoreceptors (R1–R6) |  |  |  |" in table and not any(ch.isdigit() for ch in table.split("---|")[-1].replace("R1–R6", "").replace("L1–L5", "").replace("T4/T5", ""))


def test_honesty_checks_name_groups_that_fail(tmp_path):
    bad = SCORES_CSV.replace("T4_T5,0.60,0.45", "T4_T5,0.60,0.01").replace("L1-L5,0.90,0.85", "L1-L5,0.90,0.97")
    bw, rows = write_scores(tmp_path, bad)
    main = bw.primary_rows(rows)
    beats, beats_note = bw.check_beats_controls(main)
    photo, photo_note = bw.check_none_beats_photoreceptors(main)
    assert beats is False and "Motion detectors" in beats_note
    assert photo is False and "Lamina" in photo_note


def test_honesty_checks_are_unavailable_not_passed_without_scores(tmp_path):
    import build_writeup as bw

    assert bw.check_beats_controls([])[0] is None
    assert bw.check_none_beats_photoreceptors([])[0] is None
    assert bw.check_still_scene(tmp_path)[0] is None


def test_still_scene_check_uses_the_demo_payloads(tmp_path):
    import build_writeup as bw

    (tmp_path / "results").mkdir()
    rng = np.random.default_rng(0)
    np.savez(tmp_path / "results" / "demo_payload.npz", recon_motion=rng.random((20, 721)).astype(np.float32))
    np.savez(tmp_path / "results" / "demo_payload_still.npz", recon_motion=np.full((20, 721), 0.5, np.float32))
    assert bw.check_still_scene(tmp_path)[0] is True
    np.savez(tmp_path / "results" / "demo_payload_still.npz", recon_motion=rng.random((20, 721)).astype(np.float32))
    assert bw.check_still_scene(tmp_path)[0] is False


def test_writeup_builds_with_nothing_available_and_avoids_banned_wording(tmp_path):
    import build_writeup as bw

    text = bw.build_writeup(tmp_path, {}, {}, preliminary=False)
    assert "not available yet" in text and bw.banned_phrases_in(text) == []
    assert all(limit in text for limit in bw.LIMITS)


def test_placeholder_payload_marks_writeup_preliminary(tmp_path):
    bw, _ = write_scores(tmp_path)
    np.savez(tmp_path / "results" / "demo_payload.npz", recon_motion=np.zeros((5, 721), np.float32), is_placeholder=True)
    assert "preliminary (synthetic-trained)" in bw.build_writeup(tmp_path, {}, {}, preliminary=False)


def test_sanity_notes_cannot_break_the_markdown_table(tmp_path):
    import build_writeup as bw

    sanity = {"photoreceptors_track": {"check": "photoreceptors_track", "passed": True, "note": "median |r| = 1.00"}}
    assert "median \\|r\\| = 1.00" in bw.sanity_section(tmp_path, sanity)


def test_best_lagged_correlation_finds_the_delay():
    import sanity_checks

    signal = np.sin(np.linspace(0, 12, 200))
    r, lag = sanity_checks.best_lagged_correlation(signal[:-3], -signal[:-3][np.r_[0, 0, 0, 0:194]])
    assert lag == 3 and r == pytest.approx(-1.0, abs=1e-6)


# --- build_writeup against awkward real-world files (second review) ---

def test_scores_csv_with_bom_and_capitalised_headers_is_read(tmp_path):
    text = "﻿Group,Train_Score,Test_Footage_Score,Test_Synthetic_Score,Network\nR1-R6,0.9,0.8,0.7,0\n"
    _, rows = write_scores(tmp_path, text)
    assert rows[0]["key"] == "photoreceptors" and rows[0]["test_footage_score"] == 0.8 and rows[0]["network"] == "000"


@pytest.mark.parametrize("value,expected", [
    ("flow/0000/001", "001"), ("network_000", "000"), ("001", "001"), (1, "001"), ("1.0", "001"),
    ("0.0", "000"), ("", "000"), ("nan", "000"), (None, "000"), ("NaN", "000"),
])
def test_network_labels_are_normalised(value, expected):
    import build_writeup

    assert build_writeup.network_id(value) == expected


def test_main_table_uses_network_000_even_when_listed_last(tmp_path):
    text = "group,train_score,test_footage_score,test_synthetic_score,network\nT4_T5,0.5,0.4,0.4,network_001\nT4_T5,0.6,0.45,0.5,network_000\n"
    bw, rows = write_scores(tmp_path, text)
    assert "| Motion detectors (T4/T5) | 0.600 | 0.450 | 0.500 | 0.400–0.450 |" in bw.results_table(rows)


@pytest.mark.parametrize("name,key", [
    ("R1-R6_mean", "photoreceptors"), ("photo_mean", "photoreceptors"), ("baseline_mean", "baseline"),
    ("mean", "baseline"), ("shuffled_T4T5", "shuffled"), ("T4a-d+T5a-d", "motion"),
    ("L1-L5+Mi+Tm+T4+T5", None), ("Mi_Tm+T4_T5", None), ("lamina+medulla", None), ("all", None),
])
def test_ambiguous_group_names_are_left_unknown_not_guessed(name, key):
    import build_writeup

    assert build_writeup.group_key(name) == key


def test_two_rows_for_one_table_row_keep_their_own_names(tmp_path):
    text = "group,train_score,test_footage_score,test_synthetic_score,network\nMi1,0.8,0.7,0.7,0\nTm3,0.5,0.1,0.1,0\nmean,0.2,0.2,0.2,0\n"
    bw, rows = write_scores(tmp_path, text)
    table = bw.results_table(rows)
    assert "| Medulla (Mi, Tm) |  |  |  |" in table and "| Mi1 | 0.800" in table and "| Tm3 | 0.500" in table
    passed, note = bw.check_beats_controls(bw.primary_rows(rows))
    assert passed is False and "Tm3" in note and "Mi1" not in note


def test_unrecognised_group_still_counts_in_the_honesty_checks(tmp_path):
    text = "group,train_score,test_footage_score,test_synthetic_score,network\nR1-R6,0.95,0.90,0.9,0\nall,0.99,0.95,0.9,0\nshuffled,0.1,0.1,0.1,0\n"
    bw, rows = write_scores(tmp_path, text)
    passed, note = bw.check_none_beats_photoreceptors(bw.primary_rows(rows))
    assert passed is False and "all" in note


def test_single_control_is_not_described_as_both(tmp_path):
    text = "group,train_score,test_footage_score,test_synthetic_score,network\nR1-R6,0.95,0.90,0.9,0\nshuffled,0.1,0.1,0.1,0\n"
    bw, rows = write_scores(tmp_path, text)
    passed, note = bw.check_beats_controls(bw.primary_rows(rows))
    assert passed is True and "both" not in note and "only control reported" in note


def test_networks_agree_never_passes_on_scores_alone(tmp_path):
    bw, rows = write_scores(tmp_path)
    assert bw.check_networks_agree(rows, {})[0] == bw.REPORTED
    assert bw.check_networks_agree(rows, {"second_network": {"passed": False}})[0] is False
    assert bw.check_networks_agree([], {})[0] is None


@pytest.mark.parametrize("content,expected", [
    ("true", []), ("3", []), ("null", []), ("not json", []),
    ('["gap ok"]', [("gap ok", None, "")]),
    ('{"checks": [{"check": "gap", "passed": true, "note": "a\\nb"}]}', [("gap", True, "a b")]),
    ('{"passed": true, "note": "fine"}', [("check 1", True, "fine")]),
    ('{"still": "true", "gap": {"pass": false}}', [("still", True, ""), ("gap", False, "")]),
])
def test_pipeline_checks_survive_odd_json(tmp_path, content, expected):
    import build_writeup

    path = tmp_path / "checks.json"
    path.write_text(content)
    assert build_writeup.pipeline_checks(path) == expected


def test_fake_payload_is_labelled_and_not_scored_as_still_scene(tmp_path):
    import build_writeup as bw

    (tmp_path / "results").mkdir()
    for name in ("demo_payload.npz", "demo_payload_still.npz"):
        np.savez(tmp_path / "results" / name, recon_motion=np.zeros((5, 721), np.float32), is_fake=True)
    assert bw.payload_status(tmp_path) == "fake"
    assert bw.check_still_scene(tmp_path)[0] is None
    assert "layout test only" in bw.build_writeup(tmp_path, {}, {}, preliminary=False)


def test_unedited_placeholders_are_not_printed_as_names():
    import build_writeup

    assert build_writeup.parse_pairs(["A=<name>, B=Bo", "C=<Colab link>"]) == {"B": "Bo"}


def test_corrupt_sanity_json_is_replaced_not_fatal(tmp_path):
    import sanity_checks

    path = tmp_path / "results" / "sanity" / "sanity.json"
    path.parent.mkdir(parents=True)
    path.write_text("{half written")
    sanity_checks.record(tmp_path, "uniform_gray", True, "ok")
    assert json.loads(path.read_text()) == [{"check": "uniform_gray", "passed": True, "note": "ok"}]
