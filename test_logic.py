"""Tests for the pure logic (no flyvis, no video files needed)."""
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
