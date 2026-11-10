"""Catching a wrong lens type, or a lens file that does not belong to the camera, on a few
board views before a calibration is started."""
import numpy as np
import pytest

from ds_msp.core.lie import so3_exp
from ds_msp.models.kb import KannalaBrandtModel
from ds_msp.models.radtan import RadTanModel
from ds_msp.studio import advisor

pytestmark = pytest.mark.req("FR-UI-011")
W, H = 1280, 960


def _views(model, n=10, seed=0, spread=0.9, noise=0.05):
    """Board views (6 x 5 corners, 8 cm) of ``model``, tilted and placed over the image."""
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(6) * 0.08, np.arange(5) * 0.08)
    board = np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]
    centre = board.mean(axis=0)
    points, pixels = [], []
    tries = 0
    while len(points) < n and tries < 400:
        tries += 1
        R = so3_exp(np.radians(rng.uniform(-35, 35, 3) * np.array([1, 1, 0.3])))
        direction = so3_exp(np.array([rng.uniform(-spread, spread), rng.uniform(-spread, spread), 0.0])) @ np.array([0, 0, 1.0])
        X = (board - centre) @ R.T + direction * rng.uniform(0.35, 0.6)
        uv, ok = model.project(X)
        uv = np.asarray(uv, float)
        inside = np.asarray(ok, bool).ravel() & (uv[:, 0] > 0) & (uv[:, 0] < W - 1) & (uv[:, 1] > 0) & (uv[:, 1] < H - 1)
        if inside.all():
            points.append(board)
            pixels.append(uv + noise * rng.standard_normal(uv.shape))
    assert len(points) == n
    return points, pixels


FISHEYE = KannalaBrandtModel(330.0, 330.0, 640.0, 480.0, 0.02, -0.004, 0.001, 0.0)
STANDARD = RadTanModel(900.0, 900.0, 640.0, 480.0, -0.12, 0.05, 0.0005, -0.0003, 0.0)


def test_a_standard_lens_is_measured_and_its_type_fits():
    points, pixels = _views(STANDARD, spread=0.3)
    out = advisor.advise_from_views(points, pixels, W, H, "radtan")
    assert out["verdict"] == "ok" and out["suggested"] == "radtan"
    assert out["fov_h_deg"] == pytest.approx(2 * np.degrees(np.arctan(640 / 900)), abs=3.0)
    assert out["chosen_px"] < 0.2 and "fits" in out["message"]
    # UCM fits the pixels of an ordinary lens and still biases a rig: flagged, with the better type
    ucm = advisor.advise_from_views(points, pixels, W, H, "ucm")
    assert ucm["verdict"] in ("caution", "misfit") and ucm["suggested"] in ("radtan", "kb")


def test_a_standard_lens_type_on_a_fisheye_is_caught_before_calibrating():
    points, pixels = _views(FISHEYE, spread=1.0)
    out = advisor.advise_from_views(points, pixels, W, H, "radtan")
    assert out["fov_h_deg"] > 150 and out["measured_to_deg"] > 55
    assert out["verdict"] == "misfit" and out["suggested"] == "kb"
    assert out["chosen_px"] is None or out["chosen_px"] > 2.5 * out["reference_px"]
    assert advisor.advise_from_views(points, pixels, W, H, "kb")["verdict"] == "ok"


def test_too_few_views_is_unknown_not_a_verdict():
    points, pixels = _views(STANDARD, n=2, spread=0.3)
    out = advisor.advise_from_views(points, pixels, W, H, "radtan")
    assert out["verdict"] == "unknown" and "2 of the sample photos" in out["message"]


def test_a_lens_file_is_checked_against_the_views(tmp_path, monkeypatch):
    """The right lens values pass; values of another lens, or for another image size, are named
    as not fitting."""
    from ds_msp.data.observations import RigState
    from ds_msp.io.mccalib import save_mccalib_cameras

    points, pixels = _views(STANDARD, spread=0.3)
    monkeypatch.setattr(advisor, "sample_views", lambda *a, **k: (points, pixels, (W, H), len(points)))
    other = RadTanModel(600.0, 600.0, 640.0, 480.0, -0.3, 0.1, 0.0, 0.0, 0.0)
    rig = RigState(cameras={0: STANDARD, 1: other}, T_c_g={0: np.eye(4), 1: np.eye(4)}, ref_cam_id=0,
                   object_poses={}, objects={}, img_size={0: (W, H), 1: (W, H)})
    path = tmp_path / "lenses.yml"
    save_mccalib_cameras(rig, str(path))
    good = advisor.lens_file_health({"root_path": "unused"}, 0, str(path))
    assert good["verdict"] == "ok" and good["given_px"] < 0.2, good
    wrong = advisor.lens_file_health({"root_path": "unused"}, 1, str(path))       # camera order mixed up
    assert wrong["verdict"] == "misfit" and wrong["given_px"] > 3 * wrong["reference_px"]
    assert "same camera order" in wrong["message"]
    assert advisor.lens_file_health({"root_path": "unused"}, 5, str(path))["verdict"] == "unknown"
    monkeypatch.setattr(advisor, "sample_views", lambda *a, **k: (points, pixels, (640, 480), len(points)))
    size = advisor.lens_file_health({"root_path": "unused"}, 0, str(path))
    assert size["verdict"] == "misfit" and "640 x 480" in size["message"]


def test_lens_checks_run_on_saved_detections_without_photos(tmp_path, monkeypatch):
    """With a detections file and no photo folder, the same checks read the saved board views;
    a board description the detections do not fit is named instead of judged."""
    from ds_msp.data.observations import BoardObs, RigState
    from ds_msp.io.mccalib import save_mccalib_cameras

    _points, pixels = _views(STANDARD, n=14, spread=0.3)
    obs = [BoardObs(cam_id=0, frame_id=k, board_id=0, corner_ids=np.arange(30), pts_2d=uv)
           for k, uv in enumerate(pixels)]
    obs.append(BoardObs(cam_id=1, frame_id=0, board_id=0, corner_ids=np.arange(30), pts_2d=pixels[0]))
    monkeypatch.setattr(advisor, "_keypoints", lambda *a: (obs, {0: (W, H), 1: (W, H)}))
    kp = tmp_path / "detected_keypoints_data.yml"
    kp.write_text("stands in for a detections file")
    values = {"keypoints_path": str(kp), "number_x_square": 7, "number_y_square": 6,
              "square_size": 0.08, "camera_models": ["radtan"]}
    out = advisor.lens_advice(values, 0)
    assert out["verdict"] == "ok" and out["source"] == "saved detections", out
    assert out["views"] == 12 and 60 < out["fov_h_deg"] < 85
    few = advisor.lens_advice(values, 1)
    assert few["verdict"] == "unknown" and "1 of the saved detections" in few["message"]
    wrong = advisor.lens_advice({**values, "number_x_square": 5, "number_y_square": 5}, 0)
    assert wrong["verdict"] == "unknown" and "board description" in wrong["message"]

    other = RadTanModel(600.0, 600.0, 640.0, 480.0, -0.3, 0.1, 0.0, 0.0, 0.0)
    rig = RigState(cameras={0: STANDARD, 1: other}, T_c_g={0: np.eye(4), 1: np.eye(4)}, ref_cam_id=0,
                   object_poses={}, objects={}, img_size={0: (W, H), 1: (W, H)})
    lenses = tmp_path / "lenses.yml"
    save_mccalib_cameras(rig, str(lenses))
    good = advisor.lens_file_health(values, 0, str(lenses))
    assert good["verdict"] == "ok" and "saved detections" in good["message"]
    with pytest.raises(ValueError):
        advisor.lens_advice({"number_x_square": 7}, 0)
