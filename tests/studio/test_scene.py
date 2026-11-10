"""A result shown on its own capture: frames, corners, photos, the board through the capture,
the geometry between two cameras, and what the uncertainty means for a measured point."""
import json

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from ds_msp.io.mccalib import load_camera  # noqa: E402
from ds_msp.studio import scene  # noqa: E402
from ds_msp.studio.server import create_app  # noqa: E402

pytestmark = pytest.mark.req("FR-UI-009")
RUN = "lab/run_a"


def test_frames_and_corners_come_from_the_recorded_residuals(results_root):
    folder = results_root / RUN
    frames = scene.frame_list(folder, 1)["frames"]
    assert len(frames) == 30 and [f["frame"] for f in frames] == sorted(f["frame"] for f in frames)
    one = scene.frame_corners(folder, 1, frames[3]["frame"])
    assert one["n"] == frames[3]["n"] == len(one["u"]) == len(one["dx"]) > 0
    rms = float(np.sqrt(np.mean(np.square(one["dx"]) + np.square(one["dy"]))))
    assert rms == pytest.approx(frames[3]["rms_px"], abs=2e-4)
    assert scene.frame_corners(folder, 1, 9999)["n"] == 0


def test_scene_places_the_board_where_the_cameras_see_it(results_root):
    """Projecting the board, at a frame's pose, into a camera must land on that frame's
    detected corners: the pose list and the frame numbers are matched up correctly."""
    folder = results_root / RUN
    doc = scene.run_scene(folder)
    assert len(doc["frames"]) == 30 and len(doc["boards"]) >= 1 and doc["photos"] == "none"
    report = json.loads((folder / "calibration_report.json").read_text())
    T_c_ref = np.asarray(next(e for e in report["extrinsics"] if e["cam_id"] == 1)["T_c_ref"])
    model = load_camera(str(folder / "calibrated_cameras_data.yml"), 1)
    pts = np.vstack([np.asarray(b["points"]) for b in doc["boards"]])
    f = doc["frames"][7]
    T = T_c_ref @ np.asarray(f["T"])
    uv, ok = model.project(pts @ T[:3, :3].T + T[:3, 3])
    seen = scene.frame_corners(folder, 1, f["frame"])
    found = np.c_[seen["u"], seen["v"]]
    nearest = np.min(np.linalg.norm(found[:, None, :] - np.asarray(uv)[None, :, :], axis=2), axis=1)
    assert np.median(nearest) < 0.5
    for b in doc["boards"]:                               # the outline holds every corner
        o, p = np.asarray(b["outline"]), np.asarray(b["points"])
        assert np.linalg.norm(p - o.mean(axis=0), axis=1).max() <= np.linalg.norm(o - o.mean(axis=0), axis=1).max()


def test_pair_geometry_matches_the_extrinsics(results_root):
    folder = results_root / RUN
    report = json.loads((folder / "calibration_report.json").read_text())
    ext = {e["cam_id"]: e for e in report["extrinsics"]}
    doc = scene.pair_geometry(folder, 0, 2)
    assert doc["ok"] and doc["distance"] == pytest.approx(np.linalg.norm(ext[2]["center_in_ref"]), rel=1e-9)
    assert doc["position"] == pytest.approx(ext[2]["center_in_ref"], abs=1e-9)
    # yaw, pitch, roll rebuild the rotation of camera 2's axes in camera 0's frame
    y, p, r = np.radians([doc["yaw_deg"], doc["pitch_deg"], doc["roll_deg"]])
    Ry = cv2.Rodrigues(np.array([0.0, y, 0.0]))[0]
    Rx = cv2.Rodrigues(np.array([p, 0.0, 0.0]))[0]
    Rz = cv2.Rodrigues(np.array([0.0, 0.0, r]))[0]
    np.testing.assert_allclose(Ry @ Rx @ Rz, np.asarray(ext[2]["T_c_ref"])[:3, :3].T, atol=1e-9)
    back = scene.pair_geometry(folder, 2, 0)
    assert back["distance"] == pytest.approx(doc["distance"]) and back["angle_deg"] == pytest.approx(doc["angle_deg"])
    assert scene.pair_geometry(folder, 0, 0)["ok"] is False


def test_task_uncertainty_equals_a_monte_carlo_of_the_same_errors(results_root):
    """Draw the pair error from its recorded covariance and pixel noise, triangulate, and
    compare the spread of the point with the propagated sigma."""
    folder = results_root / RUN
    report = json.loads((folder / "calibration_report.json").read_text())
    acc = json.loads((folder / "calibration_acceptance.json").read_text())
    ext = {e["cam_id"]: np.asarray(e["T_c_ref"]) for e in report["extrinsics"]}
    geo = scene.pair_geometry(folder, 0, 1)
    doc = scene.task_uncertainty(folder, 0, 1, 4.0 * geo["distance"], pixel_noise=0.4)
    assert doc["ok"] and doc["depth"] > doc["sideways"] > 0
    assert doc["depth"] ** 2 == pytest.approx(doc["from_cameras"]["depth"] ** 2 + doc["from_pixels"]["depth"] ** 2, rel=1e-6)

    unc = acc["extrinsic_uncertainty"]
    P = np.asarray(unc["pairs"]["0-1"]["covariance"]) * float(unc["t_factor"]) ** 2
    cams = {c: load_camera(str(folder / "calibrated_cameras_data.yml"), c) for c in (0, 1)}
    R0, C0 = scene._relative(ext[0], ext[1])
    look = np.array([0.0, 0.0, 1.0]) + R0.T @ np.array([0.0, 0.0, 1.0])
    look /= np.linalg.norm(look)
    X0 = 0.5 * C0 + doc["distance"] * look
    lens = [doc["lens_sigma_px"][0], doc["lens_sigma_px"][1]]
    sd = np.repeat([np.hypot(0.4, lens[0]), np.hypot(0.4, lens[1])], 2)

    def pixels(X, R, C):
        a, _ = cams[0].project(X[None, :])
        b, _ = cams[1].project((R @ (X - C))[None, :])
        return np.r_[np.ravel(a), np.ravel(b)]

    u0 = pixels(X0, R0, C0)
    h = 1e-6 * doc["distance"]
    Jx = np.stack([(pixels(X0 + h * e, R0, C0) - pixels(X0 - h * e, R0, C0)) / (2 * h) for e in np.eye(3)], axis=1)
    solve = np.linalg.solve(Jx.T @ Jx, Jx.T)
    rng = np.random.default_rng(3)
    out = []
    for d in rng.multivariate_normal(np.zeros(6), P, size=4000):
        # the true rig differs from the calibrated one by d; pixels come from the true rig
        true = pixels(X0, R0 @ cv2.Rodrigues(d[:3])[0], C0 + d[3:]) + sd * rng.standard_normal(4)
        out.append(solve @ (true - u0))                    # one Gauss-Newton step from X0
    out = np.asarray(out)
    assert np.std(out @ look) == pytest.approx(doc["depth"], rel=0.06)
    side = np.sqrt(np.linalg.eigvalsh(np.cov((out - np.outer(out @ look, look)).T))[-1])
    assert side == pytest.approx(doc["sideways"], rel=0.06)


def test_task_uncertainty_says_when_it_cannot_answer(results_root):
    folder = results_root / RUN
    assert scene.task_uncertainty(folder, 0, 0, 1.0)["ok"] is False
    assert "positive" in scene.task_uncertainty(folder, 0, 1, -1.0)["reason"]
    far = scene.task_uncertainty(folder, 0, 1, 1e-4)        # between the cameras: out of view
    assert far["ok"] is False and far["reason"]


def test_photos_are_served_only_from_inside_the_roots(tmp_path, results_root):
    folder = tmp_path / "run"
    (tmp_path / "Images" / "Cam_001").mkdir(parents=True)
    (tmp_path / "outside" / "Cam_001").mkdir(parents=True)
    folder.mkdir()
    img = np.full((60, 200), 90, np.uint8)
    inside, outside = tmp_path / "Images" / "Cam_001" / "0005.png", tmp_path / "outside" / "Cam_001" / "0006.png"
    cv2.imwrite(str(inside), img)
    cv2.imwrite(str(outside), img)
    (folder / "calibration_manifest.json").write_text(json.dumps({
        "configuration": {"resolved": {"cam_prefix": "Cam_"}},
        "inputs": {"files": [{"path": str(inside)}, {"path": str(outside)}]}}))
    idx = scene.photo_index(folder)
    assert idx["how"] == "assumed" and idx["index"] == {(0, 0): str(inside), (0, 1): str(outside)}
    assert scene.photo_path(folder, 0, 0, [tmp_path / "Images"]) == inside
    assert scene.photo_path(folder, 0, 1, [tmp_path / "Images"]) is None     # listed, but outside
    assert scene.photo_path(folder, 0, 7, [tmp_path]) is None
    jpeg = scene.photo_jpeg(inside, width=100)
    small = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert small.shape == (30, 100)


def test_recorded_photo_paths_win_over_file_numbers(tmp_path):
    """A result that records which photo each observation came from is matched exactly, also
    when the board was missing from the first photos and frame numbers were rebased."""
    from ds_msp.io.mccalib import save_mccalib_detected_keypoints
    from tests.rig._synth import make_rig

    obj, obs, img_size, _e, _m = make_rig(n_cam=2, n_frame=4, noise_px=0.0, seed=1, multi_board=False)
    for o in obs:
        o.image_path = str(tmp_path / f"Cam_{o.cam_id + 1:03d}" / f"{o.frame_id + 40:05d}.png")
    save_mccalib_detected_keypoints(obs, obj, img_size, str(tmp_path / "detected_keypoints_data.yml"))
    idx = scene.photo_index(tmp_path)
    assert idx["how"] == "recorded"
    assert idx["index"][(1, 2)].endswith("Cam_002/00042.png")


def test_routes(results_root):
    c = TestClient(create_app([str(results_root)]))
    rid = "0:" + RUN
    assert len(c.get("/api/run/scene", params={"id": rid}).json()["frames"]) == 30
    frames = c.get("/api/run/frames", params={"id": rid, "cam": 0}).json()["frames"]
    one = c.get("/api/run/frame", params={"id": rid, "cam": 0, "frame": frames[0]["frame"]}).json()
    assert one["n"] > 0 and one["photo"] is False
    assert c.get("/api/run/photo", params={"id": rid, "cam": 0, "frame": 0}).status_code == 404
    assert c.get("/api/run/pair", params={"id": rid, "a": 0, "b": 1}).json()["ok"]
    base = c.get("/api/run/pair", params={"id": rid, "a": 0, "b": 1}).json()["distance"]
    task = c.get("/api/run/task", params={"id": rid, "a": 0, "b": 1, "distance": 4 * base}).json()
    assert task["ok"] and len(task["curve"]) > 5
    assert all(a["depth"] < b["depth"] for a, b in zip(task["curve"], task["curve"][1:]))
    assert c.get("/api/run/scene", params={"id": "0:../x"}).status_code == 404


@pytest.mark.req("FR-UI-013")
def test_error_spread_counts_the_corners_beyond_each_distance(results_root):
    import numpy as np

    from ds_msp.studio.scene import _residuals, error_spread

    folder = results_root / RUN
    z = _residuals(folder)
    r = np.hypot(z["dx"], z["dy"])[z["valid"]]
    doc = error_spread(folder, max_points=100)
    assert doc["n"] == len(r) and doc["max_px"] == pytest.approx(r.max())
    grid, above = np.array(doc["grid_px"]), np.array(doc["above"])
    assert grid[0] == pytest.approx(0.01) and grid[-1] == pytest.approx(100.0)
    for k in (0, 40, 80, 120):
        assert above[k] == int((r > grid[k]).sum())
    assert (np.diff(above) <= 0).all() and above[-1] == 0
    # the sample is bounded and never loses the worst corners
    assert len(doc["dx"]) == len(doc["dy"]) == min(len(r), 100)
    assert np.hypot(doc["dx"], doc["dy"]).max() == pytest.approx(r.max(), abs=1e-4)
    assert error_spread(results_root / "nowhere")["n"] == 0
    c = TestClient(create_app([str(results_root)]))
    assert c.get("/api/run/errors", params={"id": f"0:{RUN}"}).json()["n"] == len(r)
