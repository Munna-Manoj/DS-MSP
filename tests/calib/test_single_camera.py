"""``calibrate_camera`` end to end: Board.detect() -> correspondences -> from-scratch seed ->
bundle adjustment, recovering ground-truth intrinsics.

Uses a fake :class:`~ds_msp.calib.board.Board` returning precomputed synthetic
:class:`~ds_msp.data.observations.Observation`\\ s (built the same way
``tests/calib/test_generic_calibrate.py`` builds its synthetic dataset) rather than real
rendered images -- "real image -> Observation" is already verified per board type in
``tests/calib/test_board.py``; this file is about whether :func:`calibrate_camera` correctly
wires that output into the shared backend and actually recovers ground truth, which doesn't
depend on how the Observations were produced.
"""
import cv2
import numpy as np
import pytest

from ds_msp.calib.single_camera import calibrate_camera
from ds_msp.data.observations import Observation
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.ucm import UCMModel


class _FakeBoard:
    """Satisfies the Board protocol; ignores the image paths and returns a fixed,
    precomputed set of synthetic Observations."""

    def __init__(self, obs):
        self._obs = obs

    def detect(self, image_paths, progress_cb=None):
        if progress_cb is not None:
            for i, path in enumerate(image_paths):
                progress_cb(i + 1, len(image_paths), path)
        return self._obs


def _synthetic_observations(truth, n_views=12, seed=0, rows=6, cols=8, spacing=0.08):
    g = np.mgrid[0:rows, 0:cols].reshape(2, -1).T * spacing
    board = np.column_stack([g, np.zeros(len(g))]).astype(np.float64)
    rng = np.random.default_rng(seed)
    obs = []
    for i in range(n_views):
        rvec = rng.uniform(-0.4, 0.4, 3)
        tvec = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3),
                         rng.uniform(1.2, 2.5)])
        R, _ = cv2.Rodrigues(rvec)
        Xc = (R @ board.T).T + tvec
        uv, valid = truth.project(Xc)
        uv = uv + rng.normal(0, 0.1, uv.shape)
        obs.append(Observation(points_3d=board.copy(), pixels=uv, visibility=valid, frame_id=i))
    return obs


@pytest.mark.parametrize("truth,model_name", [
    (DoubleSphereModel(700, 700, 640, 360, 0.18, 0.62), "ds"),
    (UCMModel(700, 700, 640, 360, 0.62), "ucm"),
], ids=["ds", "ucm"])
def test_calibrate_camera_recovers_ground_truth(truth, model_name):
    obs = _synthetic_observations(truth)
    board = _FakeBoard(obs)
    image_paths = [f"view_{i}" for i in range(len(obs))]   # never touched by the fake board

    result = calibrate_camera(board, image_paths, model_name, width=1280, height=720,
                              max_nfev=120)

    assert result["rms_px"] < 0.5, result["rms_px"]
    assert result["n_images_detected"] == len(obs)
    assert result["n_images_total"] == len(image_paths)


def test_calibrate_camera_raises_when_nothing_is_detected():
    board = _FakeBoard([])
    with pytest.raises(ValueError, match="no board detected"):
        calibrate_camera(board, ["a.png", "b.png"], "ds", width=640, height=480)


def test_calibrate_camera_forwards_progress_cb_per_image():
    truth = DoubleSphereModel(700, 700, 640, 360, 0.18, 0.62)
    obs = _synthetic_observations(truth)
    board = _FakeBoard(obs)
    image_paths = [f"view_{i}" for i in range(len(obs))]
    calls = []
    calibrate_camera(board, image_paths, "ds", width=1280, height=720, max_nfev=50,
                     progress_cb=lambda i, n, path: calls.append((i, n, path)))
    assert calls == [(i + 1, len(image_paths), image_paths[i]) for i in range(len(image_paths))]


# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("FR-CALIB-006")
