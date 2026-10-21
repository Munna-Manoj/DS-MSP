"""Measurement validity and frame identity at the calibration boundary."""
import numpy as np
import pytest

from ds_msp.data.observations import BoardObs, ObjectObs, Observation
from ds_msp.detect.charuco import _frame_id_from_name, _validate_frame_names, detect_image

pytestmark = pytest.mark.req("FR-CALIB-009", "FR-RIG-025", "FR-CALIB-004", "FR-CALIB-006")


@pytest.mark.parametrize("name,expected", [("cam1_frame000007.png", 7),
                                          ("cam2_frame000007.png", 7), ("000021.png", 21)])
def test_frame_number_excludes_camera_prefix(name, expected):
    assert _frame_id_from_name(name) == expected


def test_ambiguous_frame_names_fail_before_detection():
    with pytest.raises(ValueError, match="frame number"):
        _frame_id_from_name("left.png")
    with pytest.raises(ValueError, match="duplicate frame"):
        _validate_frame_names(["image_7.png", "capture000007.jpg"])


@pytest.mark.parametrize("kind", ["single", "board", "object"])
def test_nonfinite_observations_rejected(kind):
    uv = np.array([[np.nan, 2.]])
    with pytest.raises(ValueError, match="finite"):
        if kind == "single":
            Observation(np.zeros((1, 3)), uv, np.ones(1, bool))
        elif kind == "board":
            BoardObs(0, 0, 0, np.array([0]), uv)
        else:
            ObjectObs(0, 0, 0, np.array([0]), uv)


@pytest.mark.parametrize("cov", [np.zeros((4, 2, 2)), np.ones((4, 3, 3)),
                                  np.tile([[1., 2.], [0., 1.]], (4, 1, 1))])
def test_invalid_pixel_covariance_rejected(cov):
    with pytest.raises(ValueError, match="pixel_covariances"):
        Observation(np.zeros((4, 3)), np.zeros((4, 2)), np.ones(4, bool),
                    pixel_covariances=cov)


@pytest.mark.parametrize("uv,ids", [([[10, 20], [20, 20], [30, 20], [40, 20]], [0, 1, 2, 3]),
                                    ([[10, 10], [20, 10], [10, 20], [20, 20]], [0, 0, 2, 3]),
                                    ([[-1, 10], [20, 10], [10, 20], [20, 20]], [0, 1, 2, 3])])
def test_degenerate_or_invalid_detector_output_is_rejected(uv, ids):
    class Detector:
        def detectBoard(self, gray):
            return np.array(uv, np.float32).reshape(-1, 1, 2), np.array(ids), None, None
    assert detect_image([Detector()], np.zeros((64, 64), np.uint8)) == []


def test_single_camera_whitening_matches_residual_and_jacobian(monkeypatch):
    from ds_msp.geometry import calibrate_core
    from ds_msp.models.radtan import RadTanModel
    cam = RadTanModel(500, 510, 320, 240, 0, 0, 0, 0, 0)
    xy = np.mgrid[:4, :5].reshape(2, -1).T * .04
    X = np.c_[xy, np.zeros(len(xy))]
    R = np.eye(3)[None]
    t = np.array([[-.08, -.06, 1.2]])
    uv, _ = cam.project(X + t[0])
    cov = np.tile([[4., 1.], [1., 2.]], (len(X), 1, 1))
    solver = calibrate_core.schur_lm

    def checked(state, residual, linearize, retract, **kw):
        rs, As, Bs = linearize(state)
        np.testing.assert_allclose(rs[0], residual(state), atol=1e-12)
        p = len(cam.params)
        numeric = np.column_stack([(residual(retract(state, np.eye(p)[j]*1e-6, np.zeros((1, 6)))) -
                                    residual(retract(state, -np.eye(p)[j]*1e-6, np.zeros((1, 6)))))/2e-6 for j in range(p)])
        np.testing.assert_allclose(As[0], numeric, rtol=1e-5, atol=1e-6)
        numeric = np.column_stack([(residual(retract(state, np.zeros(p), np.eye(6)[j][None]*1e-6)) -
                                    residual(retract(state, np.zeros(p), -np.eye(6)[j][None]*1e-6)))/2e-6 for j in range(6)])
        np.testing.assert_allclose(Bs[0], numeric, rtol=1e-5, atol=1e-6)
        return solver(state, residual, linearize, retract, **kw)
    monkeypatch.setattr(calibrate_core, "schur_lm", checked)
    calibrate_core.bundle_adjust(type(cam), cam.params, R, t, [X], [uv + [.2, -.1]],
                                  [np.ones(len(X), bool)], pixel_covariances=[cov], max_iter=1)
