"""KB unit tests: exact agreement with cv2.fisheye (external oracle) + interop."""

import numpy as np
import cv2
import pytest

from ds_msp.models.kb import KannalaBrandtModel
from ds_msp.testing import sample_forward_points


def test_kb_matches_opencv_fisheye_project():
    m = KannalaBrandtModel.sample()
    P = sample_forward_points()
    uv, _ = m.project(P)
    obj = P.reshape(-1, 1, 3).astype(np.float64)
    cv_uv, _ = cv2.fisheye.projectPoints(
        obj, np.zeros(3), np.zeros(3), m.K, m.distortion.reshape(4, 1))
    assert np.allclose(uv, cv_uv.reshape(-1, 2), atol=1e-9)


def test_kb_K_and_distortion_are_opencv_ready():
    m = KannalaBrandtModel.sample()
    assert m.K.shape == (3, 3)
    assert m.distortion.shape == (4,)
    # Should be directly usable by cv2.fisheye.undistortPoints without error.
    pts = np.array([[[330.0, 250.0]]], dtype=np.float64)
    out = cv2.fisheye.undistortPoints(pts, m.K, m.distortion)
    assert out.shape == (1, 1, 2)


def test_kb_roundtrip():
    m = KannalaBrandtModel.sample()
    P = sample_forward_points()
    uv, v1 = m.project(P)
    rays, v2 = m.unproject(uv)
    ok = v1 & v2
    d = P[ok] / np.linalg.norm(P[ok], axis=1, keepdims=True)
    cos = np.sum(rays[ok] * d, axis=1)
    assert (cos > 1 - 1e-5).all()


@pytest.mark.req("NFR-NUM-014")
def test_point_straight_behind_the_camera_is_invalid():
    m = KannalaBrandtModel(300.0, 300.0, 640.0, 480.0, 0.01, -0.002, 0.0, 0.0)
    _uv, valid = m.project(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]))
    assert valid.tolist() == [False, True]


@pytest.mark.req("NFR-NUM-014")
def test_points_past_the_peak_radius_are_invalid():
    # k1 = -0.3: theta_d = theta - 0.3 theta^3 peaks at theta = sqrt(1/0.9) = 60.4 deg.
    m = KannalaBrandtModel(300.0, 300.0, 640.0, 480.0, -0.3, 0.0, 0.0, 0.0)
    a = np.deg2rad([50.0, 60.0, 61.0, 80.0])
    _uv, valid = m.project(np.c_[np.sin(a), np.zeros_like(a), np.cos(a)])
    assert valid.tolist() == [True, True, False, False]


@pytest.mark.req("NFR-NUM-014")
@pytest.mark.parametrize("r", [0.0, 1e-12, 1e-6])
def test_jacobian_on_and_near_the_axis(r):
    # u = fx·x/z + cx near the axis, so d(u, v)/d(x, y) -> diag(fx, fy)/z.
    m = KannalaBrandtModel(300.0, 320.0, 640.0, 480.0, 0.01, -0.002, 0.0, 0.0)
    _uv, J, _Jp, valid = m.project_jacobian(np.array([[r, 0.0, 2.0]]))
    assert valid[0]
    np.testing.assert_allclose(J[0][:, :2], [[150.0, 0.0], [0.0, 160.0]], rtol=1e-6, atol=1e-9)
