"""
ops services must work on ANY model and agree with the legacy DS implementation.
Tested across DS/UCM/EUCM/KB, the polynomial OCam and DS+ models, and the
camera-free FakeModel.
"""

import cv2
import numpy as np
import pytest

from ds_msp.ops import Undistorter, solve_pnp, solve_pnp_ransac
from ds_msp.testing import FakeModel
from ds_msp.model import DoubleSphereCamera
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.ucm import UCMModel
from ds_msp.models.eucm import EUCMModel
from ds_msp.models.kb import KannalaBrandtModel
from ds_msp.models.ocam import OCamModel
from ds_msp.models.dsplus import DSPlusModel

MODELS = [DoubleSphereModel.sample, UCMModel.sample, EUCMModel.sample,
          KannalaBrandtModel.sample, OCamModel.sample, DSPlusModel.sample,
          FakeModel.sample]


@pytest.mark.parametrize("factory", MODELS, ids=lambda f: f().name)
def test_solve_pnp_recovers_known_pose(factory):
    model = factory()
    # Build a planar target, place it in front, project with the model.
    g = np.mgrid[0:4, 0:5].reshape(2, -1).T * 0.1
    obj = np.column_stack([g, np.zeros(len(g))]).astype(np.float64)
    rvec_gt = np.array([0.05, -0.1, 0.02])
    tvec_gt = np.array([-0.15, 0.1, 2.0])
    R, _ = cv2.Rodrigues(rvec_gt)
    Xc = (R @ obj.T).T + tvec_gt
    uv, valid = model.project(Xc)
    ok, rvec, tvec = solve_pnp(model, obj[valid], uv[valid])
    assert ok
    assert np.linalg.norm(tvec - tvec_gt) < 0.05


@pytest.mark.parametrize("factory", MODELS, ids=lambda f: f().name)
def test_solve_pnp_ransac_rejects_gross_outliers(factory):
    """RANSAC PnP recovers the pose and flags the blunders where a plain solve drifts."""
    model = factory()
    g = np.mgrid[0:7, 0:7].reshape(2, -1).T * 0.1
    obj = np.column_stack([g, np.zeros(len(g))]).astype(np.float64)
    rvec_gt = np.array([0.05, -0.1, 0.02])
    tvec_gt = np.array([-0.15, 0.1, 2.0])
    R, _ = cv2.Rodrigues(rvec_gt)
    uv, valid = model.project((R @ obj.T).T + tvec_gt)
    obj_v, uv_v = obj[valid], np.asarray(uv)[valid].copy()
    # corrupt 25% of the pixels with large blunders
    rng = np.random.default_rng(0)
    out = rng.choice(len(uv_v), len(uv_v) // 4, replace=False)
    uv_v[out] += rng.uniform(-120, 120, size=(len(out), 2))
    ok, rvec, tvec, inliers = solve_pnp_ransac(model, obj_v, uv_v, seed=0)
    assert ok
    assert np.linalg.norm(tvec - tvec_gt) < 0.05            # pose recovered despite outliers
    assert not inliers[out].any()                           # every injected blunder flagged out
    assert inliers.sum() >= len(uv_v) // 2                   # the clean majority kept


@pytest.mark.parametrize("factory", MODELS, ids=lambda f: f().name)
def test_undistorter_runs_on_any_model(factory):
    model = factory()
    und = Undistorter(model, 640, 480)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out, K_new = und.undistort_image(img)
    assert out.shape == img.shape
    assert K_new.shape == (3, 3)
    # cache hit returns the same arrays
    mapx1, _, _ = und.maps()
    mapx2, _, _ = und.maps()
    assert mapx1 is mapx2


@pytest.mark.parametrize("factory", MODELS, ids=lambda f: f().name)
def test_point_distort_undistort_roundtrip(factory):
    model = factory()
    und = Undistorter(model, 640, 480)
    K_new = und.new_K(0.5)
    # central distorted pixels round-trip through undistort -> distort
    pts = np.array([[320.0, 240.0], [360.0, 250.0], [300.0, 230.0]])
    rect, v1 = und.undistort_points(pts, K_new)
    back, v2 = und.distort_points(rect, K_new)
    ok = v1 & v2
    assert ok.any()
    assert np.allclose(back[ok], pts[ok], atol=1e-3)


def test_ops_match_legacy_ds_camera():
    params = dict(fx=711.57, fy=711.24, cx=949.18, cy=518.81, xi=0.183, alpha=0.809)
    model = DoubleSphereModel(**params)
    cam = DoubleSphereCamera(**params, width=1920, height=1080)
    und = Undistorter(model, 1920, 1080)

    K_a = und.new_K(0.5)
    K_b = cam.compute_K_new(0.5)
    assert np.allclose(K_a, K_b)

    mapx_a, mapy_a, _ = und.maps(K_a)
    mapx_b, mapy_b, _ = cam.get_undistortion_maps(K_b)
    assert np.allclose(mapx_a, mapx_b, atol=1e-3)
    assert np.allclose(mapy_a, mapy_b, atol=1e-3)

# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("FR-OPS-001", "FR-OPS-003")
