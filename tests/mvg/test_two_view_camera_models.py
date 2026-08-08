"""End-to-end two-view geometry through the OCam and DS+ camera models.

``tests/mvg/test_two_view.py`` proves the geometry (essential matrix, pose recovery,
triangulation, bundle refine) on bare bearing vectors and, for one case, end-to-end through a
``DoubleSphereModel`` (``test_recover_pose_through_a_real_double_sphere_camera``). The mvg layer
is model-agnostic by construction — it only ever sees rays — but that agnosticism had never been
*exercised* for the polynomial OCam (Scaramuzza) or DS+ (UCM core + division + tilt) models.

These tests render 3D points into pixels through those two models, unproject to rays, and run the
full pipeline: eight-point essential estimation, cheirality pose recovery, metric triangulation,
robust RANSAC + angular bundle refinement. Same rigor and (noise-free) tolerances as the DS case.
"""

import numpy as np
import pytest

from ds_msp.mvg import (
    estimate_relative_pose,
    recover_pose,
    triangulate_rays,
)
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.ocam import OCamModel
from ds_msp.models.dsplus import DSPlusModel

# DS kept for parity; OCam and DS+ are the new coverage.
MODELS = [DoubleSphereModel.sample, OCamModel.sample, DSPlusModel.sample]


def _rot(axis, angle):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _rot_err_deg(R, Rhat):
    c = (np.trace(R.T @ Rhat) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _vec_ang_deg(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1)))


def _two_view_rays(cam, seed=7, n=60):
    """3D points -> two views' pixels -> unproject to rays. Returns (f1, f2, R, t)."""
    rng = np.random.default_rng(seed)
    R = _rot(rng.standard_normal(3), 0.5)
    t = rng.standard_normal(3)
    t = t / np.linalg.norm(t)
    X1 = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(-3, 3, n), rng.uniform(2, 9, n)])
    X2 = (R @ X1.T).T + t
    uv1, ok1 = cam.project(X1)
    uv2, ok2 = cam.project(X2)
    ok = ok1 & ok2
    f1, _ = cam.unproject(uv1[ok])
    f2, _ = cam.unproject(uv2[ok])
    return f1, f2, R, t, ok.sum()


@pytest.mark.parametrize("factory", MODELS, ids=lambda f: f().name)
def test_recover_pose_through_camera_model(factory):
    """3D -> pixels (two views) -> rays -> recover pose. Limited only by the project/unproject
    round-trip of the model, so ~1e-3 deg — identical bar to the DS reference test."""
    cam = factory()
    f1, f2, R, t, n = _two_view_rays(cam)
    assert n >= 8
    Rhat, that, _ = recover_pose(f1, f2)
    assert _rot_err_deg(R, Rhat) < 1e-3
    assert _vec_ang_deg(t, that) < 1e-3
    assert that @ t > 0                              # correct (non-mirrored) cheirality


@pytest.mark.parametrize("factory", MODELS, ids=lambda f: f().name)
def test_triangulated_points_are_in_front_through_camera_model(factory):
    cam = factory()
    f1, f2, R, t, n = _two_view_rays(cam, seed=9)
    Rhat, that, _ = recover_pose(f1, f2)
    _, d1, d2 = triangulate_rays(f1, f2, Rhat, that)
    assert np.all(d1 > 0) and np.all(d2 > 0)


@pytest.mark.parametrize("factory", MODELS, ids=lambda f: f().name)
def test_robust_relative_pose_through_camera_model_under_outliers(factory):
    """Full robust path (RANSAC consensus -> triangulate -> angular bundle refine) on rays from
    the model, with 25% of the second view's rays replaced by gross blunders. The estimator must
    stay sub-degree and label the outliers. Geometry is noise-free so the residual comes only
    from any single blunder that happens to satisfy the epipolar constraint."""
    cam = factory()
    rng = np.random.default_rng(11)
    R = _rot(rng.standard_normal(3), 0.5)
    t = rng.standard_normal(3)
    t = t / np.linalg.norm(t)
    X1 = np.column_stack([rng.uniform(-3, 3, 100), rng.uniform(-3, 3, 100), rng.uniform(2, 9, 100)])
    X2 = (R @ X1.T).T + t
    uv1, ok1 = cam.project(X1)
    uv2, ok2 = cam.project(X2)
    ok = ok1 & ok2
    f1, _ = cam.unproject(uv1[ok])
    f2, _ = cam.unproject(uv2[ok])

    f2c = f2.copy()
    k = int(0.25 * len(f2c))
    idx = rng.choice(len(f2c), k, replace=False)
    rnd = rng.standard_normal((k, 3))
    f2c[idx] = rnd / np.linalg.norm(rnd, axis=1, keepdims=True)
    gt_inlier = np.ones(len(f2c), dtype=bool)
    gt_inlier[idx] = False

    Rr, tr, Xr, inl = estimate_relative_pose(f1, f2c, threshold=0.005, seed=0)
    assert _rot_err_deg(R, Rr) < 0.5
    assert _vec_ang_deg(t, tr) < 2.0
    prec = (inl & gt_inlier).sum() / max(inl.sum(), 1)
    rec = (inl & gt_inlier).sum() / gt_inlier.sum()
    assert prec > 0.95 and rec > 0.9

# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("FR-MVG-001", "FR-MVG-002", "FR-MVG-003")
