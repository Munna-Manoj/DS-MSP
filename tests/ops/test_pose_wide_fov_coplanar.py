"""Regression: PnP on peripheral (>90 deg off-axis) COPLANAR correspondences (ADR-0019).

The coplanar counterpart of ``test_pose_wide_fov.py`` (ADR-0018, non-coplanar): a single
tilted planar board, imaged through a real Double Sphere model, with some corners past 90 deg
off-axis. Before ADR-0019 the coplanar branch fell back to the ``z > 0`` normalized-plane
homography, silently dropping every peripheral corner (or failing outright if too few remained);
after wiring the bearing-vector homography (``_pose_planar_bearing`` / ``_ransac_pnp_planar_bearing``)
into the coplanar branch, all four PnP entry points recover the ground-truth pose using every
valid corner, front or peripheral.
"""
import cv2
import numpy as np
import pytest

from ds_msp.core.lie import so3_exp
from ds_msp.geometry.resection import _is_coplanar
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.ops import solve_pnp, solve_pnp_ransac
from ds_msp.rig.pose_init import estimate_pose_ransac, robust_pose_irls


def _peripheral_planar_view(n=60, half=1.0, seed=0):
    """A coplanar (Z=0, object frame) board tilted enough that some corners are >90 deg
    off-axis, imaged through a real Double Sphere model."""
    m = DoubleSphereModel.sample()
    rng = np.random.default_rng(seed)
    R_gt, t_gt = so3_exp([0.0, 1.22, 0.0]), np.array([0.05, -0.02, 0.6])
    a = rng.uniform(-half, half, n)
    b = rng.uniform(-half, half, n)
    X = np.column_stack([a, b, np.zeros(n)])
    Xc = X @ R_gt.T + t_gt
    uv, val = m.project(Xc)
    return m, X[val], np.asarray(uv)[val], R_gt, t_gt


def _pose_err(R, t, R_gt, t_gt):
    ang = np.degrees(np.arccos(np.clip((np.trace(R.T @ R_gt) - 1.0) / 2.0, -1.0, 1.0)))
    return ang, float(np.linalg.norm(t - t_gt))


def test_peripheral_data_is_valid_coplanar_and_past_90deg():
    """Guard: the manufactured scenario is genuinely coplanar, DS-valid, and >90 deg."""
    m, X, uv, _, _ = _peripheral_planar_view()
    assert len(X) >= 20
    assert _is_coplanar(X)                                # a single planar board
    rays, valid = m.unproject(uv)
    assert valid.all()                                    # model certifies every observation
    off = np.degrees(np.arccos(np.clip(rays[:, 2], -1, 1)))
    assert (off > 90).sum() >= 5                           # genuinely some peripheral corners


def test_solve_pnp_recovers_peripheral_planar_pose():
    m, X, uv, R_gt, t_gt = _peripheral_planar_view()
    ok, rvec, tvec = solve_pnp(m, X, uv)
    assert ok                                             # before fix: front-only, degraded/None
    ang, terr = _pose_err(cv2.Rodrigues(rvec)[0], tvec, R_gt, t_gt)
    assert ang < 1e-3 and terr < 1e-3


def test_solve_pnp_ransac_recovers_peripheral_planar_pose():
    m, X, uv, R_gt, t_gt = _peripheral_planar_view()
    ok, rvec, tvec, inliers = solve_pnp_ransac(m, X, uv, seed=0)
    assert ok
    ang, terr = _pose_err(cv2.Rodrigues(rvec)[0], tvec, R_gt, t_gt)
    assert ang < 1e-2 and terr < 1e-2
    assert inliers.sum() >= len(X) - 1


def test_solve_pnp_ransac_accepts_four_point_peripheral_planar_minimum():
    """The planar four-point floor is applied before the general-DLT six-point floor."""
    m = DoubleSphereModel.sample()
    R_gt, t_gt = so3_exp([0.0, 1.22, 0.0]), np.array([0.05, -0.02, 0.6])
    X = np.array([
        [-0.8, -0.8, 0.0],
        [0.8, -0.8, 0.0],
        [0.8, 0.8, 0.0],
        [-0.8, 0.8, 0.0],
    ])
    Xc = X @ R_gt.T + t_gt
    uv, valid = m.project(Xc)
    rays, ray_valid = m.unproject(uv)
    assert valid.all() and ray_valid.all()
    assert (rays[:, 2] <= 0.0).sum() == 2

    ok, rvec, tvec, inliers = solve_pnp_ransac(m, X, uv, seed=0)

    assert ok
    ang, terr = _pose_err(cv2.Rodrigues(rvec)[0], tvec, R_gt, t_gt)
    assert ang < 1e-2 and terr < 1e-2
    assert inliers.all()


def test_estimate_pose_ransac_recovers_peripheral_planar_pose():
    m, X, uv, R_gt, t_gt = _peripheral_planar_view()
    T, inliers = estimate_pose_ransac(m, X, uv, seed=0)
    assert T is not None
    ang, terr = _pose_err(T[:3, :3], T[:3, 3], R_gt, t_gt)
    assert ang < 1e-2 and terr < 1e-2
    assert inliers.sum() >= len(X) - 1


def test_robust_pose_irls_recovers_peripheral_planar_pose():
    m, X, uv, R_gt, t_gt = _peripheral_planar_view()
    T = robust_pose_irls(m, X, uv, seed=0)
    assert T is not None
    ang, terr = _pose_err(T[:3, :3], T[:3, 3], R_gt, t_gt)
    assert ang < 1e-2 and terr < 1e-2


def test_solve_pnp_ransac_scores_peripheral_planar_outliers():
    """The bearing (angular) inlier metric must still flag gross blunders among a coplanar
    board's peripheral corners."""
    m, X, uv, R_gt, t_gt = _peripheral_planar_view(n=80)
    rng = np.random.default_rng(1)
    bad = rng.choice(len(uv), max(len(uv) // 5, 1), replace=False)
    uv = uv.copy()
    uv[bad] += rng.uniform(-120, 120, size=(len(bad), 2))
    ok, rvec, tvec, inliers = solve_pnp_ransac(m, X, uv, seed=0)
    assert ok
    ang, terr = _pose_err(cv2.Rodrigues(rvec)[0], tvec, R_gt, t_gt)
    assert ang < 0.5 and terr < 0.05
    assert not inliers[bad].any()


def test_narrow_fov_coplanar_unaffected():
    """Backward-consistency: a forward-hemisphere (z>0) coplanar board still solves correctly."""
    m = DoubleSphereModel.sample()
    rng = np.random.default_rng(3)
    R_gt, t_gt = so3_exp([0.05, -0.1, 0.02]), np.array([0.1, -0.05, 2.2])
    a = rng.uniform(-0.3, 0.3, 40)
    b = rng.uniform(-0.3, 0.3, 40)
    X = np.column_stack([a, b, np.zeros(40)])
    Xc = X @ R_gt.T + t_gt
    assert (Xc[:, 2] > 0).all() and _is_coplanar(X)
    uv, val = m.project(Xc)
    ok, rvec, tvec, _ = solve_pnp_ransac(m, X[val], np.asarray(uv)[val], seed=0)
    assert ok
    ang, terr = _pose_err(cv2.Rodrigues(rvec)[0], tvec, R_gt, t_gt)
    assert ang < 1e-2 and terr < 1e-2


# Traceability: PnP on bearing rays for any model (ops) + robust resection/PnP seeding (calib).
pytestmark = pytest.mark.req("FR-OPS-003", "FR-CALIB-002")
