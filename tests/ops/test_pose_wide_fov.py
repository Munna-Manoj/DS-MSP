"""Regression: PnP on peripheral (>90 deg off-axis) non-coplanar correspondences.

Reproduces the camera-agnosticism audit's failure mode (2026-07-07): synthetic Double-Sphere
correspondences at 95-112 deg off-axis — inside DS's own documented-valid range, both
``project`` and ``unproject`` certify them — drove every PnP entry point to ``ok=False``/``None``
because they filtered bearing rays with ``z > 1e-6`` and solved on the ``z = 1`` normalized
plane, which cannot represent a bearing at or past 90 deg.

Fails-before / passes-after: on ``main`` (pre-fix) all four solvers return failure on this data;
after wiring the bearing-vector DLT (``_pose_dlt_bearing`` / ``_ransac_pnp_bearing``) into the
non-coplanar branch, they recover the ground-truth pose. ADR-0019 separately extends the
coplanar branch; the forward-only path remains covered by the backward-consistency check.
"""
import cv2
import numpy as np
import pytest

from ds_msp.core.lie import so3_exp
from ds_msp.geometry.resection import _is_coplanar
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.ops import solve_pnp, solve_pnp_ransac
from ds_msp.rig.pose_init import estimate_pose_ransac, robust_pose_irls


def _peripheral_view(n=40, lo=95.0, hi=112.0, seed=0):
    """Non-coplanar world points imaged as bearings ``lo..hi`` deg off the optical axis."""
    m = DoubleSphereModel.sample()
    rng = np.random.default_rng(seed)
    R_gt, t_gt = so3_exp([0.2, -0.3, 0.15]), np.array([0.1, -0.05, -0.2])
    th = np.radians(rng.uniform(lo, hi, n))
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    depth = rng.uniform(0.5, 3.0, n)
    dirs = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
    Xc = dirs * depth[:, None]                     # camera-frame points (all z < 0)
    X = (Xc - t_gt) @ R_gt                         # world points: Xc = R_gt X + t_gt
    uv, val = m.project(Xc)
    return m, X[val], np.asarray(uv)[val], R_gt, t_gt, dirs[val]


def _pose_err(R, t, R_gt, t_gt):
    ang = np.degrees(np.arccos(np.clip((np.trace(R.T @ R_gt) - 1.0) / 2.0, -1.0, 1.0)))
    return ang, float(np.linalg.norm(t - t_gt))


def test_peripheral_data_is_valid_noncoplanar_and_past_90deg():
    """Guard: the manufactured scenario is exactly the audit's — DS-valid, >90 deg, non-coplanar."""
    m, X, uv, _, _, dirs = _peripheral_view()
    assert len(X) >= 6
    off = np.degrees(np.arccos(np.clip(dirs[:, 2], -1, 1)))
    assert off.min() > 90.0 and off.max() > 110.0        # genuinely peripheral
    assert (dirs[:, 2] <= 0).all()                        # every ray is z<=0 (old path drops all)
    assert not _is_coplanar(X)                            # non-coplanar -> bearing DLT applies
    rays, valid = m.unproject(uv)
    assert valid.all()                                    # model certifies every observation


def test_solve_pnp_recovers_peripheral_pose():
    m, X, uv, R_gt, t_gt, _ = _peripheral_view()
    ok, rvec, tvec = solve_pnp(m, X, uv)
    assert ok                                             # before fix: ok=False
    ang, terr = _pose_err(cv2.Rodrigues(rvec)[0], tvec, R_gt, t_gt)
    assert ang < 1e-3 and terr < 1e-3


def test_solve_pnp_ransac_recovers_peripheral_pose():
    m, X, uv, R_gt, t_gt, _ = _peripheral_view()
    ok, rvec, tvec, inliers = solve_pnp_ransac(m, X, uv, seed=0)
    assert ok                                             # before fix: ok=False
    ang, terr = _pose_err(cv2.Rodrigues(rvec)[0], tvec, R_gt, t_gt)
    assert ang < 1e-2 and terr < 1e-2
    assert inliers.sum() >= len(X) - 1                    # clean data -> essentially all inliers


def test_estimate_pose_ransac_recovers_peripheral_pose():
    m, X, uv, R_gt, t_gt, _ = _peripheral_view()
    T, inliers = estimate_pose_ransac(m, X, uv, seed=0)
    assert T is not None                                  # before fix: None
    ang, terr = _pose_err(T[:3, :3], T[:3, 3], R_gt, t_gt)
    assert ang < 1e-2 and terr < 1e-2
    assert inliers.sum() >= len(X) - 1


def test_robust_pose_irls_recovers_peripheral_pose():
    m, X, uv, R_gt, t_gt, _ = _peripheral_view()
    T = robust_pose_irls(m, X, uv, seed=0)                # before fix: None
    assert T is not None
    ang, terr = _pose_err(T[:3, :3], T[:3, 3], R_gt, t_gt)
    assert ang < 1e-2 and terr < 1e-2


def test_solve_pnp_ransac_scores_peripheral_outliers():
    """The bearing (angular) inlier metric must still flag gross blunders among peripheral rays."""
    m, X, uv, R_gt, t_gt, _ = _peripheral_view(n=48)
    rng = np.random.default_rng(1)
    bad = rng.choice(len(uv), len(uv) // 4, replace=False)
    uv = uv.copy()
    uv[bad] += rng.uniform(-120, 120, size=(len(bad), 2))
    ok, rvec, tvec, inliers = solve_pnp_ransac(m, X, uv, seed=0)
    assert ok
    ang, terr = _pose_err(cv2.Rodrigues(rvec)[0], tvec, R_gt, t_gt)
    assert ang < 0.5 and terr < 0.05                      # pose held despite 25% blunders
    assert not inliers[bad].any()                         # every injected blunder rejected


def test_narrow_fov_noncoplanar_unaffected():
    """Backward-consistency: a forward-hemisphere (z>0) non-coplanar view still solves correctly."""
    m = DoubleSphereModel.sample()
    rng = np.random.default_rng(3)
    R_gt, t_gt = so3_exp([0.05, -0.1, 0.02]), np.array([0.1, -0.05, 2.2])
    X = rng.uniform([-0.4, -0.4, -0.2], [0.4, 0.4, 0.2], size=(40, 3))
    Xc = X @ R_gt.T + t_gt
    assert (Xc[:, 2] > 0).all() and not _is_coplanar(X)
    uv, val = m.project(Xc)
    ok, rvec, tvec, _ = solve_pnp_ransac(m, X[val], np.asarray(uv)[val], seed=0)
    assert ok
    ang, terr = _pose_err(cv2.Rodrigues(rvec)[0], tvec, R_gt, t_gt)
    assert ang < 1e-2 and terr < 1e-2


# Traceability: PnP on bearing rays for any model (ops) + robust resection/PnP seeding (calib).
pytestmark = pytest.mark.req("FR-OPS-003", "FR-CALIB-002")
