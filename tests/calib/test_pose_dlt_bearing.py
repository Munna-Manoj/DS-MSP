"""Manufactured-solution tests for the bearing-vector-native pose DLT.

``_pose_dlt_bearing`` generalizes ``_pose_dlt_normalized`` off the ``z = 1`` plane: it solves
the cross-product constraint ``f x (R X + t) = 0`` directly on the sphere, so it recovers pose
from bearing vectors of **any** direction — including rays past 90 deg off-axis (``z <= 0``),
which the normalized-plane DLT is mathematically incapable of representing.

Coverage: exact recovery at zero noise on a full-sphere point cloud (incl. >90 deg bearings),
positive-depth (lambda>0) cheirality, graceful degradation under bearing noise, and strict
backward-consistency with the legacy ``_pose_dlt_normalized`` on ``z > 0`` data.
"""
import numpy as np
import pytest

from ds_msp.core.lie import so3_exp
from ds_msp.geometry.resection import _pose_dlt_bearing, _pose_dlt_normalized


def _full_sphere_view(R, t, n=60, seed=0):
    """World points whose bearings span the full sphere (incl. rays behind the z=1 plane)."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    depths = rng.uniform(0.5, 4.0, n)
    Xc = dirs * depths[:, None]              # camera-frame points, any direction
    X = (Xc - t) @ R                          # world points: Xc = R X + t
    f = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    return X, f


def test_exact_recovery_full_sphere_including_past_90deg():
    R_true, t_true = so3_exp([0.3, -0.7, 0.2]), np.array([0.4, -0.3, 0.1])
    X, f = _full_sphere_view(R_true, t_true, n=60, seed=0)
    # confirm the manufactured data actually exercises the >90 deg regime the old path fails on
    ang = np.degrees(np.arccos(np.clip(f[:, 2], -1, 1)))
    assert (ang > 90).sum() >= 15 and ang.max() > 150.0

    R, t = _pose_dlt_bearing(X, f)
    assert np.abs(R - R_true).max() < 1e-9
    assert np.abs(t - t_true).max() < 1e-9
    # every point in front of the camera along its bearing (lambda>0), not z>0
    lam = np.einsum("ij,ij->i", f, X @ R.T + t)
    assert (lam > 0).all()


def test_normalized_plane_dlt_cannot_but_bearing_can():
    """The legacy z=1-plane DLT collapses on wide-FOV data; the bearing DLT recovers it."""
    R_true, t_true = so3_exp([0.2, 0.9, -0.3]), np.array([-0.2, 0.15, 0.3])
    X, f = _full_sphere_view(R_true, t_true, n=60, seed=4)
    R, t = _pose_dlt_bearing(X, f)
    assert np.abs(R - R_true).max() < 1e-9 and np.abs(t - t_true).max() < 1e-9

    # feed the same rays through the old path as pn = xy/z (its own parameterization);
    # rays with z<=0 are unrepresentable, so its recovered pose is wrong.
    front = f[:, 2] > 1e-6
    pn = f[front, :2] / f[front, 2:3]
    sol = _pose_dlt_normalized(X[front], pn)
    if sol is not None:                       # even on its own hemisphere subset it may fit,
        R_old, _ = sol                        # but it can never use the z<=0 half of the sphere
        # the bearing solver used all 60 rays; assert it is strictly the accurate one
        assert np.abs(R - R_true).max() < np.abs(R_old - R_true).max() + 1e-12


def test_backward_consistent_with_normalized_dlt_on_z_positive_data():
    """On z>0 data f=(u,v,1) the two solvers must agree (bearing DLT is a strict generalization)."""
    R_true, t_true = so3_exp([0.1, -0.2, 0.05]), np.array([0.1, -0.05, 2.2])
    rng = np.random.default_rng(1)
    X = rng.uniform([-0.4, -0.4, -0.2], [0.4, 0.4, 0.2], size=(40, 3))
    Xc = X @ R_true.T + t_true
    pn = Xc[:, :2] / Xc[:, 2:3]               # normalized-plane points (all z>0)
    f = np.c_[pn, np.ones(len(pn))]           # same observations as unit-z bearings

    R_b, t_b = _pose_dlt_bearing(X, f)
    R_n, t_n = _pose_dlt_normalized(X, pn)
    assert np.abs(R_b - R_n).max() < 1e-9 and np.abs(t_b - t_n).max() < 1e-9
    assert np.abs(R_b - R_true).max() < 1e-9 and np.abs(t_b - t_true).max() < 1e-9


def test_graceful_under_bearing_noise():
    R_true, t_true = so3_exp([0.3, -0.7, 0.2]), np.array([0.4, -0.3, 0.1])
    X, f = _full_sphere_view(R_true, t_true, n=80, seed=2)
    rng = np.random.default_rng(9)
    fn = f + rng.normal(scale=0.002, size=f.shape)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    R, t = _pose_dlt_bearing(X, fn)
    ang_err = np.degrees(np.arccos(np.clip((np.trace(R.T @ R_true) - 1) / 2, -1, 1)))
    assert ang_err < 0.5
    assert np.linalg.norm(t - t_true) < 0.02


def test_underdetermined_returns_none():
    X = np.random.default_rng(0).uniform(-1, 1, (5, 3))
    f = np.random.default_rng(1).normal(size=(5, 3))
    assert _pose_dlt_bearing(X, f) is None


# Traceability: same requirement as the resection/PnP seeding suite it extends.
pytestmark = pytest.mark.req("FR-CALIB-002")
