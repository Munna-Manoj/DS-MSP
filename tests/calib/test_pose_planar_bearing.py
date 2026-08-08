"""Manufactured-solution tests for the bearing-vector-native **planar** pose solver.

``_pose_planar_bearing`` generalizes ``_pose_planar_normalized`` off the ``z = 1`` plane: it
solves the cross-product homography constraint ``f x (H . [a,b,1]) = 0`` directly on the
bearing vector, so a single **coplanar** calibration board recovers pose even when some corners
are past 90 deg off-axis (``z <= 0``) — a board imaged edge-on enough that the classic
normalized-plane homography (``pn = xy/z``) cannot represent every corner. This is the coplanar
counterpart of ``test_pose_dlt_bearing.py`` (ADR-0019 vs. ADR-0018).

Coverage: exact recovery at zero noise on a tilted board (incl. >90 deg corners),
depth-along-bearing (lambda>0) cheirality, graceful degradation under bearing noise, and strict
backward-consistency with the legacy ``_pose_planar_normalized`` on ``z > 0`` data.
"""
import numpy as np
import pytest

from ds_msp.core.lie import so3_exp
from ds_msp.geometry.resection import _pose_planar_bearing, _pose_planar_normalized


def _tilted_board_view(R, t, n=48, half=1.0, seed=0):
    """A coplanar (Z=0, object frame) board whose bearings include some past 90 deg off-axis.

    Points are generated directly in camera frame on the affine plane spanned by ``R e1``,
    ``R e2`` through ``R c0 + t`` (the exact model ``_pose_planar_bearing`` assumes), so `X`
    is coplanar in the object frame by construction for ANY ``R, t``.
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(-half, half, n)
    b = rng.uniform(-half, half, n)
    X = np.column_stack([a, b, np.zeros(n)])   # coplanar in object frame (Z=0)
    Xc = X @ R.T + t
    f = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    return X, f


# A board tilted ~70 deg from fronto-parallel, close to the camera: a genuine mix of forward
# and past-90-deg corners (verified below, not assumed).
_R_TILT = so3_exp([0.0, 1.22, 0.0])
_T_NEAR = np.array([0.05, -0.02, 0.6])


def test_scene_actually_exercises_past_90deg_corners():
    X, f = _tilted_board_view(_R_TILT, _T_NEAR, n=48, half=1.0, seed=0)
    ang = np.degrees(np.arccos(np.clip(f[:, 2], -1, 1)))
    assert (ang > 90).sum() >= 5, "manufactured scene must actually include peripheral corners"


def test_exact_recovery_with_some_corners_past_90deg():
    X, f = _tilted_board_view(_R_TILT, _T_NEAR, n=48, half=1.0, seed=0)
    R, t = _pose_planar_bearing(X, f)
    assert np.abs(R - _R_TILT).max() < 1e-8
    assert np.abs(t - _T_NEAR).max() < 1e-8
    lam = np.einsum("ij,ij->i", f, X @ R.T + t)
    assert (lam > 0).all()


def test_normalized_plane_homography_cannot_but_bearing_can():
    """The legacy z=1-plane homography collapses (or is simply inapplicable) on the corners
    past 90 deg; the bearing homography recovers the full pose from all of them."""
    X, f = _tilted_board_view(_R_TILT, _T_NEAR, n=48, half=1.0, seed=1)
    R, t = _pose_planar_bearing(X, f)
    assert np.abs(R - _R_TILT).max() < 1e-8 and np.abs(t - _T_NEAR).max() < 1e-8

    front = f[:, 2] > 1e-6
    assert front.sum() < len(f), "some corners must be excluded from the legacy hemisphere"
    pn = f[front, :2] / f[front, 2:3]
    sol = _pose_planar_normalized(X[front], pn)
    if sol is not None:
        R_old, _ = sol
        assert np.abs(R - _R_TILT).max() < np.abs(R_old - _R_TILT).max() + 1e-12


def test_backward_consistent_with_normalized_homography_on_z_positive_data():
    """On z>0 data f=(u,v,1) the two solvers must agree (bearing homography is a strict
    generalization)."""
    R_true, t_true = so3_exp([0.1, -0.15, 0.05]), np.array([0.05, -0.03, 2.0])
    X, f = _tilted_board_view(R_true, t_true, n=40, half=0.3, seed=2)
    assert (f[:, 2] > 1e-6).all(), "fronto-parallel-ish scene must stay all-forward"
    pn = f[:, :2] / f[:, 2:3]

    R_b, t_b = _pose_planar_bearing(X, f)
    R_n, t_n = _pose_planar_normalized(X, pn)
    assert np.abs(R_b - R_n).max() < 1e-9 and np.abs(t_b - t_n).max() < 1e-9
    assert np.abs(R_b - R_true).max() < 1e-8 and np.abs(t_b - t_true).max() < 1e-8


def test_graceful_under_bearing_noise():
    X, f = _tilted_board_view(_R_TILT, _T_NEAR, n=64, half=1.0, seed=3)
    rng = np.random.default_rng(9)
    fn = f + rng.normal(scale=0.002, size=f.shape)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    R, t = _pose_planar_bearing(X, fn)
    ang_err = np.degrees(np.arccos(np.clip((np.trace(R.T @ _R_TILT) - 1) / 2, -1, 1)))
    assert ang_err < 0.5
    assert np.linalg.norm(t - _T_NEAR) < 0.02


def test_underdetermined_returns_none():
    X = np.random.default_rng(0).uniform(-1, 1, (3, 3))
    X[:, 2] = 0.0
    f = np.random.default_rng(1).normal(size=(3, 3))
    assert _pose_planar_bearing(X, f) is None


# Traceability: same requirement as the resection/PnP seeding suite it extends.
pytestmark = pytest.mark.req("FR-CALIB-002")
