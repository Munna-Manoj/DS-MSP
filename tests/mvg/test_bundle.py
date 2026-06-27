"""Angular two-view bundle refinement: tightens the algebraic estimate on noisy rays."""

import numpy as np

from ds_msp.mvg import estimate_relative_pose, recover_pose
from ds_msp.mvg.bundle import angular_reprojection_error, refine_two_view


def _rot(axis, angle):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _rot_err_deg(R, Rh):
    return np.degrees(np.arccos(np.clip((np.trace(R.T @ Rh) - 1) / 2, -1, 1)))


def _scene(n=50, seed=0, noise=2e-3):
    rng = np.random.default_rng(seed)
    R = _rot(rng.standard_normal(3), 0.6)
    t = rng.standard_normal(3)
    t = t / np.linalg.norm(t)
    X1 = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-2, 2, n), rng.uniform(2, 8, n)])
    X2 = (R @ X1.T).T + t
    f1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
    f2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
    f1 = f1 + noise * rng.standard_normal(f1.shape)
    f2 = f2 + noise * rng.standard_normal(f2.shape)
    f1 /= np.linalg.norm(f1, axis=1, keepdims=True)
    f2 /= np.linalg.norm(f2, axis=1, keepdims=True)
    return f1, f2, R, t


def test_angular_error_zero_for_exact_geometry():
    f1, f2, R, t = _scene(noise=0.0)
    R0, t0, X = recover_pose(f1, f2)
    assert angular_reprojection_error(f1, f2, R0, t0, X).max() < 1e-4


def test_refinement_reduces_angular_reprojection_error():
    f1, f2, R, t = _scene(seed=1, noise=3e-3)
    R0, t0, X0 = recover_pose(f1, f2)
    before = angular_reprojection_error(f1, f2, R0, t0, X0).mean()
    Rr, tr, Xr = refine_two_view(f1, f2, R0, t0, X0)
    after = angular_reprojection_error(f1, f2, Rr, tr, Xr).mean()
    assert after < before                                  # nonlinear refinement tightens the fit


def test_refinement_is_stable_at_large_rotation():
    """A ~165° rotation puts an absolute axis-angle near the ‖r‖=π singularity; the manifold
    perturbation (δω from 0) stays well-conditioned, so refinement still converges."""
    rng = np.random.default_rng(11)
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    R = _rot(axis, np.radians(165))
    t = rng.standard_normal(3)
    t = t / np.linalg.norm(t)
    X1 = np.column_stack([rng.uniform(-2, 2, 60), rng.uniform(-2, 2, 60), rng.uniform(2, 8, 60)])
    X2 = (R @ X1.T).T + t
    keep = X2[:, 2] > 0.1
    X1, X2 = X1[keep], X2[keep]
    f1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
    f2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
    f1 += 2e-3 * rng.standard_normal(f1.shape)
    f2 += 2e-3 * rng.standard_normal(f2.shape)
    f1 /= np.linalg.norm(f1, axis=1, keepdims=True)
    f2 /= np.linalg.norm(f2, axis=1, keepdims=True)

    R0, t0, X0 = recover_pose(f1, f2)
    seed_err = _rot_err_deg(R, R0)
    Rr, tr, Xr = refine_two_view(f1, f2, R0, t0, X0)
    # the manifold step stays well-conditioned at large rotation: it converges the residual and
    # does not diverge (an absolute-axis-angle param near ‖r‖=π would stall here).
    assert angular_reprojection_error(f1, f2, Rr, tr, Xr).mean() < 0.1
    assert _rot_err_deg(R, Rr) <= seed_err + 1e-9


def test_refinement_improves_pose_accuracy_on_average():
    pose_lin, pose_ref = [], []
    for s in range(8):
        f1, f2, R, t = _scene(seed=s, noise=3e-3)
        R0, t0, X0 = recover_pose(f1, f2)
        Rr, tr, Xr = refine_two_view(f1, f2, R0, t0, X0)
        pose_lin.append(_rot_err_deg(R, R0))
        pose_ref.append(_rot_err_deg(R, Rr))
    assert np.mean(pose_ref) <= np.mean(pose_lin) + 1e-9   # refinement never worse on average


def _contaminate(f1, f2, frac, seed):
    """Replace a fraction of f2 rays with random directions (gross correspondence outliers)."""
    rng = np.random.default_rng(seed)
    f2 = f2.copy()
    k = int(frac * len(f2))
    idx = rng.choice(len(f2), k, replace=False)
    rnd = rng.standard_normal((k, 3))
    f2[idx] = rnd / np.linalg.norm(rnd, axis=1, keepdims=True)
    return f2


def test_estimate_relative_pose_is_robust_to_outliers():
    """The end-to-end estimator (RANSAC consensus → triangulate → manifold refine) must stay
    sub-degree under 25% gross correspondence outliers, where naive recover_pose+refine — which
    feeds every contaminated ray into the least-squares eight-point — blows up by tens of degrees."""
    errs_naive, errs_robust = [], []
    for s in range(8):
        f1, f2, R, t = _scene(n=100, seed=s, noise=2e-3)
        f2c = _contaminate(f1, f2, 0.25, seed=s + 100)
        R0, t0, X0 = recover_pose(f1, f2c)
        Rn, _, _ = refine_two_view(f1, f2c, R0, t0, X0)
        errs_naive.append(_rot_err_deg(R, Rn))
        Rr, _, _, inl = estimate_relative_pose(f1, f2c, threshold=0.01, seed=s)
        errs_robust.append(_rot_err_deg(R, Rr))

    assert np.mean(errs_robust) < 1.5                       # robust path stays accurate
    assert np.mean(errs_robust) < 0.2 * np.mean(errs_naive)  # and dramatically beats naive


def test_estimate_relative_pose_stays_accurate_on_clean_data():
    """No outliers: the RANSAC wrapper must not hurt. It keeps essentially all rays as inliers and
    stays sub-degree on average. (RANSAC is outlier *insurance*, not an accuracy upgrade on clean
    data — a minimal-sample essential matrix can even be a touch noisier on any single draw — so we
    average over seeds rather than assert a single lucky run.)"""
    errs = []
    for s in range(6):
        f1, f2, R, t = _scene(n=80, seed=s, noise=2e-3)
        Rr, tr, Xr, inl = estimate_relative_pose(f1, f2, threshold=0.01, seed=s)
        assert inl.sum() >= 65                              # nearly all rays are inliers
        errs.append(_rot_err_deg(R, Rr))
    assert np.mean(errs) < 0.5
