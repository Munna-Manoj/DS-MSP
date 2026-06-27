"""Robust relative pose: spherical normalization + RANSAC with an angular residual."""

import numpy as np
import pytest

from ds_msp.mvg import essential_from_rays, recover_pose
from ds_msp.mvg.ransac import ransac_relative_pose, sampson_residual


def _rot(axis, angle):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _rot_err_deg(R, Rh):
    return np.degrees(np.arccos(np.clip((np.trace(R.T @ Rh) - 1) / 2, -1, 1)))


def _scene(n=100, seed=0, cone=False):
    rng = np.random.default_rng(seed)
    R = _rot(rng.standard_normal(3), 0.6)
    t = rng.standard_normal(3)
    t = t / np.linalg.norm(t)
    if cone:                                   # clustered forward cone → ill-conditioned
        X1 = np.column_stack([rng.uniform(-0.6, 0.6, n), rng.uniform(-0.6, 0.6, n), rng.uniform(4, 6, n)])
    else:
        X1 = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-2, 2, n), rng.uniform(2, 8, n)])
    X2 = (R @ X1.T).T + t
    f1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
    f2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
    return f1, f2, R, t


def test_spherical_normalization_is_exact_in_the_noise_free_limit():
    f1, f2, _, _ = _scene(seed=0)
    Ea = essential_from_rays(f1, f2)
    Eb = essential_from_rays(f1, f2, normalize=True)
    Ea, Eb = Ea / np.linalg.norm(Ea), Eb / np.linalg.norm(Eb)
    assert min(np.linalg.norm(Ea - Eb), np.linalg.norm(Ea + Eb)) < 1e-9


def test_spherical_normalization_improves_conditioning_on_clustered_rays():
    # Single noisy trials are high-variance, so compare the MEAN error over many realizations
    # of a narrow-cone (ill-conditioned) scene: whitening should win on average.
    sig = 3e-3
    plain, norm = [], []
    for s in range(15):
        f1, f2, R, _ = _scene(seed=s, cone=True)
        rng = np.random.default_rng(100 + s)
        f1n = f1 + sig * rng.standard_normal(f1.shape)
        f2n = f2 + sig * rng.standard_normal(f2.shape)
        plain.append(_rot_err_deg(R, recover_pose(f1n, f2n, essential_from_rays(f1n, f2n))[0]))
        norm.append(_rot_err_deg(R, recover_pose(f1n, f2n, essential_from_rays(f1n, f2n, normalize=True))[0]))
    assert np.median(norm) < np.median(plain)  # whitening helps (median) on clustered rays


def test_sampson_residual_is_zero_for_perfect_correspondences():
    f1, f2, R, t = _scene(seed=3)
    E = essential_from_rays(f1, f2)
    assert sampson_residual(E, f1, f2).max() < 1e-9


def test_sampson_residual_flags_outliers():
    f1, f2, _, _ = _scene(seed=4)
    E = essential_from_rays(f1, f2)
    r = sampson_residual(E, f1, f2)
    rng = np.random.default_rng(0)
    f2_bad = f2.copy()
    f2_bad[::5] = rng.standard_normal((f2_bad[::5].shape[0], 3))     # corrupt 20%
    f2_bad = f2_bad / np.linalg.norm(f2_bad, axis=1, keepdims=True)
    r_bad = sampson_residual(E, f1, f2_bad)
    assert r.max() < 1e-9 and r_bad[::5].min() > 1e-2               # outliers are large


def test_ransac_recovers_pose_under_30pct_outliers():
    f1, f2, R, t = _scene(n=120, seed=3)
    rng = np.random.default_rng(4)
    out = rng.random(120) < 0.30
    f2o = f2.copy()
    f2o[out] = rng.standard_normal((int(out.sum()), 3))
    f2o = f2o / np.linalg.norm(f2o, axis=1, keepdims=True)

    Rh, th, inliers = ransac_relative_pose(f1, f2o, threshold=0.005, seed=0)

    assert _rot_err_deg(R, Rh) < 0.5
    assert np.degrees(np.arccos(np.clip(abs(th @ t), -1, 1))) < 2.0
    # inlier mask agrees with ground truth
    gt = ~out
    prec = (inliers & gt).sum() / max(inliers.sum(), 1)
    rec = (inliers & gt).sum() / gt.sum()
    assert prec > 0.95 and rec > 0.9


def test_ransac_beats_naive_eight_point_with_outliers():
    f1, f2, R, t = _scene(n=100, seed=7)
    rng = np.random.default_rng(2)
    out = rng.random(100) < 0.25
    f2o = f2.copy()
    f2o[out] = rng.standard_normal((int(out.sum()), 3))
    f2o = f2o / np.linalg.norm(f2o, axis=1, keepdims=True)
    naive = _rot_err_deg(R, recover_pose(f1, f2o, essential_from_rays(f1, f2o))[0])
    robust = _rot_err_deg(R, ransac_relative_pose(f1, f2o, seed=1)[0])
    assert robust < naive and robust < 0.5


def test_ransac_too_few_correspondences_raises():
    f = np.eye(3)[:5]
    with pytest.raises(ValueError, match="≥8"):
        ransac_relative_pose(f, f)
