"""Two-view geometry on bearing vectors — verified against a known synthetic scene.

Generate a random pose + 3D points, turn them into unit bearing vectors (no pixels, no
camera model — the math is model-agnostic), and check we recover the pose and structure
to numerical precision.
"""

import numpy as np
import pytest

from ds_msp.mvg import (
    decompose_essential,
    epipolar_residual,
    essential_from_rays,
    recover_pose,
    relative_pose,
    triangulate_rays,
)


def _rot(axis, angle):
    """Rodrigues rotation (no scipy/cv2 dependency)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _scene(n=40, seed=0):
    """A random two-view scene; returns rays + the ground-truth pose and points."""
    rng = np.random.default_rng(seed)
    R = _rot(rng.standard_normal(3), 0.6)
    t = rng.standard_normal(3)
    t = t / np.linalg.norm(t)                       # unit (essential is scale-free)
    # points spread in front of camera 1 (z>0), at varied depth
    X1 = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-2, 2, n), rng.uniform(2, 8, n)])
    X2 = (R @ X1.T).T + t
    keep = X2[:, 2] > 0.1                            # in front of camera 2 too
    X1, X2 = X1[keep], X2[keep]
    f1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
    f2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
    return f1, f2, R, t, X1


def _rot_err_deg(R, Rhat):
    c = (np.trace(R.T @ Rhat) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _vec_ang_deg(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1)))


def test_epipolar_constraint_holds_on_rays():
    f1, f2, R, t, _ = _scene()
    E = essential_from_rays(f1, f2)
    assert np.abs(epipolar_residual(E, f1, f2)).max() < 1e-10


def test_recover_pose_matches_ground_truth():
    f1, f2, R, t, _ = _scene(seed=1)
    Rhat, that, _ = recover_pose(f1, f2)
    assert _rot_err_deg(R, Rhat) < 1e-5
    assert _vec_ang_deg(t, that) < 1e-5              # translation direction (sign fixed by cheirality)
    assert that @ t > 0                              # correct sign, not the mirrored solution


def test_triangulation_recovers_points_at_true_scale():
    # t is unit and the ground-truth scene was built with that unit t, so triangulation
    # under the recovered (R, unit-t) reproduces the actual 3D points.
    f1, f2, R, t, X1 = _scene(seed=2)
    Rhat, that, X = recover_pose(f1, f2)
    assert np.allclose(X, X1, atol=1e-8)


def test_all_triangulated_points_are_in_front():
    f1, f2, R, t, _ = _scene(seed=3)
    Rhat, that, _ = recover_pose(f1, f2)
    _, d1, d2 = triangulate_rays(f1, f2, Rhat, that)
    assert np.all(d1 > 0) and np.all(d2 > 0)


def test_decompose_gives_four_proper_rotations():
    f1, f2, _, _, _ = _scene(seed=4)
    E = essential_from_rays(f1, f2)
    cands = decompose_essential(E)
    assert len(cands) == 4
    for R, t in cands:
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
        assert np.isclose(np.linalg.norm(t), 1.0, atol=1e-9)


def test_too_few_correspondences_raises():
    f = np.eye(3)[:3]                                # 3 rays
    with pytest.raises(ValueError, match="≥8"):
        essential_from_rays(f, f)


def test_relative_pose_is_model_agnostic_smoke():
    # Same recovery works regardless of where the rays came from — here, plain directions.
    f1, f2, R, t, _ = _scene(seed=5)
    Rhat, that = relative_pose(f1, f2)
    assert _rot_err_deg(R, Rhat) < 1e-5


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# ---------------------------------------------------------------------------
# Geometry-aware properties of the essential matrix (see docs/learn/two_view_geometry.md)
# ---------------------------------------------------------------------------

def test_essential_singular_values_are_one_one_zero():
    """E = [t]_× R has singular values (σ, σ, 0); the manifold projection sets σ=1."""
    f1, f2, _, _, _ = _scene(seed=10)
    s = np.linalg.svd(essential_from_rays(f1, f2), compute_uv=False)
    assert np.allclose(s, [1.0, 1.0, 0.0], atol=1e-9)


def test_essential_determinant_is_zero():
    f1, f2, _, _, _ = _scene(seed=11)
    assert abs(np.linalg.det(essential_from_rays(f1, f2))) < 1e-9


def test_essential_satisfies_characterization_equation():
    """A 3×3 matrix is essential iff 2 E Eᵀ E − tr(E Eᵀ) E = 0 (Huang–Faugeras)."""
    f1, f2, _, _, _ = _scene(seed=12)
    E = essential_from_rays(f1, f2)
    assert np.abs(2 * E @ E.T @ E - np.trace(E @ E.T) * E).max() < 1e-9


def test_essential_matches_skew_t_times_R_up_to_sign():
    """The recovered E equals [t]_× R built from ground truth, up to sign and scale."""
    f1, f2, R, t, _ = _scene(seed=13)
    E = essential_from_rays(f1, f2)
    E_gt = _skew(t) @ R
    E = E / np.linalg.norm(E)
    E_gt = E_gt / np.linalg.norm(E_gt)
    assert min(np.linalg.norm(E - E_gt), np.linalg.norm(E + E_gt)) < 1e-8


def test_essential_is_scale_invariant_in_the_rays():
    """Rays need not be unit: scaling each ray leaves E unchanged (it's a direction)."""
    f1, f2, _, _, _ = _scene(seed=14)
    rng = np.random.default_rng(0)
    s1 = rng.uniform(0.5, 3, (f1.shape[0], 1))
    s2 = rng.uniform(0.5, 3, (f2.shape[0], 1))
    E_a = essential_from_rays(f1, f2)
    E_b = essential_from_rays(f1 * s1, f2 * s2)
    E_a, E_b = E_a / np.linalg.norm(E_a), E_b / np.linalg.norm(E_b)
    assert min(np.linalg.norm(E_a - E_b), np.linalg.norm(E_a + E_b)) < 1e-9


def test_triangulated_points_reproject_onto_both_rays():
    """The triangulated point's direction matches f1, and (R X + t) matches f2 — to ~0°."""
    f1, f2, R, t, _ = _scene(seed=15)
    Rh, th, X = recover_pose(f1, f2)
    dir1 = X / np.linalg.norm(X, axis=1, keepdims=True)
    X2 = (Rh @ X.T).T + th
    dir2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
    ang1 = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", dir1, f1), -1, 1)))
    ang2 = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", dir2, f2), -1, 1)))
    assert ang1.max() < 1e-4 and ang2.max() < 1e-4


def test_noise_degrades_gracefully_and_monotonically():
    """Stability: pose error grows ~linearly with ray noise, no blow-up at small σ."""
    f1, f2, R, t, _ = _scene(seed=16)
    rng = np.random.default_rng(3)
    prev = -1.0
    for sig, bound in [(1e-4, 0.3), (1e-3, 3.0)]:
        f1n = f1 + sig * rng.standard_normal(f1.shape)
        f2n = f2 + sig * rng.standard_normal(f2.shape)
        Rh, _, _ = recover_pose(f1n, f2n)
        err = _rot_err_deg(R, Rh)
        assert err < bound                      # bounded
        assert err > prev                       # grows with noise
        prev = err


def test_recover_pose_through_a_real_double_sphere_camera():
    """End-to-end on a wide-FOV model: 3D points -> pixels (two views) -> unproject to
    rays -> recover pose. The rays come from `DoubleSphereModel`, proving the geometry is
    model-agnostic: nothing in `mvg` knows it's a fisheye."""
    from ds_msp.models import DoubleSphereModel

    cam = DoubleSphereModel(fx=300.0, fy=300.0, cx=320.0, cy=320.0, xi=0.3, alpha=0.6)
    rng = np.random.default_rng(7)
    R = _rot(rng.standard_normal(3), 0.5)
    t = rng.standard_normal(3)
    t = t / np.linalg.norm(t)
    X1 = np.column_stack([rng.uniform(-3, 3, 60), rng.uniform(-3, 3, 60), rng.uniform(2, 9, 60)])
    X2 = (R @ X1.T).T + t

    uv1, ok1 = cam.project(X1)
    uv2, ok2 = cam.project(X2)
    ok = ok1 & ok2
    assert ok.sum() >= 8
    f1, _ = cam.unproject(uv1[ok])
    f2, _ = cam.unproject(uv2[ok])

    Rhat, that, _ = recover_pose(f1, f2)
    assert _rot_err_deg(R, Rhat) < 1e-3          # limited only by project/unproject round-trip
    assert _vec_ang_deg(t, that) < 1e-3
    assert that @ t > 0
