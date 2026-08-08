"""Bearing-native ``robust_pose_irls`` (ADR-0020): mixed full-sphere and zero-forward cases.

Before ADR-0020, the IRLS refine residual was the normalized-plane reprojection error
(``z > 0`` only): an all-peripheral view returned the RANSAC warm-start pose completely
unrefined (``idx.size < 4`` bail-out), and mixed views refined against only their forward
subset. After the fix, the chordal bearing residual covers every usable point regardless of
``z`` sign. The statistical fixture below contains 20.7--36.0% peripheral rays across its 15
seeds; a separate fixture covers the literal zero-forward case.
"""
import numpy as np
import pytest

from ds_msp.core.lie import so3_exp
from ds_msp.geometry.resection import _is_coplanar
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.rig.pose_init import estimate_pose_gnc, estimate_pose_ransac, robust_pose_irls

R_GT, T_GT = so3_exp([0.2, -0.3, 0.15]), np.array([0.1, -0.05, -0.2])


def _pose_err(T, R_gt=R_GT, t_gt=T_GT):
    ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3].T @ R_gt) - 1) / 2, -1, 1)))
    return ang, float(np.linalg.norm(T[:3, 3] - t_gt))


def _wide_fov_scene(model, n=80, seed=0):
    """Non-coplanar world points whose bearings span the full sphere (all directions, many
    past 90 deg off-axis) under the fixed ground-truth pose."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    depth = rng.uniform(0.6, 3.0, n)
    Xc = dirs * depth[:, None]
    X = (Xc - T_GT) @ R_GT
    uv, val = model.project(Xc)
    return X[val], np.asarray(uv)[val]


def test_scene_is_genuinely_mixed_sphere_and_noncoplanar():
    model = DoubleSphereModel.sample()
    X, uv = _wide_fov_scene(model)
    rays, valid = model.unproject(uv)
    assert valid.all()
    off = np.degrees(np.arccos(np.clip(rays[:, 2], -1, 1)))
    assert (off > 90).mean() > 0.15, "scene must include meaningful peripheral coverage"
    assert not _is_coplanar(X)


def test_refine_measurably_improves_on_mixed_sphere_noisy_data():
    """Real measured numbers (15 seeds, 2px pixel noise, mixed full-sphere scene):
    median rotation error 0.162 deg (warm-start-only) -> 0.044 deg (refined), a ~73%
    reduction; median translation error 3.8mm -> 2.0mm, a ~47% reduction. This is exactly the
    Bounds below are conservative relative to the measured reduction."""
    model = DoubleSphereModel.sample()
    warm_rot, ref_rot, warm_t, ref_t = [], [], [], []
    for seed in range(15):
        X, uv = _wide_fov_scene(model, seed=seed)
        if len(X) < 10:
            continue
        rng = np.random.default_rng(1000 + seed)
        uv_noisy = uv + rng.normal(scale=2.0, size=uv.shape)
        T_warm, _ = estimate_pose_ransac(model, X, uv_noisy, seed=seed)
        if T_warm is None:
            continue
        T_ref = robust_pose_irls(model, X, uv_noisy, T0=T_warm.copy(), seed=seed)
        a1, d1 = _pose_err(T_warm)
        a2, d2 = _pose_err(T_ref)
        warm_rot.append(a1)
        ref_rot.append(a2)
        warm_t.append(d1)
        ref_t.append(d2)

    assert len(warm_rot) >= 12, "guard: most seeds must yield a usable scene"
    assert np.median(ref_rot) < 0.7 * np.median(warm_rot), (
        f"refined median {np.median(ref_rot):.4f} deg not enough better than "
        f"warm-start median {np.median(warm_rot):.4f} deg")
    assert np.median(ref_t) < 0.7 * np.median(warm_t)


def test_refine_never_much_worse_than_warm_start_per_case():
    """The full-bearing-cost safety net bounds how much worse any single case can get: it
    compares fit-to-data, not distance-to-truth, so occasional mild regression from fitting
    noise is expected (normal estimator variance) -- but never a large one."""
    model = DoubleSphereModel.sample()
    for seed in range(20):
        X, uv = _wide_fov_scene(model, seed=seed)
        if len(X) < 10:
            continue
        rng = np.random.default_rng(2000 + seed)
        uv_noisy = uv + rng.normal(scale=2.0, size=uv.shape)
        T_warm, _ = estimate_pose_ransac(model, X, uv_noisy, seed=seed)
        if T_warm is None:
            continue
        T_ref = robust_pose_irls(model, X, uv_noisy, T0=T_warm.copy(), seed=seed)
        a1, _ = _pose_err(T_warm)
        a2, _ = _pose_err(T_ref)
        assert a2 < a1 + 0.1, f"seed {seed}: refined {a2:.4f} deg vs warm {a1:.4f} deg"


def test_no_forward_points_still_refines_not_just_seeds():
    """A scene with literally zero z>0 points (impossible for the old code to refine at all)
    must still improve over the warm start, not just return it."""
    model = DoubleSphereModel.sample()
    rng = np.random.default_rng(7)
    n = 200
    # bearings confined to z < -0.3 (all strictly behind the z=0 plane -> zero forward points)
    dirs = rng.normal(size=(n, 3))
    dirs[:, 2] = -np.abs(dirs[:, 2]) - 0.3
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    depth = rng.uniform(0.6, 3.0, n)
    Xc = dirs * depth[:, None]
    X = (Xc - T_GT) @ R_GT
    uv, val = model.project(Xc)
    X, uv = X[val], np.asarray(uv)[val]
    assert len(X) >= 15
    rays, _ = model.unproject(uv)
    assert (rays[:, 2] <= 0).all(), "guard: every point must be non-forward"

    noisy = uv + np.random.default_rng(9).normal(scale=1.5, size=uv.shape)
    T_warm, _ = estimate_pose_ransac(model, X, noisy, seed=0)
    assert T_warm is not None
    T_ref = robust_pose_irls(model, X, noisy, T0=T_warm.copy(), seed=0)
    assert not np.allclose(T_ref, T_warm), "must actually refine, not pass through the seed"
    a1, _ = _pose_err(T_warm)
    a2, _ = _pose_err(T_ref)
    assert a2 <= a1 + 1e-9


def test_five_point_noncoplanar_view_uses_deterministic_small_sample_seed():
    """Four or five non-coplanar points cannot determine the six-point bearing DLT.

    The implicit pose path must use its declared deterministic SQPnP chart seed instead of
    silently optimizing from identity (the former failure was about 144 degrees / 0.75 m).
    """
    model = DoubleSphereModel.sample()
    rng = np.random.default_rng(58)
    R_gt = so3_exp(rng.uniform(-0.8, 0.8, 3))
    t_gt = np.r_[rng.uniform(-0.5, 0.5, 2), rng.uniform(0.8, 3.0)]
    X = rng.uniform([-0.8, -0.8, -0.5], [0.8, 0.8, 0.5], (5, 3))
    assert not _is_coplanar(X)
    uv, valid = model.project(X @ R_gt.T + t_gt)
    assert valid.all()
    uv = np.asarray(uv) + rng.normal(0.0, 0.5, np.asarray(uv).shape)
    assert estimate_pose_gnc(model, X, uv)[0] is None  # exercises the 4--5 point fallback

    T = robust_pose_irls(model, X, uv)

    assert T is not None
    rotation, translation = _pose_err(T, R_gt, t_gt)
    assert rotation < 0.5
    assert translation < 0.005


def test_five_point_noncoplanar_peripheral_view_uses_ray_aligned_chart():
    """The small-sample fallback must not make the physical camera's z axis a requirement."""
    model = DoubleSphereModel.sample()
    R_gt = so3_exp([0.2, -0.3, 0.15])
    t_gt = np.array([0.1, -0.05, -0.2])
    theta = np.radians([96.0, 99.0, 102.0, 105.0, 108.0])
    phi = np.radians([-10.0, -5.0, 0.0, 6.0, 12.0])
    directions = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    camera_points = directions * np.array([0.7, 1.1, 1.5, 2.0, 2.7])[:, None]
    X = (camera_points - t_gt) @ R_gt
    assert not _is_coplanar(X)
    uv, valid = model.project(camera_points)
    rays, ray_valid = model.unproject(uv)
    assert valid.all() and ray_valid.all() and (rays[:, 2] < 0.0).all()
    assert estimate_pose_gnc(model, X, uv)[0] is None

    T = robust_pose_irls(model, X, uv)

    assert T is not None
    rotation, translation = _pose_err(T)
    assert rotation < 1e-6
    assert translation < 1e-9


@pytest.mark.parametrize("n", [4, 5])
def test_small_sample_chart_finds_true_hemisphere_not_ray_mean(n):
    """An asymmetric open-hemisphere set can point opposite its normalized ray mean."""
    model = DoubleSphereModel.sample()
    angle = np.radians([89.0, -89.0, -88.0, -87.0, -86.0])
    rays = np.column_stack([np.sin(angle), np.zeros(5), np.cos(angle)])
    rays[:, 1] = [0.0, 0.01, -0.01, 0.02, -0.02]
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    camera_points = rays * np.array([0.8, 1.1, 1.4, 1.8, 2.3])[:, None]
    X = camera_points[:n]
    assert not _is_coplanar(X)
    uv, valid = model.project(camera_points[:n])
    observed, ray_valid = model.unproject(uv)
    assert valid.all() and ray_valid.all() and (observed[:, 2] > 0.0).all()
    ray_mean = observed.sum(axis=0)
    ray_mean /= np.linalg.norm(ray_mean)
    assert np.any(observed @ ray_mean <= 0.0)  # the rejected normalized-mean shortcut

    T = robust_pose_irls(model, X, uv)

    assert T is not None
    rotation, translation = _pose_err(T, np.eye(3), np.zeros(3))
    assert rotation < 1e-6
    assert translation < 1e-9


def test_four_point_seed_selects_ap3p_when_sqpnp_branch_is_catastrophic():
    """Four-point SQPnP can return a physical but wrong branch; score all deterministic roots."""
    model = DoubleSphereModel.sample()
    rng = np.random.default_rng(40127)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    rays = axis + rng.normal(scale=0.28, size=(4, 3))
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    camera_points = rays * rng.uniform(0.7, 3.0, 4)[:, None]
    R_gt = so3_exp(rng.uniform(-0.7, 0.7, 3))
    t_gt = rng.uniform(-0.4, 0.4, 3)
    X = (camera_points - t_gt) @ R_gt
    assert not _is_coplanar(X)
    uv, valid = model.project(camera_points)
    assert valid.all()
    uv = np.asarray(uv) + rng.normal(scale=0.5, size=(4, 2))

    T = robust_pose_irls(model, X, uv)

    assert T is not None
    rotation, translation = _pose_err(T, R_gt, t_gt)
    assert rotation < 1.0
    assert translation < 0.02


# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("FR-CALIB-002")
