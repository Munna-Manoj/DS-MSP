"""Deterministic GNC-TLS replacement for the robust bearing-PnP front end."""

import warnings

import cv2
import numpy as np
import pytest

from ds_msp.core.lie import so3_exp
from ds_msp.model import DoubleSphereCamera
from ds_msp.models.dsplus import DSPlusModel
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.eucm import EUCMModel
from ds_msp.models.kb import KannalaBrandtModel
from ds_msp.models.ocam import OCamModel
from ds_msp.models.radtan import RadTanModel
from ds_msp.models.ucm import UCMModel
from ds_msp.ops import solve_pnp_robust
from ds_msp.testing import FakeModel


MODELS = [
    DoubleSphereModel.sample,
    UCMModel.sample,
    EUCMModel.sample,
    KannalaBrandtModel.sample,
    OCamModel.sample,
    DSPlusModel.sample,
    RadTanModel.sample,
    FakeModel.sample,
]

WIDE_MODELS = [
    DoubleSphereModel.sample,
    UCMModel.sample,
    EUCMModel.sample,
    KannalaBrandtModel.sample,
    OCamModel.sample,
    DSPlusModel.sample,
]


def _pose_error(rvec, tvec, R_gt, t_gt):
    R = cv2.Rodrigues(rvec)[0]
    rotation = np.degrees(
        np.arccos(np.clip((np.trace(R.T @ R_gt) - 1.0) / 2.0, -1.0, 1.0))
    )
    return float(rotation), float(np.linalg.norm(tvec - t_gt))


def _contaminated_scene(kind, *, seed, outlier_fraction):
    model = DoubleSphereModel.sample()
    rng = np.random.default_rng(seed)
    n = 120
    if kind == "peripheral":
        R_gt = so3_exp([0.2, -0.3, 0.15])
        t_gt = np.array([0.1, -0.05, -0.2])
        theta = np.radians(rng.uniform(95.0, 110.0, n))
        phi = rng.uniform(0.0, 2.0 * np.pi, n)
        depth = rng.uniform(0.6, 3.0, n)
        camera_points = np.column_stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]) * depth[:, None]
        X = (camera_points - t_gt) @ R_gt
    else:
        R_gt = so3_exp([0.18, -0.22, 0.09])
        t_gt = np.array([0.12, -0.06, 2.1])
        X = (
            rng.uniform([-0.65, -0.65, -0.35], [0.65, 0.65, 0.35], (n, 3))
            if kind == "nonplanar"
            else np.column_stack([rng.uniform(-0.65, 0.65, (n, 2)), np.zeros(n)])
        )
        camera_points = X @ R_gt.T + t_gt

    uv, valid = model.project(camera_points)
    X, uv = X[valid], np.asarray(uv)[valid]
    uv = uv + rng.normal(0.0, 1.0, uv.shape)
    n_outliers = int(outlier_fraction * len(X))
    outlier_rows = rng.choice(len(X), n_outliers, replace=False)
    is_outlier = np.zeros(len(X), bool)
    is_outlier[outlier_rows] = True
    uv[outlier_rows] += rng.uniform(-80.0, 80.0, (n_outliers, 2))
    return model, X, uv, R_gt, t_gt, is_outlier


@pytest.mark.parametrize(
    ("kind", "seed", "outlier_fraction", "max_rotation_deg"),
    [
        ("nonplanar", 7, 0.70, 0.5),
        ("planar", 11, 0.60, 1.5),
        ("peripheral", 4, 0.60, 0.5),
    ],
)
def test_gnc_pnp_recovers_with_majority_outliers(
    kind, seed, outlier_fraction, max_rotation_deg
):
    model, X, uv, R_gt, t_gt, is_outlier = _contaminated_scene(
        kind, seed=seed, outlier_fraction=outlier_fraction
    )
    if kind == "peripheral":
        rays, valid = model.unproject(uv)
        assert (rays[valid & ~is_outlier, 2] < 0.0).all()

    ok, rvec, tvec, inliers = solve_pnp_robust(
        model, X, uv, noise_bound_px=4.0, max_iters=100
    )

    assert ok
    rotation, translation = _pose_error(rvec, tvec, R_gt, t_gt)
    assert rotation < max_rotation_deg
    assert translation < 0.01
    assert inliers.sum() >= 6
    assert not inliers[is_outlier].any()


def test_gnc_pnp_is_bit_deterministic():
    model, X, uv, _, _, _ = _contaminated_scene(
        "nonplanar", seed=7, outlier_fraction=0.70
    )

    a = solve_pnp_robust(model, X, uv, noise_bound_px=4.0)
    b = solve_pnp_robust(model, X, uv, noise_bound_px=4.0)

    assert a[0] and b[0]
    for lhs, rhs in zip(a[1:], b[1:]):
        np.testing.assert_array_equal(lhs, rhs)


def test_gnc_pnp_empty_input_fails_cleanly_without_runtime_warnings():
    model = DoubleSphereModel.sample()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = solve_pnp_robust(model, np.empty((0, 3)), np.empty((0, 2)))
    assert result[0] is False
    assert result[1] is None and result[2] is None
    assert result[3].shape == (0,)


def test_pixel_noise_bound_remains_calibrated_on_all_peripheral_rays():
    model = DoubleSphereModel.sample()
    rng = np.random.default_rng(1)
    R_gt = so3_exp([0.2, -0.3, 0.15])
    t_gt = np.array([0.1, -0.05, -0.2])
    theta = np.radians(rng.uniform(95.0, 110.0, 100))
    phi = rng.uniform(0.0, 2.0 * np.pi, 100)
    camera_points = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ]) * rng.uniform(0.6, 3.0, 100)[:, None]
    X = (camera_points - t_gt) @ R_gt
    uv_clean, valid = model.project(camera_points)
    assert valid.all()
    uv = np.asarray(uv_clean) + np.random.default_rng(9004).normal(0.0, 1.0, (100, 2))
    actual_pixel_error = np.linalg.norm(uv - uv_clean, axis=1)
    assert actual_pixel_error.max() < 4.0

    rays, ray_valid = model.unproject(uv)
    assert ray_valid.all() and (rays[:, 2] < 0.0).all()
    ok, rvec, tvec, inliers = solve_pnp_robust(
        model, X, uv, noise_bound_px=4.0,
    )

    assert ok
    assert inliers.all(), "a pixel bound must not become a near-axis focal approximation"
    rotation, translation = _pose_error(rvec, tvec, R_gt, t_gt)
    assert rotation < 0.1
    assert translation < 0.005


@pytest.mark.parametrize("factory", WIDE_MODELS, ids=lambda factory: factory().name)
def test_pixel_bound_is_camera_agnostic_on_all_peripheral_rays(factory):
    model = factory()
    rng = np.random.default_rng(202)
    theta = np.radians(rng.uniform(92.0, 100.0, 120))
    phi = rng.uniform(0.0, 2.0 * np.pi, 120)
    camera_points = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ]) * rng.uniform(0.8, 3.0, 120)[:, None]
    R_gt = so3_exp([0.13, -0.21, 0.08])
    t_gt = np.array([0.1, -0.04, -0.15])
    X = (camera_points - t_gt) @ R_gt
    uv, valid = model.project(camera_points)
    assert valid.all()
    uv = np.asarray(uv) + np.random.default_rng(99).normal(0.0, 1.0, (120, 2))
    rays, ray_valid = model.unproject(uv)
    assert ray_valid.all() and (rays[:, 2] < 0.0).all()

    ok, rvec, tvec, inliers = solve_pnp_robust(
        model, X, uv, noise_bound_px=4.0,
    )

    assert ok and inliers.all()
    rotation, translation = _pose_error(rvec, tvec, R_gt, t_gt)
    assert rotation < 0.1
    assert translation < 0.001


@pytest.mark.parametrize("factory", MODELS, ids=lambda factory: factory().name)
def test_gnc_pnp_is_camera_model_polymorphic(factory):
    model = factory()
    grid = np.mgrid[0:7, 0:7].reshape(2, -1).T * 0.1
    X = np.column_stack([grid, np.zeros(len(grid))]).astype(np.float64)
    rvec_gt = np.array([0.05, -0.1, 0.02])
    tvec_gt = np.array([-0.15, 0.1, 2.0])
    R_gt = cv2.Rodrigues(rvec_gt)[0]
    uv, valid = model.project(X @ R_gt.T + tvec_gt)
    X, uv = X[valid], np.asarray(uv)[valid].copy()

    rng = np.random.default_rng(0)
    outlier_rows = rng.choice(len(uv), len(uv) // 4, replace=False)
    uv[outlier_rows] += rng.uniform(-120.0, 120.0, (len(outlier_rows), 2))

    ok, rvec, tvec, inliers = solve_pnp_robust(model, X, uv)

    assert ok
    assert np.linalg.norm(rvec - rvec_gt) < 1e-6
    assert np.linalg.norm(tvec - tvec_gt) < 1e-6
    assert not inliers[outlier_rows].any()
    assert inliers.sum() >= len(X) // 2


def test_legacy_double_sphere_camera_exposes_robust_pnp_wrapper():
    cam = DoubleSphereCamera(
        fx=711.57, fy=711.24, cx=949.18, cy=518.81,
        xi=0.183, alpha=0.809, width=1920, height=1080,
    )
    grid = np.mgrid[0:7, 0:7].reshape(2, -1).T * 0.1
    X = np.column_stack([grid, np.zeros(len(grid))]).astype(np.float64)
    rvec_gt = np.array([0.05, -0.1, 0.02])
    tvec_gt = np.array([-0.15, 0.1, 2.0])
    R_gt = cv2.Rodrigues(rvec_gt)[0]
    uv, valid = cam.project(X @ R_gt.T + tvec_gt)
    assert valid.all()

    ok, rvec, tvec, inliers = cam.solve_pnp_robust(X, uv)

    assert ok and inliers.all()
    np.testing.assert_allclose(rvec, rvec_gt, atol=1e-8)
    np.testing.assert_allclose(tvec, tvec_gt, atol=1e-8)


pytestmark = pytest.mark.req("FR-OPS-003", "FR-CORE-001")
