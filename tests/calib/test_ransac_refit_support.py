"""Regression: a RANSAC consensus refit may not report an unsupported pose.

Real TUM-VI peripheral correspondences exposed the failure mode: an initial hypothesis met
the solver's support floor, fitting that consensus moved the model, and final re-scoring left
zero inliers while the public API still returned ``ok=True``.  These tests force that exact
state transition in every resection path and at the public boundary.
"""

import numpy as np
import pytest

import ds_msp.geometry.resection as resection
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.ops import solve_pnp_ransac


def _noncoplanar_points() -> np.ndarray:
    return np.array(
        [
            [-0.8, -0.4, -0.2],
            [0.7, -0.5, 0.1],
            [0.5, 0.8, -0.1],
            [-0.6, 0.7, 0.3],
            [0.1, -0.2, 0.9],
            [-0.3, 0.1, -0.8],
            [0.9, 0.4, 0.6],
        ],
        dtype=float,
    )


def _rays_for(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    camera_points = X + t
    return camera_points / np.linalg.norm(camera_points, axis=1, keepdims=True)


@pytest.mark.parametrize("refit_outcome", ["unsupported", "none", "raises"])
def test_bearing_dlt_falls_back_when_consensus_refit_collapses(monkeypatch, refit_outcome):
    X = _noncoplanar_points()
    good_t = np.array([0.0, 0.0, 4.0])
    rays = _rays_for(X, good_t)

    def fit(points, _rays):
        # Minimal hypotheses fit perfectly; the all-consensus refit is deliberately bad.
        if len(points) == 6:
            return np.eye(3), good_t
        if refit_outcome == "raises":
            raise np.linalg.LinAlgError
        if refit_outcome == "none":
            return None
        return np.eye(3), np.array([0.0, 0.0, -100.0])

    monkeypatch.setattr(resection, "_pose_dlt_bearing", fit)
    T, inliers = resection._ransac_pnp_bearing(
        X, rays, focal=200.0, thresh_px=1.0, max_iters=2, seed=0
    )

    assert T is not None
    assert inliers.all()
    np.testing.assert_allclose(T[:3, 3], good_t)


@pytest.mark.parametrize("refit_outcome", ["unsupported", "none", "raises"])
def test_planar_bearing_falls_back_when_consensus_refit_collapses(monkeypatch, refit_outcome):
    X = np.array(
        [
            [-0.8, -0.6, 0.0],
            [0.7, -0.5, 0.0],
            [0.8, 0.7, 0.0],
            [-0.6, 0.8, 0.0],
            [0.1, 0.2, 0.0],
        ],
        dtype=float,
    )
    good_t = np.array([0.0, 0.0, 4.0])
    rays = _rays_for(X, good_t)

    def fit(points, _rays):
        if len(points) == 4:
            return np.eye(3), good_t
        if refit_outcome == "raises":
            raise np.linalg.LinAlgError
        if refit_outcome == "none":
            return None
        return np.eye(3), np.array([0.0, 0.0, -100.0])

    monkeypatch.setattr(resection, "_pose_planar_bearing", fit)
    T, inliers = resection._ransac_pnp_planar_bearing(
        X, rays, focal=200.0, thresh_px=1.0, max_iters=2, seed=0
    )

    assert T is not None
    assert inliers.all()
    np.testing.assert_allclose(T[:3, 3], good_t)


@pytest.mark.parametrize("refit_outcome", ["unsupported", "none", "raises"])
def test_normalized_plane_falls_back_when_consensus_refit_collapses(monkeypatch, refit_outcome):
    X = _noncoplanar_points()
    good_t = np.array([0.0, 0.0, 4.0])
    camera_points = X + good_t
    normalized = camera_points[:, :2] / camera_points[:, 2:3]

    def fit(points, _normalized):
        if len(points) == 6:
            return np.eye(3), good_t
        if refit_outcome == "raises":
            raise np.linalg.LinAlgError
        if refit_outcome == "none":
            return None
        return np.eye(3), np.array([0.0, 0.0, -100.0])

    monkeypatch.setattr(resection, "_pose_dlt_normalized", fit)
    T, inliers = resection.ransac_pnp_normalized(
        X, normalized, focal=200.0, thresh_px=1.0, max_iters=2, seed=0
    )

    assert T is not None
    assert inliers.all()
    np.testing.assert_allclose(T[:3, 3], good_t)


def test_bearing_dlt_refit_cannot_reduce_best_consensus(monkeypatch):
    X = _noncoplanar_points()
    X[-1] = [0.0, 0.0, -3.9]
    good_t = np.array([0.0, 0.0, 4.0])
    rays = _rays_for(X, good_t)

    def fit(points, _rays):
        # The refit retains the six-point DLT floor but loses the close seventh point.
        t = good_t if len(points) == 6 else np.array([0.01, 0.0, 4.0])
        return np.eye(3), t

    monkeypatch.setattr(resection, "_pose_dlt_bearing", fit)
    T, inliers = resection._ransac_pnp_bearing(
        X, rays, focal=200.0, thresh_px=1.0, max_iters=2, seed=0
    )

    assert T is not None
    assert inliers.all()
    np.testing.assert_allclose(T[:3, 3], good_t)


def test_planar_bearing_refit_cannot_reduce_best_consensus(monkeypatch):
    X = np.array(
        [
            [-10.0, -10.0, 0.0],
            [10.0, -10.0, 0.0],
            [10.0, 10.0, 0.0],
            [-10.0, 10.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    good_t = np.array([0.0, 0.0, 4.0])
    rays = _rays_for(X, good_t)

    def fit(points, _rays):
        # The refit keeps the four-point homography floor but loses the central point.
        t = good_t if len(points) == 4 else np.array([0.03, 0.0, 4.0])
        return np.eye(3), t

    monkeypatch.setattr(resection, "_pose_planar_bearing", fit)
    T, inliers = resection._ransac_pnp_planar_bearing(
        X, rays, focal=200.0, thresh_px=1.0, max_iters=2, seed=0
    )

    assert T is not None
    assert inliers.all()
    np.testing.assert_allclose(T[:3, 3], good_t)


def test_normalized_plane_refit_cannot_reduce_best_consensus(monkeypatch):
    X = _noncoplanar_points()
    X[-1] = [0.0, 0.0, -3.9]
    good_t = np.array([0.0, 0.0, 4.0])
    camera_points = X + good_t
    normalized = camera_points[:, :2] / camera_points[:, 2:3]

    def fit(points, _normalized):
        # The refit keeps the six-point DLT floor but loses the close seventh point.
        t = good_t if len(points) == 6 else np.array([0.01, 0.0, 4.0])
        return np.eye(3), t

    monkeypatch.setattr(resection, "_pose_dlt_normalized", fit)
    T, inliers = resection.ransac_pnp_normalized(
        X, normalized, focal=200.0, thresh_px=1.0, max_iters=2, seed=0
    )

    assert T is not None
    assert inliers.all()
    np.testing.assert_allclose(T[:3, 3], good_t)


@pytest.mark.parametrize("refit_outcome", ["unsupported", "degraded", "raises"])
def test_projective_resection_falls_back_when_consensus_refit_degrades(
    monkeypatch, refit_outcome
):
    X = _noncoplanar_points()
    X[-1] = [0.0, 0.0, -3.9]
    good_P = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 4.0]]
    )
    camera_points = np.c_[X, np.ones(len(X))] @ good_P.T
    uv = camera_points[:, :2] / camera_points[:, 2:3]

    def fit(points, _uv):
        if len(points) == 6:
            return good_P
        if refit_outcome == "raises":
            raise np.linalg.LinAlgError
        refit_P = good_P.copy()
        refit_P[0, 3] = 0.01 if refit_outcome == "degraded" else 100.0
        return refit_P

    monkeypatch.setattr(resection, "dlt_projection", fit)
    P, inliers = resection.ransac_resection(
        X, uv, thresh_px=0.005, max_iters=2, seed=0
    )

    assert P is not None
    assert inliers.all()
    np.testing.assert_allclose(P, good_P)


@pytest.mark.parametrize("requested_sample", [4, 5])
def test_projective_resection_enforces_six_point_dlt_floor(requested_sample):
    X = _noncoplanar_points()[:5]
    camera_points = X + np.array([0.0, 0.0, 4.0])
    uv = camera_points[:, :2] / camera_points[:, 2:3]

    P, inliers = resection.ransac_resection(X, uv, min_sample=requested_sample)

    assert P is None
    assert not inliers.any()


def test_public_ransac_success_requires_final_solver_support(monkeypatch):
    model = DoubleSphereModel.sample()
    X = _noncoplanar_points()
    uv, valid = model.project(X + np.array([0.0, 0.0, 4.0]))
    assert valid.all()

    def unsupported_refit(points, normalized, **kwargs):
        del normalized, kwargs
        return np.eye(4), np.zeros(len(points), dtype=bool)

    monkeypatch.setattr(resection, "ransac_pnp_normalized", unsupported_refit)
    ok, rvec, tvec, inliers = solve_pnp_ransac(model, X, uv, seed=0)

    assert not ok
    assert rvec is None and tvec is None
    assert not inliers.any()


pytestmark = pytest.mark.req("FR-OPS-003", "FR-CALIB-002")
