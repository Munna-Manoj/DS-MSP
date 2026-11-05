"""Executable checks of the guide's conditional linear information arguments.

These verify algebra on full-rank synthetic systems, not nonlinear convergence
or the uncertainty coverage of a fitted real camera.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.req("FR-CALIB-001")


def test_kb_prediction_bound_and_worst_case_noise():
    theta = np.linspace(.05, 1.4, 40)
    design = theta[:, None] ** np.arange(1, 10, 2)
    inverse = np.linalg.pinv(design)
    coefficients = np.array([400, 12, -2.4, .28, -.06])
    epsilon = .1
    for angle in (.3, 1.3, 1.65):
        query = angle ** np.arange(1, 10, 2)
        leverage = query @ inverse
        # Noise aligned with the prediction functional attains Cauchy-Schwarz.
        noise = epsilon * leverage / np.linalg.norm(leverage)
        estimated = inverse @ (design @ coefficients + noise)
        error = abs(query @ (estimated - coefficients))
        bound = epsilon * np.linalg.norm(leverage)
        assert error <= bound + 1e-8
        np.testing.assert_allclose(error, bound, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("pose_rank", [4, 6])
def test_independent_pose_marginalization_adds_positive_semidefinite_information(pose_rank):
    rng = np.random.default_rng(781)
    A = rng.normal(size=(32, 8))  # whitened intrinsic Jacobian
    B = rng.normal(size=(32, 6))  # whitened nuisance-pose Jacobian
    if pose_rank < 6:
        B[:, pose_rank:] = B[:, :6-pose_rank]
    projector = np.eye(32) - B @ np.linalg.pinv(B)
    added = A.T @ projector @ A
    np.testing.assert_allclose(projector @ projector, projector, atol=1e-12)
    assert np.linalg.eigvalsh(added).min() >= -1e-12
    # Eliminating the independent nuisance pose equals projecting its span out.
    for delta in rng.normal(size=(5, 8)):
        pose = -np.linalg.pinv(B) @ A @ delta
        residual = A @ delta + B @ pose
        np.testing.assert_allclose(residual @ residual, delta @ added @ delta)
    prior = rng.normal(size=(8, 8))
    information = prior.T @ prior + np.eye(8)
    reduction = np.linalg.inv(information) - np.linalg.inv(information + added)
    assert np.linalg.eigvalsh(reduction).min() >= -1e-12
