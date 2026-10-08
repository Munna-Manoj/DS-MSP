"""SO(3)/SE(3) Lie primitives — round-trips, cv2 agreement, and the right-Jacobian identity."""

import cv2
import numpy as np
import pytest

from ds_msp.core.lie import (
    hat,
    se3_adjoint,
    se3_exp,
    se3_log,
    so3_exp,
    so3_log,
    so3_right_jacobian,
    vee,
)


def test_hat_vee_are_inverse():
    w = np.array([0.3, -1.2, 0.7])
    assert np.allclose(vee(hat(w)), w)
    assert np.allclose(hat(w), -hat(w).T)                  # skew-symmetric


def test_so3_exp_matches_cv2_rodrigues():
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = rng.standard_normal(3) * rng.uniform(0, 3)
        R_cv, _ = cv2.Rodrigues(w)
        assert np.allclose(so3_exp(w), R_cv, atol=1e-10)


def test_so3_exp_log_roundtrip_including_small_and_large():
    rng = np.random.default_rng(1)
    for theta in [0.0, 1e-9, 1e-4, 0.5, 2.0, 3.0]:         # below π so axis sign is unambiguous
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        w = theta * axis
        assert np.allclose(so3_log(so3_exp(w)), w, atol=1e-7)


def test_so3_exp_is_a_rotation():
    rng = np.random.default_rng(2)
    for _ in range(20):
        R = so3_exp(rng.standard_normal(3) * 2)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_so3_log_near_pi_recovers_angle():
    axis = np.array([0.2, -0.9, 0.3])
    axis /= np.linalg.norm(axis)
    R = so3_exp((np.pi - 1e-7) * axis)
    w = so3_log(R)
    assert np.isclose(np.linalg.norm(w), np.pi, atol=1e-4)
    # axis matches up to sign
    assert np.allclose(np.abs(w / np.linalg.norm(w)), np.abs(axis), atol=1e-3)


def test_right_jacobian_matches_finite_difference():
    """∂(Exp(w) v)/∂w = -Exp(w) [v]_× J_r(w)."""
    rng = np.random.default_rng(3)
    w = rng.standard_normal(3) * 0.7
    v = rng.standard_normal(3)
    analytic = -so3_exp(w) @ hat(v) @ so3_right_jacobian(w)
    num = np.zeros((3, 3))
    eps = 1e-6
    for j in range(3):
        d = np.zeros(3)
        d[j] = eps
        num[:, j] = (so3_exp(w + d) @ v - so3_exp(w - d) @ v) / (2 * eps)
    assert np.allclose(analytic, num, atol=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_se3_exp_log_roundtrip(seed):
    rng = np.random.default_rng(seed)
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    phi = axis * rng.uniform(0, 2.8)                       # rotation angle < π (principal range)
    xi = np.concatenate([rng.standard_normal(3), phi])
    T = se3_exp(xi)
    assert np.allclose(T[3], [0, 0, 0, 1])
    assert np.allclose(se3_log(T), xi, atol=1e-7)
    # exp is a valid rigid transform
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-10)


def _random_se3(rng):
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    xi = np.concatenate([rng.standard_normal(3), axis * rng.uniform(0, 2.5)])
    return se3_exp(xi)


@pytest.mark.req("FR-CORE-002")
@pytest.mark.parametrize("seed", range(5))
def test_se3_adjoint_conjugation_identity(seed):
    """T·Exp(ξ)·T⁻¹ = Exp(Ad_T·ξ) for random poses and tangents."""
    rng = np.random.default_rng(100 + seed)
    T = _random_se3(rng)
    xi = np.concatenate([rng.standard_normal(3), rng.standard_normal(3) * 0.5])
    lhs = T @ se3_exp(xi) @ np.linalg.inv(T)
    rhs = se3_exp(se3_adjoint(T) @ xi)
    assert np.allclose(lhs, rhs, atol=1e-10)


@pytest.mark.req("FR-CORE-002")
@pytest.mark.parametrize("seed", range(5))
def test_se3_adjoint_is_a_homomorphism(seed):
    """Ad_{AB} = Ad_A · Ad_B."""
    rng = np.random.default_rng(200 + seed)
    A, B = _random_se3(rng), _random_se3(rng)
    assert np.allclose(se3_adjoint(A @ B), se3_adjoint(A) @ se3_adjoint(B), atol=1e-10)
