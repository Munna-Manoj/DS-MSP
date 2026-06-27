"""SO(3) / SE(3) Lie-group primitives (pure NumPy) — manifold-correct pose optimization.

Optimizing a rotation as a *flat* axis-angle vector ``r ∈ ℝ³`` is biased for large rotations and
breaks down near ``‖r‖ = π`` (the double-cover singularity). The fix is to optimize on the
manifold: keep the pose as a matrix and update it by a **local perturbation** retracted through
the exponential map, ``R ← R · exp([δω]_×)`` with ``δω`` small. This module provides the
``exp``/``log`` maps and the SO(3) right Jacobian that make that retraction (and its analytic
derivative) exact. Used by `mvg.bundle.refine_two_view` and `calib.bundle` (manifold mode).

Numerically safe near ``θ = 0`` (Taylor series) and ``θ = π`` (`log` via the largest diagonal).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def hat(w: np.ndarray) -> np.ndarray:
    """so(3) hat: axis-angle vector ``(3,)`` → skew-symmetric ``[w]_× (3,3)``."""
    w = np.asarray(w, float)
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def vee(W: np.ndarray) -> np.ndarray:
    """Inverse of :func:`hat`: skew-symmetric ``(3,3)`` → axis-angle vector ``(3,)``."""
    W = np.asarray(W, float)
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Exp map ``ℝ³ → SO(3)`` (Rodrigues), numerically safe at ``θ = 0``."""
    w = np.asarray(w, float)
    theta2 = float(w @ w)
    Wx = hat(w)
    if theta2 < _EPS ** 2:                                  # Taylor: I + [w]_× + ½[w]_×²
        return np.eye(3) + Wx + 0.5 * (Wx @ Wx)
    theta = np.sqrt(theta2)
    return (np.eye(3) + (np.sin(theta) / theta) * Wx
            + ((1 - np.cos(theta)) / theta2) * (Wx @ Wx))


def so3_log(R: np.ndarray) -> np.ndarray:
    """Log map ``SO(3) → ℝ³``, safe at ``θ = 0`` and ``θ = π``."""
    R = np.asarray(R, float)
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(c)
    if theta < _EPS:
        return vee(R - R.T) * 0.5                           # small angle
    if np.pi - theta < 1e-5:                                # near π: axis from the largest diagonal
        A = (R + np.eye(3)) / 2.0                           # ≈ a aᵀ
        k = int(np.argmax(np.diag(A)))
        axis = A[:, k] / np.sqrt(max(A[k, k], _EPS))
        axis = axis / np.linalg.norm(axis)
        return theta * axis
    return (theta / (2.0 * np.sin(theta))) * vee(R - R.T)


def so3_right_jacobian(w: np.ndarray) -> np.ndarray:
    """Right Jacobian ``J_r(w)`` of SO(3): ``∂/∂δ Log(Exp(w)·Exp(δ))⁻¹·… ``; relates a tangent
    perturbation to the exp-map derivative. ``∂(Exp(w)v)/∂w = -Exp(w)[v]_× J_r(w)``."""
    w = np.asarray(w, float)
    theta2 = float(w @ w)
    Wx = hat(w)
    if theta2 < _EPS ** 2:
        return np.eye(3) - 0.5 * Wx + (1.0 / 6.0) * (Wx @ Wx)
    theta = np.sqrt(theta2)
    a = (1 - np.cos(theta)) / theta2
    b = (theta - np.sin(theta)) / (theta2 * theta)
    return np.eye(3) - a * Wx + b * (Wx @ Wx)


def so3_left_jacobian(w: np.ndarray) -> np.ndarray:
    """Left Jacobian ``J_l(w) = J_r(w)ᵀ = J_r(-w)``."""
    return so3_right_jacobian(np.asarray(w, float)).T


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exp map ``ℝ⁶ → SE(3)``. ``xi = [ρ (3), φ (3)]`` (translation tangent, then rotation).
    Returns a ``4×4`` homogeneous transform ``[[R, J_l(φ)ρ], [0, 1]]``."""
    xi = np.asarray(xi, float)
    rho, phi = xi[:3], xi[3:]
    R = so3_exp(phi)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = so3_left_jacobian(phi) @ rho
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """Inverse of :func:`se3_exp`: ``SE(3) → ℝ⁶`` as ``[ρ, φ]``."""
    T = np.asarray(T, float)
    phi = so3_log(T[:3, :3])
    rho = np.linalg.solve(so3_left_jacobian(phi), T[:3, 3])
    return np.concatenate([rho, phi])
