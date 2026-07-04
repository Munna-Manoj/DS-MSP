"""
Pure Enhanced Unified Camera Model (EUCM) math (Khomutenko et al. 2016).

EUCM generalizes UCM with a second parameter ``beta`` that scales the radial
term inside the sphere distance: ``d = sqrt(beta*(x^2+y^2) + z^2)``.

Self-contained: numpy only (enforced by the independence gate).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def eucm_project(points_3d: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                 alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame points. Returns ``(u, v, valid)``."""
    x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]
    d = np.sqrt(beta * (x*x + y*y) + z*z)
    den = alpha * d + (1.0 - alpha) * z

    valid = den > 1e-8
    den = np.maximum(den, 1e-8)
    u = fx * x / den + cx
    v = fy * y / den + cy
    return u, v, valid


def eucm_unproject(points_2d: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                   alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject pixels to unit rays. Returns ``(rays, valid)``."""
    mx = (points_2d[..., 0] - cx) / fx
    my = (points_2d[..., 1] - cy) / fy
    r2 = mx*mx + my*my

    s = 1.0 - (2.0 * alpha - 1.0) * beta * r2
    valid = s >= 0
    s = np.maximum(s, 0.0)
    # Denominator is >= (1-alpha) for alpha in [0,1) and only reaches exactly 0 at the legal
    # optimizer boundary alpha=1 combined with a ray exactly at the sphere-intersection edge
    # (s=0) — a real 0/0 on wide-FOV fits (see ds_math.py/ucm_math.py's identical formula for
    # the derivation). Gate it explicitly rather than let a NaN ray fall through as valid=True.
    den = alpha * np.sqrt(s) + (1.0 - alpha)
    valid = valid & (den > 1e-9)
    mz = (1.0 - beta * alpha*alpha * r2) / np.maximum(den, 1e-9)

    ray = np.stack([mx, my, mz], axis=-1)
    norm = np.linalg.norm(ray, axis=-1, keepdims=True)
    ray = ray / np.maximum(norm, 1e-10)
    ray[~valid] = 0.0
    return ray, valid


def eucm_project_jacobian(points_3d: np.ndarray, fx: float, fy: float,
                          cx: float, cy: float, alpha: float, beta: float
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analytic Jacobian. Returns ``(u, v, J_point(N,2,3), J_param(N,2,6), valid)``.

    Parameter order: ``(fx, fy, cx, cy, alpha, beta)``.
    """
    x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]
    rho2 = x*x + y*y
    d = np.sqrt(beta * rho2 + z*z)
    d = np.maximum(d, 1e-12)
    den = alpha * d + (1.0 - alpha) * z
    valid = den > 1e-8

    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    inv = 1.0 / den
    inv2 = inv * inv
    u = fx * x * inv + cx
    v = fy * y * inv + cy

    dden_dx = alpha * beta * x / d
    dden_dy = alpha * beta * y / d
    dden_dz = alpha * z / d + (1.0 - alpha)
    dden_dalpha = d - z
    dden_dbeta = alpha * rho2 / (2.0 * d)

    J_point = np.empty(points_3d.shape[:-1] + (2, 3), dtype=np.float64)
    J_point[..., 0, 0] = fx * (den - x * dden_dx) * inv2
    J_point[..., 0, 1] = fx * (-x * dden_dy) * inv2
    J_point[..., 0, 2] = fx * (-x * dden_dz) * inv2
    J_point[..., 1, 0] = fy * (-y * dden_dx) * inv2
    J_point[..., 1, 1] = fy * (den - y * dden_dy) * inv2
    J_point[..., 1, 2] = fy * (-y * dden_dz) * inv2

    J_param = np.zeros(points_3d.shape[:-1] + (2, 6), dtype=np.float64)
    J_param[..., 0, 0] = x * inv                       # du/dfx
    J_param[..., 0, 2] = 1.0                           # du/dcx
    J_param[..., 0, 4] = -fx * x * inv2 * dden_dalpha  # du/dalpha
    J_param[..., 0, 5] = -fx * x * inv2 * dden_dbeta   # du/dbeta
    J_param[..., 1, 1] = y * inv                       # dv/dfy
    J_param[..., 1, 3] = 1.0                           # dv/dcy
    J_param[..., 1, 4] = -fy * y * inv2 * dden_dalpha  # dv/dalpha
    J_param[..., 1, 5] = -fy * y * inv2 * dden_dbeta   # dv/dbeta
    return u, v, J_point, J_param, valid
