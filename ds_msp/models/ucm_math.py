"""
Pure Unified Camera Model (UCM) math.

UCM is the single-sphere unified projection (Geyer/Barreto/Mei). It is the
Double Sphere model with ``xi = 0``, parameterized by ``alpha`` only.

Self-contained: numpy only, no internal imports (enforced by the independence
gate), so it stands alone and is independently testable.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _w(alpha: float) -> float:
    """Half-space coefficient (DS condition with xi = 0)."""
    return (1.0 - alpha) / alpha if alpha > 0.5 else alpha / (1.0 - alpha)


def ucm_project(points_3d: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame points. Returns ``(u, v, valid)``."""
    x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]
    d = np.sqrt(x*x + y*y + z*z)
    den = alpha * d + (1.0 - alpha) * z

    valid = (z > -_w(alpha) * d) & (den > 1e-8)
    den = np.maximum(den, 1e-8)
    u = fx * x / den + cx
    v = fy * y / den + cy
    return u, v, valid


def ucm_unproject(points_2d: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                  alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject pixels to unit rays. Returns ``(rays, valid)``."""
    mx = (points_2d[..., 0] - cx) / fx
    my = (points_2d[..., 1] - cy) / fy
    r2 = mx*mx + my*my

    s = 1.0 - (2.0 * alpha - 1.0) * r2
    valid = s >= 0
    s = np.maximum(s, 0.0)
    # Denominator is >= (1-alpha) for alpha in [0,1) and only reaches exactly 0 at the legal
    # optimizer boundary alpha=1 (calib_param.py bounds alpha to [0,1]) combined with a ray
    # exactly at the sphere-intersection edge (s=0) — a real 0/0 on wide-FOV fits. Unlike the
    # DS model, there is no downstream check that would catch the resulting NaN: `valid` here
    # was `s >= 0` only, so a NaN ray could previously be returned tagged `valid=True`.
    den = alpha * np.sqrt(s) + (1.0 - alpha)
    valid = valid & (den > 1e-9)
    mz = (1.0 - alpha*alpha * r2) / np.maximum(den, 1e-9)

    ray = np.stack([mx, my, mz], axis=-1)
    norm = np.linalg.norm(ray, axis=-1, keepdims=True)
    ray = ray / np.maximum(norm, 1e-10)
    ray[~valid] = 0.0
    return ray, valid


def ucm_project_jacobian(points_3d: np.ndarray, fx: float, fy: float,
                         cx: float, cy: float, alpha: float
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analytic Jacobian. Returns ``(u, v, J_point(N,2,3), J_param(N,2,5), valid)``.

    Parameter order: ``(fx, fy, cx, cy, alpha)``.
    """
    x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]
    d = np.sqrt(x*x + y*y + z*z)
    d = np.maximum(d, 1e-12)
    den = alpha * d + (1.0 - alpha) * z
    valid = (z > -_w(alpha) * d) & (den > 1e-8)

    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    inv = 1.0 / den
    inv2 = inv * inv
    u = fx * x * inv + cx
    v = fy * y * inv + cy

    dden_dx = alpha * x / d
    dden_dy = alpha * y / d
    dden_dz = alpha * z / d + (1.0 - alpha)
    dden_dalpha = d - z

    J_point = np.empty(points_3d.shape[:-1] + (2, 3), dtype=np.float64)
    J_point[..., 0, 0] = fx * (den - x * dden_dx) * inv2
    J_point[..., 0, 1] = fx * (-x * dden_dy) * inv2
    J_point[..., 0, 2] = fx * (-x * dden_dz) * inv2
    J_point[..., 1, 0] = fy * (-y * dden_dx) * inv2
    J_point[..., 1, 1] = fy * (den - y * dden_dy) * inv2
    J_point[..., 1, 2] = fy * (-y * dden_dz) * inv2

    J_param = np.zeros(points_3d.shape[:-1] + (2, 5), dtype=np.float64)
    J_param[..., 0, 0] = x * inv                       # du/dfx
    J_param[..., 0, 2] = 1.0                           # du/dcx
    J_param[..., 0, 4] = -fx * x * inv2 * dden_dalpha  # du/dalpha
    J_param[..., 1, 1] = y * inv                       # dv/dfy
    J_param[..., 1, 3] = 1.0                           # dv/dcy
    J_param[..., 1, 4] = -fy * y * inv2 * dden_dalpha  # dv/dalpha
    return u, v, J_point, J_param, valid
