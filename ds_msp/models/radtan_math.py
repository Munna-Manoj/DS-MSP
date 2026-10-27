"""
Pure pinhole + radial-tangential (Brown-Conrady) math — the OpenCV pinhole model.

Distortion order matches OpenCV ``distCoeffs = [k1, k2, p1, p2, k3]``:
  a = x/z, b = y/z, r2 = a^2 + b^2
  radial = 1 + k1 r2 + k2 r2^2 + k3 r2^3
  x' = a*radial + 2 p1 a b + p2 (r2 + 2 a^2)
  y' = b*radial + p1 (r2 + 2 b^2) + 2 p2 a b
  u = fx x' + cx,  v = fy y' + cy

Narrow-FOV model: only valid for z > 0 (cannot represent >= 90 deg).
Self-contained: numpy only (enforced by the independence gate).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def radtan_project(points_3d, fx, fy, cx, cy, k1, k2, p1, p2, k3
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame points. Returns ``(u, v, valid)``."""
    x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]
    valid = z > 1e-9
    zsafe = np.where(valid, z, 1.0)
    a = x / zsafe
    b = y / zsafe
    r2 = a*a + b*b
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xp = a * radial + 2.0*p1*a*b + p2*(r2 + 2.0*a*a)
    yp = b * radial + p1*(r2 + 2.0*b*b) + 2.0*p2*a*b
    u = fx * xp + cx
    v = fy * yp + cy
    u = np.where(valid, u, 0.0)
    v = np.where(valid, v, 0.0)
    return u, v, valid


def radtan_unproject(points_2d, fx, fy, cx, cy, k1, k2, p1, p2, k3
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject pixels to unit rays (iterative distortion inverse)."""
    a0 = (points_2d[..., 0] - cx) / fx
    b0 = (points_2d[..., 1] - cy) / fy
    a, b = a0.copy(), b0.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(20):
            r2 = a*a + b*b
            radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
            # A radial factor at or below 0 is past the point where distortion folds the
            # image back; hold such points (they fail the check below) instead of dividing.
            radial = np.where(radial > 1e-12, radial, np.nan)
            dx = 2.0*p1*a*b + p2*(r2 + 2.0*a*a)
            dy = p1*(r2 + 2.0*b*b) + 2.0*p2*a*b
            a = (a0 - dx) / radial
            b = (b0 - dy) / radial
        # validity: forward re-distortion must reproduce the input (NaN compares False)
        r2 = a*a + b*b
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        xp = a*radial + 2.0*p1*a*b + p2*(r2 + 2.0*a*a)
        yp = b*radial + p1*(r2 + 2.0*b*b) + 2.0*p2*a*b
        valid = (np.abs(xp - a0) < 1e-6) & (np.abs(yp - b0) < 1e-6) & (radial > 0)
    a = np.where(valid, a, 0.0)
    b = np.where(valid, b, 0.0)

    rays = np.stack([a, b, np.ones_like(a)], axis=-1)
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    rays[~valid] = 0.0
    return rays, valid


def radtan_project_jacobian(points_3d, fx, fy, cx, cy, k1, k2, p1, p2, k3
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analytic Jacobian. Returns ``(u, v, J_point(N,2,3), J_param(N,2,9), valid)``.

    Parameter order: ``(fx, fy, cx, cy, k1, k2, p1, p2, k3)``.
    """
    x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]
    valid = z > 1e-9
    zsafe = np.where(valid, z, 1.0)
    a = x / zsafe
    b = y / zsafe
    r2 = a*a + b*b
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    dradial = k1 + r2 * (2.0*k2 + 3.0*k3*r2)        # d(radial)/d(r2)

    xp = a * radial + 2.0*p1*a*b + p2*(r2 + 2.0*a*a)
    yp = b * radial + p1*(r2 + 2.0*b*b) + 2.0*p2*a*b
    u = fx * xp + cx
    v = fy * yp + cy

    # derivatives of (xp, yp) w.r.t normalized (a, b)
    dxp_da = radial + 2.0*a*a*dradial + 2.0*p1*b + 6.0*p2*a
    dxp_db = 2.0*a*b*dradial + 2.0*p1*a + 2.0*p2*b
    dyp_da = 2.0*a*b*dradial + 2.0*p1*a + 2.0*p2*b
    dyp_db = radial + 2.0*b*b*dradial + 6.0*p1*b + 2.0*p2*a

    # a = x/z, b = y/z
    inv_z = 1.0 / zsafe
    J_point = np.zeros(points_3d.shape[:-1] + (2, 3), dtype=np.float64)
    J_point[..., 0, 0] = fx * dxp_da * inv_z
    J_point[..., 0, 1] = fx * dxp_db * inv_z
    J_point[..., 0, 2] = -fx * (dxp_da * a + dxp_db * b) * inv_z
    J_point[..., 1, 0] = fy * dyp_da * inv_z
    J_point[..., 1, 1] = fy * dyp_db * inv_z
    J_point[..., 1, 2] = -fy * (dyp_da * a + dyp_db * b) * inv_z

    r4 = r2 * r2
    r6 = r4 * r2
    J_param = np.zeros(points_3d.shape[:-1] + (2, 9), dtype=np.float64)
    J_param[..., 0, 0] = xp                 # du/dfx
    J_param[..., 0, 2] = 1.0                # du/dcx
    J_param[..., 0, 4] = fx * a * r2        # du/dk1
    J_param[..., 0, 5] = fx * a * r4        # du/dk2
    J_param[..., 0, 6] = fx * (2.0*a*b)     # du/dp1
    J_param[..., 0, 7] = fx * (r2 + 2.0*a*a)  # du/dp2
    J_param[..., 0, 8] = fx * a * r6        # du/dk3
    J_param[..., 1, 1] = yp                 # dv/dfy
    J_param[..., 1, 3] = 1.0                # dv/dcy
    J_param[..., 1, 4] = fy * b * r2        # dv/dk1
    J_param[..., 1, 5] = fy * b * r4        # dv/dk2
    J_param[..., 1, 6] = fy * (r2 + 2.0*b*b)  # dv/dp1
    J_param[..., 1, 7] = fy * (2.0*a*b)     # dv/dp2
    J_param[..., 1, 8] = fy * b * r6        # dv/dk3

    u = np.where(valid, u, 0.0)
    v = np.where(valid, v, 0.0)
    return u, v, J_point, J_param, valid
