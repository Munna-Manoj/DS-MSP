"""
Pure Kannala-Brandt (KB / equidistant) math — the OpenCV ``cv2.fisheye`` model.

theta = atan2(r, z),  r = sqrt(x^2 + y^2)
theta_d = theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9
u = fx * theta_d * x/r + cx,   v = fy * theta_d * y/r + cy

Unprojection solves theta_d(theta) = ru for theta by Newton-Raphson.
Self-contained: numpy only (enforced by the independence gate).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _theta_d(theta: np.ndarray, k1, k2, k3, k4) -> np.ndarray:
    t2 = theta * theta
    return theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))


def _dtheta_d(theta: np.ndarray, k1, k2, k3, k4) -> np.ndarray:
    t2 = theta * theta
    return 1.0 + t2 * (3*k1 + t2 * (5*k2 + t2 * (7*k3 + t2 * 9*k4)))


def _theta_max(k1, k2, k3, k4) -> float:
    """Largest angle up to which ``theta_d`` grows with ``theta``, capped at pi.

    It is the first positive zero of ``d(theta_d)/d(theta)``. Past it the image radius
    shrinks again, so two rays share a pixel and the projection cannot be inverted.
    """
    # d(theta_d)/d(theta) = 1 + 3k1 s + 5k2 s^2 + 7k3 s^3 + 9k4 s^4 with s = theta^2.
    roots = np.roots([9.0 * k4, 7.0 * k3, 5.0 * k2, 3.0 * k1, 1.0])
    s = roots.real[(np.abs(roots.imag) <= 1e-9 * np.abs(roots)) & (roots.real > 0)]
    return float(min(np.pi, np.sqrt(s.min()))) if s.size else float(np.pi)


def _kb_valid(r, z, theta, k1, k2, k3, k4):
    """Inside the one-to-one range, and not on the axis behind the camera (``theta = pi``)
    or at the camera centre."""
    return (theta < _theta_max(k1, k2, k3, k4)) & ((r > 0) | (z > 0))


def kb_project(points_3d: np.ndarray, fx, fy, cx, cy, k1, k2, k3, k4
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame points. Returns ``(u, v, valid)``."""
    x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]
    r = np.sqrt(x*x + y*y)
    theta = np.arctan2(r, z)
    td = _theta_d(theta, k1, k2, k3, k4)
    # theta_d / r tends to 1/z on the axis; use that limit within 1e-9 rad of it.
    ax = r <= 1e-9 * z
    scale = np.where(ax, 1.0 / np.where(ax, z, 1.0), td / np.where(ax | (r == 0), 1.0, r))
    u = fx * scale * x + cx
    v = fy * scale * y + cy
    valid = _kb_valid(r, z, theta, k1, k2, k3, k4) & np.isfinite(u) & np.isfinite(v)
    return u, v, valid


def kb_unproject(points_2d: np.ndarray, fx, fy, cx, cy, k1, k2, k3, k4
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject pixels to unit rays via Newton. Returns ``(rays, valid)``."""
    mx = (points_2d[..., 0] - cx) / fx
    my = (points_2d[..., 1] - cy) / fy
    ru = np.sqrt(mx*mx + my*my)
    rusafe = np.maximum(ru, 1e-12)

    # theta_d rises on [0, theta_max], so the root is bracketed there: Newton steps that
    # leave the bracket are replaced by bisection, which cannot pick the wrong root.
    tmax = _theta_max(k1, k2, k3, k4)
    lo, hi = np.zeros_like(ru), np.full_like(ru, tmax)
    theta = np.clip(ru, 0.0, tmax)
    for _ in range(60):
        f = _theta_d(theta, k1, k2, k3, k4) - ru
        if np.all(np.abs(f) < 1e-14):
            break
        lo = np.where(f < 0, theta, lo)
        hi = np.where(f > 0, theta, hi)
        fp = _dtheta_d(theta, k1, k2, k3, k4)
        step = theta - f / np.where(fp > 0, fp, 1.0)
        theta = np.where((fp > 0) & (step > lo) & (step < hi), step, 0.5 * (lo + hi))
    residual = np.abs(_theta_d(theta, k1, k2, k3, k4) - ru)
    valid = (residual < 1e-6) & (theta < tmax)

    sin_t = np.sin(theta)
    rays = np.stack([sin_t * mx / rusafe, sin_t * my / rusafe, np.cos(theta)], axis=-1)
    # on-axis pixels (ru ~ 0) -> straight ahead
    onaxis = ru < 1e-9
    if np.any(onaxis):
        rays[onaxis] = np.array([0.0, 0.0, 1.0])
    rays[~valid] = 0.0
    return rays, valid


def kb_project_jacobian(points_3d: np.ndarray, fx, fy, cx, cy, k1, k2, k3, k4
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analytic Jacobian. Returns ``(u, v, J_point(N,2,3), J_param(N,2,8), valid)``.

    Parameter order: ``(fx, fy, cx, cy, k1, k2, k3, k4)``.
    """
    x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]
    r_true = np.sqrt(x*x + y*y)
    ax = r_true <= 1e-9 * z                  # on the axis: the formulas below are 0/0
    r = np.where(ax | (r_true == 0), 1.0, r_true)
    r3 = r * r * r
    d2 = r*r + z*z                       # = x^2+y^2+z^2
    d2 = np.maximum(d2, 1e-12)
    theta = np.arctan2(r_true, z)
    t2 = theta * theta
    td = _theta_d(theta, k1, k2, k3, k4)
    dp = _dtheta_d(theta, k1, k2, k3, k4)

    g = x / r                            # x/r
    h = y / r                            # y/r
    u = fx * td * g + cx
    v = fy * td * h + cy

    # d theta / d(point)
    dtheta_dx = (z / d2) * (x / r)
    dtheta_dy = (z / d2) * (y / r)
    dtheta_dz = -r / d2

    # d(x/r)/d(point), d(y/r)/d(point)
    dg_dx = y*y / r3
    dg_dy = -x*y / r3
    dh_dx = -x*y / r3
    dh_dy = x*x / r3

    J_point = np.empty(points_3d.shape[:-1] + (2, 3), dtype=np.float64)
    J_point[..., 0, 0] = fx * (dp * dtheta_dx * g + td * dg_dx)
    J_point[..., 0, 1] = fx * (dp * dtheta_dy * g + td * dg_dy)
    J_point[..., 0, 2] = fx * (dp * dtheta_dz * g)
    J_point[..., 1, 0] = fy * (dp * dtheta_dx * h + td * dh_dx)
    J_point[..., 1, 1] = fy * (dp * dtheta_dy * h + td * dh_dy)
    J_point[..., 1, 2] = fy * (dp * dtheta_dz * h)

    # d(u,v)/d params. d theta_d / d k_i = theta^(2i+1).
    t3 = theta * t2
    t5 = t3 * t2
    t7 = t5 * t2
    t9 = t7 * t2
    J_param = np.zeros(points_3d.shape[:-1] + (2, 8), dtype=np.float64)
    J_param[..., 0, 0] = td * g          # du/dfx
    J_param[..., 0, 2] = 1.0             # du/dcx
    J_param[..., 0, 4] = fx * g * t3     # du/dk1
    J_param[..., 0, 5] = fx * g * t5
    J_param[..., 0, 6] = fx * g * t7
    J_param[..., 0, 7] = fx * g * t9
    J_param[..., 1, 1] = td * h          # dv/dfy
    J_param[..., 1, 3] = 1.0             # dv/dcy
    J_param[..., 1, 4] = fy * h * t3     # dv/dk1
    J_param[..., 1, 5] = fy * h * t5
    J_param[..., 1, 6] = fy * h * t7
    J_param[..., 1, 7] = fy * h * t9

    # On the axis u = fx x/z + cx to second order in r/z, so the limits are exact there.
    if np.any(ax):
        za = np.where(ax, z, 1.0)
        lim = np.zeros_like(J_point)
        lim[..., 0, 0] = fx / za
        lim[..., 0, 2] = -fx * x / (za * za)
        lim[..., 1, 1] = fy / za
        lim[..., 1, 2] = -fy * y / (za * za)
        J_point = np.where(ax[..., None, None], lim, J_point)
        u = np.where(ax, fx * x / za + cx, u)
        v = np.where(ax, fy * y / za + cy, v)
        J_param[..., 0, 0] = np.where(ax, x / za, J_param[..., 0, 0])
        J_param[..., 1, 1] = np.where(ax, y / za, J_param[..., 1, 1])
        # d theta_d / d k_i = theta^(2i+1), and (x/r)·theta^3 is O(r^2): zero on the axis.
        J_param[..., 4:] = np.where(ax[..., None, None], 0.0, J_param[..., 4:])

    valid = (_kb_valid(r_true, z, theta, k1, k2, k3, k4)
             & np.isfinite(u) & np.isfinite(v) & np.isfinite(J_point).all(axis=(-2, -1)))
    return u, v, J_point, J_param, valid
