"""
Pure Double Sphere math (Usenko et al. 2018).

This is the standalone, allocation-conscious math layer: closed-form projection,
unprojection, and the analytic Jacobian, operating on raw NumPy arrays with **no
dependency on any camera class or the rest of the package**. It can be imported
and used directly (``from ds_msp.models.ds_math import ds_project``) without
constructing any object.

Imports: numpy only (enforced by tests/contract/test_independence.py).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def ds_project(points_3d: np.ndarray, fx: float, fy: float, cx: float, cy: float,
               xi: float, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standalone projection function.
    Returns: u, v, valid
    """
    x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]

    d1 = np.sqrt(x*x + y*y + z*z)
    z1 = z + xi * d1
    d2 = np.sqrt(x*x + y*y + z1*z1)
    den = alpha * d2 + (1.0 - alpha) * z1

    # Projectability is a half-space condition, NOT simply z > 0.
    # Using z > 0 would clip the field of view to < 180 deg and silently drop
    # the very wide-angle rays the Double Sphere model exists to represent.
    # Per Usenko et al. 2018 (Eq. 43-45): the point is projectable iff
    #     z > -w2 * d1,
    # with w1 piecewise in alpha and w2 derived from w1 and xi.
    if alpha > 0.5:
        w1 = (1.0 - alpha) / alpha
    else:
        w1 = alpha / (1.0 - alpha)
    w2 = (w1 + xi) / np.sqrt(max(2.0 * w1 * xi + xi * xi + 1.0, 1e-12))

    valid = (z > -w2 * d1) & (den > 1e-8)
    den = np.maximum(den, 1e-8)

    u = fx * x / den + cx
    v = fy * y / den + cy

    return u, v, valid


def ds_project_jacobian(points_3d: np.ndarray, fx: float, fy: float,
                        cx: float, cy: float, xi: float, alpha: float
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Analytic Jacobian of the Double Sphere projection.

    Closed-form derivatives are exact, allocation-free, and far cheaper than
    finite differences (no step-size tuning, no cancellation), which makes
    Levenberg-Marquardt / Gauss-Newton calibration both faster and more robust.

    Returns
    -------
    u, v : (...,) arrays
        Projected pixel coordinates.
    J_point : (..., 2, 3) array
        d(u, v) / d(x, y, z).
    J_intr : (..., 2, 6) array
        d(u, v) / d(fx, fy, cx, cy, xi, alpha).
    valid : (...,) bool array
        Projectability mask (same condition as ds_project).
    """
    x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]

    d1 = np.sqrt(x*x + y*y + z*z)
    d1 = np.maximum(d1, 1e-12)
    z1 = z + xi * d1
    d2 = np.sqrt(x*x + y*y + z1*z1)
    d2 = np.maximum(d2, 1e-12)
    den = alpha * d2 + (1.0 - alpha) * z1

    if alpha > 0.5:
        w1 = (1.0 - alpha) / alpha
    else:
        w1 = alpha / (1.0 - alpha)
    w2 = (w1 + xi) / np.sqrt(max(2.0 * w1 * xi + xi * xi + 1.0, 1e-12))
    valid = (z > -w2 * d1) & (den > 1e-8)

    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    inv = 1.0 / den
    inv2 = inv * inv
    u = fx * x * inv + cx
    v = fy * y * inv + cy

    # Shared sub-expressions for the denominator's derivatives.
    A = alpha * z1 / d2 + (1.0 - alpha)      # appears in d(den)/dxi and d(den)/dz
    B = 1.0 + xi * z1 / d1
    Cz = 1.0 + xi * z / d1

    dden_dx = alpha * x * B / d2 + (1.0 - alpha) * xi * x / d1
    dden_dy = alpha * y * B / d2 + (1.0 - alpha) * xi * y / d1
    dden_dz = Cz * A
    dden_dxi = d1 * A
    dden_dalpha = d2 - z1

    # Jacobian w.r.t. the 3D point.
    J_point = np.empty(points_3d.shape[:-1] + (2, 3), dtype=np.float64)
    J_point[..., 0, 0] = fx * (den - x * dden_dx) * inv2
    J_point[..., 0, 1] = fx * (-x * dden_dy) * inv2
    J_point[..., 0, 2] = fx * (-x * dden_dz) * inv2
    J_point[..., 1, 0] = fy * (-y * dden_dx) * inv2
    J_point[..., 1, 1] = fy * (den - y * dden_dy) * inv2
    J_point[..., 1, 2] = fy * (-y * dden_dz) * inv2

    # Jacobian w.r.t. intrinsics [fx, fy, cx, cy, xi, alpha].
    J_intr = np.zeros(points_3d.shape[:-1] + (2, 6), dtype=np.float64)
    J_intr[..., 0, 0] = x * inv          # du/dfx
    J_intr[..., 0, 2] = 1.0              # du/dcx
    J_intr[..., 0, 4] = -fx * x * inv2 * dden_dxi
    J_intr[..., 0, 5] = -fx * x * inv2 * dden_dalpha
    J_intr[..., 1, 1] = y * inv          # dv/dfy
    J_intr[..., 1, 3] = 1.0              # dv/dcy
    J_intr[..., 1, 4] = -fy * y * inv2 * dden_dxi
    J_intr[..., 1, 5] = -fy * y * inv2 * dden_dalpha

    return u, v, J_point, J_intr, valid


def ds_unproject(points_2d: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                 xi: float, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standalone unprojection function.
    Returns: rays, valid
    """
    u, v = points_2d[..., 0], points_2d[..., 1]

    mx = (u - cx) / fx
    my = (v - cy) / fy
    r2 = mx*mx + my*my

    # Validity check 1: Sphere intersection check
    s = 1.0 - (2.0 * alpha - 1.0) * r2
    valid_s = s >= 0
    s = np.maximum(s, 0.0)

    # Closed-form unprojection. The denominator is >= (1-alpha) for alpha in [0,1) and only
    # reaches exactly 0 at the boundary alpha=1 (a legal optimizer bound, calib_param.py's
    # alpha in [0,1]) combined with a ray exactly at the sphere-intersection edge (s=0) — a
    # real, reachable 0/0 on wide-FOV fits, not a hypothetical. Clamp explicitly and gate it
    # in `valid` instead of relying on the resulting NaN happening to compare False below.
    den = alpha * np.sqrt(s) + (1.0 - alpha)
    valid_den = den > 1e-9
    mz = (1.0 - alpha*alpha * r2) / np.maximum(den, 1e-9)

    # Validity check 2: Feasible ray scale check (prevents nan in sqrt)
    sqrt_arg = mz*mz + (1.0 - xi*xi) * r2
    valid_sqrt = sqrt_arg >= 0

    # Combined validity mask
    valid = valid_s & valid_den & valid_sqrt

    # Safe square root calculation
    sqrt_safe = np.sqrt(np.maximum(sqrt_arg, 0.0))
    k = (mz * xi + sqrt_safe) / np.maximum(mz*mz + r2, 1e-10)

    ray = np.stack([k * mx, k * my, k * mz - xi], axis=-1)
    norm = np.linalg.norm(ray, axis=-1, keepdims=True)
    ray = ray / np.maximum(norm, 1e-10)

    # Safely zero out invalid rays to prevent NaN propagation
    ray[~valid] = 0.0

    return ray, valid
