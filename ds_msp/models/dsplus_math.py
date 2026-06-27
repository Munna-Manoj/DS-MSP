"""
Pure DS+ math — a closed-form-invertible staged fisheye model.

Composition (read right-to-left; a 3D bearing enters on the right, a pixel exits
on the left)::

    pixel  =  K . H_tau . D_lambda . S_alpha(bearing)

  S : Unified Camera Model core (Geyer/Mei; = Double Sphere with xi=0) -> (mx,my)
  D : division-model radial layer (Fitzgibbon, CVPR 2001), 2 terms (lambda1,lambda2)
  H : projective tilt / Scheimpflug layer (cf. OpenCV CALIB_TILTED_MODEL), (tau_x,tau_y)
  K : pinhole (fx,fy,cx,cy)

This is the UCM-core, two-axis-tilt specialization of the DS+ derivation in
``.ai/experiments/2026-06-27-dsplus-derivation/`` (Double-Sphere's xi is dropped;
it is the proven near-null direction for the target lens). Parameter order is
``(fx, fy, cx, cy, alpha, lambda1, lambda2, tau_x, tau_y)``.

Self-contained: numpy only, no internal imports (enforced by the independence
gate), so it stands alone and is independently testable.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _w(alpha: float) -> float:
    """UCM half-space coefficient (DS condition with xi = 0)."""
    return (1.0 - alpha) / alpha if alpha > 0.5 else alpha / (1.0 - alpha)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------
def dsplus_project(points_3d: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                   alpha: float, lambda1: float, lambda2: float,
                   tau_x: float, tau_y: float
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame points. Returns ``(u, v, valid)``."""
    x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]

    # Stage S: UCM sphere.
    d = np.sqrt(x * x + y * y + z * z)
    den = alpha * d + (1.0 - alpha) * z
    valid_s = (z > -_w(alpha) * d) & (den > 1e-9)
    dens = np.maximum(den, 1e-9)
    mx, my = x / dens, y / dens

    # Stage D: division radial (Fitzgibbon).
    rho2 = mx * mx + my * my
    g = 1.0 + lambda1 * rho2 + lambda2 * rho2 * rho2
    valid_d = (g > 1e-9) & (1.0 - lambda1 * rho2 - 3.0 * lambda2 * rho2 * rho2 > 0.0)
    gs = np.where(np.abs(g) < 1e-12, 1e-12, g)
    dx, dy = mx / gs, my / gs

    # Stage H: tilt homography (2-axis Scheimpflug).
    w = tau_x * dx + tau_y * dy + 1.0
    valid_h = np.abs(w) > 1e-9
    ws = np.where(np.abs(w) < 1e-12, 1e-12, w)
    hx, hy = dx / ws, dy / ws

    # Stage K: pinhole.
    u = fx * hx + cx
    v = fy * hy + cy
    return u, v, valid_s & valid_d & valid_h


# ---------------------------------------------------------------------------
# Analytic Jacobian (chain rule through the 4 stages)
# ---------------------------------------------------------------------------
def dsplus_project_jacobian(points_3d: np.ndarray, fx: float, fy: float,
                            cx: float, cy: float, alpha: float,
                            lambda1: float, lambda2: float,
                            tau_x: float, tau_y: float
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray]:
    """Analytic Jacobian. Returns ``(u, v, J_point(N,2,3), J_param(N,2,9), valid)``.

    Parameter order: ``(fx, fy, cx, cy, alpha, lambda1, lambda2, tau_x, tau_y)``.
    """
    x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]

    # Stage S: UCM sphere.
    d = np.sqrt(x * x + y * y + z * z)
    d = np.maximum(d, 1e-12)
    den = alpha * d + (1.0 - alpha) * z
    valid_s = (z > -_w(alpha) * d) & (den > 1e-9)
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    inv = 1.0 / den
    inv2 = inv * inv
    mx, my = x * inv, y * inv

    # Stage D: division radial.
    rho2 = mx * mx + my * my
    g = 1.0 + lambda1 * rho2 + lambda2 * rho2 * rho2
    valid_d = (g > 1e-9) & (1.0 - lambda1 * rho2 - 3.0 * lambda2 * rho2 * rho2 > 0.0)
    g = np.where(np.abs(g) < 1e-12, 1e-12, g)
    g2 = g * g
    dx, dy = mx / g, my / g

    # Stage H: tilt homography.
    w = tau_x * dx + tau_y * dy + 1.0
    valid_h = np.abs(w) > 1e-9
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    w2 = w * w
    hx, hy = dx / w, dy / w

    # Stage K: pinhole.
    u = fx * hx + cx
    v = fy * hy + cy

    shape = points_3d.shape[:-1]

    # --- stage Jacobian blocks ---
    # J_S = d(mx,my)/d(x,y,z)  (2x3)
    dden_dx = alpha * x / d
    dden_dy = alpha * y / d
    dden_dz = alpha * z / d + (1.0 - alpha)
    JS = np.empty(shape + (2, 3), dtype=np.float64)
    JS[..., 0, 0] = (den - x * dden_dx) * inv2
    JS[..., 0, 1] = (-x * dden_dy) * inv2
    JS[..., 0, 2] = (-x * dden_dz) * inv2
    JS[..., 1, 0] = (-y * dden_dx) * inv2
    JS[..., 1, 1] = (den - y * dden_dy) * inv2
    JS[..., 1, 2] = (-y * dden_dz) * inv2

    # J_D = d(dx,dy)/d(mx,my)  (2x2), c = (1/r) dg/dr so dg/dmx = c*mx
    c = 2.0 * lambda1 + 4.0 * lambda2 * rho2
    JD = np.empty(shape + (2, 2), dtype=np.float64)
    JD[..., 0, 0] = (g - c * mx * mx) / g2
    JD[..., 0, 1] = (-c * mx * my) / g2
    JD[..., 1, 0] = (-c * mx * my) / g2
    JD[..., 1, 1] = (g - c * my * my) / g2

    # J_H = d(hx,hy)/d(dx,dy)  (2x2)
    JH = np.empty(shape + (2, 2), dtype=np.float64)
    JH[..., 0, 0] = (w - dx * tau_x) / w2
    JH[..., 0, 1] = (-dx * tau_y) / w2
    JH[..., 1, 0] = (-dy * tau_x) / w2
    JH[..., 1, 1] = (w - dy * tau_y) / w2

    Kdiag = np.zeros(shape + (2, 2), dtype=np.float64)
    Kdiag[..., 0, 0] = fx
    Kdiag[..., 1, 1] = fy

    # J_point = K @ H @ D @ S
    J_point = Kdiag @ JH @ JD @ JS

    # --- J_param ---
    J_param = np.zeros(shape + (2, 9), dtype=np.float64)
    J_param[..., 0, 0] = hx          # du/dfx
    J_param[..., 1, 1] = hy          # dv/dfy
    J_param[..., 0, 2] = 1.0         # du/dcx
    J_param[..., 1, 3] = 1.0         # dv/dcy

    # alpha (stage S): K @ H @ D @ d(mx,my)/dalpha
    dden_dalpha = d - z
    dp_dalpha = np.empty(shape + (2, 1), dtype=np.float64)
    dp_dalpha[..., 0, 0] = -x * dden_dalpha * inv2
    dp_dalpha[..., 1, 0] = -y * dden_dalpha * inv2
    J_alpha = Kdiag @ JH @ JD @ dp_dalpha
    J_param[..., :, 4] = J_alpha[..., :, 0]

    # lambda1,lambda2 (stage D): K @ H @ d(dx,dy)/dlambda
    dD_dlam = np.empty(shape + (2, 2), dtype=np.float64)
    dD_dlam[..., 0, 0] = -mx * rho2 / g2          # d dx/d lambda1
    dD_dlam[..., 0, 1] = -mx * rho2 * rho2 / g2   # d dx/d lambda2
    dD_dlam[..., 1, 0] = -my * rho2 / g2
    dD_dlam[..., 1, 1] = -my * rho2 * rho2 / g2
    J_lam = Kdiag @ JH @ dD_dlam
    J_param[..., :, 5] = J_lam[..., :, 0]
    J_param[..., :, 6] = J_lam[..., :, 1]

    # tau_x,tau_y (stage H): K @ d(hx,hy)/dtau
    dH_dtau = np.empty(shape + (2, 2), dtype=np.float64)
    dH_dtau[..., 0, 0] = -dx * dx / w2     # d hx/d tau_x
    dH_dtau[..., 0, 1] = -dx * dy / w2     # d hx/d tau_y
    dH_dtau[..., 1, 0] = -dy * dx / w2
    dH_dtau[..., 1, 1] = -dy * dy / w2
    J_tau = Kdiag @ dH_dtau
    J_param[..., :, 7] = J_tau[..., :, 0]
    J_param[..., :, 8] = J_tau[..., :, 1]

    return u, v, J_point, J_param, valid_s & valid_d & valid_h


# ---------------------------------------------------------------------------
# Closed-form unprojection (pixel -> unit ray), reverse order
# ---------------------------------------------------------------------------
def _quartic_real_roots(c2, c1, c0):
    """Real roots of ``rho^4 + c2 rho^2 + c1 rho + c0 = 0`` (depressed, no cubic).

    Vectorized general-quartic radical (Ferrari). Returns a list of 4 complex
    root arrays.
    """
    p = c2
    q = c1
    D0 = c2 * c2 + 12.0 * c0
    D1 = 2.0 * c2 ** 3 + 27.0 * c1 * c1 - 72.0 * c2 * c0
    inner = D1 * D1 - 4.0 * D0 ** 3 + 0j
    Qc = ((D1 + np.sqrt(inner)) / 2.0) ** (1.0 / 3.0)
    bad = np.abs(Qc) < 1e-300
    if np.any(bad):
        Qc = np.where(bad, ((D1 - np.sqrt(inner)) / 2.0) ** (1.0 / 3.0), Qc)
    S = 0.5 * np.sqrt((-2.0 / 3.0) * p + (Qc + D0 / Qc) / 3.0 + 0j)
    roots = []
    for s1 in (+1.0, -1.0):
        for s2 in (+1.0, -1.0):
            term = -4.0 * S * S - 2.0 * p - s1 * (q / S)
            roots.append(s1 * S + s2 * 0.5 * np.sqrt(term + 0j))
    return roots


def _invert_division(s, lambda1: float, lambda2: float):
    """Recover undistorted radius ``rho`` solving ``s = rho/(1+l1 rho^2+l2 rho^4)``.

    1-term (``lambda2==0``) -> quadratic radical; 2-term -> quartic radical
    (Ferrari). Picks the real positive root closest to ``s`` (small-distortion
    branch). Returns ``rho`` (>= 0).
    """
    s = np.asarray(s, dtype=np.float64)
    nz = s > 1e-12
    if abs(lambda2) < 1e-15:
        if abs(lambda1) < 1e-15:
            return s.copy()
        disc = np.maximum(1.0 - 4.0 * lambda1 * s * s, 0.0)
        rho_nz = (1.0 - np.sqrt(disc)) / (2.0 * lambda1 * np.maximum(s, 1e-300))
        return np.where(nz, rho_nz, s)
    # 2-term quartic: lambda2 s rho^4 + lambda1 s rho^2 - rho + s = 0
    # divide by (lambda2 s): rho^4 + (l1/l2) rho^2 - 1/(l2 s) rho + 1/l2 = 0
    c2 = (lambda1 / lambda2) * np.ones_like(s)
    c1 = np.where(nz, -1.0 / (lambda2 * np.maximum(s, 1e-300)), 0.0)
    c0 = (1.0 / lambda2) * np.ones_like(s)
    roots = _quartic_real_roots(c2, c1, c0)
    R = np.stack(roots, axis=-1)
    real_mask = np.abs(R.imag) < 1e-6
    Rr = np.where(real_mask, R.real, np.inf)
    Rr = np.where(Rr > 0, Rr, np.inf)
    # Pick the SMALLEST positive real root: the principal branch continuously
    # connected to rho=0 (rho -> s as s -> 0). A "closest-to-s" rule fails once
    # two positive roots exist (large normalized radius rd >~ 1.8): the spurious
    # larger root can sit nearer s than the physical one, flipping the recovered
    # ray. The principal branch is always the smallest positive root within the
    # forward-injective domain (guarded by valid_d in dsplus_project).
    idx = np.argmin(Rr, axis=-1)
    rho_q = np.take_along_axis(Rr, idx[..., None], axis=-1)[..., 0]
    return np.where(nz, rho_q, s)


def dsplus_unproject(points_2d: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                     alpha: float, lambda1: float, lambda2: float,
                     tau_x: float, tau_y: float) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject pixels to unit rays (closed form). Returns ``(rays, valid)``."""
    u, v = points_2d[..., 0], points_2d[..., 1]

    # K^-1
    xp = (u - cx) / fx
    yp = (v - cy) / fy

    # H^-1 : tilt inverse (same projective form, sign flipped)
    wpr = 1.0 - tau_x * xp - tau_y * yp
    valid_h = np.abs(wpr) > 1e-9
    wprs = np.where(np.abs(wpr) < 1e-12, 1e-12, wpr)
    xg = xp / wprs
    yg = yp / wprs

    # D^-1 : recover undistorted radius from the distorted radius
    rd = np.hypot(xg, yg)
    rho = _invert_division(rd, lambda1, lambda2)
    scale = np.where(rd > 1e-12, rho / np.maximum(rd, 1e-300), 1.0)
    mx = xg * scale
    my = yg * scale

    # S^-1 : UCM (DS with xi=0) closed form
    r2 = mx * mx + my * my
    ss = 1.0 - (2.0 * alpha - 1.0) * r2
    valid_s = ss >= 0
    ss = np.maximum(ss, 0.0)
    mz = (1.0 - alpha * alpha * r2) / (alpha * np.sqrt(ss) + (1.0 - alpha))

    ray = np.stack([mx, my, mz], axis=-1)
    norm = np.linalg.norm(ray, axis=-1, keepdims=True)
    ray = ray / np.maximum(norm, 1e-10)
    valid = valid_s & valid_h
    ray[~valid] = 0.0
    return ray, valid
