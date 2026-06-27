"""
Pure EUCM+ math — a *truly* closed-form-invertible staged fisheye model.

Composition (read right-to-left; a 3D bearing enters on the right, a pixel exits
on the left)::

    pixel  =  K . H_tau . D_lambda . S_(alpha,beta)(bearing)

  S : Enhanced Unified Camera Model core (Khomutenko et al. 2016), the UCM core
      with a radial weight ``beta`` inside the sphere distance -> (x, y)
  D : division-model radial layer (Fitzgibbon, CVPR 2001), 1 term (lambda1)
  H : projective tilt / Scheimpflug layer (cf. OpenCV CALIB_TILTED_MODEL), (tau_x,tau_y)
  K : pinhole (fx, fy, cx, cy)

EUCM+ is the sqrt-only sibling of DS+ (``dsplus_math``): it swaps the UCM core for
the EUCM core (adding ``beta``) and keeps a *single* division term so that the whole
inverse is solvable with square roots alone — no cube root, no ``np.roots``, no
Newton iteration. The 1-term division inverse is a quadratic (one ``sqrt``); the
EUCM inverse is itself sqrt-only; the tilt inverse is linear.

Parameter order is ``(fx, fy, cx, cy, alpha, beta, lambda1, tau_x, tau_y)``.

Self-contained: numpy only, no internal imports (enforced by the independence
gate), so it stands alone and is independently testable.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------
def eucmplus_project(points_3d: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                     alpha: float, beta: float, lambda1: float,
                     tau_x: float, tau_y: float
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame points. Returns ``(u, v, valid)``."""
    x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]

    # Stage S: EUCM sphere.
    rho2 = x * x + y * y
    d = np.sqrt(beta * rho2 + z * z)
    den = alpha * d + (1.0 - alpha) * z
    valid_s = den > 1e-9
    dens = np.maximum(den, 1e-9)
    mx, my = x / dens, y / dens

    # Stage D: division radial (Fitzgibbon, 1 term).
    s2 = mx * mx + my * my
    g = 1.0 + lambda1 * s2
    valid_d = np.abs(g) > 1e-9
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
def eucmplus_project_jacobian(points_3d: np.ndarray, fx: float, fy: float,
                              cx: float, cy: float, alpha: float, beta: float,
                              lambda1: float, tau_x: float, tau_y: float
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray]:
    """Analytic Jacobian. Returns ``(u, v, J_point(N,2,3), J_param(N,2,9), valid)``.

    Parameter order: ``(fx, fy, cx, cy, alpha, beta, lambda1, tau_x, tau_y)``.
    """
    x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]

    # Stage S: EUCM sphere.
    rho2 = x * x + y * y
    d = np.sqrt(beta * rho2 + z * z)
    d = np.maximum(d, 1e-12)
    den = alpha * d + (1.0 - alpha) * z
    valid_s = den > 1e-9
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    inv = 1.0 / den
    inv2 = inv * inv
    mx, my = x * inv, y * inv

    # Stage D: division radial (1 term).
    s2 = mx * mx + my * my
    g = 1.0 + lambda1 * s2
    valid_d = np.abs(g) > 1e-9
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
    dden_dx = alpha * beta * x / d
    dden_dy = alpha * beta * y / d
    dden_dz = alpha * z / d + (1.0 - alpha)
    JS = np.empty(shape + (2, 3), dtype=np.float64)
    JS[..., 0, 0] = (den - x * dden_dx) * inv2
    JS[..., 0, 1] = (-x * dden_dy) * inv2
    JS[..., 0, 2] = (-x * dden_dz) * inv2
    JS[..., 1, 0] = (-y * dden_dx) * inv2
    JS[..., 1, 1] = (den - y * dden_dy) * inv2
    JS[..., 1, 2] = (-y * dden_dz) * inv2

    # J_D = d(dx,dy)/d(mx,my)  (2x2), c = (1/r) dg/dr so dg/dmx = c*mx
    c = 2.0 * lambda1
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

    # beta (stage S): K @ H @ D @ d(mx,my)/dbeta
    dden_dbeta = alpha * rho2 / (2.0 * d)
    dp_dbeta = np.empty(shape + (2, 1), dtype=np.float64)
    dp_dbeta[..., 0, 0] = -x * dden_dbeta * inv2
    dp_dbeta[..., 1, 0] = -y * dden_dbeta * inv2
    J_beta = Kdiag @ JH @ JD @ dp_dbeta
    J_param[..., :, 5] = J_beta[..., :, 0]

    # lambda1 (stage D): K @ H @ d(dx,dy)/dlambda1
    dD_dlam = np.empty(shape + (2, 1), dtype=np.float64)
    dD_dlam[..., 0, 0] = -mx * s2 / g2          # d dx/d lambda1
    dD_dlam[..., 1, 0] = -my * s2 / g2          # d dy/d lambda1
    J_lam = Kdiag @ JH @ dD_dlam
    J_param[..., :, 6] = J_lam[..., :, 0]

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
# Closed-form unprojection (pixel -> unit ray), reverse order. SQRT-ONLY:
# no cube root, no polynomial root finder, no iteration.
# ---------------------------------------------------------------------------
def eucmplus_unproject(points_2d: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                       alpha: float, beta: float, lambda1: float,
                       tau_x: float, tau_y: float) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject pixels to unit rays (closed form, sqrt-only). Returns ``(rays, valid)``."""
    u, v = points_2d[..., 0], points_2d[..., 1]

    # K^-1
    xt = (u - cx) / fx
    yt = (v - cy) / fy

    # H^-1 : tilt inverse (linear; same projective form, sign flipped)
    denom = 1.0 - tau_x * xt - tau_y * yt
    valid_h = np.abs(denom) > 1e-9
    denoms = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    wt = 1.0 / denoms
    xg = xt * wt
    yg = yt * wt

    # D^-1 : recover undistorted radius from the distorted radius (1-term division,
    # quadratic -> one sqrt). rd = r / (1 + lambda1 r^2) =>
    # lambda1 rd r^2 - r + rd = 0 => r = (1 - sqrt(1 - 4 lambda1 rd^2)) / (2 lambda1 rd).
    rd = np.hypot(xg, yg)
    disc = 1.0 - 4.0 * lambda1 * rd * rd
    valid_d = disc >= 0.0
    discs = np.maximum(disc, 0.0)
    nz = rd > 1e-12
    if abs(lambda1) < 1e-15:
        r = rd.copy()
    else:
        r_nz = (1.0 - np.sqrt(discs)) / (2.0 * lambda1 * np.maximum(rd, 1e-300))
        r = np.where(nz, r_nz, rd)  # limit r -> rd as rd -> 0
    scale = np.where(nz, r / np.maximum(rd, 1e-300), 1.0)
    mx = xg * scale
    my = yg * scale

    # S^-1 : EUCM closed form (sqrt-only)
    r2 = mx * mx + my * my
    ss = 1.0 - (2.0 * alpha - 1.0) * beta * r2
    valid_s = ss >= 0.0
    ss = np.maximum(ss, 0.0)
    mz = (1.0 - beta * alpha * alpha * r2) / (alpha * np.sqrt(ss) + (1.0 - alpha))

    ray = np.stack([mx, my, mz], axis=-1)
    norm = np.linalg.norm(ray, axis=-1, keepdims=True)
    ray = ray / np.maximum(norm, 1e-10)
    valid = valid_s & valid_d & valid_h
    ray[~valid] = 0.0
    return ray, valid
