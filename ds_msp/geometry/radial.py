r"""Lens-independent camera seeding from the radial alignment constraint.

For any central camera with a radially symmetric lens, a pixel offset from the principal point
points in the same azimuth as the camera-frame ray, whatever the radial function is. With
``(x, y)`` the pixel offset and ``(X_c, Y_c)`` the camera-frame point,

    x Y_c - y X_c = 0,    x X_c + y Y_c > 0.                                        (1)

(1) holds for incidence angles past 90 degrees, where a pinhole bearing does not exist. It is
linear in the first two rows of ``[R | t]`` (Tsai 1987; Thirthala and Pollefeys 2005), so these
are found without knowing the focal length or the distortion. The remaining unknowns, each view's
``t_z`` and the lens, are linear as well once the lens is written as a back-projection polynomial
``g`` (Scaramuzza et al. 2006), with ``rho`` the pixel radius and ``R_c = hypot(X_c, Y_c)``:

    (x, y, g(rho)) parallel to (X_c, Y_c, Z_c)  =>  g(rho) R_c - rho t_z = rho (r_31 X + r_32 Y + r_33 Z).

A negative ``g`` is a ray behind the image plane, so the same equation covers fisheye lenses.
No focal length is needed and nothing here is seeded: every step is a deterministic linear solve.

A planar view has two solutions for the third rotation row. The polynomial is shared across
views, so each view takes the branch the shared lens explains best.

Limits: the principal point is an input; (1) degrades as its error approaches the radius of the
observed points. Tangential distortion and decentring break (1) slightly. This module provides a
seed for a later nonlinear refinement and makes no accuracy claim of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

#: Powers of ``rho / rho_scale`` in the back-projection polynomial; the linear term is left out
#: because a radially symmetric lens has ``dg/drho = 0`` on the axis (Scaramuzza et al. 2006).
BACKPROJECTION_POWERS: Tuple[int, ...] = (0, 2, 3, 4)

#: A view is planar when its smallest point-cloud singular value is below this fraction of the
#: largest. Printed boards sit at machine precision; fused multi-board targets are far above it.
_PLANAR_TOL = 1e-6

#: IRLS passes in the radial resection and the shared lens solve.
_IRLS_PASSES = 4

#: Least-median samples per view. At 30% blunders a 7-point sample is clean 8% of the time, so 40
#: samples reach a clean one 96% of the time, and a view that misses is dropped by the lens step.
_LMEDS_SAMPLES = 40
#: Points per least-median sample: one above the planar minimum of 6 rows, which damps noise.
_LMEDS_POINTS = 7

#: Minimum correspondences for a view: 5 for the planar null space plus redundancy to reject noise.
MIN_VIEW_POINTS = 8


@dataclass
class RadialView:
    """Outcome of one view. ``R``/``t`` map target to camera; ``t[2]`` comes from the shared lens."""
    index: int
    R: np.ndarray
    t: np.ndarray
    inliers: np.ndarray
    residual: float


@dataclass
class RadialSeed:
    """Lens-independent seed for one camera: poses, and rays for every inlier pixel."""
    views: List[RadialView]
    coeffs: np.ndarray
    rho_scale: float
    cx: float
    cy: float
    rays: np.ndarray
    pixels: np.ndarray

    @property
    def paraxial_focal(self) -> float:
        """Focal length on the axis in pixels; the polynomial's constant term in image units."""
        return float(self.coeffs[0] * self.rho_scale)

    def backproject(self, pixels: np.ndarray) -> np.ndarray:
        """Unit rays for pixels under the fitted polynomial."""
        xy = np.asarray(pixels, float) - (self.cx, self.cy)
        rho = np.linalg.norm(xy, axis=1)
        rays = np.column_stack([xy, _poly(rho / self.rho_scale, self.coeffs) * self.rho_scale])
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _poly(u: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    return sum(c * u ** p for c, p in zip(coeffs, BACKPROJECTION_POWERS))


def _is_planar(X: np.ndarray) -> bool:
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    return s[-1] <= _PLANAR_TOL * max(s[0], 1e-12)


def _plane_frame(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(B, o)`` such that ``X = o + P @ B`` with ``P`` having a zero third column."""
    o = X.mean(0)
    _, _, Vt = np.linalg.svd(X - o)
    B = Vt.copy()
    if np.linalg.det(B) < 0:
        B[2] *= -1
    return B, o


def _weighted_null(A: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    W = A if weights is None else A * weights[:, None]
    return np.linalg.svd(W.T @ W)[2][-1]


def _radial_rows_planar(uv: np.ndarray, P: np.ndarray, w: Optional[np.ndarray]):
    """Candidates ``(R, t12)`` from the planar constraint, ``P`` being ``(N, 2)`` plane coordinates."""
    x, y = uv.T
    a, b = P.T
    A = np.column_stack([-y * a, -y * b, x * a, x * b, -y, x])
    n = _weighted_null(A, w)
    m = n[:4].reshape(2, 2)                                # scaled [[r11, r12], [r21, r22]]
    tt = n[4:]
    p, q = float(m[:, 0] @ m[:, 0]), float(m[:, 1] @ m[:, 1])
    det2 = float(np.linalg.det(m) ** 2)
    disc = max((p + q) ** 2 - 4.0 * det2, 0.0)
    if det2 > 1e-30:
        s2 = ((p + q) - np.sqrt(disc)) / (2.0 * det2)     # the other root makes r31^2 negative
    else:
        s2 = 1.0 / max(p + q, 1e-30)
    s = float(np.sqrt(max(s2, 0.0)))
    if s == 0.0:
        return []
    m2, tt2 = s * m, s * tt
    # Equation (1)'s positivity fixes the overall sign, for any incidence angle.
    Xc, Yc = (m2 @ P.T) + tt2[:, None]
    if np.median(x * Xc + y * Yc) < 0.0:
        m2, tt2 = -m2, -tt2
    r31 = np.sqrt(max(1.0 - (m2[0, 0] ** 2 + m2[1, 0] ** 2), 0.0))
    r32 = np.sqrt(max(1.0 - (m2[0, 1] ** 2 + m2[1, 1] ** 2), 0.0))
    # Orthogonality of the first two columns picks the relative sign of (r31, r32).
    if -(m2[:, 0] @ m2[:, 1]) < 0.0:
        r32 = -r32
    out = []
    for sign in (1.0, -1.0):
        c1 = np.array([m2[0, 0], m2[1, 0], sign * r31])
        c2 = np.array([m2[0, 1], m2[1, 1], sign * r32])
        u1 = c1 / max(np.linalg.norm(c1), 1e-12)
        u2 = c2 - (u1 @ c2) * u1
        u2 /= max(np.linalg.norm(u2), 1e-12)
        out.append((np.column_stack([u1, u2, np.cross(u1, u2)]), tt2.copy()))
    return out


def _radial_rows_general(uv: np.ndarray, X: np.ndarray, w: Optional[np.ndarray]):
    """Candidate ``(R, t12)`` from the non-coplanar constraint (8 unknowns, one solution)."""
    x, y = uv.T
    A = np.column_stack([-y * X[:, 0], -y * X[:, 1], -y * X[:, 2],
                         x * X[:, 0], x * X[:, 1], x * X[:, 2], -y, x])
    n = _weighted_null(A, w)
    r1, r2, tt = n[:3], n[3:6], n[6:]
    s = 2.0 / max(np.linalg.norm(r1) + np.linalg.norm(r2), 1e-30)
    r1, r2, tt = s * r1, s * r2, s * tt
    Xc, Yc = (np.vstack([r1, r2]) @ X.T) + tt[:, None]
    if np.median(x * Xc + y * Yc) < 0.0:
        r1, r2, tt = -r1, -r2, -tt
    U, _, Vt = np.linalg.svd(np.vstack([r1, r2, np.cross(r1, r2)]))
    R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    return [(R, tt)]


def _alignment_residual(uv: np.ndarray, X: np.ndarray, R: np.ndarray, t12: np.ndarray) -> np.ndarray:
    """Pixel distance of each point from the radial line of its predicted camera-frame azimuth."""
    XY = (R[:2] @ X.T).T + t12
    n = np.maximum(np.linalg.norm(XY, axis=1), 1e-12)
    return np.abs(uv[:, 0] * XY[:, 1] - uv[:, 1] * XY[:, 0]) / n


def _scale_band(residual: np.ndarray, floor: float) -> float:
    return max(floor, 3.0 * 1.4826 * float(np.median(residual)))


def radial_view_candidates(uv: np.ndarray, X: np.ndarray, *, floor_px: float = 1.0):
    """IRLS radial resection of one view.

    ``uv`` are pixel offsets from the principal point. Returns ``[(R, t12, inlier_mask), ...]``,
    two candidates for a planar view and one otherwise, empty when the view is degenerate. Weights
    use a truncated (hard) rule on the residual, re-scaled from its median each pass.
    """
    uv, X = np.asarray(uv, float), np.asarray(X, float)
    if len(uv) < MIN_VIEW_POINTS:
        return []
    planar = _is_planar(X)
    if planar:
        B, o = _plane_frame(X)
        target = (X - o) @ B.T
    else:
        target = X

    def solve(w):
        return (_radial_rows_planar(uv, target[:, :2], w) if planar
                else _radial_rows_general(uv, X, w))

    def residual_of(w):
        found = solve(w)
        return (found, _alignment_residual(uv, target, found[0][0], found[0][1])) if found else ([], None)

    # High breakdown first: a least-squares start cannot survive many blunders, as they shape the
    # very residuals that would have to expose them. Samples come from a generator seeded by the
    # data size only, so the same view always gives the same answer.
    rng = np.random.default_rng(len(uv))
    n_sample = max(_LMEDS_POINTS, 8 if not planar else _LMEDS_POINTS)
    best_w, best_median = None, np.inf
    if len(uv) > n_sample:
        for _ in range(_LMEDS_SAMPLES):
            w0 = np.zeros(len(uv))
            w0[rng.choice(len(uv), n_sample, replace=False)] = 1.0
            found, res = residual_of(w0)
            if res is None:
                continue
            median = float(np.median(res))
            if median < best_median:
                best_median, best_w = median, w0
    w = None
    if best_w is not None:
        band = _scale_band(residual_of(best_w)[1], floor_px)
        w = (residual_of(best_w)[1] <= band).astype(float)
        if w.sum() < MIN_VIEW_POINTS:
            return []
    sols = []
    for _ in range(_IRLS_PASSES):
        sols, res = residual_of(w)
        if not sols:
            return []
        band = _scale_band(res, floor_px)
        w_new = (res <= band).astype(float)
        if w_new.sum() < MIN_VIEW_POINTS:
            return []
        if w is not None and np.array_equal(w, w_new):
            break
        w = w_new
    inl = np.asarray(w, bool) if w is not None else np.ones(len(uv), bool)
    out = []
    for R, tt in sols:
        if planar:
            # Back to the target's own frame: X_c = R P + t with P = (X - o) B^T.
            R_full = R @ B
            t_full = np.array([tt[0], tt[1], 0.0]) - R_full @ o
            out.append((R_full, t_full[:2].copy(), inl))
        else:
            out.append((R, tt, inl))
    return out


def _design_rows(uv: np.ndarray, X: np.ndarray, R: np.ndarray, t12: np.ndarray, rho_scale: float):
    """Rows of ``g(rho) R_c - rho t_z = rho Z_p`` divided by ``rho_scale * max(R_c)``.

    Returns ``(powers (N, K), rho (N,), rhs (N,))`` so a caller can place the per-view ``t_z``
    column. The division makes the rows of different views comparable."""
    XY = (R[:2] @ X.T).T + t12
    Rc = np.linalg.norm(XY, axis=1)
    Zp = (R[2] @ X.T)
    rho = np.linalg.norm(uv, axis=1)
    u = rho / rho_scale
    # Dividing by the view's point radius makes the equation dimensionless across target scales.
    k = max(float(np.median(Rc)), 1e-12)
    powers = np.column_stack([(u ** p) * Rc / k for p in BACKPROJECTION_POWERS])
    return powers, u / k, u * Zp / k


def _ray_angle_residual(uv: np.ndarray, X: np.ndarray, R: np.ndarray, t: np.ndarray,
                        coeffs: np.ndarray, rho_scale: float) -> np.ndarray:
    """Angle (radians) between each back-projected pixel ray and the camera-frame target point."""
    rho = np.linalg.norm(uv, axis=1)
    pred = np.column_stack([uv, _poly(rho / rho_scale, coeffs) * rho_scale])
    pc = (R @ X.T).T + t
    pred /= np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
    pc /= np.maximum(np.linalg.norm(pc, axis=1, keepdims=True), 1e-12)
    return np.arctan2(np.linalg.norm(np.cross(pred, pc), axis=1), np.sum(pred * pc, axis=1))


def _solve_tz(powers, u, rhs, coeffs, mask) -> Tuple[float, float]:
    """Best ``t_z`` for a fixed lens: 1-D least squares of the shared equation, with its RMS."""
    a = u[mask]
    if a.size == 0 or a @ a <= 0.0:
        return 0.0, float("inf")
    b = powers[mask] @ coeffs - rhs[mask]
    tz = float(a @ b / (a @ a))
    return tz, float(np.sqrt(np.mean((a * tz - b) ** 2)))


def _joint_lens(rows, mask_of) -> Optional[np.ndarray]:
    """Shared lens coefficients with every view's ``t_z`` eliminated (variable projection)."""
    K = len(BACKPROJECTION_POWERS)
    AtA, Atb = np.zeros((K, K)), np.zeros(K)
    for i, (powers, u, rhs) in enumerate(rows):
        m = mask_of(i)
        if m.sum() < 2:
            continue
        P, a, b = powers[m], u[m], rhs[m]
        aa = float(a @ a)
        if aa <= 0.0:
            continue
        # Project out the view's t_z column: M = I - a a^T / (a^T a).
        Pp = P - np.outer(a, a @ P) / aa
        bp = b - a * (a @ b) / aa
        AtA += Pp.T @ Pp
        Atb += Pp.T @ bp
    if not np.isfinite(AtA).all() or np.trace(AtA) <= 0.0:
        return None
    try:
        return np.linalg.lstsq(AtA, Atb, rcond=1e-12)[0]
    except np.linalg.LinAlgError:
        return None


def radial_seed(object_points: Sequence[np.ndarray], pixels: Sequence[np.ndarray],
                principal_point: Tuple[float, float], *, rho_scale: Optional[float] = None,
                rounds: int = 5, floor_px: float = 1.0) -> Optional[RadialSeed]:
    """Lens-independent poses and rays for one camera from known-target views.

    ``object_points[i]`` is ``(N_i, 3)`` in that view's target frame (one printed board, or a
    fused object known to be accurate) and ``pixels[i]`` the matching ``(N_i, 2)`` detections.
    Views need not share a frame: each gets its own pose. Returns ``None`` when no consistent lens
    is found, so the caller keeps its other seeds.

    Each planar view starts on the branch that alone gives the lower residual; the views then
    alternate between a joint lens solve and re-picking each view's branch and ``t_z``.
    """
    if not len(object_points):
        return None
    pp = np.asarray(principal_point, float)
    data = []
    if rho_scale is None:
        rho_scale = max(float(np.max(np.linalg.norm(np.asarray(p, float) - pp, axis=1)))
                        for p in pixels if len(p))
    for i, (X, uv_abs) in enumerate(zip(object_points, pixels)):
        X = np.asarray(X, float)
        uv = np.asarray(uv_abs, float) - pp
        if len(uv) < MIN_VIEW_POINTS:
            continue
        cands = radial_view_candidates(uv, X, floor_px=floor_px)
        if not cands:
            continue
        rows = [_design_rows(uv, X, R, t12, rho_scale) for R, t12, _ in cands]
        data.append(dict(index=i, X=X, uv=uv, cands=cands, rows=rows, pick=0))
    if len(data) < 2:
        return None

    # Single-view fits only choose a starting branch, so a two-term lens is enough for each.
    for d in data:
        best, best_rms = 0, float("inf")
        for j, (powers, u, rhs) in enumerate(d["rows"]):
            m = d["cands"][j][2]
            A = np.column_stack([powers[m][:, :2], -u[m]])
            sol, *_ = np.linalg.lstsq(A, rhs[m], rcond=None)
            rms = float(np.sqrt(np.mean((A @ sol - rhs[m]) ** 2)))
            if sol[0] > 0.0 and rms < best_rms:
                best, best_rms = j, rms
        d["pick"] = best

    keep = np.ones(len(data), bool)
    coeffs = None
    for _ in range(max(rounds, 1)):
        rows = [d["rows"][d["pick"]] for d, k in zip(data, keep) if k]
        masks = [d["cands"][d["pick"]][2] & d.get("lens_inliers", True)
                 for d, k in zip(data, keep) if k]
        coeffs_new = _joint_lens(rows, lambda i: masks[i])
        if coeffs_new is None or coeffs_new[0] <= 0.0:
            return None if coeffs is None else _finish(data, keep, coeffs, rho_scale, pp)
        coeffs = coeffs_new
        changed = False
        # A blunder along the radial line passes equation (1) and is visible only here.
        for d in data:
            powers, u, rhs = d["rows"][d["pick"]]
            base = d["cands"][d["pick"]][2]
            tz = _solve_tz(powers, u, rhs, coeffs, base)[0]
            e = np.abs(powers @ coeffs - u * tz - rhs)
            band = _scale_band(e[base], 1e-12)
            mask = e <= band
            changed |= not np.array_equal(mask, d.get("lens_inliers"))
            d["lens_inliers"] = mask
        rms_views = []
        for d in data:
            scored = []
            for j, (powers, u, rhs) in enumerate(d["rows"]):
                scored.append(_solve_tz(powers, u, rhs, coeffs, d["cands"][j][2] & (
                    d["lens_inliers"] if j == d["pick"] else True))[1])
            pick = int(np.argmin(scored))
            changed |= pick != d["pick"]
            d["pick"] = pick
            rms_views.append(scored[pick])
        rms_views = np.asarray(rms_views)
        # A view the shared lens cannot explain (mis-decoded board) must not shape that lens.
        band = _scale_band(rms_views, 1e-9)
        keep_new = rms_views <= band
        if keep_new.sum() < 2:
            keep_new = keep
        changed |= not np.array_equal(keep_new, keep)
        keep = keep_new
        if not changed:
            break
    return _finish(data, keep, coeffs, rho_scale, pp)


def _finish(data, keep, coeffs, rho_scale, pp) -> Optional[RadialSeed]:
    views, ray_list, pix_list = [], [], []
    for d, k in zip(data, keep):
        if not k:
            continue
        R, t12, inl = d["cands"][d["pick"]]
        inl = inl & d.get("lens_inliers", True)
        powers, u, rhs = d["rows"][d["pick"]]
        tz, rms = _solve_tz(powers, u, rhs, coeffs, inl)
        t = np.array([t12[0], t12[1], tz])
        err = _ray_angle_residual(d["uv"], d["X"], R, t, coeffs, rho_scale)
        views.append(RadialView(index=d["index"], R=R, t=t, inliers=inl, residual=float(np.median(err))))
        Xc = (R @ d["X"].T).T + t
        ray_list.append(Xc[inl] / np.maximum(np.linalg.norm(Xc[inl], axis=1, keepdims=True), 1e-12))
        pix_list.append(d["uv"][inl] + pp)
    if not views:
        return None
    return RadialSeed(views=views, coeffs=np.asarray(coeffs, float), rho_scale=float(rho_scale),
                      cx=float(pp[0]), cy=float(pp[1]),
                      rays=np.vstack(ray_list), pixels=np.vstack(pix_list))
