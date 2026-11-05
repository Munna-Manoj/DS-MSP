r"""Fit a camera model to known rays by a global search over its shape parameters.

Every model here projects as ``u = fx m_x(ray; s) + cx``, ``v = fy m_y(ray; s) + cy`` with shape
parameters ``s``. For fixed ``s`` and centre the focal lengths are linear least-squares solutions,
so the problem separates (Golub and Pereyra 1973): the nonconvexity lives in ``s`` alone. When
``s`` has one or two parameters, a grid over ``s`` with ``fx, fy`` eliminated covers every basin
instead of relying on a local solve from one start. This matters for Double Sphere, where fits of
opposite ``xi`` sign are separate local minima of the same pixel objective.

Higher-dimensional shapes (Kannala-Brandt, RadTan) enter linearly or nearly so, and start from
zero. The result is a seed; the caller refines it jointly with the poses.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Type

import cv2
import numpy as np

from ..core.optimize import box_active, lm_solve
from ..geometry.radial import radial_seed
from .bundle import calibrate

#: Grid points per shape parameter.
_GRID_POINTS = 9
#: Grid candidates polished by a local solve. More than one, as the best-on-grid basin can differ
#: from the best polished basin when two basins are close.
_POLISH = 4
#: Largest number of correspondences used; a seed does not need all of them.
_MAX_POINTS = 3000
#: Shape parameters above this count are not gridded.
_MAX_GRID_DIMS = 2
#: Median residual (px) above which a caller should not use a fit from
#: :func:`calibrate_from_views` as a camera's seed. Failing it only selects the caller's slower
#: fallback seed, so the bound is loose: fits on the real and rendered data seen so far have a
#: median of 0.06 to 0.25 px.
MAX_SEED_MEDIAN_PX = 5.0


def _shape_grid(lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Grid over bounded shape parameters, log-spaced when the range spans over a decade and is
    positive (EUCM ``beta``), kept away from the bounds where models degenerate."""
    axes = []
    for lo, hi in zip(lb, ub):
        if lo > 0.0 and hi / lo > 10.0:
            axes.append(np.geomspace(max(lo, 0.05), hi * 0.5, _GRID_POINTS))
        else:
            margin = 0.1 * (hi - lo)
            axes.append(np.linspace(lo + margin, hi - margin, _GRID_POINTS))
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([m.ravel() for m in mesh])


def _focal_solve(cls: Type, shape: np.ndarray, rays: np.ndarray, xy: np.ndarray):
    """``(fx, fy, residual_px, valid)`` for a fixed shape, by linear least squares."""
    unit = np.concatenate([[1.0, 1.0, 0.0, 0.0], shape])
    try:
        m, valid = cls.from_params(unit).project(rays)
    except ValueError:
        return None
    m = np.asarray(m, float)
    valid = np.asarray(valid, bool).ravel() & np.isfinite(m).all(axis=1)
    if valid.sum() < 6:
        return None
    den = np.sum(m[valid] ** 2, axis=0)
    if np.any(den <= 0.0):
        return None
    f = np.sum(m[valid] * xy[valid], axis=0) / den
    if np.any(f <= 0.0):
        return None
    err = np.full(len(rays), np.inf)
    err[valid] = np.linalg.norm(m[valid] * f - xy[valid], axis=1)
    return f[0], f[1], err, valid


def _trimmed_cost(err: np.ndarray, band: float) -> float:
    """Truncated squared error, so invalid points (infinite error) and outliers count, boundedly."""
    return float(np.mean(np.minimum(err, band) ** 2))


def _polish(cls: Type, params0: np.ndarray, rays: np.ndarray, pixels: np.ndarray,
            band: float, max_iter: int) -> Optional[np.ndarray]:
    """Local intrinsics-only Levenberg-Marquardt on the ray-to-pixel objective."""
    lb, ub = cls.param_bounds()

    def residual(p):
        uv, valid = cls.from_params(p).project(rays)
        r = np.asarray(uv, float) - pixels
        r[~np.asarray(valid, bool).ravel()] = band
        return r.ravel()

    def jacobian(p):
        uv, _jp, jq, valid = cls.from_params(p).project_jacobian(rays)
        J = np.where(np.asarray(valid, bool).reshape(-1, 1, 1), jq, 0.0)
        return J.reshape(len(rays) * 2, len(p))

    res = lm_solve(np.clip(params0, lb, ub), residual, jacobian,
                   lambda p, d: np.clip(p + d, lb, ub), block=2, max_iter=max_iter,
                   robust_kernel="huber", robust_scale=band,
                   bound_active=lambda p, g: box_active(p, lb, ub, g))
    return res.state if np.isfinite(res.state).all() else None


def fit_model_candidates(cls: Type, rays: np.ndarray, pixels: np.ndarray, cx: float, cy: float,
                         *, band_px: float = 3.0, max_iter: int = 40, max_basins: int = 3
                         ) -> List[Tuple[object, float]]:
    """Distinct local minima of ``cls`` for unit ``rays`` and ``pixels``, best first. Deterministic.

    Two candidates are distinct when a shape parameter differs by over 5% of its range: close
    basins can swap rank once the poses are free, so the caller decides between them on the full
    problem. ``band_px`` truncates the selection cost; it is a distance for comparing candidates
    and does not bound the accuracy of the result. Returns ``[(model, trimmed_rms_px), ...]``.
    """
    rays, pixels = np.asarray(rays, float), np.asarray(pixels, float)
    if len(rays) < 6:
        return []
    if len(rays) > _MAX_POINTS:
        pick = np.linspace(0, len(rays) - 1, _MAX_POINTS).astype(int)
        rays, pixels = rays[pick], pixels[pick]
    xy = pixels - (cx, cy)
    lb, ub = cls.param_bounds()
    n_shape = len(lb) - 4
    if 0 < n_shape <= _MAX_GRID_DIMS:
        shapes = _shape_grid(lb[4:], ub[4:])
    else:
        shapes = np.zeros((1, n_shape))
    scored = []
    for s in shapes:
        sol = _focal_solve(cls, s, rays, xy)
        if sol is None:
            continue
        fx, fy, err, _valid = sol
        scored.append((_trimmed_cost(err, band_px), np.concatenate([[fx, fy, cx, cy], s])))
    scored.sort(key=lambda item: item[0])
    # Polish the best grid point of each grid cell neighbourhood, not the best few points overall,
    # which would all sit in one basin.
    span = np.where(np.isfinite(ub[4:] - lb[4:]), ub[4:] - lb[4:], 1.0)
    seeds: List[np.ndarray] = []
    for _, p0 in scored:
        if all(np.max(np.abs(p0[4:] - q[4:]) / span, initial=0.0) > 0.15 for q in seeds):
            seeds.append(p0)
        if len(seeds) >= _POLISH:
            break
    found: List[Tuple[float, np.ndarray]] = []
    for p0 in seeds:
        p = _polish(cls, p0, rays, pixels, band_px, max_iter)
        if p is None:
            continue
        uv, valid = cls.from_params(p).project(rays)
        err = np.full(len(rays), np.inf)
        ok = np.asarray(valid, bool).ravel()
        err[ok] = np.linalg.norm(np.asarray(uv, float)[ok] - pixels[ok], axis=1)
        found.append((_trimmed_cost(err, band_px), p))
    found.sort(key=lambda item: item[0])
    out: List[Tuple[float, np.ndarray]] = []
    for cost, p in found:
        if all(np.max(np.abs(p[4:] - q[4:]) / span, initial=0.0) > 0.05 for _, q in out):
            out.append((cost, p))
    return [(cls.from_params(p), float(np.sqrt(c))) for c, p in out[:max_basins]]


def calibrate_from_views(cls: Type, object_points: Sequence[np.ndarray],
                         pixels: Sequence[np.ndarray], width: int, height: int,
                         *, principal_point: Optional[Tuple[float, float]] = None,
                         loss: str = "soft_l1", f_scale: float = 1.0, max_nfev: int = 150,
                         pixel_covariances: Optional[Sequence] = None,
                         band_px: float = 3.0) -> Optional[Dict]:
    """Calibrate one camera from views of accurately known targets, using no prior intrinsics.

    ``object_points[i]`` must be known exactly up to that view's pose. A single printed board is;
    a fused multi-board target is not until its inter-board poses have been estimated, so pass
    each board as its own view. Each lens basin from :func:`fit_model_candidates` gets a full
    refinement with free poses, and the lowest truncated pixel cost wins.

    Returns the :func:`~ds_msp.calib.bundle.calibrate` result of the winner, extended with
    ``view_indices`` (the views that were used) and ``basins`` (candidates refined), or ``None``
    when no consistent lens exists, so the caller can fall back to another seed.
    """
    cx, cy = principal_point if principal_point is not None else principal_point_guess(width, height)
    seed = radial_seed(object_points, pixels, (cx, cy))
    if seed is None:
        return None
    candidates = fit_model_candidates(cls, seed.rays, seed.pixels, cx, cy, band_px=band_px)
    if not candidates:
        return None
    anchor = seed.paraxial_focal
    used = [v.index for v in seed.views]
    X = [np.asarray(object_points[i], float) for i in used]
    uv = [np.asarray(pixels[i], float) for i in used]
    vis = [np.asarray(v.inliers, bool) for v in seed.views]
    cov = None if pixel_covariances is None else [pixel_covariances[i] for i in used]
    best, best_cost = None, np.inf
    for model, _ in candidates:
        res = calibrate(model, X, uv, vis, loss=loss, f_scale=f_scale, max_nfev=max_nfev,
                        multi_start=False, pixel_covariances=cov)
        # A rough lens can seed a planar view on its mirror branch, and a redescending or bounded
        # kernel then gives that whole view little pull. Seeding the poses again from the refined
        # lens picks the right branch; keep the pass only if it lowers the truncated cost.
        cost = _solution_cost(res, X, uv, vis, band_px)
        again = calibrate(res["model"], X, uv, vis, loss=loss, f_scale=f_scale, max_nfev=max_nfev,
                          multi_start=False, pixel_covariances=cov)
        again_cost = _solution_cost(again, X, uv, vis, band_px)
        if again_cost < cost:
            res, cost = again, again_cost
        # A refine can trade focal for distortion and absorb blunders, leaving a low residual at a
        # tiny focal. The seed's focal comes from a linear solve on rejected data, so it anchors
        # the basin; the band is wide because a seed is only good to tens of percent.
        if not 0.5 * anchor < _axial_focal(res["model"]) < 2.0 * anchor:
            continue
        if cost < best_cost:
            best, best_cost = res, cost
    if best is None:
        return None
    return dict(best, view_indices=used, basins=len(candidates), trimmed_rms_px=float(np.sqrt(best_cost)))


def _axial_focal(model) -> float:
    """Pixels per radian on the optical axis, whatever the projection model."""
    theta = 1e-4
    uv, _ = model.project(np.array([[np.sin(theta), 0.0, np.cos(theta)], [0.0, 0.0, 1.0]]))
    uv = np.asarray(uv, float)
    return float(abs(uv[0, 0] - uv[1, 0]) / theta)


def _solution_cost(res: Dict, X, uv, vis, band_px: float) -> float:
    """Truncated squared pixel error of a fit, counting points that left the valid domain."""
    model, errs = res["model"], []
    for (rvec, t), Xi, uvi, mask in zip(res["poses"], X, uv, vis):
        R = cv2.Rodrigues(np.asarray(rvec, float))[0]
        pred, ok = model.project(Xi[mask] @ R.T + np.asarray(t, float))
        e = np.full(int(mask.sum()), np.inf)
        ok = np.asarray(ok, bool).ravel()
        e[ok] = np.linalg.norm(np.asarray(pred, float)[ok] - uvi[mask][ok], axis=1)
        errs.append(e)
    return _trimmed_cost(np.concatenate(errs), band_px) if errs else float("inf")


def principal_point_guess(width: int, height: int) -> Tuple[float, float]:
    """Image centre in the pixel-centre convention. Lens centring errors are refined later."""
    return (width - 1) / 2.0, (height - 1) / 2.0
