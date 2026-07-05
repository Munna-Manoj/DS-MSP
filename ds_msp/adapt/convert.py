"""
Camera model conversion ("adapter").

Converts an already-calibrated source model to a target model **without images or
recalibration**: sample pixels -> unproject with the source -> seed the target ->
refine with Levenberg-Marquardt using each model's **analytic** param Jacobian
(no autodiff). Mirrors the fisheye-calib-adapter pipeline in pure Python.

Global optimum, any scenario
----------------------------
A single linear seed plus one local refine can stall in a poor basin when the
target's shape parameters are far from their seed (e.g. a strong-``xi`` Double
Sphere, a DS+ whose ``lambda1``/``lambda2`` are far from their neutral seed, or a
polynomial OCam whose higher coefficients matter). To find the *global* optimum
regardless of the source, the refine runs as a **multi-start**: the linear seed,
a **deterministic sweep** that walks each finitely-bounded shape parameter across
its range, and ``n_restarts`` additional random dispersed seeds — keeping the
lowest-cost fit. The deterministic sweep makes the result reproducible and removes
the random-restart lottery that previously let some targets settle in a wrong
basin and fail even to self-convert. Only the *shape* parameters are seeded; the
intrinsics (fx, fy, cx, cy) start from their closed-form linear seed.

Decoupled by dependency injection: ``convert`` takes the source instance and the
target *class*, so this module imports no concrete model — only the contract and
SciPy. Works with any model satisfying ``CameraModel``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Type

import numpy as np
from scipy.optimize import least_squares

from ..core.contracts import CameraModel
from .evaluate import reprojection_report
from .sampling import sample_image_grid

# Intrinsics are nailed by the linear seed; only these "shape" parameters are
# dispersed across restarts.
_INTRINSIC_NAMES = frozenset({"fx", "fy", "cx", "cy"})


# Fractions of each finite bound span used for the deterministic shape sweep.
_SWEEP_FRACS = (0.2, 0.5, 0.8)


def _shape_seeds(target_cls: Type[CameraModel], base: np.ndarray,
                 n_restarts: int, rng: np.random.Generator) -> List[np.ndarray]:
    """Linear seed + a deterministic shape sweep + ``n_restarts`` random dispersed seeds.

    The deterministic sweep moves *one* finitely-bounded shape parameter at a time to a few
    fractions of its range (others held at the linear seed). This guarantees a shape optimum
    that sits *far* from the linear seed is reached without relying on the luck of the random
    restarts — e.g. a DS+ whose ``lambda1``/``lambda2`` are far from their neutral seed would
    otherwise self-convert into a wrong basin. The random restarts are retained on top for
    joint (multi-parameter) basins; they keep the previous behaviour for ``n_restarts>0``.
    """
    seeds = [base.copy()]
    lb, ub = target_cls.param_bounds()
    shape_idx = [i for i, n in enumerate(target_cls.param_names)
                 if n not in _INTRINSIC_NAMES]
    if not shape_idx:
        return seeds
    # Deterministic per-parameter sweep across each finite bound range.
    for i in shape_idx:
        lo, hi = lb[i], ub[i]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        for frac in _SWEEP_FRACS:
            p = base.copy()
            p[i] = lo + frac * (hi - lo)
            seeds.append(np.clip(p, lb, ub))
    # Random dispersed restarts (joint perturbations), as before.
    for _ in range(max(0, n_restarts)):
        p = base.copy()
        for i in shape_idx:
            lo, hi = lb[i], ub[i]
            s = base[i]
            # Sample within a window around the seed, clipped to bounds. The
            # window is the smaller of half the (finite) bound span or a scale
            # tied to the seed magnitude, so tightly-bounded params (alpha, xi)
            # explore their whole range while large ones (OCam a-coeffs) stay local.
            span = 0.5 * (hi - lo)
            w = min(span, max(abs(s), 1.0))
            p[i] = rng.uniform(max(lo, s - w), min(hi, s + w))
        seeds.append(np.clip(p, lb, ub))
    return seeds


def convert(source: CameraModel, target_cls: Type[CameraModel], *,
            width: int, height: int, n_samples: int = 500,
            max_fov_deg: Optional[float] = None,
            n_restarts: int = 4, seed: int = 0,
            verbose: bool = False) -> Tuple[CameraModel, dict]:
    """Fit ``target_cls`` parameters to reproduce ``source`` over the image.

    Parameters
    ----------
    source : CameraModel
        The calibrated source model.
    target_cls : type[CameraModel]
        The model class to convert into.
    width, height : int
        Image size used for sampling and reporting.
    n_samples : int
        Approximate number of grid samples used for the fit.
    max_fov_deg : float, optional
        Restrict the fitted FOV (full angle). Useful when the target is narrower
        than the source (e.g. converting a >180 deg fisheye into a pinhole-like
        model) so the fit is not dragged by unrepresentable rays.
    n_restarts : int
        Number of *random* dispersed shape-parameter restarts, in addition to the
        linear seed and the always-on deterministic shape sweep. ``0`` keeps just
        the linear seed plus the deterministic sweep (already multi-start and
        reproducible); higher values add random joint-perturbation restarts.
    seed : int
        RNG seed for the restart dispersion, so conversions are reproducible.

    Returns
    -------
    (target, report) : the fitted model and a quality report (see
    ``reprojection_report``). ``report["n_restarts"]`` records the restart count.
    """
    # 1. sample pixels -> source bearing rays (forward hemisphere only)
    pixels = sample_image_grid(width, height, n_samples)
    rays, valid = source.unproject(pixels)
    keep = valid & (rays[:, 2] > 1e-6)
    if max_fov_deg is not None:
        ang = np.degrees(np.arccos(np.clip(rays[:, 2], -1.0, 1.0)))
        keep &= ang <= (max_fov_deg / 2.0)
    rays, pixels = rays[keep], pixels[keep]
    if len(rays) < len(target_cls.param_names):
        raise ValueError(
            f"Too few forward correspondences ({len(rays)}) to fit "
            f"{target_cls.name}; increase n_samples or max_fov_deg."
        )

    # 2. linear seed: inherit intrinsics from the source, SVD-seed distortion
    target = target_cls.from_params(np.zeros(len(target_cls.param_names)))
    target.initialize_from_correspondences(source.K, rays, pixels)
    lb, ub = target_cls.param_bounds()
    base = np.clip(target.params, lb, ub)

    # 3. nonlinear refine, NaN-safe so out-of-domain rays never poison the solve.
    def residual(p):
        """``scipy.optimize.least_squares`` residual vector for ``p``: flattened pixel
        error ``(target.project(rays) - pixels).ravel()``, shape ``(2*len(rays),)``. Rays
        that fall outside ``target``'s projectable domain produce a non-finite pixel,
        replaced with a large constant (``1e6``) so the solver is pushed away from
        out-of-domain parameter regions instead of being poisoned by NaN."""
        uv, _ = target_cls.from_params(p).project(rays)
        r = uv - pixels
        return np.where(np.isfinite(r), r, 1e6).ravel()

    def jac(p):
        """Analytic Jacobian of :func:`residual` w.r.t. ``p``, shape
        ``(2*len(rays), len(p))``, from ``target_cls.project_jacobian``. Non-finite
        entries (same out-of-domain rays as :func:`residual`) are zeroed rather than left
        as NaN, so they contribute no gradient instead of poisoning the normal equations."""
        _, _, j_param, _ = target_cls.from_params(p).project_jacobian(rays)
        j = j_param.reshape(-1, p.size)
        return np.where(np.isfinite(j), j, 0.0)

    # 3b. multi-start: refine from the linear seed plus dispersed shape seeds,
    #     keep the lowest-cost fit (the global optimum in practice).
    rng = np.random.default_rng(seed)
    best_x: np.ndarray = base
    best_cost = np.inf
    best_success = False
    for x0 in _shape_seeds(target_cls, base, n_restarts, rng):
        res = least_squares(residual, x0, jac=jac, bounds=(lb, ub),
                            method="trf", x_scale="jac",
                            verbose=2 if verbose else 0)
        if res.cost < best_cost:
            best_cost, best_x, best_success = res.cost, res.x, res.success
    target = target_cls.from_params(best_x)

    # 4. evaluate over the (possibly FOV-restricted) image region
    report = reprojection_report(source, target, width, height,
                                 max_fov_deg=max_fov_deg, gt_params=None)
    report["converged"] = bool(best_success)
    report["n_restarts"] = int(n_restarts)
    report["source_model"] = source.name
    report["target_model"] = target_cls.name
    return target, report
