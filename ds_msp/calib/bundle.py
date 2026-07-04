# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025-2026 Munna-Manoj. Robust auto-initialized bundle-adjustment
# calibration — part of the DS-MSP robust calibration engine
# (https://github.com/Munna-Manoj/DS-MSP). NONCOMMERCIAL use only, with attribution —
# see LICENSE-NONCOMMERCIAL.txt and LICENSING.md. The rest of DS-MSP is MIT.
"""
Generic, robustly-initialized bundle-adjustment calibration for ANY camera model.

Given checkerboard correspondences and an initial model (for the model *type* + a rough
intrinsics seed), jointly refines intrinsics and per-image extrinsics by Levenberg-
Marquardt using the model's **analytic** projection Jacobian (no autodiff). Works for any
``CameraModel`` (DS / UCM / EUCM / KB / RadTan / OCam / DS⁺ / EUCM⁺) with **no
Kannala-Brandt dependency** in the init path.

Robust by default (FR-CALIB-001, NFR-NUM-006):
  * **Pose seeding** branches on target geometry and resolves the planar two-fold (mirror)
    ambiguity by full-board reprojection, so near-fronto-parallel views cannot seed the
    wrong basin and inflate mean / p95 error (the failure mode the median hides).
  * The BA loop runs a redescending robust kernel with **MAD auto-scale** (50%-breakdown,
    re-estimated each iteration), so a few mis-detected corners are down-weighted, never
    dropped. A **graduated-non-convexity** anneal is available (``gnc=True``) for pathological
    bad-init basins, but is **off by default**: with the two-fold seeding above the iterate
    already starts in the right basin, and a wide early kernel would only re-admit the gross
    outliers the auto-scale kernel is there to reject (measured: it degrades focal accuracy).
  * Reports **median, mean, p95 and max** reprojection error — not just RMS — so a fit that
    is good "on average" but has a flipped view is visible, not hidden.

The model-agnostic BA loop lives in :mod:`ds_msp.geometry.calibrate_core`; planar/robust
pose seeding lives in :mod:`ds_msp.geometry.resection`. The model type comes from the
injected ``init_model`` (no concrete-model import). cv2 is used only to emit ``(rvec, tvec)``.
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from ..core.contracts import CameraModel
from ..geometry.calibrate_core import bundle_adjust

#: Map the historical SciPy ``loss`` names to the in-house IRLS kernels (back-compat).
_LOSS_TO_KERNEL = {
    "linear": "none", "huber": "huber", "cauchy": "cauchy",
    "soft_l1": "pseudo_huber",
}


def _coplanar(X: np.ndarray, tol: float = 1e-3) -> bool:
    """True if the board points lie on a plane (single ChArUco board ⇒ coplanar)."""
    Xc = np.asarray(X, float) - np.asarray(X, float).mean(0)
    if len(Xc) < 4:
        return True
    s = np.linalg.svd(Xc, compute_uv=False)
    return s[-1] <= tol * max(s[0], 1e-12)


def _pose_candidates(Xv: np.ndarray, pn: np.ndarray):
    """Candidate object->camera ``(rvec, tvec)`` poses on the normalized plane (``K = I``).

    A **planar** board has a two-fold (mirror) pose ambiguity that near-fronto-parallel views
    cannot resolve locally; IPPE (Collins & Bartoli) returns *both* solutions, so the caller
    can pick by reprojection rather than gamble on one and seed the wrong basin. A
    non-coplanar (fused) target uses the iterative DLT (single solution).
    """
    K = np.eye(3)
    if _coplanar(Xv):
        try:
            n, rvecs, tvecs, _ = cv2.solvePnPGeneric(Xv, pn, K, None, flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            n = 0
        if n:
            return [(rvecs[i].ravel(), tvecs[i].ravel()) for i in range(n)]
    # Fallback (non-coplanar, or coplanar where IPPE could not solve). The iterative solver
    # raises on a sparse/degenerate view ("DLT needs >= 6 points"); treat that as "no
    # candidate" so the caller seeds a default pose instead of crashing the whole calibration.
    try:
        ok, rv, tv = cv2.solvePnP(Xv, pn, K, None)
    except cv2.error:
        return []
    return [(rv.ravel(), tv.ravel())] if ok else []


def _normalized_reproj_err(Xv: np.ndarray, pn: np.ndarray, rvec, tvec) -> float:
    """Mean reprojection error of a pose on the normalized plane (∞ if too few in front)."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    Xc = (R @ Xv.T).T + np.asarray(tvec, float)
    z = Xc[:, 2]
    front = z > 1e-9
    if front.sum() < 0.5 * len(Xv):
        return float("inf")
    proj = Xc[front, :2] / z[front, None]
    return float(np.mean(np.linalg.norm(proj - pn[front], axis=1)))


def _seed_poses(init_model, X_world_list, keypoints_list, visibility_list):
    """Per-view object->camera (rvec, tvec) seeds, two-fold-disambiguated.

    Unprojects pixels to normalized rays with ``init_model`` (only the ray *direction*
    matters for seeding, so a rough intrinsics seed is fine), enumerates the planar two-fold
    pose candidates, and keeps the one with the lowest reprojection on the full board — so a
    near-fronto-parallel view cannot seed the mirror basin and inflate mean / p95. The global
    BA then resolves any residual ambiguity via cross-view consistency + the GNC anneal.
    """
    rvecs, tvecs = [], []
    for Xw, uv, vis in zip(X_world_list, keypoints_list, visibility_list):
        vis = np.asarray(vis, bool)
        Xv = Xw[vis].astype(np.float64)
        uvv = uv[vis].astype(np.float64)
        rays, vr = init_model.unproject(uvv)
        use = vr & (rays[:, 2] > 1e-6)
        chosen = None
        if use.sum() >= 4:
            Xu = Xv[use]
            pn = (rays[use, :2] / rays[use, 2:3]).astype(np.float64)
            best = float("inf")
            for rv, tv in _pose_candidates(Xu, pn):
                e = _normalized_reproj_err(Xu, pn, rv, tv)
                if e < best:
                    best, chosen = e, (np.asarray(rv, float), np.asarray(tv, float))
        if chosen is None:
            rvecs.append(np.zeros(3))
            tvecs.append(np.array([0.0, 0.0, 1.5]))
        else:
            rvecs.append(chosen[0])
            tvecs.append(chosen[1])
    return rvecs, tvecs


def _reproj_errors(model, poses, X_world_list, keypoints_list, masks) -> np.ndarray:
    """Per-observation pixel error magnitudes over valid, visible corners (1-D array)."""
    errs = []
    for (rvec, tv), Xw, uv, vis in zip(poses, X_world_list, keypoints_list, masks):
        R, _ = cv2.Rodrigues(np.asarray(rvec, float))
        uvp, valid = model.project((R @ Xw.T).T + tv)
        mm = vis & valid
        d = uvp[mm] - uv[mm]
        errs.append(np.linalg.norm(d, axis=1))
    return np.concatenate(errs) if errs else np.empty(0)


def _shape_seeds(cls, base: np.ndarray, n_restarts: int, seed: int, *, force: bool = True):
    """Multi-start seeds: the base intrinsics with the **shape** parameters (index ≥ 4 by the
    ``[fx, fy, cx, cy, …]`` convention) dispersed across their bounds. Focal/principal point are
    held at the base seed (they are well-constrained by the data); the shape parameters own the
    basins a single local refine can fall into (the DS ``ξ`` fold, EUCM ``α→1``, and — corrected
    2026-07-02 — a *plain distortion tail on under-constrained data too*).

    History (why this always disperses, no model-class shortcut): an earlier version of this
    function skipped dispersion for models with no *documented* degenerate basin (RadTan/KB/
    OCam — measured "identical median/RMS" on a well-conditioned real 2-camera / 99-frame
    dataset). At scale on less-constrained data (100 synthetic cameras, ~20 frames/camera) that
    turned out to be **wrong**: forcing the single base seed left 56/100 RadTan cameras with
    roughly double the reprojection error and zero improved — not a degenerate-basin collapse,
    just an ordinary local optimum a single local refine can land in when the data is thinner.
    Restoring full multi-start for every model (verified: matches the true pre-change baseline
    exactly, mean/median/max all within 1e-4px, 0/100 cameras degraded) fixed it.
    ``force`` is kept (default ``True``) as the call-site-level override this history earned;
    a future, *data-conditioned* (not model-class-conditioned) skip would need the same rigor
    (true-baseline comparison at realistic scale, not just a small/well-conditioned case) before
    shipping."""
    lb, ub = cls.param_bounds()
    seeds = [base.copy()]
    if n_restarts <= 0 or len(base) <= 4:
        return seeds
    if not force:
        return seeds
    rng = np.random.default_rng(seed)
    for _ in range(n_restarts):
        p = base.copy()
        for j in range(4, len(base)):
            lo, hi = lb[j], ub[j]
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                p[j] = rng.uniform(lo + 0.15 * (hi - lo), hi - 0.15 * (hi - lo))
        seeds.append(np.clip(p, lb, ub))
    return seeds


def _seed_and_fit(cls, params0, X_world_list, keypoints_list, visibility_list,
                  *, kernel, scale, gnc_start, gnc_iters, max_iter):
    """Seed per-view poses for ``params0`` then run the BA loop. Returns ``(params, Rb, t, out)``."""
    seed_model = cls.from_params(params0)
    rvecs, tvecs = _seed_poses(seed_model, X_world_list, keypoints_list, visibility_list)
    R0 = np.stack([cv2.Rodrigues(np.asarray(r, float))[0] for r in rvecs])
    t0 = np.stack([np.asarray(t, float) for t in tvecs])
    return bundle_adjust(cls, params0, R0, t0,
                         X_world_list, keypoints_list, visibility_list,
                         kernel=kernel, scale=scale, gnc_start=gnc_start,
                         gnc_iters=gnc_iters, max_iter=max_iter)


def calibrate(init_model: CameraModel,
              X_world_list: List[np.ndarray],
              keypoints_list: List[np.ndarray],
              visibility_list: List[np.ndarray],
              *, max_nfev: int = 200, verbose: int = 0,
              robust: str = "cauchy", robust_scale: "float | str" = "auto",
              gnc: bool = False, multi_start: bool = True, n_restarts: int = 4,
              seed: int = 0, force_multistart: bool = True,
              loss: str | None = None, f_scale: float | None = None) -> Dict:
    """Calibrate any model from checkerboard correspondences — robust by default.

    Parameters
    ----------
    robust : str
        Redescending IRLS kernel applied in the BA loop: ``"cauchy"`` (default),
        ``"huber"``, ``"geman_mcclure"``, ``"barron"``, or ``"none"`` for plain L2.
        A robust kernel keeps *every* corner but **down-weights** large residuals instead of
        letting one mis-localized corner drag the L2 fit.
    robust_scale : float | str
        Inlier scale (px) where down-weighting begins, or ``"auto"`` (default) to
        re-estimate it each iteration from the residual MAD (50%-breakdown).
    gnc : bool
        Run a graduated-non-convexity anneal (default ``False``): start with a wide kernel
        that dissolves spurious minima, then sharpen. Useful only for pathological bad-init
        basins; with two-fold seeding it is unnecessary and re-admits gross outliers, so it is
        off by default (enable it for a known-hard initialization).
    multi_start : bool
        Model-aware multi-start auto-init (default ``True``): screen the base seed plus
        ``n_restarts`` seeds with the **shape** parameters dispersed across their bounds, keep
        the one with the lowest robust (median) reprojection, then refine it fully. This is what
        makes a *poor* ``init_model`` (only the type + a rough focal) converge — it rescues
        wrong-basin shape seeds (the DS ``ξ`` fold, etc.) and is a no-op when the base seed is
        already good. Determinism is preserved via ``seed``. Applies to **every** model,
        including RadTan/KB/OCam: an earlier version of this code skipped dispersion for models
        with no *documented* degenerate basin on the theory that a plain distortion tail is
        well-conditioned — true on rich, real data, measurably **false** at scale on thinner
        synthetic data (56/100 RadTan cameras with ~2x the reprojection error in one real test —
        see :func:`_shape_seeds`'s docstring). Set ``multi_start=False`` only when you have
        independently verified it is safe for your data, not by model class alone.
    n_restarts : int
        Number of dispersed shape seeds for ``multi_start`` (default 4).
    force_multistart : bool
        Kept for API stability (default ``True``, matching the always-disperse default above);
        see :func:`_shape_seeds`.

    Backward compatibility: passing the SciPy-style ``loss`` (``"linear"``/``"huber"``/
    ``"soft_l1"``/``"cauchy"``) and/or ``f_scale`` reproduces the pre-robust-default
    behaviour (fixed scale, GNC off) so existing callers stay stable.

    Returns a dict with the refined ``model``, per-image ``poses`` as ``(rvec, tvec)``,
    ``success``, ``n_obs``, and reprojection statistics over valid observations:
    ``rms_px``, ``mean_px``, ``median_px``, ``p95_px``, ``max_px`` (all in pixels and
    independent of the kernel, so they stay comparable across configurations).
    """
    # Back-compat: an explicit loss/f_scale pins the legacy fixed-scale, no-GNC path.
    legacy = loss is not None or f_scale is not None
    if loss is not None:
        robust = _LOSS_TO_KERNEL.get(loss, loss)
    if legacy:
        gnc = False
        robust_scale = f_scale if f_scale is not None else 1.0

    cls = type(init_model)
    n_img = len(X_world_list)
    masks = [np.asarray(v, bool) for v in visibility_list]

    kernel = robust
    if kernel == "none":
        scale_arg, gnc_start, gnc_iters = 1.0, 0.0, 0
    else:
        scale_arg = robust_scale
        # Auto-scale GNC: start the kernel ~3x wide and anneal to the calibrated scale over
        # ~20% of the iteration budget (a few annealing steps, then the sharp robust fit).
        gnc_start, gnc_iters = (3.0, max(8, max_nfev // 5)) if gnc else (0.0, 0)

    fit_kw = dict(kernel=kernel, scale=scale_arg, gnc_start=gnc_start, gnc_iters=gnc_iters)

    # Model-aware multi-start auto-init: cheaply screen the base seed + shape-dispersed seeds
    # (a short BA each), score by the *robust* (median) reprojection so a wrong shape basin is
    # rejected, then run the full refine from the winning seed. With a single seed this reduces
    # to one full fit, so non-multistart behaviour is unchanged.
    seeds = _shape_seeds(cls, init_model.params, n_restarts, seed,
                         force=force_multistart) if multi_start else [
        np.asarray(init_model.params, float)]
    if len(seeds) > 1:
        screen_iter = max(15, max_nfev // 5)
        best_seed, best_score = seeds[0], float("inf")
        for p0 in seeds:
            pr, Rr, tr, _ = _seed_and_fit(cls, p0, X_world_list, keypoints_list,
                                          visibility_list, max_iter=screen_iter, **fit_kw)
            mr = cls.from_params(pr)
            poses_r = [(cv2.Rodrigues(Rr[i])[0].ravel(), tr[i]) for i in range(n_img)]
            er = _reproj_errors(mr, poses_r, X_world_list, keypoints_list, masks)
            score = float(np.median(er)) if er.size else float("inf")
            if score < best_score:
                best_seed, best_score = p0, score
    else:
        best_seed = seeds[0]

    params, Rb, t, out = _seed_and_fit(cls, best_seed, X_world_list, keypoints_list,
                                       visibility_list, max_iter=max_nfev, **fit_kw)
    model = cls.from_params(params)

    # Absolute (rvec, tvec) per image so downstream code (cv2.Rodrigues) is unaffected.
    poses = [(cv2.Rodrigues(Rb[i])[0].ravel(), np.asarray(t[i], float)) for i in range(n_img)]

    # True reprojection stats over valid observations — computed directly so they mean the
    # same thing under any kernel. mean/p95 expose flipped views that the median hides.
    errs = _reproj_errors(model, poses, X_world_list, keypoints_list, masks)
    if errs.size:
        stats = {
            "rms_px": float(np.sqrt(np.mean(errs ** 2))),
            "mean_px": float(np.mean(errs)),
            "median_px": float(np.median(errs)),
            "p95_px": float(np.percentile(errs, 95)),
            "max_px": float(np.max(errs)),
        }
    else:
        nan = float("nan")
        stats = {"rms_px": nan, "mean_px": nan, "median_px": nan, "p95_px": nan, "max_px": nan}

    return {"model": model, "poses": poses, "success": bool(out.success),
            "n_obs": int(errs.size), **stats}
