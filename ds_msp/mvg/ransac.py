"""Robust relative pose on bearing vectors — RANSAC with an angular (Sampson) residual.

The eight-point estimator (``two_view.essential_from_rays``) is least-squares: a few mismatched
rays wreck it. This wraps it in RANSAC, scoring with a **Sampson distance on the sphere** that is
an angle in radians (so the inlier threshold is FOV-independent — the right currency for a fisheye,
unlike a pixel threshold).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .two_view import _as_rays, essential_from_rays, recover_pose


def sampson_residual(E: np.ndarray, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """Symmetric angular epipolar distance per correspondence (radians, small-angle).

    First-order (Sampson) approximation of how far each ray pair is from satisfying
    ``f2ᵀ E f1 = 0``, with the gradient taken in the **tangent planes** of the unit rays so the
    result is an angle, not an algebraic residual.
    """
    E = np.asarray(E, float)
    f1 = _as_rays(f1)
    f2 = _as_rays(f2)
    num = np.einsum("ij,jk,ik->i", f2, E, f1)          # f2ᵀ E f1
    Ef1 = f1 @ E.T                                      # epipolar normal in cam 2, (N,3)
    Etf2 = f2 @ E                                       # epipolar normal in cam 1, (N,3)
    g2 = Ef1 - np.einsum("ij,ij->i", Ef1, f2)[:, None] * f2     # tangent component at f2
    g1 = Etf2 - np.einsum("ij,ij->i", Etf2, f1)[:, None] * f1   # tangent component at f1
    denom = np.sqrt(np.sum(g1 * g1, axis=1) + np.sum(g2 * g2, axis=1))
    return np.abs(num) / np.maximum(denom, 1e-12)


def ransac_relative_pose(
    f1: np.ndarray, f2: np.ndarray, *,
    threshold: float = 0.005, max_iters: int = 1000, confidence: float = 0.999,
    seed: int = 0, refine: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust ``(R, t)`` from ray correspondences via RANSAC over the eight-point.

    Parameters
    ----------
    threshold : float
        Inlier cutoff on the Sampson **angle** (radians); ~0.005 ≈ 0.3°.
    max_iters, confidence :
        RANSAC budget; iterations are cut short once `confidence` that an all-inlier sample was
        drawn is reached (adaptive).
    refine : bool
        Re-fit the essential matrix on all inliers (with spherical normalization) before pose
        recovery.

    Returns
    -------
    (R, t, inliers) : the pose and a boolean inlier mask over the input correspondences.
    """
    f1 = _as_rays(f1)
    f2 = _as_rays(f2)
    n = f1.shape[0]
    if n < 8:
        raise ValueError(f"need ≥8 correspondences, got {n}")
    rng = np.random.default_rng(seed)

    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    iters = max_iters
    it = 0
    while it < iters:
        it += 1
        idx = rng.choice(n, 8, replace=False)
        try:
            E = essential_from_rays(f1[idx], f2[idx])
        except (np.linalg.LinAlgError, ValueError):
            continue
        inl = sampson_residual(E, f1, f2) < threshold
        c = int(inl.sum())
        if c > best_count:
            best_count, best_inliers = c, inl
            # adaptive stop: enough iterations to have hit an all-inlier sample
            w = max(best_count / n, 1e-6)
            denom = np.log(max(1.0 - w ** 8, 1e-12))
            if denom < 0:
                iters = min(max_iters, int(np.log(1.0 - confidence) / denom) + 1)

    if best_count < 8:
        raise RuntimeError("RANSAC failed to find an 8-point consensus; check threshold/data")

    fin1, fin2 = f1[best_inliers], f2[best_inliers]
    E = essential_from_rays(fin1, fin2, normalize=refine)
    R, t, _ = recover_pose(fin1, fin2, E)
    return R, t, best_inliers
