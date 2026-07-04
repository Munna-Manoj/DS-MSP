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
    result is an **angle in radians**, not the unitless algebraic residual of
    :func:`~ds_msp.mvg.epipolar_residual`. This is the FOV-independent scoring function
    :func:`ransac_relative_pose` thresholds against — a fixed radian cutoff means the same
    angular tolerance at the image centre and at the rim of a wide-FOV lens, unlike a pixel
    threshold.

    Parameters
    ----------
    E : (3, 3) ndarray
        Essential matrix candidate.
    f1, f2 : (N, 3) ndarray
        Unit (or non-unit; renormalized internally) bearing vectors in camera 1 and camera 2.

    Returns
    -------
    (N,) ndarray
        Non-negative Sampson angle (radians) per correspondence; small for a correct ``E`` and
        a genuine correspondence, large (up to ``~pi``) for a mismatch.
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

    Repeatedly samples 8 correspondences, fits :func:`~ds_msp.mvg.essential_from_rays`, and
    scores every correspondence with :func:`sampson_residual`; keeps the largest inlier
    consensus and (optionally) re-fits on it. The iteration budget is adaptive: once the best
    inlier fraction seen implies an all-inlier 8-sample has probably already been drawn (at
    the given ``confidence``), remaining iterations are skipped.

    Parameters
    ----------
    f1, f2 : (N, 3) ndarray
        Unit (or non-unit; renormalized internally) bearing vectors in camera 1 and camera 2,
        ``N >= 8``.
    threshold : float, default 0.005
        Inlier cutoff on the Sampson **angle** (radians); ``0.005 rad ~ 0.3 deg``.
    max_iters : int, default 1000
        Maximum RANSAC iterations (upper bound; adaptive stopping usually exits sooner).
    confidence : float, default 0.999
        Target probability of having drawn at least one all-inlier 8-sample; controls the
        adaptive early stop.
    seed : int, default 0
        Seed for the internal ``numpy.random.default_rng`` (deterministic sampling).
    refine : bool, default True
        Re-fit the essential matrix on all inliers, with spherical whitening
        (``essential_from_rays(..., normalize=True)``), before the final pose recovery.

    Returns
    -------
    R : (3, 3) ndarray
        Rotation mapping camera 1 to camera 2 (``X2 = R @ X1 + t``), ``det(R) = +1``.
    t : (3,) ndarray
        Unit-length translation direction.
    inliers : (N,) bool ndarray
        Consensus inlier mask over the input correspondences.

    Raises
    ------
    ValueError
        If fewer than 8 correspondences are given.
    RuntimeError
        If no 8-point sample ever reaches an 8-correspondence consensus (degenerate or
        all-outlier data).

    Examples
    --------
    Recovering rotation to ``~0.11 deg`` from 30%-corrupted ray matches (full walkthrough with
    the naive-vs-RANSAC comparison table: [Chapter 8, §5](../learn/08_two_view_geometry_on_rays.md#5-make-it-robust-ransac-against-wrong-matches)):

    >>> import numpy as np
    >>> from ds_msp.mvg import ransac_relative_pose
    >>> def rot_err_deg(A, B):
    ...     return np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1)))
    >>> rng = np.random.default_rng(3)
    >>> R_true = np.eye(3)  # no rotation, for a self-contained deterministic example
    >>> t_true = np.array([0.1, 0.0, 0.0])
    >>> X1 = np.column_stack([rng.uniform(-2, 2, 120), rng.uniform(-2, 2, 120),
    ...                       rng.uniform(2, 8, 120)])
    >>> X2 = (R_true @ X1.T).T + t_true
    >>> f1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
    >>> f2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
    >>> rng2 = np.random.default_rng(4)
    >>> outlier = rng2.random(120) < 0.30
    >>> f2[outlier] = rng2.standard_normal((int(outlier.sum()), 3))
    >>> f2 /= np.linalg.norm(f2, axis=1, keepdims=True)
    >>> R, t, inliers = ransac_relative_pose(f1, f2, threshold=0.005, seed=0)
    >>> bool(rot_err_deg(R_true, R) < 1.0)
    True
    >>> int(inliers.sum()) >= 80
    True
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
