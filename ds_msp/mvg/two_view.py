"""Two-view geometry on unit bearing vectors — essential matrix, pose, triangulation.

Conventions
-----------
- ``f1, f2`` are **unit bearing vectors** (rays), shape ``(N, 3)``, one correspondence per
  row: ``f1[i]`` in camera 1, ``f2[i]`` in camera 2, looking at the same 3D point.
- Relative pose ``(R, t)`` maps a point from camera 1 to camera 2: ``X2 = R @ X1 + t``.
  ``t`` is recovered only up to scale, so it is returned **unit-length**; its sign is fixed
  by cheirality (the point must lie in front of both cameras).
- "In front" for a wide-FOV camera means **positive depth along the bearing vector**
  (``λ > 0``), not ``z > 0`` — a ray past 90° is still a valid observation.

The calibrated epipolar constraint is ``f2ᵀ E f1 = 0`` with ``E = [t]_× R`` (rank 2). It
holds for bearing vectors of any central camera, which is why nothing here is pinhole-specific.

References
----------
Hartley, R. and Zisserman, A., *Multiple View Geometry in Computer Vision*, 2nd ed.,
Cambridge University Press, 2004 — the eight-point algorithm (§11.3) applied here to
bearing vectors instead of pixels, and the four-fold ``E`` decomposition (§9.6.2, Result
9.19) implemented in `decompose_essential`.

Derivations, proofs, and numerical-stability notes: see the explanation page
[Two-view geometry](../explain/two_view_geometry.md) and the tutorial
[Chapter 8 — Two-view geometry on rays](../learn/08_two_view_geometry_on_rays.md).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

_W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _as_rays(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"rays must have shape (N, 3), got {f.shape}")
    n = np.linalg.norm(f, axis=1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("zero-length bearing vector")
    return f / n


def epipolar_residual(E: np.ndarray, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """Algebraic epipolar residual ``f2ᵀ E f1`` for each ray correspondence.

    This is the raw algebraic constraint, not an angle — use :func:`~ds_msp.mvg.sampson_residual`
    for a threshold that is meaningful in radians (e.g. for RANSAC).

    Parameters
    ----------
    E : (3, 3) ndarray
        Essential matrix, e.g. from :func:`essential_from_rays`.
    f1, f2 : (N, 3) ndarray
        Unit (or non-unit; renormalized internally) bearing vectors in camera 1 and camera 2,
        one correspondence per row.

    Returns
    -------
    (N,) ndarray
        ``f2[i] @ E @ f1[i]`` for each row ``i``. Zero to float64 round-off for an exact ``E``
        and noise-free correspondences.

    Examples
    --------
    >>> import numpy as np
    >>> from ds_msp.mvg import essential_from_rays, epipolar_residual
    >>> rng = np.random.default_rng(0)
    >>> X1 = rng.uniform(-2, 2, (10, 3)) + np.array([0, 0, 5])  # points in front of camera 1
    >>> X2 = X1 + np.array([1.0, 0.0, 0.0])                    # camera 2 shifted along +x
    >>> f1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
    >>> f2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
    >>> E = essential_from_rays(f1, f2)
    >>> float(np.abs(epipolar_residual(E, f1, f2)).max()) < 1e-10
    True
    """
    f1 = _as_rays(f1)
    f2 = _as_rays(f2)
    return np.einsum("ij,jk,ik->i", f2, np.asarray(E, float), f1)


def _whiten(f: np.ndarray, reg: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
    """Spherical pre-conditioning: a linear map ``T`` balancing the ray spread.

    Returns ``(f @ Tᵀ, T)`` with ``T = (Cov + εI)^{-1/2}``, ``Cov = (1/N) Σ fᵢ fᵢᵀ`` and
    ``ε = reg·λ_max``. Balancing the bearing-vector covariance better conditions the eight-point
    design matrix when the rays cluster in a narrow cone (the 360-8PA idea, arXiv:2104.10900).

    The ``εI`` regularization is essential and not cosmetic: for a *very* narrow cone ``Cov`` is
    near-singular, and an unregularized ``Cov^{-1/2}`` would amplify the near-degenerate axis by a
    huge factor and make the estimate **worse**. Tying ε to ``λ_max`` caps that amplification, so
    whitening never hurts (it just does less when there's little spread to balance). Does **not**
    re-unitize — the constraint ``(T₂f₂)ᵀ E' (T₁f₁) = 0`` recovers ``E = T₂ᵀ E' T₁``.
    """
    cov = (f.T @ f) / f.shape[0]
    w, V = np.linalg.eigh(cov)
    w = w + reg * w.max()
    T = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    return f @ T.T, T


def essential_from_rays(f1: np.ndarray, f2: np.ndarray, *, normalize: bool = False
                        ) -> np.ndarray:
    """Essential matrix from >=8 ray correspondences (eight-point on bearing vectors).

    Solves ``f2ᵀ E f1 = 0`` in the least-squares (smallest-singular-vector) sense — the
    eight-point algorithm (Hartley & Zisserman, §11.3) applied to unit bearing vectors instead
    of pixels — then projects the raw least-squares solution onto the essential manifold
    (singular values forced to ``(1, 1, 0)``).

    Parameters
    ----------
    f1, f2 : (N, 3) ndarray
        Unit (or non-unit; renormalized internally) bearing vectors in camera 1 and camera 2,
        one correspondence per row, ``N >= 8``.
    normalize : bool, default False
        Apply spherical whitening (see the module-private ``_whiten``) before the solve and
        undo it after. Leave off for clean data — it changes nothing in the noise-free limit
        (verified in [Chapter 8](../learn/08_two_view_geometry_on_rays.md), residual
        ``5.69e-16`` either way). Turn on for noisy / narrow-baseline rays where conditioning
        matters; pixel-domain Hartley normalization does not apply to bearing vectors, so this
        is its spherical analogue.

    Returns
    -------
    (3, 3) ndarray
        Essential matrix ``E`` with singular values ``(1, 1, 0)``, satisfying
        ``f2ᵀ E f1 ≈ 0`` for the input correspondences.

    Raises
    ------
    ValueError
        If fewer than 8 correspondences are given, or if ``f1``/``f2`` are not shape ``(N, 3)``
        or contain a zero-length vector (checked in the shared ``_as_rays`` validator).

    Examples
    --------
    >>> import numpy as np
    >>> from ds_msp.mvg import essential_from_rays, epipolar_residual
    >>> rng = np.random.default_rng(0)
    >>> X1 = rng.uniform(-2, 2, (10, 3)) + np.array([0, 0, 5])
    >>> X2 = X1 + np.array([1.0, 0.0, 0.0])
    >>> f1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
    >>> f2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
    >>> E = essential_from_rays(f1, f2)
    >>> E.shape
    (3, 3)
    >>> np.round(np.linalg.svd(E, compute_uv=False), 6)
    array([1., 1., 0.])
    """
    f1 = _as_rays(f1)
    f2 = _as_rays(f2)
    if f1.shape[0] < 8:
        raise ValueError(f"need ≥8 correspondences, got {f1.shape[0]}")
    g1, g2 = f1, f2
    T1 = T2 = None
    if normalize:
        g1, T1 = _whiten(f1)
        g2, T2 = _whiten(f2)
    # Each row: coefficients of vec(E) (row-major) in g2ᵀ E g1 = 0  →  kron(g2, g1).
    A = g2[:, [0, 0, 0, 1, 1, 1, 2, 2, 2]] * np.tile(g1, 3)
    _, _, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3, 3)
    if normalize:
        E = T2.T @ E @ T1                       # map back to un-whitened coordinates
    # Project onto the essential manifold: equal non-zero singular values, rank 2.
    U, _, Vt2 = np.linalg.svd(E)
    return U @ np.diag([1.0, 1.0, 0.0]) @ Vt2


def decompose_essential(E: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """The four ``(R, t)`` candidates consistent with an essential matrix.

    Every essential matrix factors as ``E = [t]_× R`` for exactly two rotations and two signs
    of ``t`` (Hartley & Zisserman, §9.6.2, Result 9.19) — ambiguous without an extra
    cheirality test, which :func:`recover_pose` applies to pick the one physical solution.

    Parameters
    ----------
    E : (3, 3) ndarray
        Essential matrix, e.g. from :func:`essential_from_rays`. Need not already be exactly
        rank 2 / equal-singular-value — only its left/right singular bases are used.

    Returns
    -------
    list of (ndarray, ndarray)
        The four ``(R, t)`` candidates ``[(R1, t), (R1, -t), (R2, t), (R2, -t)]``. Each ``R``
        is a ``(3, 3)`` rotation matrix (``det = +1``); ``t`` is a ``(3,)`` unit vector
        (translation is recovered only up to scale).

    Examples
    --------
    >>> import numpy as np
    >>> from ds_msp.mvg import essential_from_rays, decompose_essential
    >>> rng = np.random.default_rng(0)
    >>> X1 = rng.uniform(-2, 2, (10, 3)) + np.array([0, 0, 5])
    >>> X2 = X1 + np.array([1.0, 0.0, 0.0])
    >>> f1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
    >>> f2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
    >>> E = essential_from_rays(f1, f2)
    >>> candidates = decompose_essential(E)
    >>> len(candidates)
    4
    """
    U, _, Vt = np.linalg.svd(np.asarray(E, float))
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    R1 = U @ _W @ Vt
    R2 = U @ _W.T @ Vt
    t = U[:, 2]
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def triangulate_rays(f1: np.ndarray, f2: np.ndarray, R: np.ndarray, t: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Midpoint triangulation of ray pairs under pose ``(R, t)``.

    For each correspondence, finds the two points (one on ray 1, one on ray 2, expressed in
    camera-1 frame) that minimize the distance between the rays, and returns their midpoint —
    the closed-form linear method (see e.g. Hartley & Sturm, *Triangulation*, CVIU 1997, §2.1).

    Parameters
    ----------
    f1, f2 : (N, 3) ndarray
        Unit (or non-unit; renormalized internally) bearing vectors in camera 1 and camera 2.
    R, t : (3, 3), (3,) ndarray
        Relative pose mapping a point from camera 1 to camera 2: ``X2 = R @ X1 + t``
        (the convention returned by :func:`recover_pose` / :func:`decompose_essential`).

    Returns
    -------
    X : (N, 3) ndarray
        Triangulated 3D points, in the **camera-1** frame.
    depth1 : (N,) ndarray
        Signed distance of each point along ``f1`` from camera 1's centre.
    depth2 : (N,) ndarray
        Signed distance of each point along ``f2`` from camera 2's centre. A point is in front
        of both cameras iff ``depth1 > 0`` and ``depth2 > 0`` — the ray-cheirality test used by
        :func:`recover_pose` (positive depth *along the bearing ray*, not ``z > 0``: valid for
        rays past 90° off-axis on a wide-FOV camera).

    Notes
    -----
    Near-parallel ray pairs (``|f1 · d2| -> 1``, i.e. almost no parallax) make the 2x2 linear
    system nearly singular; the denominator is floored at ``1e-12`` rather than raising, so the
    result degrades gracefully (a poorly-conditioned but finite point) instead of dividing by
    zero.
    """
    f1 = _as_rays(f1)
    f2 = _as_rays(f2)
    R = np.asarray(R, float)
    t = np.asarray(t, float).reshape(3)
    # Both rays expressed in camera-1 frame.
    c2 = -R.T @ t                       # camera-2 centre in camera-1 frame
    d2 = f2 @ R                         # = (R.T @ f2.T).T, ray-2 directions in camera-1 frame
    w0 = -c2                            # o1 - o2, with o1 = 0
    b = np.einsum("ij,ij->i", f1, d2)   # f1·d2  (a = f1·f1 = 1, c = d2·d2 = 1)
    d = f1 @ w0                          # f1·w0
    e = d2 @ w0                          # d2·w0
    denom = 1.0 - b * b                  # a*c - b^2
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    lam1 = (b * e - d) / denom           # (b*e - c*d)/denom, c = 1
    lam2 = (e - b * d) / denom           # (a*e - b*d)/denom, a = 1
    P1 = lam1[:, None] * f1
    P2 = c2[None, :] + lam2[:, None] * d2
    X = 0.5 * (P1 + P2)
    return X, lam1, lam2


def recover_pose(f1: np.ndarray, f2: np.ndarray, E: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relative pose ``(R, t)`` and triangulated points from ray correspondences.

    Computes ``E`` (if not supplied), enumerates the four :func:`decompose_essential`
    candidates, and via cheirality picks the one with the most :func:`triangulate_rays` points
    in front of **both** cameras.

    Parameters
    ----------
    f1, f2 : (N, 3) ndarray
        Unit (or non-unit; renormalized internally) bearing vectors in camera 1 and camera 2,
        ``N >= 8`` when ``E`` is not supplied (needed by :func:`essential_from_rays`).
    E : (3, 3) ndarray, optional
        A precomputed essential matrix (e.g. re-fit on a RANSAC inlier set by
        :func:`ransac_relative_pose`). If ``None`` (default), computed from ``f1``, ``f2`` via
        :func:`essential_from_rays` with ``normalize=False``.

    Returns
    -------
    R : (3, 3) ndarray
        Rotation mapping camera 1 to camera 2 (``X2 = R @ X1 + t``), ``det(R) = +1``.
    t : (3,) ndarray
        Unit-length translation direction; its scale is unobservable from two views.
    X : (N, 3) ndarray
        Triangulated 3D points, in the **camera-1** frame.

    Examples
    --------
    Exact recovery on noise-free rays (see
    [Chapter 8](../learn/08_two_view_geometry_on_rays.md) for the full walkthrough, including a
    real Double Sphere camera round-trip asserted to ``< 1e-3`` degrees rotation error):

    >>> import numpy as np
    >>> from ds_msp.mvg import recover_pose
    >>> rng = np.random.default_rng(1)
    >>> R_true = np.eye(3)
    >>> t_true = np.array([1.0, 0.0, 0.0])
    >>> X1 = rng.uniform(-2, 2, (40, 3)) + np.array([0, 0, 5])
    >>> X2 = (R_true @ X1.T).T + t_true
    >>> f1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
    >>> f2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
    >>> R, t, X = recover_pose(f1, f2)
    >>> bool(np.allclose(R, R_true, atol=1e-8))
    True
    """
    f1 = _as_rays(f1)
    f2 = _as_rays(f2)
    if E is None:
        E = essential_from_rays(f1, f2)
    best = None
    for R, t in decompose_essential(E):
        _, d1, d2 = triangulate_rays(f1, f2, R, t)
        n_front = int(np.sum((d1 > 0) & (d2 > 0)))
        if best is None or n_front > best[0]:
            best = (n_front, R, t)
    _, R, t = best
    X, _, _ = triangulate_rays(f1, f2, R, t)
    return R, t, X


def relative_pose(f1: np.ndarray, f2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience: ``(R, t)`` from ray correspondences (eight-point + cheirality)."""
    R, t, _ = recover_pose(f1, f2)
    return R, t
