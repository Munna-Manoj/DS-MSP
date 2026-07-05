# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025-2026 Munna-Manoj. Robust from-scratch intrinsic seed + RANSAC
# resection — part of the DS-MSP robust calibration engine
# (https://github.com/Munna-Manoj/DS-MSP). NONCOMMERCIAL use only, with attribution —
# see LICENSE-NONCOMMERCIAL.txt and LICENSING.md. The rest of DS-MSP is MIT.
"""From-scratch robust intrinsic seed + RANSAC PnP (pure NumPy, no OpenCV).

Why this exists
---------------
The rig front-end used to seed each camera's focal/principal point with
``cv2.calibrateCamera`` and seed per-view poses with ``cv2.solvePnP``. Both are
**non-robust** (plain L2 / single DLT): a handful of gross mis-decoded corners
(40 px blunders) drags the focal seed to garbage and lands per-view poses in the
wrong basin, after which even the downstream IRLS bundle adjuster cannot climb
out — the rig diverges past ~6-10 % gross outliers (one camera's extrinsic
collapses entirely).

The robustness has to live in the *seed*, before any reweighting can help. This
module replaces those primitives with **RANSAC over a linear DLT model**, which
rejects gross outliers by construction:

* :func:`ransac_resection` — RANSAC a 3x4 camera matrix on the genuinely-3D
  target (multi-board), then RQ-decompose it into ``K, R, t``. The intrinsic seed
  is the robust median of the per-view ``K`` over the inlier views.
* :func:`ransac_pnp_normalized` — RANSAC a pose on the normalized image plane
  (pixels already unprojected to bearings through the camera's own model), the
  drop-in robust replacement for ``cv2.solvePnP`` seeding.

Everything here is closed-form linear algebra (SVD); the only randomness is the
RANSAC sampling, seeded for reproducibility. Downstream, ``calib.bundle.calibrate``
still does the metric refinement under IRLS — this module only has to land the
seed in the right basin despite the blunders.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Hartley normalization (conditions the DLT so the SVD is well-posed)
# --------------------------------------------------------------------------- #
def _normalize_2d(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Translate to centroid, scale to mean distance √2. Returns (T, pts_h_norm)."""
    c = pts.mean(axis=0)
    d = np.linalg.norm(pts - c, axis=1)
    s = np.sqrt(2.0) / max(float(d.mean()), 1e-12)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    ph = np.c_[pts, np.ones(len(pts))] @ T.T
    return T, ph


def _normalize_3d(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Translate to centroid, scale to mean distance √3. Returns (U, pts_h_norm)."""
    c = pts.mean(axis=0)
    d = np.linalg.norm(pts - c, axis=1)
    s = np.sqrt(3.0) / max(float(d.mean()), 1e-12)
    U = np.array([[s, 0, 0, -s * c[0]], [0, s, 0, -s * c[1]],
                  [0, 0, s, -s * c[2]], [0, 0, 0, 1.0]])
    ph = np.c_[pts, np.ones(len(pts))] @ U.T
    return U, ph


def dlt_projection(X: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Normalized DLT estimate of the 3x4 camera matrix ``P`` (``uv ~ P·[X;1]``).

    Solves the 2N x 12 homogeneous system ``A vec(P) = 0`` by SVD (smallest right-singular
    vector) after Hartley normalization, then de-normalizes. The returned ``P`` is only defined
    up to scale.

    Parameters
    ----------
    X : (N, 3) ndarray
        World-frame 3D points, genuinely non-coplanar (see :func:`_is_coplanar` /
        :func:`ransac_pnp_normalized`'s planar branch for coplanar targets, where this general
        DLT is degenerate).
    uv : (N, 2) ndarray
        Corresponding image points (pixels, or normalized coordinates with ``K = I``).

    Returns
    -------
    (3, 4) ndarray
        Camera matrix ``P`` such that ``[u, v, 1]ᵀ ~ P @ [X, 1]ᵀ`` up to scale.

    Notes
    -----
    Needs **at least 6** correspondences for ``A`` to be non-degenerate (``2N >= 11`` free
    unknowns), but this is **not checked**: fewer points still run without error and silently
    return a meaningless ``P``. Callers needing that guarantee should go through
    :func:`ransac_resection`, which enforces ``min_sample`` before calling this.
    """
    X = np.asarray(X, float)
    uv = np.asarray(uv, float)
    U, Xn = _normalize_3d(X)
    T, un = _normalize_2d(uv)
    n = len(X)
    A = np.zeros((2 * n, 12))
    Xh = Xn                                   # (n,4) homogeneous, normalized
    u, v = un[:, 0], un[:, 1]
    A[0::2, 0:4] = -Xh
    A[0::2, 8:12] = u[:, None] * Xh
    A[1::2, 4:8] = -Xh
    A[1::2, 8:12] = v[:, None] * Xh
    _, _, Vt = np.linalg.svd(A)
    Pn = Vt[-1].reshape(3, 4)
    # de-normalize: u = T·P_real·X  and  un = T·u, Xn = U·X  ⇒  P_real = T⁻¹·Pn·U
    P = np.linalg.inv(T) @ Pn @ U
    return P


def _rq3(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """RQ decomposition of a 3x3 matrix: ``M = R_up · Q`` with ``R_up`` upper-triangular
    and ``Q`` orthogonal. Implemented via a flipped QR."""
    P = np.array([[0, 0, 1.0], [0, 1, 0], [1, 0, 0]])      # reversal permutation
    Mt = P @ M
    Q0, R0 = np.linalg.qr(Mt.T)
    R = P @ R0.T @ P
    Q = P @ Q0.T
    return R, Q


def decompose_P(P: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Factor ``P = K[R|t]`` into intrinsics ``K`` (``K[2,2]=1``), rotation ``R`` (det +1),
    and translation ``t``. Returns ``None`` if the factorization is degenerate.
    """
    H = P[:, :3]
    if abs(np.linalg.det(H)) < 1e-12:
        return None
    K, R = _rq3(H)
    # force a positive diagonal on K (sign ambiguity of RQ): H = K·S·S·R with S=diag(±1)
    d = np.diag(K)
    S = np.diag(np.where(d >= 0, 1.0, -1.0))
    K = K @ S
    R = S @ R
    lam = K[2, 2]                                           # DLT scale: H = λ·K_true·R
    if abs(lam) < 1e-12:
        return None
    K = K / lam                                            # normalize K[2,2] = 1
    Pn = P / lam                                           # rescale P to match → Pn = K[R|t]
    if np.linalg.det(R) < 0:                                # det(R)=+1 resolves the 1-bit sign
        R = -R
        Pn = -Pn
    if K[0, 0] < 0 or K[1, 1] < 0:
        return None
    t = np.linalg.inv(K) @ Pn[:, 3]
    return K, R, t


def _reproj_err(P: np.ndarray, X: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Per-point reprojection error (px) of a 3x4 ``P`` (NaN/behind → +inf)."""
    Xh = np.c_[X, np.ones(len(X))]
    proj = Xh @ P.T
    if np.median(proj[:, 2]) < 0:            # DLT P is sign-ambiguous; orient so scene is forward
        proj = -proj
    z = proj[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uvp = proj[:, :2] / z[:, None]
    e = np.linalg.norm(uvp - uv, axis=1)
    e[~np.isfinite(e) | (z <= 0)] = np.inf
    return e


def ransac_resection(X: np.ndarray, uv: np.ndarray, *, thresh_px: float = 3.0,
                     max_iters: int = 300, confidence: float = 0.999,
                     min_sample: int = 6, seed: int = 0
                     ) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """RANSAC a 3x4 camera matrix robust to gross outliers.

    Samples ``min_sample`` correspondences, fits a DLT ``P``, scores by reprojection
    inliers, and refits ``P`` on the full consensus set. Returns ``(P | None, inlier_mask)``.
    """
    X = np.asarray(X, float)
    uv = np.asarray(uv, float)
    n = len(X)
    if n < min_sample:
        return None, np.zeros(n, bool)
    rng = np.random.default_rng(seed)
    best_inl = np.zeros(n, bool)
    iters = max_iters
    it = 0
    while it < iters and it < max_iters:
        it += 1
        sample = rng.choice(n, min_sample, replace=False)
        try:
            P = dlt_projection(X[sample], uv[sample])
        except np.linalg.LinAlgError:
            continue
        inl = _reproj_err(P, X, uv) < thresh_px
        if inl.sum() > best_inl.sum():
            best_inl = inl
            frac = float(np.clip(inl.mean(), 1e-6, 1.0))
            if frac >= 1.0:
                break
            den = np.log1p(-frac ** min_sample)
            if den < -1e-12:
                iters = min(max_iters, int(np.log1p(-confidence) / den) + 1)
    if best_inl.sum() < min_sample:
        return None, best_inl
    try:
        P = dlt_projection(X[best_inl], uv[best_inl])
    except np.linalg.LinAlgError:
        return None, best_inl
    # final inlier set under the refit
    best_inl = _reproj_err(P, X, uv) < thresh_px
    return P, best_inl


def intrinsics_seed(objpts_list: List[np.ndarray], imgpts_list: List[np.ndarray],
                    w: int, h: int, *, thresh_px: float = 3.0, seed: int = 0
                    ) -> Tuple[np.ndarray, List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]]:
    """Robust pinhole intrinsic seed ``K`` from a genuinely-3D target, no OpenCV.

    RANSAC-resects every view, RQ-decomposes each into ``K_i, R_i, t_i``, and returns
    the **robust median** ``K`` over the views whose decomposition is plausible plus the
    per-view ``(K_i, R_i, t_i)`` (``None`` for views that failed). Gross outliers are
    rejected inside each view's RANSAC, so the focal seed never sees the blunders.
    """
    diag = float(np.hypot(w, h))
    Ks, poses = [], []
    for i, (X, uv) in enumerate(zip(objpts_list, imgpts_list)):
        P, inl = ransac_resection(np.asarray(X, float), np.asarray(uv, float),
                                  thresh_px=thresh_px, seed=seed + i)
        dec = decompose_P(P) if P is not None and inl.sum() >= 6 else None
        if dec is None:
            poses.append(None)
            continue
        K, R, t = dec
        fx, fy = K[0, 0], K[1, 1]
        # only let plausibly-focused views vote for the consensus intrinsics
        if 0.2 * diag < fx < 5.0 * diag and 0.2 * diag < fy < 5.0 * diag:
            Ks.append([fx, fy, K[0, 2], K[1, 2]])
        poses.append((K, R, t))
    if Ks:
        fx, fy, cx, cy = np.median(np.array(Ks), axis=0)
    else:
        fx = fy = float(w)
        cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return K, poses


# --------------------------------------------------------------------------- #
# RANSAC PnP on the normalized plane (drop-in for cv2.solvePnP seeding)
# --------------------------------------------------------------------------- #
def _pose_dlt_normalized(X: np.ndarray, pn: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Closed-form pose from ≥6 (3D point, normalized 2D) pairs with ``K = I``.

    DLT for ``P = [R|t]`` then project the 3x3 block back onto SO(3) via SVD and fix
    the scale/sign from the orthogonalization. Returns ``(R, t)`` or ``None``.
    """
    if len(X) < 6:
        return None
    uv = pn                                  # normalized coords behave like pixels with K=I
    U, Xn = _normalize_3d(np.asarray(X, float))
    n = len(X)
    A = np.zeros((2 * n, 12))
    Xh = Xn
    A[0::2, 0:4] = -Xh
    A[0::2, 8:12] = uv[:, 0:1] * Xh
    A[1::2, 4:8] = -Xh
    A[1::2, 8:12] = uv[:, 1:2] * Xh
    _, _, Vt = np.linalg.svd(A)
    Pn = Vt[-1].reshape(3, 4)
    P = Pn @ U                               # de-normalize 3D side (2D side is K=I already)
    M = P[:, :3]
    # nearest rotation to M (up to scale); recover scale from singular values
    Uu, s, Vh = np.linalg.svd(M)
    R = Uu @ Vh
    if np.linalg.det(R) < 0:
        R = -R
        P = -P
    scale = float(s.mean())
    if scale < 1e-12:
        return None
    t = P[:, 3] / scale
    if t[2] < 0:                             # scene must be in front of the camera
        R, t = R, t                          # sign already fixed via det; depth sign:
    # enforce positive depth by flipping the homogeneous sign if needed
    if t[2] < 0:
        return None
    return R, t


def _is_coplanar(X: np.ndarray, tol: float = 1e-3) -> bool:
    """True if the 3-D points lie on a plane (smallest PCA extent ≪ the in-plane extent).

    A single ChArUco board is coplanar (all ``Z = 0`` in board frame); a fused multi-board
    object with a tilted board is not. The pose solver must branch on this: the general 3×4
    DLT is **degenerate for coplanar points**, so a planar target needs a homography/IPPE pose.
    """
    Xc = np.asarray(X, float) - np.asarray(X, float).mean(0)
    if len(Xc) < 4:
        return True
    s = np.linalg.svd(Xc, compute_uv=False)
    return s[-1] <= tol * max(s[0], 1e-12)


def _pose_planar_normalized(X: np.ndarray, pn: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Pose of a **coplanar** target from (3D, normalized-2D) pairs via a plane **homography**
    (``K = I``), pure NumPy.

    A point on the plane is ``P = c0 + a·e1 + b·e2`` (plane basis from PCA); under ``K = I`` its
    camera ray is ``Xc = a·(R e1) + b·(R e2) + (R c0 + t) = H·[a, b, 1]ᵀ``. Fit ``H`` by DLT,
    then recover ``R, t`` from its columns (Zhang's planar pose). Degeneracy-free for a board,
    unlike the general 3×4 DLT."""
    X = np.asarray(X, float)
    pn = np.asarray(pn, float)
    if len(X) < 4:
        return None
    c0 = X.mean(0)
    Xc = X - c0
    _, _, Vt = np.linalg.svd(Xc)
    e1, e2, nrm = Vt[0], Vt[1], Vt[2]
    a, b = Xc @ e1, Xc @ e2                       # 2-D plane coordinates
    # homography DLT: [a,b,1] -> pn (2 rows/point), null-space of the 2n x 9 design matrix.
    n = len(X)
    M = np.zeros((2 * n, 9))
    one = np.ones(n)
    P = np.column_stack([a, b, one])             # (n,3) plane homog coords
    M[0::2, 0:3] = -P
    M[0::2, 6:9] = pn[:, 0:1] * P
    M[1::2, 3:6] = -P
    M[1::2, 6:9] = pn[:, 1:2] * P
    _, _, Vh = np.linalg.svd(M)
    H = Vh[-1].reshape(3, 3)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    s = 0.5 * (np.linalg.norm(h1) + np.linalg.norm(h2))
    if s < 1e-12:
        return None
    if h3[2] < 0:                                # enforce positive depth (g0_z > 0)
        H, h1, h2, h3 = -H, -h1, -h2, -h3
    g1, g2, g0 = h1 / s, h2 / s, h3 / s          # R e1, R e2, R c0 + t
    g3 = np.cross(g1, g2)
    G = np.column_stack([g1, g2, g3])
    Uu, _, Vv = np.linalg.svd(G)                 # nearest rotation [R e1, R e2, R nrm]
    Rg = Uu @ np.diag([1.0, 1.0, np.linalg.det(Uu @ Vv)]) @ Vv
    R = Rg @ np.column_stack([e1, e2, nrm]).T    # R maps object axes -> camera
    t = g0 - R @ c0
    if t[2] <= 0:
        return None
    return R, t


def ransac_pnp_normalized(X: np.ndarray, pn: np.ndarray, *, focal: float = 1.0,
                          thresh_px: float = 3.0, max_iters: int = 300,
                          confidence: float = 0.999, min_sample: int = 6,
                          seed: int = 0) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """RANSAC pose on the normalized plane. ``thresh_px`` is interpreted in pixels via
    ``focal`` (the model focal) so the gate matches the pixel-domain blunders. Returns
    ``(T_cam_obj (4,4) | None, inlier_mask)``.

    Branches on target geometry: a **coplanar** board uses the IPPE planar solver (the general
    3×4 DLT is degenerate for coplanar points — it returns garbage poses, ~1700 px reprojection
    on a wide-FOV board); a non-coplanar (fused multi-board) object uses the DLT."""
    X = np.asarray(X, float)
    pn = np.asarray(pn, float)
    n = len(X)
    if n < min_sample:
        return None, np.zeros(n, bool)
    coplanar = _is_coplanar(X)
    min_sample = 4 if coplanar else min_sample
    solve = _pose_planar_normalized if coplanar else _pose_dlt_normalized
    thr = thresh_px / max(focal, 1e-9)       # normalized-plane tolerance
    rng = np.random.default_rng(seed)
    best_inl = np.zeros(n, bool)
    iters, it = max_iters, 0

    def _err(R, t):
        Xc = X @ R.T + t
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            proj = Xc[:, :2] / z[:, None]
        e = np.linalg.norm(proj - pn, axis=1)
        e[~np.isfinite(e) | (z <= 0)] = np.inf
        return e

    while it < iters and it < max_iters:
        it += 1
        sample = rng.choice(n, min_sample, replace=False)
        sol = solve(X[sample], pn[sample])
        if sol is None:
            continue
        R, t = sol
        inl = _err(R, t) < thr
        if inl.sum() > best_inl.sum():
            best_inl = inl
            frac = float(np.clip(inl.mean(), 1e-6, 1.0))
            if frac >= 1.0:
                break
            den = np.log1p(-frac ** min_sample)
            if den < -1e-12:
                iters = min(max_iters, int(np.log1p(-confidence) / den) + 1)
    if best_inl.sum() < min_sample:
        return None, best_inl
    sol = solve(X[best_inl], pn[best_inl])
    if sol is None:
        return None, best_inl
    R, t = sol
    best_inl = _err(R, t) < thr
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, best_inl
