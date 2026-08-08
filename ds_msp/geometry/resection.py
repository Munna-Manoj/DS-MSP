"""From-scratch robust resection and bearing-native PnP (pure NumPy, no OpenCV).

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
module supplies two robust-estimator families around the same linear geometry:

* :func:`ransac_resection` — RANSAC a 3x4 camera matrix on the genuinely-3D
  target (multi-board), then RQ-decompose it into ``K, R, t``. The intrinsic seed
  is the robust median of the per-view ``K`` over the inlier views.
* :func:`gnc_pnp_bearings` — deterministic GNC-TLS pose estimation on full-sphere
  unit bearings. This is the preferred per-view pose seed.
* :func:`ransac_pnp_normalized` — the compatibility RANSAC pose engine retained
  for callers that explicitly request random minimal-set consensus.

The foundation stays NumPy-only. Downstream calibration still performs metric
refinement; these estimators only have to land it in the right basin despite
gross correspondence blunders.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..core.lie import hat, se3_exp
from ..core.optimize import gnc_tls_solve, lm_solve
from .bearing import chordal_bearing_residual_jacobian


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
    min_sample = max(6, min_sample)
    if n < min_sample:
        return None, np.zeros(n, bool)
    rng = np.random.default_rng(seed)
    best_inl = np.zeros(n, bool)
    best_P = None
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
            best_P = P.copy()
            frac = float(np.clip(inl.mean(), 1e-6, 1.0))
            if frac >= 1.0:
                break
            den = np.log1p(-frac ** min_sample)
            if den < -1e-12:
                iters = min(max_iters, int(np.log1p(-confidence) / den) + 1)
    if best_inl.sum() < min_sample or best_P is None:
        return None, best_inl
    try:
        P_refit = dlt_projection(X[best_inl], uv[best_inl])
    except np.linalg.LinAlgError:
        return best_P, best_inl
    refit_inl = _reproj_err(P_refit, X, uv) < thresh_px
    # Consensus refitting is optional polishing. Never replace a supported RANSAC hypothesis
    # with a refit that fails or reduces the measured consensus.
    if refit_inl.sum() >= best_inl.sum():
        return P_refit, refit_inl
    return best_P, best_inl


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


def _pose_dlt_bearing(X: np.ndarray, rays: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Closed-form pose from >=6 (3D point, **bearing vector**) pairs, any ray direction.

    Bearing-vector-native ("cross-product") DLT for absolute pose. Unlike
    :func:`_pose_dlt_normalized` — which divides each ray by its ``z`` to land it on the
    ``z = 1`` plane and therefore **cannot represent any bearing past 90 deg off-axis**
    (``z <= 0``) — this solves directly on the sphere and handles wide-FOV rays of any sign.

    Model
    -----
    For a world point ``X_i`` seen as bearing ``f_i`` under pose ``(R, t)``, the camera-frame
    point ``R X_i + t`` is parallel to ``f_i`` (same direction, up to the unknown positive
    depth ``lambda_i``): ``R X_i + t = lambda_i f_i``, ``lambda_i > 0``. Eliminating the depth
    gives the exact linear constraint

        f_i x (R X_i + t) = [f_i]_x (P [X_i; 1]) = 0,   P = [R | t]  (3x4).

    The cross product yields 3 rows linear in ``vec(P)``, of which 2 are independent (a cross
    product is rank-2: ``f_i . (f_i x w) = 0``), so each correspondence contributes 2 equations
    — the same count as the classic point-DLT. Stacking ``A vec(P) = 0`` and taking the SVD
    null space recovers ``P`` up to scale; ``R`` follows by projecting ``P[:, :3]`` onto SO(3),
    the scale from its singular values, and the sign is fixed by requiring ``det(R) = +1`` (this
    already forces the correct, scene-in-front branch — see the cheirality note below).

    Reduces to :func:`_pose_dlt_normalized` when every ``f_i = (u_i, v_i, 1)``: rows 1-2 become
    exactly ``p1.X = u p3.X`` and ``p2.X = v p3.X`` (the two normalized-plane DLT equations) and
    row 3 is their dependent combination — so this is a strict generalization, identical on
    ``z > 0`` data and additionally valid for ``z <= 0``.

    Cheirality is enforced on the **depth along the bearing** ``lambda_i = f_i . (R X_i + t)``
    (``lambda > 0``), per the wide-FOV convention in ``ds_msp.mvg.two_view`` — **not** ``z > 0``.

    Scope: this handles the **non-coplanar** case only (the general 3x4 DLT is degenerate for
    coplanar points, exactly like :func:`_pose_dlt_normalized`); coplanar targets use the
    bearing-vector homography path instead, :func:`_pose_planar_bearing` (ADR-0019).

    References
    ----------
    Hartley & Zisserman, *Multiple View Geometry*, 2nd ed., general camera resectioning DLT
    (§7.1), here written with the cross-product ``f x PX = 0`` form on bearing vectors instead
    of pixel homogeneous coordinates; Kneip & Furgale, *OpenGV*, ICRA 2014, §III (generalized
    absolute-pose DLT on bearing vectors).

    Parameters
    ----------
    X : (N, 3) ndarray
        World-frame 3D points, genuinely non-coplanar, ``N >= 6``.
    rays : (N, 3) ndarray
        Bearing vectors (unit or non-unit; renormalized internally), any ``z`` sign.

    Returns
    -------
    (R, t) or None
        ``R`` (3x3, ``det = +1``) and ``t`` (3,) with ``R X_i + t = lambda_i f_i``,
        ``lambda_i > 0``. ``None`` if under-determined, degenerate, or non-physical
        (majority of depths behind the camera).
    """
    X = np.asarray(X, float)
    f = np.asarray(rays, float)
    if len(X) < 6:
        return None
    nrm = np.linalg.norm(f, axis=1, keepdims=True)
    if np.any(nrm < 1e-12):
        return None
    f = f / nrm                              # unit bearings (sphere, any z sign)
    U, Xn = _normalize_3d(X)                 # Hartley-condition the 3D side; Xn is (n,4) homog
    n = len(X)
    fx, fy, fz = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    # f x (P.[X;1]) = 0 -> 3 rows/point in vec(P) = [p1(4), p2(4), p3(4)]
    A = np.zeros((3 * n, 12))
    A[0::3, 4:8] = -fz * Xn                   # row1:  fy*(p3.X) - fz*(p2.X) = 0
    A[0::3, 8:12] = fy * Xn
    A[1::3, 0:4] = fz * Xn                    # row2:  fz*(p1.X) - fx*(p3.X) = 0
    A[1::3, 8:12] = -fx * Xn
    A[2::3, 0:4] = -fy * Xn                   # row3:  fx*(p2.X) - fy*(p1.X) = 0  (dependent)
    A[2::3, 4:8] = fx * Xn
    _, _, Vt = np.linalg.svd(A)
    Pn = Vt[-1].reshape(3, 4)
    P = Pn @ U                               # de-normalize the 3D side (bearing side is K=I)
    M = P[:, :3]
    Uu, s, Vh = np.linalg.svd(M)             # nearest rotation to M (= scale * R)
    R = Uu @ Vh
    if np.linalg.det(R) < 0:                 # det(R)=+1 fixes the homogeneous sign
        R = -R
        P = -P
    scale = float(s.mean())
    if scale < 1e-12:
        return None
    t = P[:, 3] / scale
    # Cheirality on depth-along-bearing (lambda>0), NOT z>0: valid for rays past 90 deg.
    lam = np.einsum("ij,ij->i", f, X @ R.T + t)
    if np.median(lam) < 0:
        return None
    return R, t


def _is_coplanar(X: np.ndarray, tol: float = 1e-3) -> bool:
    """True if the 3-D points lie on a plane (smallest PCA extent ≪ the in-plane extent).

    A single ChArUco board is coplanar (all ``Z = 0`` in board frame); a fused multi-board
    object with a tilted board is not. The pose solver must branch on this: the general 3×4
    DLT is **degenerate for coplanar points**, so a planar target needs a homography/IPPE pose.
    """
    X = np.asarray(X, float)
    if len(X) < 4:
        return True
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    return s[-1] <= tol * max(s[0], 1e-12)


def bearing_pose_residual_jacobian(
    T: np.ndarray,
    X: np.ndarray,
    rays: np.ndarray,
    *,
    focal: float = 1.0,
    whiteners: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chordal bearing residual and left-SE(3) Jacobian for absolute pose.

    The predicted direction for world point ``X_i`` is
    ``d_i = normalize(T[:3, :3] @ X_i + T[:3, 3])`` and the residual is
    ``focal * (d_i - f_i)``. It is finite over the complete sphere and has squared norm
    ``2 focal**2 * (1 - cos(theta_i))``. With a fixed concentration, minimizing this cost
    is also maximum likelihood under an isotropic von Mises--Fisher bearing model. Optional
    per-observation ``whiteners`` replace the scalar focal with a local anisotropic metric,
    such as :func:`projection_bearing_whiteners`, while preserving the signed chordal ray.

    The Jacobian uses the left perturbation consumed by :func:`refine_pose_bearings`,
    ``T <- exp(delta) @ T``. Invalid zero-length/non-finite rows are returned with zero
    residual and Jacobian and ``valid=False``.

    Returns
    -------
    residual : (N, 3) ndarray
        Pixel-equivalent chordal residual blocks.
    jacobian : (N, 3, 6) ndarray
        Derivative with respect to ``delta = [translation, rotation]``.
    valid : (N,) ndarray of bool
        Rows with finite, non-zero predicted and observed directions.
    """
    T = np.asarray(T, float)
    X = np.asarray(X, float)
    rays = np.asarray(rays, float)
    if T.shape != (4, 4):
        raise ValueError("T must have shape (4, 4)")
    if X.ndim != 2 or X.shape[1:] != (3,) or rays.shape != X.shape:
        raise ValueError("X and rays must have matching shape (N, 3)")
    if not np.isfinite(focal) or focal <= 0.0:
        raise ValueError("focal must be finite and positive")
    if whiteners is not None:
        whiteners = np.asarray(whiteners, float)
        if whiteners.shape != (len(X), 3, 3):
            raise ValueError("whiteners must have shape (N, 3, 3)")
        if not np.isfinite(whiteners).all():
            raise ValueError("whiteners must be finite")

    camera_points = X @ T[:3, :3].T + T[:3, 3]
    residual, dres_dpoint, valid = chordal_bearing_residual_jacobian(
        camera_points, rays, eps=1e-12
    )
    jacobian = np.zeros((len(X), 3, 6), dtype=float)
    for i in np.flatnonzero(valid):
        dpoint_ddelta = np.hstack((np.eye(3), -hat(camera_points[i])))
        jacobian[i] = dres_dpoint[i] @ dpoint_ddelta
    if whiteners is None:
        return focal * residual, focal * jacobian, valid
    residual = np.einsum("nij,nj->ni", whiteners, residual)
    jacobian = np.einsum("nij,njk->nik", whiteners, jacobian)
    return residual, jacobian, valid


def bearing_pose_angular_error(T: np.ndarray, X: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """Per-correspondence angular pose error in radians over the complete bearing sphere."""
    residual, _, valid = bearing_pose_residual_jacobian(T, X, rays)
    # For unit directions, ||d-f|| = 2 sin(theta/2). This avoids a separate duplicate
    # normalization path and is more accurate than acos(dot) close to a perfect fit.
    half_chord = np.clip(0.5 * np.linalg.norm(residual, axis=1), 0.0, 1.0)
    error = 2.0 * np.arcsin(half_chord)
    error[~valid] = np.inf
    return error


def refine_pose_bearings(
    T0: np.ndarray,
    X: np.ndarray,
    rays: np.ndarray,
    *,
    focal: float = 1.0,
    whiteners: Optional[np.ndarray] = None,
    max_iter: int = 30,
) -> np.ndarray:
    """Refine an absolute pose directly on unit bearings with manifold LM.

    Callers unproject pixels once, then this camera-neutral consensus-polish primitive
    minimizes the 3-component chordal residual without projecting through a concrete lens
    model or dividing by bearing ``z``. The ordinary least-squares objective is appropriate
    after a robust estimator has selected a clean consensus. :func:`lm_solve` accepts only
    cost-reducing steps, so numerical failure safely leaves ``T0`` unchanged; public robust
    callers also rescore support over all observations before accepting the result.
    """
    T0 = np.asarray(T0, float)
    X = np.asarray(X, float)
    rays = np.asarray(rays, float)
    if T0.shape != (4, 4) or not np.isfinite(T0).all():
        raise ValueError("T0 must be a finite array with shape (4, 4)")
    if X.ndim != 2 or X.shape[1:] != (3,) or rays.shape != X.shape:
        raise ValueError("X and rays must have matching shape (N, 3)")
    if len(X) < 4:
        raise ValueError("at least four bearing correspondences are required")
    if whiteners is not None:
        whiteners = np.asarray(whiteners, float)
        if whiteners.shape != (len(X), 3, 3):
            raise ValueError("whiteners must have shape (N, 3, 3)")

    def residual(T: object) -> np.ndarray:
        e, _, _ = bearing_pose_residual_jacobian(
            np.asarray(T), X, rays, focal=focal, whiteners=whiteners
        )
        return e.reshape(-1)

    def jacobian(T: object) -> np.ndarray:
        _, J, _ = bearing_pose_residual_jacobian(
            np.asarray(T), X, rays, focal=focal, whiteners=whiteners
        )
        return J.reshape(-1, 6)

    def retract(T: object, delta: np.ndarray) -> np.ndarray:
        return se3_exp(delta) @ np.asarray(T)

    result = lm_solve(
        T0.copy(), residual, jacobian, retract, block=3, max_iter=max_iter,
        robust_kernel="none", tol=1e-10,
    )
    candidate = np.asarray(result.state, float)
    return candidate if candidate.shape == (4, 4) and np.isfinite(candidate).all() else T0.copy()


def guarded_refine_pose_bearings(
    T: np.ndarray,
    X: np.ndarray,
    rays: np.ndarray,
    inliers: np.ndarray,
    *,
    focal: float,
    threshold: float,
    whiteners: Optional[np.ndarray] = None,
    max_iter: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Polish a bearing consensus without weakening its robust-estimator support.

    The first candidate is fit to the current consensus. A bounded least-trimmed refinement
    then keeps the same number of smallest-residual observations while allowing threshold-edge
    membership to change; this avoids bias from freezing an imperfect discrete consensus.
    The result is rescored over all correspondences and accepted only if it preserves support
    and does not increase the truncated residual (MSAC-style) score. A failed or non-finite
    refinement is therefore a harmless no-op.
    """
    inliers = np.asarray(inliers, bool)
    if inliers.shape != (len(X),):
        raise ValueError("inliers must have shape (N,)")
    if inliers.sum() < 4:
        return np.asarray(T, float).copy(), inliers.copy()
    whiteners = None if whiteners is None else np.asarray(whiteners, float)
    if whiteners is not None and whiteners.shape != (len(X), 3, 3):
        raise ValueError("whiteners must have shape (N, 3, 3)")
    X = np.asarray(X)
    rays = np.asarray(rays)
    support = int(inliers.sum())
    try:
        candidate = refine_pose_bearings(
            T, X[inliers], rays[inliers],
            focal=focal,
            whiteners=None if whiteners is None else whiteners[inliers],
            max_iter=max_iter,
        )
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return np.asarray(T, float).copy(), inliers.copy()

    def block_error(pose: np.ndarray) -> np.ndarray:
        residual, _, valid = bearing_pose_residual_jacobian(
            pose, X, rays, focal=focal, whiteners=whiteners
        )
        error = np.linalg.norm(residual, axis=1)
        error[~valid] = np.inf
        return error

    # One bounded least-trimmed update is the deterministic local-optimization analogue of
    # replacing a threshold-edge RANSAC/GNC member with a better-supported observation. Keep
    # cardinality fixed here; the final threshold rescore below may still grow the consensus.
    error_candidate = block_error(candidate)
    finite = np.isfinite(error_candidate)
    if finite.sum() >= support:
        order = np.argsort(error_candidate, kind="stable")
        trimmed = np.zeros(len(X), dtype=bool)
        trimmed[order[:support]] = True
        if not np.array_equal(trimmed, inliers):
            try:
                candidate = refine_pose_bearings(
                    candidate, X[trimmed], rays[trimmed],
                    focal=focal,
                    whiteners=None if whiteners is None else whiteners[trimmed],
                    max_iter=max_iter,
                )
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                return np.asarray(T, float).copy(), inliers.copy()

    error_before = block_error(T)
    error_after = block_error(candidate)
    candidate_inliers = error_after < threshold
    score_before = float(np.minimum(error_before ** 2, threshold ** 2).sum())
    score_after = float(np.minimum(error_after ** 2, threshold ** 2).sum())
    if candidate_inliers.sum() >= inliers.sum() and score_after <= score_before:
        return candidate, candidate_inliers
    return np.asarray(T, float).copy(), inliers.copy()


def gnc_pnp_bearings(
    X: np.ndarray,
    rays: np.ndarray,
    *,
    focal: float = 1.0,
    noise_bound_px: float = 3.0,
    max_outer: int = 100,
    inner_max_iter: int = 2,
    refine: bool = True,
    whiteners: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Deterministic GNC-TLS absolute pose on full-sphere bearing vectors.

    This is the robust non-minimal alternative to minimal-set RANSAC. A bearing DLT
    (non-coplanar) or bearing homography (coplanar) supplies a deterministic all-data seed;
    :func:`gnc_tls_solve` then alternates weighted chordal pose updates with closed-form TLS
    weights against the explicit ``noise_bound_px``. When local pixel ``whiteners`` are
    supplied, the bound is in locally calibrated pixel units at every observed ray; otherwise
    the residual uses the legacy scalar-``focal`` chordal approximation. The final weights form
    a hard consensus mask, optionally followed by :func:`guarded_refine_pose_bearings`.

    The nonlinear weighted pose step uses the repository's manifold LM, so this implementation
    deliberately does **not** claim the certifiable/no-initial-guess guarantee available when
    every GNC variable update has a globally optimal non-minimal solver. It is nevertheless
    deterministic, uses all measurements rather than random minimal subsets, and inherits the
    same full-sphere residual as the rest of the bearing backend.
    """
    X = np.asarray(X, float)
    rays = np.asarray(rays, float)
    if X.ndim != 2 or X.shape[1:] != (3,) or rays.shape != X.shape:
        raise ValueError("X and rays must have matching shape (N, 3)")
    if not np.isfinite(noise_bound_px) or noise_bound_px <= 0.0:
        raise ValueError("noise_bound_px must be finite and positive")
    if whiteners is not None:
        whiteners = np.asarray(whiteners, float)
        if whiteners.shape != (len(X), 3, 3) or not np.isfinite(whiteners).all():
            raise ValueError("whiteners must be finite with shape (N, 3, 3)")
    coplanar = _is_coplanar(X)
    support_floor = 4 if coplanar else 6
    if len(X) < support_floor:
        return None, np.zeros(len(X), bool)

    solve_linear = _pose_planar_bearing if coplanar else _pose_dlt_bearing
    try:
        seed = solve_linear(X, rays)
    except np.linalg.LinAlgError:
        seed = None
    if seed is None:
        return None, np.zeros(len(X), bool)
    T0 = np.eye(4)
    T0[:3, :3], T0[:3, 3] = seed

    def residual(T: object) -> np.ndarray:
        e, _, _ = bearing_pose_residual_jacobian(
            np.asarray(T), X, rays, focal=focal, whiteners=whiteners
        )
        return e.reshape(-1)

    def jacobian(T: object) -> np.ndarray:
        _, J, _ = bearing_pose_residual_jacobian(
            np.asarray(T), X, rays, focal=focal, whiteners=whiteners
        )
        return J.reshape(-1, 6)

    def retract(T: object, delta: np.ndarray) -> np.ndarray:
        return se3_exp(delta) @ np.asarray(T)

    try:
        result = gnc_tls_solve(
            T0, residual, jacobian, retract,
            noise_bound=noise_bound_px, block=3, max_outer=max_outer,
            inner_max_iter=inner_max_iter,
        )
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return None, np.zeros(len(X), bool)
    T = np.asarray(result.state, float)
    if T.shape != (4, 4) or not np.isfinite(T).all() or result.weights is None:
        return None, np.zeros(len(X), bool)

    final_residual, _, final_valid = bearing_pose_residual_jacobian(
        T, X, rays, focal=focal, whiteners=whiteners
    )
    final_error = np.linalg.norm(final_residual, axis=1)
    inliers = (
        (np.asarray(result.weights) > 0.5)
        & final_valid
        & (final_error < noise_bound_px)
    )
    if inliers.sum() < support_floor:
        return None, inliers
    if refine:
        T, inliers = guarded_refine_pose_bearings(
            T, X, rays, inliers, focal=focal, threshold=noise_bound_px,
            whiteners=whiteners,
        )
    if inliers.sum() < support_floor:
        return None, inliers
    return T, inliers


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
    e1, e2 = Vt[0], Vt[1]
    nrm = np.cross(e1, e2)                        # right-handed normal, NOT Vt[2] (SVD's sign
    # on the smallest singular vector is arbitrary -- Vt[2] can be antiparallel to cross(e1,e2),
    # which silently turns [e1,e2,Vt[2]] into a reflection and R below into a mirrored, wrong
    # pose. Manifests for large board tilts; verified via manufactured recovery test.
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


def _pose_planar_bearing(X: np.ndarray, rays: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Closed-form pose of a **coplanar** target from (3D, **bearing vector**) pairs, any ray
    direction.

    Bearing-vector-native generalization of :func:`_pose_planar_normalized`, exactly as
    :func:`_pose_dlt_bearing` generalizes :func:`_pose_dlt_normalized` — replaces the
    homogeneous normalized-plane target ``[u, v, 1]`` with the true bearing ``f`` (any ``z``
    sign), so a single planar calibration board imaged past 90 deg off-axis is handled instead
    of silently restricted to its forward-hemisphere corners.

    Model
    -----
    A point on the plane is ``P = c0 + a·e1 + b·e2`` (plane basis from PCA, as in the legacy
    method). Under pose ``(R, t)`` its **camera-frame** point is exactly
    ``Xc = a·(R e1) + b·(R e2) + (R c0 + t) = H·[a, b, 1]``, ``H = [R e1 | R e2 | R c0 + t]``
    (3x3) — this holds regardless of ``z``'s sign; the legacy method's ``pn = Xc / Xc_z``
    additionally assumes ``Xc_z > 0`` so it can drop the (redundant, scale-invariant) ``Xc_z``
    factor, which is exactly what breaks past 90 deg. Since the bearing ``f`` is parallel to
    ``Xc`` (``f ∥ Xc``, any sign, any scale), the cross-product constraint
    ``f × (H·[a,b,1]) = 0`` holds directly — 3 rows linear in ``vec(H)``, rank-2 (2 independent
    equations), the same count as the legacy homography DLT's 2 rows/point.

    Reduces to :func:`_pose_planar_normalized` when every ``f_i ∥ (u_i, v_i, 1)`` with
    ``z > 0``: the null space of a homogeneous linear system is invariant to per-row positive
    scaling, so using ``f_i`` (any positive multiple of ``(u_i, v_i, 1)``) in place of
    ``(u_i, v_i, 1)`` recovers the identical ``H`` (up to the same overall scale ambiguity both
    methods already resolve the same way).

    ``H``'s column decomposition into ``(R, t)`` is **identical** to the legacy method (it
    depends only on ``H``'s structure, not on what was used to fit it) — except the sign
    disambiguation and cheirality check, which used ``h3[2] < 0`` / ``t[2] <= 0`` (assuming the
    plane origin and every point project with ``z > 0``). Generalized here to the same
    depth-along-bearing convention as :func:`_pose_dlt_bearing`: ``lambda = f · (R X + t) > 0``,
    valid for any ray direction.

    References
    ----------
    Zhang, *A Flexible New Technique for Camera Calibration*, TPAMI 2000 (planar homography
    pose decomposition — the column-recovery step, unchanged here); the bearing-vector
    cross-product generalization follows the same principle as Hartley & Zisserman §7.1 /
    Kneip & Furgale OpenGV ICRA 2014 §III already applied to the non-coplanar case in
    :func:`_pose_dlt_bearing`.

    Parameters
    ----------
    X : (N, 3) ndarray
        World-frame 3D points, genuinely coplanar (see :func:`_is_coplanar`), ``N >= 4``.
    rays : (N, 3) ndarray
        Bearing vectors (unit or non-unit; renormalized internally), any ``z`` sign.

    Returns
    -------
    (R, t) or None
        ``R`` (3x3, ``det = +1``) and ``t`` (3,) with ``R X_i + t = lambda_i f_i``,
        ``lambda_i > 0``. ``None`` if under-determined or non-physical.
    """
    X = np.asarray(X, float)
    f = np.asarray(rays, float)
    if len(X) < 4:
        return None
    nrm = np.linalg.norm(f, axis=1, keepdims=True)
    if np.any(nrm < 1e-12):
        return None
    f = f / nrm                              # unit bearings (any z sign)
    c0 = X.mean(0)
    Xc = X - c0
    _, _, Vt = np.linalg.svd(Xc)
    e1, e2 = Vt[0], Vt[1]
    nrm3 = np.cross(e1, e2)                  # right-handed normal, NOT Vt[2] (arbitrary sign;
    # see the identical fix + comment in _pose_planar_normalized above)
    a, b = Xc @ e1, Xc @ e2
    n = len(X)
    P = np.column_stack([a, b, np.ones(n)])   # (n,3) plane homogeneous coords
    fx, fy, fz = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    # f x (H.P) = 0 -> 3 rows/point in vec(H) = [h_row0(3), h_row1(3), h_row2(3)]
    M = np.zeros((3 * n, 9))
    M[0::3, 3:6] = -fz * P                    # row1:  fy*(row2.P) - fz*(row1.P) = 0
    M[0::3, 6:9] = fy * P
    M[1::3, 0:3] = fz * P                     # row2:  fz*(row0.P) - fx*(row2.P) = 0
    M[1::3, 6:9] = -fx * P
    M[2::3, 0:3] = -fy * P                    # row3:  fx*(row1.P) - fy*(row0.P) = 0 (dependent)
    M[2::3, 3:6] = fx * P
    _, _, Vh = np.linalg.svd(M)
    H = Vh[-1].reshape(3, 3)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    s = 0.5 * (np.linalg.norm(h1) + np.linalg.norm(h2))
    if s < 1e-12:
        return None
    # Sign disambiguation via depth-along-bearing (lambda>0), not h3[2]<0 -- valid past 90 deg.
    lam0 = np.einsum("ij,ij->i", f, (H @ P.T).T)
    if np.median(lam0) < 0:
        H, h1, h2, h3 = -H, -h1, -h2, -h3
    g1, g2, g0 = h1 / s, h2 / s, h3 / s        # R e1, R e2, R c0 + t
    g3 = np.cross(g1, g2)
    G = np.column_stack([g1, g2, g3])
    Uu, _, Vv = np.linalg.svd(G)
    Rg = Uu @ np.diag([1.0, 1.0, np.linalg.det(Uu @ Vv)]) @ Vv
    R = Rg @ np.column_stack([e1, e2, nrm3]).T
    t = g0 - R @ c0
    # Final cheirality on the full point set (lambda>0), matching _pose_dlt_bearing.
    lam = np.einsum("ij,ij->i", f, X @ R.T + t)
    if np.median(lam) < 0:
        return None
    return R, t


def _ransac_pnp_planar_bearing(X: np.ndarray, rays: np.ndarray, *, focal: float = 1.0,
                               thresh_px: float = 3.0, max_iters: int = 300,
                               confidence: float = 0.999, min_sample: int = 4,
                               seed: int = 0, whiteners: Optional[np.ndarray] = None
                               ) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """RANSAC pose on **bearing vectors** for a **coplanar** target (a single calibration
    board), valid for the full sphere. The non-coplanar analogue of :func:`_ransac_pnp_bearing`
    — same full-sphere inlier metric, same reduction-to-legacy guarantee, minimal sample 4 (a
    homography) instead of 6 (a general 3x4 DLT). With ``whiteners``, ``thresh_px`` is evaluated
    in the camera's local pixel metric at each observed bearing."""
    X = np.asarray(X, float)
    f = np.asarray(rays, float)
    n = len(X)
    if n < min_sample:
        return None, np.zeros(n, bool)
    nrm = np.linalg.norm(f, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = f / nrm
    if whiteners is not None:
        whiteners = np.asarray(whiteners, float)
        if whiteners.shape != (n, 3, 3) or not np.isfinite(whiteners).all():
            raise ValueError("whiteners must be finite with shape (N, 3, 3)")
    thr = thresh_px if whiteners is not None else thresh_px / max(focal, 1e-9)
    rng = np.random.default_rng(seed)
    best_inl = np.zeros(n, bool)
    best_sol = None
    iters, it = max_iters, 0

    def _ang_err(R, t):
        Pc = X @ R.T + t
        d = np.linalg.norm(Pc, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            fp = Pc / d[:, None]
        if whiteners is None:
            cos = np.clip(np.einsum("ij,ij->i", fp, f), -1.0, 1.0)
            e = np.arccos(cos)
        else:
            chord = fp - f
            e = np.linalg.norm(np.einsum("nij,nj->ni", whiteners, chord), axis=1)
        e[~np.isfinite(d) | (d < 1e-12)] = np.inf
        return e

    while it < iters and it < max_iters:
        it += 1
        sample = rng.choice(n, min_sample, replace=False)
        try:
            sol = _pose_planar_bearing(X[sample], f[sample])
        except np.linalg.LinAlgError:
            continue
        if sol is None:
            continue
        R, t = sol
        inl = _ang_err(R, t) < thr
        if inl.sum() > best_inl.sum():
            best_inl = inl
            best_sol = (R.copy(), t.copy())
            frac = float(np.clip(inl.mean(), 1e-6, 1.0))
            if frac >= 1.0:
                break
            den = np.log1p(-frac ** min_sample)
            if den < -1e-12:
                iters = min(max_iters, int(np.log1p(-confidence) / den) + 1)
    if best_inl.sum() < min_sample or best_sol is None:
        return None, best_inl
    R, t = best_sol
    try:
        sol = _pose_planar_bearing(X[best_inl], f[best_inl])
    except np.linalg.LinAlgError:
        sol = None
    if sol is not None:
        R_refit, t_refit = sol
        refit_inl = _ang_err(R_refit, t_refit) < thr
        # Consensus refitting is optional polishing. Never replace the supported hypothesis
        # with a failed refit or one that reduces its measured consensus.
        if refit_inl.sum() >= best_inl.sum():
            R, t, best_inl = R_refit, t_refit, refit_inl
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, best_inl


def _ransac_pnp_bearing(X: np.ndarray, rays: np.ndarray, *, focal: float = 1.0,
                        thresh_px: float = 3.0, max_iters: int = 300,
                        confidence: float = 0.999, min_sample: int = 6,
                        seed: int = 0, whiteners: Optional[np.ndarray] = None
                        ) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """RANSAC pose on **bearing vectors** (non-coplanar target), valid for the full sphere.

    The minimal-sample model is :func:`_pose_dlt_bearing` (cross-product DLT), so peripheral
    rays past 90 deg off-axis (``z <= 0``) can both **seed** the hypothesis and be **scored**
    as inliers/outliers — the normalized-plane engine (:func:`ransac_pnp_normalized`'s DLT
    branch) cannot, because ``pn = xy/z`` is undefined/unstable there.

    Inlier metric
    -------------
    The base error is the complete-sphere bearing discrepancy between
    ``f_pred = normalize(R X + t)`` and ``f_obs``. With per-observation ``whiteners`` from the
    camera projection Jacobian, its chord is measured in a locally calibrated pixel metric, so
    ``thresh_px`` remains meaningful at the image periphery. Without whiteners, compatibility
    callers retain the earlier angular gate ``thresh_px / focal``. Both forms stay finite at
    ``z = 0`` and penalize the antipodal ray rather than dividing by bearing ``z``.

    Returns ``(T_cam_obj (4,4) | None, inlier_mask)`` over all ``len(X)`` input rows.
    """
    X = np.asarray(X, float)
    f = np.asarray(rays, float)
    n = len(X)
    if n < min_sample:
        return None, np.zeros(n, bool)
    nrm = np.linalg.norm(f, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = f / nrm
    if whiteners is not None:
        whiteners = np.asarray(whiteners, float)
        if whiteners.shape != (n, 3, 3) or not np.isfinite(whiteners).all():
            raise ValueError("whiteners must be finite with shape (N, 3, 3)")
    thr = (thresh_px if whiteners is not None else
           thresh_px / max(focal, 1e-9))
    rng = np.random.default_rng(seed)
    best_inl = np.zeros(n, bool)
    best_sol = None
    iters, it = max_iters, 0

    def _ang_err(R, t):
        Pc = X @ R.T + t
        d = np.linalg.norm(Pc, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            fp = Pc / d[:, None]
        if whiteners is None:
            cos = np.clip(np.einsum("ij,ij->i", fp, f), -1.0, 1.0)
            e = np.arccos(cos)
        else:
            chord = fp - f
            e = np.linalg.norm(np.einsum("nij,nj->ni", whiteners, chord), axis=1)
        e[~np.isfinite(d) | (d < 1e-12)] = np.inf
        return e

    while it < iters and it < max_iters:
        it += 1
        sample = rng.choice(n, min_sample, replace=False)
        try:
            sol = _pose_dlt_bearing(X[sample], f[sample])
        except np.linalg.LinAlgError:
            continue
        if sol is None:
            continue
        R, t = sol
        inl = _ang_err(R, t) < thr
        if inl.sum() > best_inl.sum():
            best_inl = inl
            best_sol = (R.copy(), t.copy())
            frac = float(np.clip(inl.mean(), 1e-6, 1.0))
            if frac >= 1.0:
                break
            den = np.log1p(-frac ** min_sample)
            if den < -1e-12:
                iters = min(max_iters, int(np.log1p(-confidence) / den) + 1)
    if best_inl.sum() < min_sample or best_sol is None:
        return None, best_inl
    R, t = best_sol
    try:
        sol = _pose_dlt_bearing(X[best_inl], f[best_inl])
    except np.linalg.LinAlgError:
        sol = None
    if sol is not None:
        R_refit, t_refit = sol
        refit_inl = _ang_err(R_refit, t_refit) < thr
        # Keep the best supported minimal-sample hypothesis when optional consensus refitting
        # fails or reduces support, including on noisy all-peripheral data.
        if refit_inl.sum() >= best_inl.sum():
            R, t, best_inl = R_refit, t_refit, refit_inl
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, best_inl


def ransac_pnp_normalized(X: np.ndarray, pn: np.ndarray, *, focal: float = 1.0,
                          thresh_px: float = 3.0, max_iters: int = 300,
                          confidence: float = 0.999, min_sample: int = 6,
                          seed: int = 0, rays: Optional[np.ndarray] = None,
                          whiteners: Optional[np.ndarray] = None,
                          ) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """RANSAC pose from a camera's unprojected observations.

    Bearing callers can supply per-ray projection ``whiteners`` so ``thresh_px`` is evaluated
    in a local pixel metric over the full sphere. Without them, the historical scalar-focal
    angular conversion is retained. Returns ``(T_cam_obj (4,4) | None, inlier_mask)``.

    Branches on target geometry **and** the available observation form:

    * **Coplanar** board with ``rays`` supplied → the bearing-vector homography engine
      (:func:`_ransac_pnp_planar_bearing`, ADR-0019), valid for the **full sphere** including a
      board imaged edge-on enough to put corners past 90 deg off-axis (``z <= 0``).
    * **Non-coplanar** target with ``rays`` supplied → the bearing-vector DLT engine
      (:func:`_ransac_pnp_bearing`, ADR-0018), valid for the **full sphere** including peripheral
      rays past 90 deg (``z <= 0``); see that function for the inlier metric.
    * ``rays=None`` (legacy callers) → the normalized-plane engine on ``pn`` — IPPE homography
      for coplanar, general 3×4 DLT otherwise — ``z > 0`` only, unchanged behaviour.

    Parameters
    ----------
    pn : (N, 2) normalized-plane points (``xy/z``). Used by the legacy (``rays=None``) paths;
        ignored when a bearing path is taken.
    rays : (N, 3) optional bearing vectors (any ``z`` sign). When given, switches on the
        full-sphere bearing engine (coplanar or not) so ``z <= 0`` points are handled.
    whiteners : (N, 3, 3) optional
        Fixed local bearing-to-pixel metrics aligned with ``rays``.
    """
    X = np.asarray(X, float)
    pn = np.asarray(pn, float)
    n = len(X)
    coplanar = _is_coplanar(X)
    # Select the solver before enforcing its sample floor: a planar homography is
    # determined by four correspondences, while either general DLT needs at least six.
    min_sample = 4 if coplanar else max(6, min_sample)
    if n < min_sample:
        return None, np.zeros(n, bool)
    if rays is not None:
        if coplanar:
            return _ransac_pnp_planar_bearing(X, np.asarray(rays, float), focal=focal,
                                              thresh_px=thresh_px, max_iters=max_iters,
                                              confidence=confidence, min_sample=4, seed=seed,
                                              whiteners=whiteners)
        return _ransac_pnp_bearing(X, np.asarray(rays, float), focal=focal, thresh_px=thresh_px,
                                   max_iters=max_iters, confidence=confidence,
                                   min_sample=min_sample, seed=seed, whiteners=whiteners)
    solve = _pose_planar_normalized if coplanar else _pose_dlt_normalized
    thr = thresh_px / max(focal, 1e-9)       # normalized-plane tolerance
    rng = np.random.default_rng(seed)
    best_inl = np.zeros(n, bool)
    best_sol = None
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
        try:
            sol = solve(X[sample], pn[sample])
        except np.linalg.LinAlgError:
            continue
        if sol is None:
            continue
        R, t = sol
        inl = _err(R, t) < thr
        if inl.sum() > best_inl.sum():
            best_inl = inl
            best_sol = (R.copy(), t.copy())
            frac = float(np.clip(inl.mean(), 1e-6, 1.0))
            if frac >= 1.0:
                break
            den = np.log1p(-frac ** min_sample)
            if den < -1e-12:
                iters = min(max_iters, int(np.log1p(-confidence) / den) + 1)
    if best_inl.sum() < min_sample or best_sol is None:
        return None, best_inl
    R, t = best_sol
    try:
        sol = solve(X[best_inl], pn[best_inl])
    except np.linalg.LinAlgError:
        sol = None
    if sol is not None:
        R_refit, t_refit = sol
        refit_inl = _err(R_refit, t_refit) < thr
        # Apply the same safe-polish rule as the bearing paths: a failed or degraded refit
        # cannot erase the already-supported RANSAC hypothesis.
        if refit_inl.sum() >= best_inl.sum():
            R, t, best_inl = R_refit, t_refit, refit_inl
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, best_inl
