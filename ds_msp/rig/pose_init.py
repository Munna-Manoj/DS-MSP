"""Robust pose initialization — model-aware RANSAC PnP (``ransacP3PDistortion`` +
``BoardObs::estimatePose``, geometrytools.cpp:710 / BoardObs.cpp:121) and object-in-rig
pose averaging (``CameraGroupObs::computeObjectsPose``, CameraGroupObs.cpp:42).

The DS-MSP twist: instead of MC-Calib's per-distortion-type ``undistortPoints`` branch,
unproject pixels with the camera's *own* model (every model exposes ``unproject``) and
solve/refine directly on bearing vectors across the model-valid sphere.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..core.contracts import CameraModel
from ..core.lie import hat, se3_exp
from ..core.robust import auto_kernel_scale, gnc_scale, robust_weight, studentized_sq
from ..geometry.bearing import chordal_bearing_residual_jacobian
from .averaging import average_rotation


def _focal(model: CameraModel) -> float:
    K = model.K
    return 0.5 * (abs(K[0, 0]) + abs(K[1, 1]))


def bearing_chordal_residual_jacobian(
    T: np.ndarray, X: np.ndarray, f: np.ndarray, foc: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Chordal angular residual and its analytic Jacobian for a single-view pose.

    Exposed (not just inlined in :func:`robust_pose_irls`) so it can be finite-difference
    checked directly, mirroring :func:`ds_msp.rig.bundle.build_problem`'s callbacks.

    ``e_i = focal * (d_i - f_i)``, where
    ``d_i = normalize(T[:3,:3] @ X_i + T[:3,3])`` and ``f_i`` is the observed unit bearing.
    Its squared norm is ``2 focal² (1 - cos(theta_i))``: zero only for the same ray and
    monotone through the full 180-degree angular domain.  No division by ``z`` is involved.

    Jacobian is w.r.t. the **left**-perturbation pose tangent ``delta`` used by this module's
    update rule ``T <- se3_exp(delta) @ T``: ``d(Pc)/d(delta) = [I | -hat(Pc)]``, chained through
    ``d(normalize(Pc))/d(Pc) = (I - d d^T)/|Pc|``.

    Parameters
    ----------
    T : (4, 4) ndarray
        Current pose, ``T_cam_obj``.
    X : (N, 3) ndarray
        World points.
    f : (N, 3) ndarray
        Observed **unit** bearings (any ``z`` sign), same order as ``X``.
    foc : float
        Focal length scale, so the residual reads in pixel-equivalent units near the axis.

    Returns
    -------
    e : (N, 3) ndarray
        Chordal residual per point. Zero rows where ``|Pc_i| < 1e-9`` (degenerate).
    J : (N, 3, 6) ndarray
        ``d(e)/d(delta)`` per point, zero rows at the same degenerate points.
    """
    n = len(X)
    Pc = (T[:3, :3] @ X.T).T + T[:3, 3]
    e, de_dpc, good = chordal_bearing_residual_jacobian(Pc, f, eps=1e-9)
    J = np.zeros((n, 3, 6))
    for i in range(n):
        if not good[i]:
            continue
        Gi = np.hstack([np.eye(3), -hat(Pc[i])])
        J[i] = foc * de_dpc[i] @ Gi
    return foc * e, J


def robust_pose_irls(
    model: CameraModel,
    object_pts: np.ndarray,
    image_pts: np.ndarray,
    T0: Optional[np.ndarray] = None,
    *,
    kernel: str = "cauchy",
    max_iter: int = 15,
    gnc_iters: int = 5,
    gnc_start: float = 4.0,
    studentize: bool = True,
    seed: int = 0,
) -> Optional[np.ndarray]:
    """Refine a single-view pose by IRLS on a **chordal bearing residual** — full sphere, any
    ``z`` sign, keeps every usable point (ADR-0020).

    The residual is the difference between predicted and observed unit bearings,
    ``e_i = focal · (d_i - f_i)``.  Its squared norm is
    ``2 focal² (1 - cos(theta_i))``: it is zero exactly when the rays agree and grows
    monotonically to the antipode.  It is finite for every direction, involves no division by
    ``z``, and agrees with the classic pixel residual to first order near the optical axis.
    The shared formulation also drives :func:`ds_msp.mvg.bundle.refine_two_view` and
    :func:`ds_msp.rig.bundle.build_problem`'s ``residual_mode="angular"``.

    Outliers are down-weighted by a redescending kernel (``cauchy`` by default) with a
    MAD-auto scale and a short graduated-non-convexity anneal, not rejected: the answer
    uses all correspondences, mirroring the down-weight-don't-drop philosophy of the global
    BA (and the robust PnP path). ``studentize=True`` additionally inflates the residual of
    high-leverage points (the self-masking outliers a residual kernel cannot see).

    Returns the refined ``T_cam_obj`` (4x4), or ``None`` only when the view has too few
    unprojectable points to define a pose at all (insufficient data, not outlier rejection).
    The pose is warm-started from RANSAC P3P when ``T0`` is not given (a bearing-vector RANSAC,
    coplanar or not, so it seeds correctly even when every point is past 90 deg — ADR-0018,
    ADR-0019). As a cheap safety net against a pathological Gauss-Newton step, the return value
    is whichever of {warm-start, refined} scores lower on the same full bearing-angle cost —
    never worse than not refining at all.
    """
    X = np.asarray(object_pts, float)
    uv = np.asarray(image_pts, float)
    rays, ok = model.unproject(uv)
    ok = np.asarray(ok).ravel().astype(bool)            # valid rays (any z sign)
    if ok.sum() < 4:
        return None
    foc = _focal(model)

    if T0 is None:                                       # RANSAC warm-start (bearing if NC)
        T0, _ = estimate_pose_ransac(model, X, uv, seed=seed)
    T = np.eye(4) if T0 is None else T0.copy()

    Xv = X[ok]
    f = rays[ok]
    f = f / np.linalg.norm(f, axis=1, keepdims=True)     # unit observed bearings, any z sign
    n = len(Xv)

    for it in range(max_iter):
        e, J = bearing_chordal_residual_jacobian(T, Xv, f, foc)
        Pc = (T[:3, :3] @ Xv.T).T + T[:3, 3]
        good = np.linalg.norm(Pc, axis=1) > 1e-9         # degenerate only at the camera center

        s = np.einsum("nk,nk->n", e, e)                  # squared residual per point
        Jflat = J.reshape(3 * n, 6)
        if studentize and good.sum() > 8:
            s = studentized_sq(Jflat, e.reshape(-1), block=3)
        scale = auto_kernel_scale(np.sqrt(np.maximum(s, 0.0)), kernel)
        if gnc_iters > 0:
            scale = gnc_scale(it, gnc_iters, gnc_start * scale, scale)
        w = robust_weight(s, kernel, scale)              # per-point IRLS weight (n,)
        w[~good] = 0.0

        W = np.repeat(w, 3)
        H = Jflat.T @ (W[:, None] * Jflat) + 1e-9 * np.eye(6)
        g = Jflat.T @ (W * e.reshape(-1))
        try:
            delta = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        T = se3_exp(delta) @ T
        if np.linalg.norm(delta) < 1e-9:
            break

    def _full_bearing_cost(Tc: np.ndarray) -> float:
        Xc = (Tc[:3, :3] @ Xv.T).T + Tc[:3, 3]
        norm = np.linalg.norm(Xc, axis=1)
        norm[norm < 1e-12] = 1e-12
        cos_ang = np.einsum("ij,ij->i", Xc / norm[:, None], f)
        return float(np.mean(1.0 - np.clip(cos_ang, -1.0, 1.0)))

    T0_full = np.eye(4) if T0 is None else T0
    if _full_bearing_cost(T) > _full_bearing_cost(T0_full):
        return T0_full
    return T


def estimate_pose_ransac(
    model: CameraModel,
    object_pts: np.ndarray,
    image_pts: np.ndarray,
    *,
    thresh_px: float = 3.0,
    max_iters: int = 1000,
    confidence: float = 0.99,
    min_inliers: int = 4,
    seed: int = 0,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """RANSAC P3P of ``object_pts`` (N,3) against ``image_pts`` (N,2) pixels.

    RANSACs a bearing-vector engine (:func:`ds_msp.geometry.resection.ransac_pnp_normalized`
    with ``rays=...``), valid across the full sphere — peripheral rays past 90 deg (``z <= 0``)
    seed and are scored (angular inlier metric) rather than dropped: a DLT for a
    **non-coplanar** target (ADR-0018), a homography for a **coplanar** board (ADR-0019). Falls
    back to the legacy ``cv2`` P3P path on the ``z > 0`` normalized plane only when there are
    too few valid points for the bearing-native minimal sample.

    Returns ``(T_cam_obj (4,4) | None, inliers (N,) bool)``. ``None`` when fewer than
    ``min_inliers`` points survive — MC-Calib invalidates a BoardObs below 4
    (BoardObs.cpp:149).
    """
    from ..geometry.resection import _is_coplanar, ransac_pnp_normalized

    object_pts = np.asarray(object_pts, float)
    image_pts = np.asarray(image_pts, float)
    n_all = len(object_pts)
    rays, ok = model.unproject(image_pts)
    ok = np.asarray(ok).ravel().astype(bool)
    valid_idx = np.where(ok)[0]
    coplanar_target = valid_idx.size >= 4 and _is_coplanar(object_pts[ok])

    # Bearing-vector RANSAC over the full valid ray set (any z sign) -- DLT or homography.
    if valid_idx.size >= (4 if coplanar_target else 6):
        Xv_all = object_pts[ok]
        rays_v = rays[ok]
        with np.errstate(divide="ignore", invalid="ignore"):
            pn_all = rays_v[:, :2] / rays_v[:, 2:3]
        T, inl = ransac_pnp_normalized(Xv_all, pn_all, focal=_focal(model), thresh_px=thresh_px,
                                       max_iters=max_iters, confidence=confidence, seed=seed,
                                       rays=rays_v)
        if T is None or inl.sum() < min_inliers:
            return None, np.zeros(n_all, bool)
        inliers = np.zeros(n_all, bool)
        inliers[valid_idx[inl]] = True
        return T, inliers

    ok = ok & (rays[:, 2] > 1e-6)
    idx = np.where(ok)[0]
    if len(idx) < min_inliers:
        return None, np.zeros(n_all, bool)

    Xv = object_pts[idx]
    pnv = (rays[idx, :2] / rays[idx, 2:3]).astype(np.float64)
    thresh = thresh_px / _focal(model)            # pixel tol -> normalized-plane tol
    rng = np.random.default_rng(seed)
    n = len(Xv)

    best_inl, best_rvec, best_tvec = None, None, None
    it, iters = 0, max_iters
    K_eye = np.eye(3)
    while it < iters and it < max_iters:
        it += 1
        if n == 4:
            sample = np.arange(4)
        else:
            sample = rng.choice(n, 4, replace=False)
        try:
            okp, rvec, tvec = cv2.solvePnP(Xv[sample], pnv[sample], K_eye, None,
                                           flags=cv2.SOLVEPNP_P3P)
        except cv2.error:
            continue
        if not okp:
            continue
        proj, _ = cv2.projectPoints(Xv, rvec, tvec, K_eye, None)
        err = np.linalg.norm(proj.reshape(-1, 2) - pnv, axis=1)
        inl = err < thresh
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best_rvec, best_tvec = inl, rvec, tvec
            frac = float(np.clip(inl.mean(), 1e-6, 1.0))
            if frac >= 1.0:
                break
            fr3 = frac ** 3                        # adaptive iteration count, exponent 3 (cpp:300)
            # guard: tiny frac makes (1 - fr3) round to 1.0 -> log 0 -> div-by-zero
            iters = (max_iters if fr3 < 1e-9
                     else min(max_iters, int(np.log(1 - confidence) / np.log(1 - fr3)) + 1))

    if best_inl is None or best_inl.sum() < min_inliers:
        return None, np.zeros(n_all, bool)

    rvec, tvec = best_rvec, best_tvec
    if best_inl.sum() >= 6:                        # DLT refine needs >=6; else keep hypothesis
        okf, rv, tv = cv2.solvePnP(Xv[best_inl], pnv[best_inl], K_eye, None,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        if okf:
            rvec, tvec = rv, tv
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    inliers = np.zeros(n_all, bool)
    inliers[idx[best_inl]] = True
    return T, inliers


def average_object_pose_in_group(
    T_c_o_per_cam: List[Tuple[int, np.ndarray]],
    T_c_g: dict,
    ref_cam_id: int,
) -> np.ndarray:
    """Recover an object's pose in the group frame (``T_g_o``) from one or more cameras.

    Mirrors ``CameraGroupObs::computeObjectsPose``: if the reference camera sees the
    object, use it directly; otherwise average across the non-ref cameras with Markley
    rotation averaging and **arithmetic-mean** translation (CameraGroupObs.cpp:95).
    """
    # T_g_o = inv(T_c_g) @ T_c_o   (object->cam lifted into the group frame)
    lifted = []
    for cam_id, T_c_o in T_c_o_per_cam:
        T_g_o = np.linalg.inv(T_c_g[cam_id]) @ T_c_o
        if cam_id == ref_cam_id:
            return T_g_o
        lifted.append(T_g_o)
    R = average_rotation([T[:3, :3] for T in lifted])
    t = np.mean(np.array([T[:3, 3] for T in lifted]), axis=0)
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = t
    return out
