"""Robust pose initialization — model-aware RANSAC PnP (``ransacP3PDistortion`` +
``BoardObs::estimatePose``, geometrytools.cpp:710 / BoardObs.cpp:121) and object-in-rig
pose averaging (``CameraGroupObs::computeObjectsPose``, CameraGroupObs.cpp:42).

The DS-MSP twist: instead of MC-Calib's per-distortion-type ``undistortPoints`` branch,
unproject pixels with the camera's *own* model (every model exposes ``unproject``) and
PnP on the normalized plane — the same trick ``calib.bundle._seed_poses`` already uses.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..core.contracts import CameraModel
from ..core.lie import hat, se3_exp
from ..core.robust import auto_kernel_scale, gnc_scale, robust_weight, studentized_sq
from .averaging import average_rotation


def _focal(model: CameraModel) -> float:
    K = model.K
    return 0.5 * (abs(K[0, 0]) + abs(K[1, 1]))


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
    """Refine a single-view pose by IRLS on the normalized plane — **keeps every point**.

    Outliers are down-weighted by a redescending kernel (``cauchy`` by default) with a
    MAD-auto scale and a short graduated-non-convexity anneal, not rejected: the answer
    uses all correspondences, mirroring the down-weight-don't-drop philosophy of the global
    BA (and the robust PnP path). ``studentize=True`` additionally inflates the residual of
    high-leverage points (the self-masking outliers a residual kernel cannot see).

    Returns the refined ``T_cam_obj`` (4x4), or ``None`` only when the view has too few
    unprojectable points to define a pose at all (insufficient data, not outlier rejection).
    The pose is warm-started from RANSAC P3P when ``T0`` is not given.
    """
    X = np.asarray(object_pts, float)
    uv = np.asarray(image_pts, float)
    rays, ok = model.unproject(uv)
    ok = ok & (rays[:, 2] > 1e-6)
    if ok.sum() < 4:
        return None
    idx = np.where(ok)[0]
    Xv = X[idx]
    pn = rays[idx, :2] / rays[idx, 2:3]                  # normalized observations
    foc = _focal(model)

    if T0 is None:                                       # RANSAC P3P warm-start (init only)
        T0, _ = estimate_pose_ransac(model, X, uv, seed=seed)
    T = np.eye(4) if T0 is None else T0.copy()

    n = len(Xv)
    for it in range(max_iter):
        Pc = (T[:3, :3] @ Xv.T).T + T[:3, 3]             # (n,3) camera-frame points
        Z = Pc[:, 2]
        good = Z > 1e-6
        if good.sum() < 4:
            break
        proj = np.zeros((n, 2))
        proj[good] = Pc[good, :2] / Z[good, None]
        e = (proj - pn) * foc                            # residual in pixels (n,2)
        # 2x6 Jacobian per point on the normalized plane, scaled to pixels.
        J = np.zeros((n, 2, 6))
        invZ = np.zeros(n)
        invZ[good] = 1.0 / Z[good]
        Jp = np.zeros((n, 2, 3))
        Jp[:, 0, 0] = invZ
        Jp[:, 0, 2] = -Pc[:, 0] * invZ * invZ
        Jp[:, 1, 1] = invZ
        Jp[:, 1, 2] = -Pc[:, 1] * invZ * invZ
        for i in range(n):                               # [I | -hat(P)] left perturbation
            Gi = np.hstack([np.eye(3), -hat(Pc[i])])
            J[i] = foc * Jp[i] @ Gi
        e[~good] = 0.0
        J[~good] = 0.0

        s = np.einsum("nk,nk->n", e, e)                  # squared residual per point (px^2)
        Jflat = J.reshape(2 * n, 6)
        if studentize and good.sum() > 8:
            s = studentized_sq(Jflat, e.reshape(-1), block=2)
        scale = auto_kernel_scale(np.sqrt(np.maximum(s, 0.0)), kernel)
        if gnc_iters > 0:
            scale = gnc_scale(it, gnc_iters, gnc_start * scale, scale)
        w = robust_weight(s, kernel, scale)              # per-point IRLS weight (n,)
        w[~good] = 0.0

        W = np.repeat(w, 2)
        H = Jflat.T @ (W[:, None] * Jflat) + 1e-9 * np.eye(6)
        g = Jflat.T @ (W * e.reshape(-1))
        try:
            delta = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        T = se3_exp(delta) @ T
        if np.linalg.norm(delta) < 1e-9:
            break
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

    Returns ``(T_cam_obj (4,4) | None, inliers (N,) bool)``. ``None`` when fewer than
    ``min_inliers`` points survive — MC-Calib invalidates a BoardObs below 4
    (BoardObs.cpp:149).
    """
    object_pts = np.asarray(object_pts, float)
    image_pts = np.asarray(image_pts, float)
    n_all = len(object_pts)
    rays, ok = model.unproject(image_pts)
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
