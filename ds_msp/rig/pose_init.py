"""Robust pose initialization — model-aware bearing GNC-TLS (replacing
``ransacP3PDistortion`` +
``BoardObs::estimatePose``, geometrytools.cpp:710 / BoardObs.cpp:121) and object-in-rig
pose averaging (``CameraGroupObs::computeObjectsPose``, CameraGroupObs.cpp:42).

The DS-MSP twist: instead of MC-Calib's per-distortion-type ``undistortPoints`` branch,
unproject pixels with the camera's *own* model (every model exposes ``unproject``) and
solve/refine directly on bearing vectors across the model-valid sphere.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..core.contracts import CameraModel
from ..core.lie import se3_exp, so3_exp
from ..core.robust import (
    auto_kernel_scale,
    gnc_scale,
    robust_weight,
    studentized_sq,
)
from ..geometry.bearing import projection_bearing_whiteners
from ..geometry.resection import (
    bearing_pose_residual_jacobian,
    gnc_pnp_bearings,
    refine_pose_bearings,
)
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
    e, J, _ = bearing_pose_residual_jacobian(T, X, f, focal=foc)
    return e, J


def estimate_pose_gnc(
    model: CameraModel,
    object_pts: np.ndarray,
    image_pts: np.ndarray,
    *,
    noise_bound_px: float = 3.0,
    max_iters: int = 100,
    min_inliers: int = 4,
    refine: bool = True,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Deterministic GNC-TLS pose seed on model-unprojected full-sphere bearings."""
    object_pts = np.asarray(object_pts, float)
    image_pts = np.asarray(image_pts, float)
    rays, valid = model.unproject(image_pts)
    valid = np.asarray(valid).ravel().astype(bool)
    valid_idx = np.flatnonzero(valid)
    rays_valid = np.asarray(rays, float)[valid]
    _, J_point, _, projection_valid = model.project_jacobian(rays_valid)
    whiteners, metric_valid = projection_bearing_whiteners(
        rays_valid, J_point, projection_valid
    )
    valid_idx = valid_idx[metric_valid]
    T, inliers_valid = gnc_pnp_bearings(
        object_pts[valid][metric_valid], rays_valid[metric_valid],
        noise_bound_px=noise_bound_px, max_outer=max_iters, refine=refine,
        whiteners=whiteners[metric_valid],
    )
    inliers = np.zeros(len(object_pts), bool)
    inliers[valid_idx[inliers_valid]] = True
    if T is None or inliers.sum() < min_inliers:
        return None, inliers
    return T, inliers


def _small_sample_pose_seed(
    model: CameraModel,
    object_pts: np.ndarray,
    image_pts: np.ndarray,
) -> Optional[np.ndarray]:
    """Deterministic 4--5 point seed when the six-point bearing DLT is unavailable.

    SQPnP is applied only after rotating the observed bearings into a well-conditioned virtual
    perspective chart. The chart rotation is a coordinate change, not a camera-model
    approximation: the returned pose is transformed back before bearing-native IRLS. A set that
    does not fit inside a numerically usable open hemisphere has no supported chart and is rejected
    rather than refined from an arbitrary identity pose.
    """
    def hemisphere_axis(unit_rays: np.ndarray) -> Optional[np.ndarray]:
        """Minimum-norm center of ``f_i @ c >= 1``, or ``None`` if infeasible.

        In three dimensions the strictly convex quadratic program has at most three linearly
        independent active constraints. Enumerating those active sets is exact for this tiny
        (four- or five-ray) problem and avoids treating the normalized ray mean as a spherical
        cap center, which is false for strongly asymmetric fields of view.
        """
        best: Optional[np.ndarray] = None
        best_norm_sq = np.inf
        for active_count in range(1, min(3, len(unit_rays)) + 1):
            for active in combinations(range(len(unit_rays)), active_count):
                A = unit_rays[np.asarray(active)]
                gram = A @ A.T
                if np.linalg.matrix_rank(gram, tol=1e-12) < active_count:
                    continue
                try:
                    multipliers = np.linalg.solve(gram, np.ones(active_count))
                except np.linalg.LinAlgError:
                    continue
                if np.any(multipliers < -1e-10):
                    continue
                candidate = A.T @ multipliers
                if np.any(unit_rays @ candidate < 1.0 - 1e-9):
                    continue
                norm_sq = float(candidate @ candidate)
                if norm_sq < best_norm_sq - 1e-10:
                    best, best_norm_sq = candidate, norm_sq
        if best is None or not np.isfinite(best_norm_sq) or best_norm_sq <= 1e-24:
            return None
        return best / np.sqrt(best_norm_sq)

    X = np.asarray(object_pts, float)
    uv = np.asarray(image_pts, float)
    rays, valid = model.unproject(uv)
    valid = np.asarray(valid).ravel().astype(bool)
    if valid.sum() not in (4, 5):
        return None
    X = X[valid]
    rays = np.asarray(rays, float)[valid]
    norm = np.linalg.norm(rays, axis=1)
    if not np.isfinite(rays).all() or np.any(norm <= 1e-12):
        return None
    rays = rays / norm[:, None]
    _, J_project, _, projection_valid = model.project_jacobian(rays)
    whiteners, metric_valid = projection_bearing_whiteners(
        rays, J_project, projection_valid
    )
    X = X[metric_valid]
    rays = rays[metric_valid]
    whiteners = whiteners[metric_valid]
    if len(X) not in (4, 5):
        return None

    # A valid perspective chart must preserve the sign of every ray; otherwise xy/z identifies
    # a bearing with its antipode. Solve for a true common open-hemisphere center rather than
    # assuming the normalized mean has that property.
    axis = hemisphere_axis(rays)
    if axis is None or np.any(rays @ axis <= 1e-6):
        return None

    z_axis = np.array([0.0, 0.0, 1.0])
    cosine = float(np.clip(axis @ z_axis, -1.0, 1.0))
    if cosine > 1.0 - 1e-12:
        chart_rotation = np.eye(3)
    elif cosine < -1.0 + 1e-12:
        chart_rotation = so3_exp(np.array([np.pi, 0.0, 0.0]))
    else:
        rotation_axis = np.cross(axis, z_axis)
        rotation_axis /= np.linalg.norm(rotation_axis)
        chart_rotation = so3_exp(rotation_axis * np.arccos(cosine))

    chart_rays = rays @ chart_rotation.T
    if np.any(chart_rays[:, 2] <= 1e-6):
        return None
    normalized = chart_rays[:, :2] / chart_rays[:, 2:3]

    # SQPnP is accurate for five points but can select a catastrophic four-point branch.
    # Enumerate its candidate together with every AP3P branch at n=4, transform each back to
    # the physical camera frame, locally refine each on the same bearing objective, and then
    # choose by the original locally pixel-calibrated score. Refining before comparing matters:
    # raw algebraic PnP scores can rank a wrong root just ahead of the correct root even though
    # the correct basin reaches a two-orders-of-magnitude lower geometric cost. This stays
    # deterministic and does not use random minimal-set consensus.
    flags = [cv2.SOLVEPNP_SQPNP]
    if len(X) == 4:
        flags.append(cv2.SOLVEPNP_AP3P)
    candidates: List[Tuple[float, np.ndarray]] = []
    for flag in flags:
        try:
            result = cv2.solvePnPGeneric(
                X, normalized, np.eye(3), None, flags=flag
            )
        except cv2.error:
            continue
        if not isinstance(result, tuple) or len(result) < 3 or not result[0]:
            continue
        for rvec, tvec in zip(result[1], result[2]):
            R_chart = cv2.Rodrigues(np.asarray(rvec, float))[0]
            R = chart_rotation.T @ R_chart
            t = chart_rotation.T @ np.asarray(tvec, float).ravel()
            camera_points = X @ R.T + t
            point_norm = np.linalg.norm(camera_points, axis=1)
            if (not np.isfinite(R).all() or not np.isfinite(t).all()
                    or np.any(~np.isfinite(point_norm)) or np.any(point_norm <= 1e-12)):
                continue
            depth_along_ray = np.einsum("ij,ij->i", rays, camera_points)
            if np.any(depth_along_ray <= 0.0):
                continue
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            try:
                T = refine_pose_bearings(
                    T, X, rays, whiteners=whiteners, max_iter=30
                )
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                continue
            camera_points = X @ T[:3, :3].T + T[:3, 3]
            point_norm = np.linalg.norm(camera_points, axis=1)
            if np.any(~np.isfinite(point_norm)) or np.any(point_norm <= 1e-12):
                continue
            depth_along_ray = np.einsum("ij,ij->i", rays, camera_points)
            if np.any(depth_along_ray <= 0.0):
                continue
            predicted = camera_points / point_norm[:, None]
            residual = np.einsum("nij,nj->ni", whiteners, predicted - rays)
            score = float(np.einsum("ij,ij->", residual, residual))
            if not np.isfinite(score):
                continue
            candidates.append((score, T))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


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
    """Refine a single-view pose by IRLS on a **whitened chordal bearing residual** — full
    sphere, any ``z`` sign, keeps every usable point (ADR-0020, ADR-0021).

    The residual is the difference between predicted and observed unit bearings,
    ``e_i = W_i · (d_i - f_i)``. ``W_i`` is fixed from the model's analytic projection
    Jacobian at the observed ray: on the sphere tangent it matches pixel error to first order,
    while its radial completion keeps the antipodal signed ray costly. The residual is finite
    for every direction and involves no division by ``z``. Its unwhitened base remains the
    shared chordal formulation used by :func:`ds_msp.mvg.bundle.refine_two_view` and
    :func:`ds_msp.rig.bundle.build_problem`'s ``residual_mode="angular"``.

    Outliers are down-weighted by a redescending kernel (``cauchy`` by default) with a
    MAD-auto scale and a short graduated-non-convexity anneal, not rejected: the answer
    uses all correspondences, mirroring the down-weight-don't-drop philosophy of the global
    BA (and the robust PnP path). ``studentize=True`` additionally inflates the residual of
    high-leverage points (the self-masking outliers a residual kernel cannot see).

    Returns the refined ``T_cam_obj`` (4x4), or ``None`` when the view has too few usable points
    or no supported deterministic seed. The pose is warm-started from deterministic GNC-TLS when
    ``T0`` is not given (bearing-native, coplanar or not, so it seeds correctly even when every
    point is past 90 deg). A genuinely non-coplanar 4--5 point view cannot determine the six-point
    bearing DLT; when its rays fit a numerically usable open hemisphere, a deterministic SQPnP
    solve in a ray-aligned virtual perspective chart supplies the seed instead. As a cheap
    safety net against a pathological Gauss-Newton step, an explicitly supplied ``T0`` keeps the
    historical full-bearing L2 acceptance guard (appropriate when the caller already selected a
    clean consensus). An implicit robust seed uses a truncated final score; otherwise a gross
    outlier could make the guard reject a good robust solution precisely because it stopped
    fitting that outlier.

    ``seed`` is retained for source compatibility with the former RANSAC warm start; neither
    deterministic seed path uses it.
    """
    X = np.asarray(object_pts, float)
    uv = np.asarray(image_pts, float)
    rays, ok = model.unproject(uv)
    ok = np.asarray(ok).ravel().astype(bool)            # valid rays (any z sign)
    if ok.sum() < 4:
        return None
    explicit_warm_start = T0 is not None
    if T0 is None:                                       # deterministic full-sphere robust seed
        T0, _ = estimate_pose_gnc(model, X, uv)
        if T0 is None:
            T0 = _small_sample_pose_seed(model, X, uv)
        if T0 is None:
            return None
    T = T0.copy()

    Xv = X[ok]
    f = rays[ok]
    f = f / np.linalg.norm(f, axis=1, keepdims=True)     # unit observed bearings, any z sign
    _, J_project, _, projection_valid = model.project_jacobian(f)
    whiteners, metric_valid = projection_bearing_whiteners(
        f, J_project, projection_valid
    )
    Xv = Xv[metric_valid]
    f = f[metric_valid]
    whiteners = whiteners[metric_valid]
    if len(Xv) < 4:
        return None
    n = len(Xv)
    comparison_scale = 1.0

    for it in range(max_iter):
        e, J, _ = bearing_pose_residual_jacobian(
            T, Xv, f, whiteners=whiteners
        )
        Pc = (T[:3, :3] @ Xv.T).T + T[:3, 3]
        good = np.linalg.norm(Pc, axis=1) > 1e-9         # degenerate only at the camera center

        s = np.einsum("nk,nk->n", e, e)                  # squared residual per point
        Jflat = J.reshape(3 * n, 6)
        if studentize and good.sum() > 8:
            s = studentized_sq(Jflat, e.reshape(-1), block=3)
        scale = auto_kernel_scale(np.sqrt(np.maximum(s, 0.0)), kernel)
        if gnc_iters > 0:
            scale = gnc_scale(it, gnc_iters, gnc_start * scale, scale)
        comparison_scale = scale
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

    def _robust_bearing_cost(Tc: np.ndarray) -> float:
        e, J, _ = bearing_pose_residual_jacobian(
            Tc, Xv, f, whiteners=whiteners
        )
        s = np.einsum("nk,nk->n", e, e)
        if explicit_warm_start:
            return float(s.sum())
        if studentize and n > 8:
            s = studentized_sq(J.reshape(3 * n, 6), e.reshape(-1), block=3)
        return float(np.minimum(s, comparison_scale ** 2).sum())

    # Left-multiplying many floating-point SO(3) factors can leave the rotation a few ulps
    # off the group even when the geometric optimum is exact.  One Newton--Schulz polar
    # correction removes that accumulated roundoff without changing the estimated basin or
    # introducing an angle-chart singularity (important for poses near 180 degrees).
    candidate = T.copy()
    R = candidate[:3, :3]
    candidate[:3, :3] = 0.5 * R @ (3.0 * np.eye(3) - R.T @ R)

    T0_full = T0
    if _robust_bearing_cost(candidate) > _robust_bearing_cost(T0_full):
        return T0_full
    return candidate


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
    """Compatibility RANSAC pose of ``object_pts`` against ``image_pts``.

    RANSACs a bearing-vector engine (:func:`ds_msp.geometry.resection.ransac_pnp_normalized`
    with ``rays=...``), valid across the full sphere — peripheral rays past 90 deg (``z <= 0``)
    seed and are scored in the model-derived local pixel metric rather than dropped: a DLT for a
    **non-coplanar** target (ADR-0018), a homography for a **coplanar** board (ADR-0019). Falls
    back to the legacy ``cv2`` P3P path on the ``z > 0`` normalized plane only when there are
    too few valid points for the bearing-native minimal sample, where P3P is used.

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
    rays_valid = np.asarray(rays, float)[ok]
    _, J_point, _, projection_valid = model.project_jacobian(rays_valid)
    whiteners, metric_valid = projection_bearing_whiteners(
        rays_valid, J_point, projection_valid
    )
    metric_idx = valid_idx[metric_valid]
    coplanar_target = metric_idx.size >= 4 and _is_coplanar(object_pts[metric_idx])

    # Bearing-vector RANSAC over the full valid ray set (any z sign) -- DLT or homography.
    if metric_idx.size >= (4 if coplanar_target else 6):
        Xv_all = object_pts[metric_idx]
        rays_v = rays_valid[metric_valid]
        whiteners_v = whiteners[metric_valid]
        with np.errstate(divide="ignore", invalid="ignore"):
            pn_all = rays_v[:, :2] / rays_v[:, 2:3]
        T, inl = ransac_pnp_normalized(Xv_all, pn_all, focal=_focal(model), thresh_px=thresh_px,
                                       max_iters=max_iters, confidence=confidence, seed=seed,
                                       rays=rays_v, whiteners=whiteners_v)
        if T is None or inl.sum() < min_inliers:
            return None, np.zeros(n_all, bool)
        inliers = np.zeros(n_all, bool)
        inliers[metric_idx[inl]] = True
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
