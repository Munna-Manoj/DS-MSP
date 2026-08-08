"""
Pose estimation that works on ANY camera model.

Depends only on the CameraModel contract (project/unproject) + OpenCV — never on
a concrete model. Unprojects pixels to bearing rays, then solves PnP.

Coverage by target geometry
---------------------------
Both target geometries are full-sphere (ADR-0018, ADR-0019): peripheral rays past 90 deg
off-axis (``z <= 0``) are used, not dropped.

- **Non-coplanar** targets (fused multi-board / genuinely-3D scenes) use a bearing-vector DLT
  (:func:`ds_msp.geometry.resection._pose_dlt_bearing`).
- **Coplanar** targets (a single planar board) use a bearing-vector homography
  (:func:`ds_msp.geometry.resection._pose_planar_bearing`), the coplanar analogue.

The plain ``solve_pnp`` keeps the established normalized-plane solve for all-forward data and
switches to bearings when peripheral rays require it. ``solve_pnp_robust`` is the recommended
deterministic GNC-TLS estimator. ``solve_pnp_ransac`` retains classic random minimal-set
consensus for compatibility and explicitly requested experiments.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from ..core.contracts import CameraModel


def _pixel_whitened_bearings(
    model: CameraModel, rays: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-ray local pixel metrics through the camera-model contract."""
    from ..geometry.bearing import projection_bearing_whiteners

    _, J_point, _, projection_valid = model.project_jacobian(np.asarray(rays, float))
    return projection_bearing_whiteners(rays, J_point, projection_valid)


def solve_pnp(model: CameraModel, object_points: np.ndarray, image_points: np.ndarray,
              method: int = cv2.SOLVEPNP_ITERATIVE
              ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """Estimate pose from 3D-2D correspondences for any fisheye/omni model.

    With peripheral rays past 90 deg (``z <= 0`` — which the pinhole normalized-plane solve
    cannot represent), this switches to a bearing-vector solve on the full valid ray set so the
    wide-FOV points are used rather than dropped: a DLT
    (:func:`ds_msp.geometry.resection._pose_dlt_bearing`) for a non-coplanar target, or a
    homography (:func:`ds_msp.geometry.resection._pose_planar_bearing`) for a coplanar one
    (ADR-0018, ADR-0019). Otherwise (all rays forward, or too few points for the bearing-native
    minimal sample) it keeps the ``cv2.solvePnP`` path on the ``z > 0`` normalized plane.

    Parameters
    ----------
    model : CameraModel
        Any model implementing the contract.
    object_points : (N, 3) world points.
    image_points : (N, 2) distorted pixels.

    Returns
    -------
    tuple
        ``(success, rvec, tvec)`` with squeezed ``(3,)`` vectors, or ``(False, None, None)``
        if fewer than 4 usable points remain or the solve fails.
    """
    from ..geometry.resection import _is_coplanar, _pose_dlt_bearing, _pose_planar_bearing

    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)

    rays, valid = model.unproject(image_points)
    valid = np.asarray(valid).ravel().astype(bool)
    Xv = object_points[valid]
    rays_v = rays[valid]
    if len(Xv) < 4:
        return False, None, None

    # Peripheral (z<=0) rays: the pinhole normalized-plane solve cannot represent them — solve
    # on bearing vectors directly across the full sphere.
    if np.any(rays_v[:, 2] <= 1e-6):
        coplanar = _is_coplanar(Xv)
        sol = (_pose_planar_bearing(Xv, rays_v) if coplanar else
               _pose_dlt_bearing(Xv, rays_v) if len(Xv) >= 6 else None)
        if sol is not None:
            R, t = sol
            rvec = cv2.Rodrigues(np.ascontiguousarray(R))[0].squeeze()
            return True, rvec, t
        if coplanar:
            return False, None, None
        # non-coplanar with < 6 points: fall through to the legacy z>0-only solve below.

    usable = rays_v[:, 2] > 1e-6
    Xv, rays_v = Xv[usable], rays_v[usable]
    if len(Xv) < 4:
        return False, None, None
    pts_norm = rays_v[:, :2] / rays_v[:, 2:3]
    success, rvec, tvec = cv2.solvePnP(
        Xv, pts_norm.astype(np.float64),
        np.eye(3, dtype=np.float64), np.zeros(5, dtype=np.float64), flags=method)
    if not success:
        return False, None, None
    return True, rvec.squeeze(), tvec.squeeze()


def solve_pnp_robust(model: CameraModel, object_points: np.ndarray, image_points: np.ndarray,
                     *, noise_bound_px: float = 3.0, max_iters: int = 100,
                     refine: bool = True
                     ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """Deterministic, high-breakdown PnP for any central camera model.

    Pixels are unprojected once through the model and every subsequent optimization operation
    is performed on unit bearings. The model's analytic ``project_jacobian`` calibrates a local
    per-ray pixel metric, so ``noise_bound_px`` stays meaningful at the image periphery instead
    of relying on a near-axis scalar-focal approximation. A non-coplanar target starts from the
    full-sphere bearing DLT; a coplanar target starts from the bearing homography. GNC-TLS then
    uses all correspondences in a deterministic graduated solve and returns a hard inlier mask
    without random minimal-set sampling. The optional final consensus polish uses the same
    signed chordal bearing residual and cannot reduce support or worsen the truncated full-data
    score.

    This is the recommended robust PnP entry point. :func:`solve_pnp_ransac` remains available
    as an explicitly named compatibility API for callers that require classic RANSAC behavior.

    Parameters
    ----------
    model : CameraModel
        Any central model implementing ``unproject`` and analytic ``project_jacobian``.
    object_points : (N, 3) ndarray
        World/object-frame points.
    image_points : (N, 2) ndarray
        Corresponding distorted pixels.
    noise_bound_px : float, default 3.0
        Hard TLS inlier bound in the model-derived local pixel metric. Set this from the
        expected feature noise/gating policy, not from the contaminated sample median.
    max_iters : int, default 100
        Maximum GNC graduation levels.
    refine : bool, default True
        Run a final guarded least-squares polish on the hard GNC consensus.

    Returns
    -------
    tuple
        ``(success, rvec, tvec, inliers)`` with an ``(N,)`` mask. Invalid model observations
        are always ``False``. At least four coplanar or six non-coplanar valid bearings are
        required.
    """
    from ..geometry.resection import gnc_pnp_bearings

    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    n = len(object_points)
    rays, valid = model.unproject(image_points)
    valid = np.asarray(valid).ravel().astype(bool)
    idx = np.flatnonzero(valid)
    rays_valid = np.asarray(rays, float)[valid]
    whiteners, metric_valid = _pixel_whitened_bearings(model, rays_valid)
    idx = idx[metric_valid]
    T, inliers_valid = gnc_pnp_bearings(
        object_points[valid][metric_valid], rays_valid[metric_valid],
        noise_bound_px=noise_bound_px, max_outer=max_iters, refine=refine,
        whiteners=whiteners[metric_valid],
    )
    mask = np.zeros(n, bool)
    mask[idx[inliers_valid]] = True
    if T is None:
        return False, None, None, mask
    rvec = cv2.Rodrigues(np.ascontiguousarray(T[:3, :3]))[0].squeeze()
    return True, rvec, T[:3, 3].copy(), mask


def solve_pnp_ransac(model: CameraModel, object_points: np.ndarray, image_points: np.ndarray,
                     *, thresh_px: float = 3.0, max_iters: int = 300,
                     confidence: float = 0.999, seed: int = 0, refine: bool = True
                     ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """Outlier-robust PnP for any camera model.

    Unlike :func:`solve_pnp` (a single non-robust solve), this rejects gross-outlier
    correspondences with RANSAC before estimating the pose, so a handful of mis-detected /
    mismatched points don't drag the result. The pixels are unprojected through the model
    (the same step :func:`solve_pnp` uses), then a RANSAC pose is fit
    (:func:`ds_msp.geometry.resection.ransac_pnp_normalized`):

    * a **non-coplanar** target uses the **bearing-vector DLT** engine (ADR-0018) — full-sphere,
      so peripheral rays past 90 deg (``z <= 0``) both seed and are scored in a model-derived
      local pixel metric; no forward-hemisphere restriction.
    * a **coplanar** board uses the **bearing-vector homography** engine (ADR-0019) — same
      full-sphere coverage, minimal sample 4.

    With ``refine=True``, both bearing paths polish the consensus directly on unit rays with a
    full-sphere chordal residual and manifold LM. The candidate is rescored over every valid
    correspondence and is accepted only when neither support nor the truncated local-pixel bearing
    score degrades; failed or harmful polish leaves the supported RANSAC pose unchanged. The fixed
    metric comes from the camera contract before optimization; no iterative camera projection or
    division by bearing ``z`` occurs in this path. With too few valid
    points for the bearing-native minimal sample, the solver falls back to the legacy
    plane-homography/DLT engine on the ``z > 0`` normalized plane, where ``refine=True`` uses
    the established OpenCV consensus polish on front-facing correspondences only.

    Parameters
    ----------
    object_points : (N, 3) world points.
    image_points : (N, 2) distorted pixels.
    thresh_px : inlier gate in pixels. On the bearing path, the camera contract's analytic
        projection Jacobian constructs a fixed local pixel metric at each observed ray.
    refine : bool, default True
        Polish the RANSAC consensus. The bearing-capable path refines directly on unit rays;
        ``False`` returns the consensus-refitted linear hypothesis.

    Returns
    -------
    tuple
        ``(success, rvec, tvec, inliers)`` where ``inliers`` is an ``(N,)`` boolean mask over
        ``image_points`` (``False`` for rejected outliers or invalid observations). Model-valid
        rays past 90 degrees are retained by the bearing path.
    """
    from ..geometry.resection import (
        _is_coplanar,
        guarded_refine_pose_bearings,
        ransac_pnp_normalized,
    )

    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    n = len(object_points)
    rays, valid = model.unproject(image_points)
    valid = np.asarray(valid).ravel().astype(bool)
    focal = float(np.asarray(model.K, float)[0, 0])

    idx_valid = np.flatnonzero(valid)
    rays_valid = np.asarray(rays, float)[valid]
    whiteners_valid, metric_valid = _pixel_whitened_bearings(model, rays_valid)
    idx_metric = idx_valid[metric_valid]
    X_metric = object_points[valid][metric_valid]
    rays_metric = rays_valid[metric_valid]
    whiteners_metric = whiteners_valid[metric_valid]
    coplanar_target = idx_metric.size >= 4 and _is_coplanar(X_metric)
    bearing = idx_metric.size >= (4 if coplanar_target else 6)

    if bearing:
        # Full-sphere path: keep every valid ray (incl. z<=0); pn is unused by the engine.
        X = X_metric
        rays_v = rays_metric
        with np.errstate(divide="ignore", invalid="ignore"):
            pn = (rays_v[:, :2] / rays_v[:, 2:3]).astype(np.float64)
        sub = idx_metric
        T, inl = ransac_pnp_normalized(X, pn, focal=focal, thresh_px=thresh_px,
                                       max_iters=max_iters, confidence=confidence, seed=seed,
                                       rays=rays_v, whiteners=whiteners_metric)
        support_floor = 4 if coplanar_target else 6
        if T is None or inl.sum() < support_floor:
            return False, None, None, np.zeros(n, bool)
        if refine:
            T, inl = guarded_refine_pose_bearings(
                T, X, rays_v, inl, focal=1.0, threshold=thresh_px,
                whiteners=whiteners_metric,
            )
        rvec = cv2.Rodrigues(np.ascontiguousarray(T[:3, :3]))[0]
        tvec = T[:3, 3].reshape(3, 1)
    else:
        usable = valid & (rays[:, 2] > 1e-6)
        sub = np.flatnonzero(usable)
        if sub.size < 4:
            return False, None, None, np.zeros(n, bool)
        X = object_points[usable]
        pn = (rays[usable, :2] / rays[usable, 2:3]).astype(np.float64)
        T, inl = ransac_pnp_normalized(X, pn, focal=focal, thresh_px=thresh_px,
                                       max_iters=max_iters, confidence=confidence, seed=seed)
        support_floor = 4 if _is_coplanar(X) else 6
        if T is None or inl.sum() < support_floor:
            return False, None, None, np.zeros(n, bool)
        rvec = cv2.Rodrigues(T[:3, :3])[0]
        tvec = T[:3, 3].reshape(3, 1)
        if refine and inl.sum() >= 4:
            ok, rv, tv = cv2.solvePnP(X[inl], pn[inl], np.eye(3, dtype=np.float64),
                                      np.zeros(5, dtype=np.float64), rvec.copy(), tvec.copy(),
                                      useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                rvec, tvec = rv, tv

    mask = np.zeros(n, bool)
    mask[sub[inl]] = True
    return True, rvec.squeeze(), tvec.squeeze(), mask
