"""
Pose estimation that works on ANY camera model.

Depends only on the CameraModel contract (project/unproject) + OpenCV — never on
a concrete model. Unprojects pixels to bearing rays, then solves PnP.

Coverage by target geometry
---------------------------
Both target geometries are full-sphere (ADR-0018, ADR-0019): peripheral rays past 90 deg
off-axis (``z <= 0``) are used, not dropped.

- **Non-coplanar** targets (fused multi-board / genuinely-3D scenes) use a bearing-vector DLT
  (:func:`ds_msp.geometry.resection._pose_dlt_bearing`), with an angular inlier metric in the
  RANSAC path.
- **Coplanar** targets (a single planar board) use a bearing-vector homography
  (:func:`ds_msp.geometry.resection._pose_planar_bearing`), the coplanar analogue.

The plain ``solve_pnp`` keeps the established normalized-plane solve for all-forward data and
switches to bearings when peripheral rays require it. ``solve_pnp_ransac`` uses the bearing
engine whenever its 4-point planar or 6-point non-coplanar sample floor is available; only an
undersized set falls back to usable forward correspondences on the normalized plane.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from ..core.contracts import CameraModel


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
      so peripheral rays past 90 deg (``z <= 0``) both seed and are scored (angular inlier
      metric); no forward-hemisphere restriction.
    * a **coplanar** board uses the **bearing-vector homography** engine (ADR-0019) — same
      full-sphere coverage, minimal sample 4.

    Both bearing paths skip ``refine``: the pinhole ``cv2.solvePnP`` polish cannot represent
    ``z <= 0`` points, and the bearing engine already refits on the consensus set internally.
    With too few valid points for the bearing-native minimal sample, this falls back to the
    legacy plane-homography/DLT engine on the ``z > 0`` normalized plane, then ``refine=True``
    polishes on the consensus set — front-facing correspondences only.

    Parameters
    ----------
    object_points : (N, 3) world points.
    image_points : (N, 2) distorted pixels.
    thresh_px : inlier gate in pixels (converted to an angular tolerance ``thresh_px/focal`` on
        the bearing path — the exact analogue near the optical axis).

    Returns
    -------
    tuple
        ``(success, rvec, tvec, inliers)`` where ``inliers`` is an ``(N,)`` boolean mask over
        ``image_points`` (``False`` for rejected outliers or invalid observations). Model-valid
        rays past 90 degrees are retained by the bearing path.
    """
    from ..geometry.resection import _is_coplanar, ransac_pnp_normalized

    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    n = len(object_points)
    rays, valid = model.unproject(image_points)
    valid = np.asarray(valid).ravel().astype(bool)
    focal = float(np.asarray(model.K, float)[0, 0])

    idx_valid = np.flatnonzero(valid)
    coplanar_target = idx_valid.size >= 4 and _is_coplanar(object_points[valid])
    bearing = idx_valid.size >= (4 if coplanar_target else 6)

    if bearing:
        # Full-sphere path: keep every valid ray (incl. z<=0); pn is unused by the engine.
        X = object_points[valid]
        rays_v = rays[valid]
        with np.errstate(divide="ignore", invalid="ignore"):
            pn = (rays_v[:, :2] / rays_v[:, 2:3]).astype(np.float64)
        sub = idx_valid
        T, inl = ransac_pnp_normalized(X, pn, focal=focal, thresh_px=thresh_px,
                                       max_iters=max_iters, confidence=confidence, seed=seed,
                                       rays=rays_v)
        support_floor = 4 if coplanar_target else 6
        if T is None or inl.sum() < support_floor:
            return False, None, None, np.zeros(n, bool)
        rvec = cv2.Rodrigues(np.ascontiguousarray(T[:3, :3]))[0]
        tvec = T[:3, 3].reshape(3, 1)
        # No cv2 polish: it would drop the z<=0 inliers. The bearing engine already refit on
        # the consensus set.
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
