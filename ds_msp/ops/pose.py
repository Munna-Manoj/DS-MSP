"""
Pose estimation that works on ANY camera model.

Depends only on the CameraModel contract (project/unproject) + OpenCV — never on
a concrete model. Unprojects to bearing rays, keeps the front-facing valid ones,
and solves PnP in the normalized plane.
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
        if fewer than 4 points survive the front-facing filter or the solve fails.
    """
    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)

    rays, valid = model.unproject(image_points)
    usable = valid & (rays[:, 2] > 1e-6)
    if not usable.all():
        object_points = object_points[usable]
        rays = rays[usable]
        if len(object_points) < 4:
            return False, None, None

    pts_norm = rays[:, :2] / rays[:, 2:3]
    success, rvec, tvec = cv2.solvePnP(
        object_points, pts_norm.astype(np.float64),
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
    (the same step :func:`solve_pnp` uses), then a RANSAC pose is fit on the normalized plane
    (:func:`ds_msp.geometry.resection.ransac_pnp_normalized` — a plane homography for a coplanar
    board, a 3x4 DLT otherwise); with ``refine=True`` the pose is polished on the consensus set.

    Parameters
    ----------
    object_points : (N, 3) world points.
    image_points : (N, 2) distorted pixels.
    thresh_px : inlier reprojection gate in pixels.

    Returns
    -------
    tuple
        ``(success, rvec, tvec, inliers)`` where ``inliers`` is an ``(N,)`` boolean mask over
        ``image_points`` (``False`` for points dropped as outliers or as non-front-facing
        rays). Like any plane-based PnP it uses front-facing (``z > 0``) correspondences only.
    """
    from ..geometry.resection import ransac_pnp_normalized          # NC robust engine

    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    n = len(object_points)
    rays, valid = model.unproject(image_points)
    usable = np.asarray(valid).ravel().astype(bool) & (rays[:, 2] > 1e-6)
    idx = np.flatnonzero(usable)
    if idx.size < 4:
        return False, None, None, np.zeros(n, bool)

    X = object_points[usable]
    pn = (rays[usable, :2] / rays[usable, 2:3]).astype(np.float64)
    focal = float(np.asarray(model.K, float)[0, 0])
    T, inl = ransac_pnp_normalized(X, pn, focal=focal, thresh_px=thresh_px,
                                   max_iters=max_iters, confidence=confidence, seed=seed)
    if T is None:
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
    mask[idx[inl]] = True
    return True, rvec.squeeze(), tvec.squeeze(), mask
