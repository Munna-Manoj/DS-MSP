"""
Double Sphere Camera Model - OpenCV-style Wrapper
=================================================

This module provides a functional interface for the Double Sphere camera model
that mimics the `cv2.fisheye` module signatures. This allows for easy integration
into existing OpenCV-based pipelines.

The distortion coefficients `D` are assumed to be `[xi, alpha]`.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from .model import DoubleSphereCamera, ds_project, ds_unproject, balanced_pinhole_K

def projectPoints(objectPoints: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                  K: np.ndarray, D: np.ndarray,
                  alpha: float = 0.0, jacobian: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Project 3D points to image plane.
    
    Mimics cv2.fisheye.projectPoints.
    
    Parameters
    ----------
    objectPoints : (N, 1, 3) or (N, 3) array
        3D points in world coordinates.
    rvec : (3, 1) or (3,) array
        Rotation vector.
    tvec : (3, 1) or (3,) array
        Translation vector.
    K : (3, 3) array
        Camera matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1].
    D : (2,) or (4,) array
        Distortion coefficients [xi, alpha].
    alpha : float, optional
        Skew parameter (not supported, ignored).
    jacobian : optional
        Not supported, always returns None.
        
    Returns
    -------
    imagePoints : (N, 1, 2) array
        Projected 2D points.
    jacobian : None
    """
    objectPoints = np.atleast_2d(objectPoints.squeeze())
    rvec = np.array(rvec).flatten()
    tvec = np.array(tvec).flatten()
    K = np.array(K)
    D = np.array(D).flatten()
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    xi, alpha_ds = D[0], D[1]
    
    # Transform points: P_cam = R * P_world + t
    R, _ = cv2.Rodrigues(rvec)
    points_cam = (R @ objectPoints.T).T + tvec

    u, v, _ = ds_project(points_cam, fx, fy, cx, cy, xi, alpha_ds)
    points_2d = np.stack([u, v], axis=-1)

    # Format output to match cv2 (N, 1, 2)
    return points_2d.reshape(-1, 1, 2), None

def undistortPoints(distorted: np.ndarray, K: np.ndarray, D: np.ndarray,
                    R: Optional[np.ndarray] = None, P: Optional[np.ndarray] = None
                   ) -> np.ndarray:
    """
    Undistort 2D points.

    Mimics ``cv2.fisheye.undistortPoints``: unprojects each distorted pixel to a
    unit bearing ray through the Double Sphere model, optionally rotates it by
    ``R``, then reprojects to the ``z=1`` normalized plane (or through ``P`` if
    given).

    Parameters
    ----------
    distorted : ndarray of shape (N, 1, 2) or (N, 2)
        Distorted 2D pixels in the original image.
    K : ndarray of shape (3, 3)
        Camera matrix ``[[fx,0,cx],[0,fy,cy],[0,0,1]]``.
    D : ndarray of shape (2,)
        Distortion coefficients ``[xi, alpha]``.
    R : ndarray of shape (3, 3), optional
        Rectification rotation applied to the unprojected rays before the
        final projection. Defaults to no rotation.
    P : ndarray of shape (3, 3), optional
        New camera matrix used to re-project the rectified rays to pixels. If
        omitted, points are returned as normalized coordinates ``(x/z, y/z)``
        with no principal point or focal length applied (matches
        ``cv2.fisheye.undistortPoints`` with ``P=None``).

    Returns
    -------
    ndarray of shape (N, 1, 2)
        Rectified points: normalized coordinates if ``P`` is ``None``,
        otherwise pixels in the ``P`` frame. Rows whose ray failed to
        unproject are **not** flagged — this function has no validity output,
        unlike :meth:`ds_msp.ops.undistort.Undistorter.undistort_points`.
    """
    distorted = np.atleast_2d(distorted.squeeze())
    K = np.array(K)
    D = np.array(D).flatten()
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    xi, alpha_ds = D[0], D[1]
    
    # Unproject to unit rays
    rays, valid = ds_unproject(distorted, fx, fy, cx, cy, xi, alpha_ds)

    # Apply rectification if provided
    if R is not None:
        rays = (R @ rays.T).T

    # Normalize rays (z=1)
    rays_norm = rays / (rays[:, 2:3] + 1e-10)
    
    # Project using new camera matrix P or return normalized coordinates
    if P is not None:
        fx_new, fy_new = P[0, 0], P[1, 1]
        cx_new, cy_new = P[0, 2], P[1, 2]
        u = fx_new * rays_norm[:, 0] + cx_new
        v = fy_new * rays_norm[:, 1] + cy_new
        undistorted = np.stack([u, v], axis=-1)
    else:
        # Return normalized coordinates (x, y) where z=1
        undistorted = rays_norm[:, :2]
        
    return undistorted.reshape(-1, 1, 2)

def distortPoints(undistorted: np.ndarray, K: np.ndarray, D: np.ndarray,
                  alpha: float = 0.0
                 ) -> np.ndarray:
    """
    Distort 2D points.
    
    Mimics cv2.fisheye.distortPoints.
    
    Input points are interpreted as normalized image-plane coordinates (x, y)
    on the z = 1 plane, matching the convention of `undistortPoints` with no
    `P` matrix. They are lifted to 3D rays (x, y, 1) and projected through the
    Double Sphere model defined by K and D.

    Parameters
    ----------
    undistorted : (N, 1, 2) or (N, 2) array
        Undistorted points on the normalized plane (z = 1).

    Returns
    -------
    distorted : (N, 1, 2) array
    """
    undistorted = np.atleast_2d(undistorted.squeeze())
    K = np.array(K)
    D = np.array(D).flatten()

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    xi, alpha_ds = D[0], D[1]

    # Lift normalized points to 3D rays (x, y, 1) and project.
    points_3d = np.column_stack([undistorted, np.ones(len(undistorted))])
    u, v, _ = ds_project(points_3d, fx, fy, cx, cy, xi, alpha_ds)
    distorted_pts = np.stack([u, v], axis=-1)

    return distorted_pts.reshape(-1, 1, 2)

def initUndistortRectifyMap(K: np.ndarray, D: np.ndarray, R: np.ndarray, P: np.ndarray,
                            size: Tuple[int, int], m1type: int
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute undistortion and rectification maps.

    Mimics ``cv2.fisheye.initUndistortRectifyMap``: for every destination pixel,
    lifts it to a normalized ray via ``P``, rotates by ``R.T`` into the original
    camera frame, then projects through the Double Sphere model (``K``, ``D``) to
    find the source pixel — the standard backward-remap pipeline.

    Parameters
    ----------
    K : ndarray of shape (3, 3)
        Camera matrix of the original (distorted) image.
    D : ndarray of shape (2,)
        Distortion coefficients ``[xi, alpha]``.
    R : ndarray of shape (3, 3)
        Rectification rotation (pass ``np.eye(3)`` for none).
    P : ndarray of shape (3, 3)
        New camera matrix for the destination (undistorted) image.
    size : tuple of (int, int)
        Destination image ``(width, height)``, pixels.
    m1type : int
        Requested map type. Only ``cv2.CV_16SC2`` is honored (converted via
        ``cv2.convertMaps``); any other value (including ``cv2.CV_32FC1``)
        returns the maps as separate ``float32`` arrays regardless of the
        requested type — unlike real OpenCV, this does not support a combined
        fixed-point ``(map1, map2)`` pair for other type codes.

    Returns
    -------
    map1 : ndarray of shape (height, width)
        Source-image ``u`` (or fixed-point packed ``(u, v)`` if
        ``m1type == cv2.CV_16SC2``) for each destination pixel.
    map2 : ndarray of shape (height, width)
        Source-image ``v`` for each destination pixel, or the ``CV_16SC2``
        interpolation-weight map when ``m1type == cv2.CV_16SC2``.
    """
    w, h = size
    K = np.array(K)
    D = np.array(D).flatten()
    P = np.array(P)
    R = np.array(R) if R is not None else np.eye(3)
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    xi, alpha_ds = D[0], D[1]
    
    fx_new, fy_new = P[0, 0], P[1, 1]
    cx_new, cy_new = P[0, 2], P[1, 2]

    # Build the destination (undistorted) pixel grid and back-map each pixel
    # to a source location in the distorted image, the standard remap pipeline:
    #   1. undistorted pixel (u', v') -> normalized ray via P^-1
    #   2. rotate by R^T into the original camera frame
    #   3. project to the distorted pixel (u, v) via K, D
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x, y, indexing='xy')

    mx = (x_grid - cx_new) / fx_new
    my = (y_grid - cy_new) / fy_new
    rays_new = np.stack([mx, my, np.ones_like(mx)], axis=-1)

    rays_orig = np.einsum('ij,hwj->hwi', R.T, rays_new)

    u, v, _ = ds_project(rays_orig, fx, fy, cx, cy, xi, alpha_ds)
    map1 = u.astype(np.float32)
    map2 = v.astype(np.float32)

    if m1type == cv2.CV_16SC2:
        map1, map2 = cv2.convertMaps(map1, map2, m1type)
        
    return map1, map2

def undistortImage(distorted: np.ndarray, K: np.ndarray, D: np.ndarray,
                   Knew: Optional[np.ndarray] = None, new_size: Optional[Tuple[int, int]] = None
                  ) -> np.ndarray:
    """
    Undistort a fisheye image into a rectified pinhole view.

    Mimics ``cv2.fisheye.undistortImage``: builds the backward-remap maps with
    :func:`initUndistortRectifyMap` and applies them with ``cv2.remap``. See
    the [Undistort a fisheye image](../how-to/undistort_images.md) how-to for a
    worked example including the ``balance`` field-of-view trade-off.

    Parameters
    ----------
    distorted : ndarray of shape (H, W, ...)
        Source (distorted) image.
    K : ndarray of shape (3, 3)
        Camera matrix of ``distorted``.
    D : ndarray of shape (2,)
        Distortion coefficients ``[xi, alpha]``.
    Knew : ndarray of shape (3, 3), optional
        New camera matrix for the output image. Defaults to ``K`` itself if
        omitted — unlike a well-chosen balanced matrix, this typically leaves
        heavy cropping or black borders on a wide fisheye; prefer passing a
        matrix from :func:`estimateNewCameraMatrixForUndistortRectify`.
    new_size : tuple of (int, int), optional
        Output image ``(width, height)``. Defaults to ``distorted``'s own size.

    Returns
    -------
    ndarray of shape (new_size[1], new_size[0], ...)
        The rectified image. Pixels with no valid source ray are filled per
        ``cv2.remap``'s default border handling.

    Raises
    ------
    AttributeError
        If ``distorted`` is not a real array (e.g. ``None`` from a failed
        ``cv2.imread``) — ``distorted.shape`` fails before any remap math runs.
    """
    h, w = distorted.shape[:2]
    if new_size is None:
        new_size = (w, h)
    if Knew is None:
        # Mirror cv2.fisheye.undistortImage: fall back to the input K when no
        # new matrix is given. For a wide fisheye, prefer passing a Knew from
        # estimateNewCameraMatrixForUndistortRectify to avoid heavy cropping.
        Knew = K

    map1, map2 = initUndistortRectifyMap(K, D, np.eye(3), Knew, new_size, cv2.CV_32FC1)
    return cv2.remap(distorted, map1, map2, cv2.INTER_LINEAR)

def estimateNewCameraMatrixForUndistortRectify(K: np.ndarray, D: np.ndarray,
                                               image_size: Tuple[int, int],
                                               R: Optional[np.ndarray] = None,
                                               balance: float = 0.0,
                                               new_size: Optional[Tuple[int, int]] = None,
                                               fov_scale: float = 1.0
                                              ) -> np.ndarray:
    """
    Estimate a balanced pinhole camera matrix for undistortion/rectification.

    Mimics ``cv2.fisheye.estimateNewCameraMatrixForUndistortRectify``'s call
    signature, but the underlying rule is this library's own
    :func:`ds_msp.core.pinhole.balanced_pinhole_K`: the new focal length is
    ``(fx + fy) / 2 * (0.4 + 0.4 * balance)`` — ``balance=0.0`` gives ``0.4x``
    the average input focal (widest field of view, most black border) and
    ``balance=1.0`` gives ``0.8x`` (tightest crop, no border). The principal
    point is placed at the output image center. ``R`` and ``fov_scale`` are
    accepted for signature compatibility but are **ignored** — unlike real
    OpenCV, this does not support a rectification rotation or an explicit FOV
    scale.

    Parameters
    ----------
    K : ndarray of shape (3, 3)
        Camera matrix of the original (distorted) image.
    D : ndarray of shape (2,)
        Distortion coefficients. Unused by this implementation (the new focal
        is derived from ``K`` alone), accepted for signature compatibility.
    image_size : tuple of (int, int)
        Original image ``(width, height)``. Only used to default ``new_size``.
    R : ndarray of shape (3, 3), optional
        Ignored (accepted for signature compatibility with ``cv2.fisheye``).
    balance : float, default=0.0
        Field-of-view/border trade-off in ``[0, 1]``; see above.
    new_size : tuple of (int, int), optional
        Output image ``(width, height)``. Defaults to ``image_size``.
    fov_scale : float, default=1.0
        Ignored (accepted for signature compatibility with ``cv2.fisheye``).

    Returns
    -------
    ndarray of shape (3, 3)
        The new pinhole camera matrix, ``fx_new == fy_new``.
    """
    w, h = image_size
    if new_size is None:
        new_size = (w, h)

    K = np.array(K)
    out_w, out_h = new_size
    fx, fy = K[0, 0], K[1, 1]

    return balanced_pinhole_K(fx, fy, out_w, out_h, balance)

def solvePnP(objectPoints: np.ndarray, imagePoints: np.ndarray,
             K: np.ndarray, D: np.ndarray,
             rvec: Optional[np.ndarray] = None, tvec: Optional[np.ndarray] = None,
             useExtrinsicGuess: bool = False, flags: int = cv2.SOLVEPNP_ITERATIVE
            ) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Find an object pose from 3D-2D point correspondences on a fisheye image.

    Mimics ``cv2.solvePnP``'s call signature: builds a
    :class:`~ds_msp.model.DoubleSphereCamera` from ``K``/``D`` and delegates to
    :meth:`~ds_msp.model.DoubleSphereCamera.solve_pnp`, which unprojects to
    bearing rays and solves PnP in the normalized plane (see
    [Solve PnP on a fisheye](../how-to/solve_pnp_on_fisheye.md)). ``rvec``,
    ``tvec``, and ``useExtrinsicGuess`` are accepted for signature
    compatibility but **ignored** — this implementation always solves from
    scratch and cannot warm-start from an extrinsic guess.

    Parameters
    ----------
    objectPoints : ndarray of shape (N, 3) or (N, 1, 3)
        3D points in world/object coordinates.
    imagePoints : ndarray of shape (N, 2) or (N, 1, 2)
        Corresponding distorted fisheye pixels.
    K : ndarray of shape (3, 3)
        Camera matrix.
    D : ndarray of shape (2,)
        Distortion coefficients ``[xi, alpha]``.
    rvec, tvec : ndarray, optional
        Ignored (accepted for signature compatibility with ``cv2.solvePnP``).
    useExtrinsicGuess : bool, default=False
        Ignored (accepted for signature compatibility with ``cv2.solvePnP``).
    flags : int, default=cv2.SOLVEPNP_ITERATIVE
        OpenCV PnP method passed through to the underlying solve.

    Returns
    -------
    success : bool
        Whether the solve succeeded. ``False`` if fewer than 4 correspondences
        survive the front-facing (``z > 1e-6``) unprojection filter, or the
        underlying ``cv2.solvePnP`` call fails.
    rvec : ndarray of shape (3, 1)
        Rodrigues rotation vector. Zeros if ``success`` is ``False``.
    tvec : ndarray of shape (3, 1)
        Translation vector, same units as ``objectPoints``. Zeros if
        ``success`` is ``False``.
    """
    objectPoints = np.atleast_2d(objectPoints.squeeze())
    imagePoints = np.atleast_2d(imagePoints.squeeze())
    K = np.array(K)
    D = np.array(D).flatten()
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    xi, alpha_ds = D[0], D[1]
    
    # Use dummy size, doesn't affect PnP
    cam = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha_ds, 2000, 2000)
    
    success, r, t = cam.solve_pnp(objectPoints, imagePoints, method=flags)
    
    if not success:
        return False, np.zeros((3, 1)), np.zeros((3, 1))
        
    return True, r.reshape(3, 1), t.reshape(3, 1)
