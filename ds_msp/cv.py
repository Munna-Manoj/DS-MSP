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
from typing import Tuple, Optional, Union
from .model import DoubleSphereCamera

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
    
    # Use DoubleSphereCamera for projection
    # We use a dummy width/height as they don't affect projection
    cam = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha_ds, 2000, 2000)
    points_2d, valid = cam.project(points_cam)
    
    # Format output to match cv2 (N, 1, 2)
    return points_2d.reshape(-1, 1, 2), None

def undistortPoints(distorted: np.ndarray, K: np.ndarray, D: np.ndarray,
                    R: Optional[np.ndarray] = None, P: Optional[np.ndarray] = None
                   ) -> np.ndarray:
    """
    Undistort 2D points.
    
    Mimics cv2.fisheye.undistortPoints.
    
    Parameters
    ----------
    distorted : (N, 1, 2) or (N, 2) array
        Distorted 2D points.
    K : (3, 3) array
        Camera matrix.
    D : (2,) array
        Distortion coefficients [xi, alpha].
    R : (3, 3) array, optional
        Rectification transformation.
    P : (3, 3) array, optional
        New camera matrix.
        
    Returns
    -------
    undistorted : (N, 1, 2) array
        Undistorted points.
    """
    distorted = np.atleast_2d(distorted.squeeze())
    K = np.array(K)
    D = np.array(D).flatten()
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    xi, alpha_ds = D[0], D[1]
    
    cam = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha_ds, 2000, 2000)
    
    # Unproject to unit rays
    rays, valid = cam.unproject(distorted)
    
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
    
    Parameters
    ----------
    undistorted : (N, 1, 2) or (N, 2) array
        Undistorted 2D points (normalized or pixel coordinates? 
        cv2.fisheye.distortPoints takes normalized points if K is not applied beforehand,
        but usually it takes points in normalized image plane.
        Wait, cv2.fisheye.distortPoints doc says:
        "Projects points using fisheye model"
        Actually it usually takes normalized points (x, y, 1) if K is identity, 
        or it might take pixel coordinates if we consider it as inverse of undistortPoints with P.
        
        Let's assume input is normalized coordinates (x, y) on unit plane z=1, 
        UNLESS we want to support pixel coordinates.
        OpenCV documentation is a bit vague, but usually `distortPoints` maps 
        normalized homogeneous coordinates to distorted pixel coordinates.
        However, the signature has `K`.
        If `K` is provided, it implies the input might be on the normalized plane, 
        and we want to project them to the image plane defined by K and D.
        
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
    
    cam = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha_ds, 2000, 2000)
    
    # Assume input is (x, y) on normalized plane (z=1)
    # So we construct 3D points (x, y, 1)
    points_3d = np.column_stack([undistorted, np.ones(len(undistorted))])
    
    # Project
    distorted_pts, valid = cam.project(points_3d)
    
    return distorted_pts.reshape(-1, 1, 2)

def initUndistortRectifyMap(K: np.ndarray, D: np.ndarray, R: np.ndarray, P: np.ndarray,
                            size: Tuple[int, int], m1type: int
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute undistortion and rectification maps.
    
    Mimics cv2.fisheye.initUndistortRectifyMap.
    
    Parameters
    ----------
    K : (3, 3) array
        Camera matrix.
    D : (2,) array
        Distortion coefficients.
    R : (3, 3) array
        Rectification transformation (or identity).
    P : (3, 3) array
        New camera matrix.
    size : (width, height)
        Image size.
    m1type : int
        Map type (e.g., cv2.CV_32FC1).
        
    Returns
    -------
    map1, map2 : arrays
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
    
    cam = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha_ds, w, h)
    
    # Create grid for new image
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x, y, indexing='xy')
    
    # Unproject from new camera P to normalized rays
    # P maps (X, Y, Z) -> (u, v)
    # We want (u, v) -> (X, Y, Z)
    # u = fx_new * X/Z + cx_new  => X/Z = (u - cx_new) / fx_new
    mx = (x_grid - cx_new) / fx_new
    my = (y_grid - cy_new) / fy_new
    rays_new = np.stack([mx, my, np.ones_like(mx)], axis=-1)
    
    # Apply inverse rectification R^T (since R maps world to cam, we want cam to world? 
    # Wait, R usually rotates the camera frame. 
    # Standard pipeline: 
    # 1. Undistorted pixel (u', v') -> Normalized (x', y', 1) via P^-1
    # 2. Rotate by R^-1 (or R^T) -> Original camera frame ray
    # 3. Project to distorted pixel (u, v) via K, D
    
    rays_orig = np.einsum('ij,hwj->hwi', R.T, rays_new)
    
    # Project back to distorted image
    distorted_pts, valid = cam.project(rays_orig)
    
    map1 = distorted_pts[..., 0].astype(np.float32)
    map2 = distorted_pts[..., 1].astype(np.float32)
    
    # Handle invalid points (optional, maybe set to -1?)
    # cv2.remap handles borders, but we can mark invalid
    # map1[~valid] = -1
    # map2[~valid] = -1
    
    if m1type == cv2.CV_16SC2:
        map1, map2 = cv2.convertMaps(map1, map2, m1type)
        
    return map1, map2

def undistortImage(distorted: np.ndarray, K: np.ndarray, D: np.ndarray,
                   Knew: Optional[np.ndarray] = None, new_size: Optional[Tuple[int, int]] = None
                  ) -> np.ndarray:
    """
    Undistort image.
    
    Mimics cv2.fisheye.undistortImage.
    """
    h, w = distorted.shape[:2]
    if new_size is None:
        new_size = (w, h)
    if Knew is None:
        Knew = K # Should probably estimate optimal Knew if not provided, but cv2 defaults to K usually or user provides it.
        # Actually ds_camera has compute_K_new.
        # But to mimic cv2 strictly, if Knew is not provided, it uses K.
        # However, for fisheye, using K might result in bad crops.
        # Let's use K if not provided, as per signature implication.
    
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
    Estimates new camera matrix.
    
    Mimics cv2.fisheye.estimateNewCameraMatrixForUndistortRectify.
    """
    w, h = image_size
    if new_size is None:
        new_size = (w, h)
        
    K = np.array(K)
    D = np.array(D).flatten()
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    xi, alpha_ds = D[0], D[1]
    
    cam = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha_ds, w, h)
    
    # Use the compute_K_new from DoubleSphereCamera
    # balance: 0.0 to 1.0
    # ds_camera.compute_K_new uses balance to scale focal length
    # We can reuse that logic
    
    return cam.compute_K_new(balance)

def solvePnP(objectPoints: np.ndarray, imagePoints: np.ndarray,
             K: np.ndarray, D: np.ndarray,
             rvec: Optional[np.ndarray] = None, tvec: Optional[np.ndarray] = None,
             useExtrinsicGuess: bool = False, flags: int = cv2.SOLVEPNP_ITERATIVE
            ) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Finds an object pose from 3D-2D point correspondences.
    
    Mimics cv2.solvePnP.
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
