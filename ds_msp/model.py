"""
Double Sphere Camera Model - Clean Implementation
==================================================

A minimal, production-ready implementation of the Double Sphere camera model
for fisheye cameras. Easy to understand, test, and integrate into other projects.

Author: Advanced 3D Vision
License: MIT
"""

import numpy as np
import cv2
import json
from typing import Tuple, Optional


class DoubleSphereCamera:
    """
    Double Sphere Camera Model (Usenko et al., 2018)
    
    A camera model for wide-angle and fisheye lenses with closed-form
    unprojection, making it ideal for real-time applications and PnP.
    
    Parameters
    ----------
    fx, fy : float
        Focal lengths in pixels
    cx, cy : float
        Principal point coordinates
    xi, alpha : float
        Double Sphere distortion parameters
    width, height : int
        Image dimensions in pixels
    
    Examples
    --------
    >>> # Create camera from calibration
    >>> cam = DoubleSphereCamera(
    ...     fx=711.57, fy=711.24, cx=949.18, cy=518.81,
    ...     xi=0.183, alpha=0.809, width=1920, height=1080
    ... )
    >>> 
    >>> # Undistort image
    >>> img_undist, K_new = cam.undistort_image(img)
    >>> 
    >>> # Solve PnP
    >>> success, rvec, tvec = cam.solve_pnp(points_3d, points_2d)
    """
    
    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 xi: float, alpha: float,
                 width: Optional[int] = None, height: Optional[int] = None,
                 is_flip: bool = False):
        # The projection model needs only the 6 intrinsics. `width`/`height` are
        # used solely by the image-level helpers (undistortion maps, K_new); they
        # are optional so the model can be built for pure project/unproject/PnP
        # without inventing meaningless image dimensions.
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.xi = xi
        self.alpha = alpha
        self.width = width
        self.height = height
        self.is_flip = is_flip

        # Cache for undistortion
        self._mapx = None
        self._mapy = None
        self._K_new = None

    def _require_dims(self, what: str) -> None:
        if self.width is None or self.height is None:
            raise ValueError(
                f"{what} requires image dimensions; construct with "
                f"width=... and height=... (only needed for image-level ops)."
            )

    @property
    def K(self) -> np.ndarray:
        """Pinhole intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    @property
    def D(self) -> np.ndarray:
        """Double Sphere distortion coefficients [xi, alpha]."""
        return np.array([self.xi, self.alpha], dtype=np.float64)

    def __repr__(self) -> str:
        dims = f", width={self.width}, height={self.height}" if self.width else ""
        return (f"DoubleSphereCamera(fx={self.fx:.3f}, fy={self.fy:.3f}, "
                f"cx={self.cx:.3f}, cy={self.cy:.3f}, xi={self.xi:.4f}, "
                f"alpha={self.alpha:.4f}{dims})")
    
    @classmethod
    def from_json(cls, json_path: str):
        """Load camera from calibration JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 1. Top-level calibration output format
        if 'fx' in data and 'fy' in data and 'cx' in data and 'cy' in data:
            fx = data['fx']
            fy = data['fy']
            cx = data['cx']
            cy = data['cy']
            xi = data.get('xi', 0.0)
            alpha = data.get('alpha', 0.5)
            width = data.get('image_width', data.get('width', 1920))
            height = data.get('image_height', data.get('height', 1080))
            is_flip = data.get('is_flip', False)
            return cls(fx, fy, cx, cy, xi, alpha, width, height, is_flip)
            
        # 2. Nested intrinsics format
        elif 'intrinsics' in data:
            intrinsic = data['intrinsics']
            width = data.get('image_width', 640)
            height = data.get('image_height', 480)
            return cls(
                fx=intrinsic['fx'], fy=intrinsic['fy'],
                cx=intrinsic['cx'], cy=intrinsic['cy'],
                xi=intrinsic['xi'], alpha=intrinsic['alpha'],
                width=width, height=height
            )
            
        # 3. Third-party nested resolution format
        else:
            try:
                cam_data = list(data.values())[0]
                intrinsic = cam_data['intrinsics'][0]['intrinsics']
                resolution = cam_data['resolution'][0]
                width, height = resolution[0], resolution[1]
                return cls(
                    fx=intrinsic['fx'], fy=intrinsic['fy'],
                    cx=intrinsic['cx'], cy=intrinsic['cy'],
                    xi=intrinsic['xi'], alpha=intrinsic['alpha'],
                    width=width, height=height
                )
            except (KeyError, IndexError, TypeError) as e:
                raise ValueError(f"Unsupported calibration JSON format: {e}")
    
    # ========================================================================
    # Core Projection/Unprojection
    # ========================================================================

    def project(self, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points to 2D pixel coordinates.
        """
        u, v, valid = ds_project(points_3d, self.fx, self.fy, self.cx, self.cy, self.xi, self.alpha)
        
        # Flip x-coordinates if driver provides flipped images
        if self.is_flip:
            u = (self.width - 1) - u
        
        return np.stack([u, v], axis=-1), valid

    def project_jacobian(
        self, points_3d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Project points with the analytic point/intrinsic Jacobians.

        This keeps the legacy facade compatible with model-agnostic services that consume the
        modern :class:`~ds_msp.core.contracts.CameraModel` derivative contract.
        """
        u, v, J_point, J_intr, valid = ds_project_jacobian(
            np.asarray(points_3d, dtype=np.float64),
            self.fx, self.fy, self.cx, self.cy, self.xi, self.alpha,
        )
        if self.is_flip:
            self._require_dims("flipped projection")
            u = (self.width - 1) - u
            J_point = J_point.copy()
            J_intr = J_intr.copy()
            J_point[..., 0, :] *= -1.0
            J_intr[..., 0, :] *= -1.0
        return np.stack([u, v], axis=-1), J_point, J_intr, valid
    
    def unproject(self, points_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unproject 2D pixels to 3D unit rays (closed-form).
        """
        u, v = points_2d[..., 0], points_2d[..., 1]
        
        # Flip x-coordinates if driver provides flipped images
        if self.is_flip:
            u = (self.width - 1) - u
            
        return ds_unproject(np.stack([u, v], axis=-1), self.fx, self.fy, self.cx, self.cy, self.xi, self.alpha)



    
    # ========================================================================
    # Image Undistortion
    # ========================================================================
    
    def compute_K_new(self, balance: float = 0.5) -> np.ndarray:
        """
        Compute optimal K matrix for undistorted image.
        
        Parameters
        ----------
        balance : float
            0.0 = more FOV (40% of original focal length)
            0.5 = balanced (60% of original) - default
            1.0 = less FOV (80% of original)
            
        Returns
        -------
        K_new : (3, 3) array
            New intrinsic matrix
        """
        self._require_dims("compute_K_new")
        return balanced_pinhole_K(self.fx, self.fy, self.width, self.height, balance)
    
    def get_undistortion_maps(self, K_new: Optional[np.ndarray] = None
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate undistortion maps for cv2.remap.
        
        Parameters
        ----------
        K_new : (3, 3) array, optional
            New intrinsic matrix. If None, computed automatically.
            
        Returns
        -------
        mapx, mapy : (H, W) arrays
            Maps for cv2.remap
        K_new : (3, 3) array
            Intrinsic matrix used
        """
        self._require_dims("get_undistortion_maps")
        if K_new is None:
            K_new = self.compute_K_new()

        if self._mapx is not None and self._K_new is not None:
            if np.array_equal(K_new, self._K_new):
                return self._mapx, self._mapy, self._K_new
        
        fx_new, fy_new = K_new[0, 0], K_new[1, 1]
        cx_new, cy_new = K_new[0, 2], K_new[1, 2]
        
        # Create undistorted pixel grid
        x = np.arange(self.width, dtype=np.float32)
        y = np.arange(self.height, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(x, y, indexing='xy')
        
        # Convert to normalized coordinates and create rays
        mx = (x_grid - cx_new) / fx_new
        my = (y_grid - cy_new) / fy_new
        rays = np.stack([mx, my, np.ones_like(mx)], axis=-1)
        
        # Project back to distorted image
        distorted_pts, valid = self.project(rays)
        
        mapx = distorted_pts[..., 0].astype(np.float32)
        mapy = distorted_pts[..., 1].astype(np.float32)
        mapx[~valid] = -1
        mapy[~valid] = -1
        
        self._mapx, self._mapy, self._K_new = mapx, mapy, K_new
        
        return mapx, mapy, K_new
    
    def undistort_image(self, img: np.ndarray, K_new: Optional[np.ndarray] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Undistort fisheye image to pinhole.
        
        Parameters
        ----------
        img : (H, W, C) or (H, W) array
            Input distorted image
        K_new : (3, 3) array, optional
            New intrinsic matrix
            
        Returns
        -------
        img_undist : array
            Undistorted image
        K_new : (3, 3) array
            Intrinsic matrix for undistorted image
        """
        mapx, mapy, K_new = self.get_undistortion_maps(K_new)
        img_undist = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return img_undist, K_new
    
    # ========================================================================
    # Keypoint Transformation
    # ========================================================================
    
    def undistort_points(self, points_dist: np.ndarray, K_new: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform keypoints from distorted to undistorted space."""
        rays, valid = self.unproject(points_dist)
        rays_norm = rays / (rays[:, 2:3] + 1e-10)
        
        u = K_new[0, 0] * rays_norm[:, 0] + K_new[0, 2]
        v = K_new[1, 1] * rays_norm[:, 1] + K_new[1, 2]
        
        return np.column_stack([u, v]), valid
    
    def distort_points(self, points_undist: np.ndarray, K_new: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform keypoints from undistorted to distorted space."""
        mx = (points_undist[:, 0] - K_new[0, 2]) / K_new[0, 0]
        my = (points_undist[:, 1] - K_new[1, 2]) / K_new[1, 1]
        rays = np.column_stack([mx, my, np.ones(len(mx))])
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        
        return self.project(rays)
    
    # ========================================================================
    # PnP Pose Estimation
    # ========================================================================
    
    def solve_pnp(self, points_3d: np.ndarray, points_2d: np.ndarray,
                  method: int = cv2.SOLVEPNP_ITERATIVE
                 ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Solve PnP for fisheye camera.
        
        This method handles fisheye distortion by unprojecting to rays and selecting a
        target-geometry-aware PnP solver.
        
        Parameters
        ----------
        points_3d : (N, 3) array
            3D points in world coordinates
        points_2d : (N, 2) array
            2D keypoints in distorted image
        method : int
            OpenCV PnP method used by the forward normalized-plane fallback (for example,
            ``cv2.SOLVEPNP_ITERATIVE``). Bearing DLT/homography paths do not use this flag.
            
        Returns
        -------
        success : bool
        rvec : (3,) array or None
            Rotation vector
        tvec : (3,) array or None
            Translation vector

        Notes
        -----
        Rays past 90 degrees off-axis (``z <= 0``) are solved directly on bearing vectors:
        a DLT for non-coplanar targets (ADR-0018) or a bearing homography for coplanar
        targets such as a single board (ADR-0019). Forward-only data keeps the established
        normalized-plane path. See :func:`ds_msp.ops.pose.solve_pnp`.
        """
        from .ops import solve_pnp as _solve_pnp
        return _solve_pnp(self, points_3d, points_2d, method=method)

    def solve_pnp_robust(self, points_3d: np.ndarray, points_2d: np.ndarray,
                         *, noise_bound_px: float = 3.0, max_iters: int = 100,
                         refine: bool = True
                         ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """Deterministic high-breakdown PnP using GNC-TLS directly on unit bearings.

        Returns ``(success, rvec, tvec, inliers)`` with an ``(N,)`` boolean mask. This is the
        recommended robust estimator; the analytic projection Jacobian supplies a fixed local
        pixel metric at every observed ray. :meth:`solve_pnp_ransac` remains available when
        classic random minimal-set consensus is explicitly required. See
        :func:`ds_msp.ops.solve_pnp_robust`.
        """
        from .ops import solve_pnp_robust as _robust
        return _robust(
            self, points_3d, points_2d, noise_bound_px=noise_bound_px,
            max_iters=max_iters, refine=refine,
        )

    def solve_pnp_ransac(self, points_3d: np.ndarray, points_2d: np.ndarray,
                         *, thresh_px: float = 3.0, max_iters: int = 300,
                         confidence: float = 0.999, seed: int = 0, refine: bool = True
                         ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """Outlier-robust PnP — rejects gross-outlier correspondences with RANSAC before
        solving, then refines the pose on the consensus set. Returns
        ``(success, rvec, tvec, inliers)`` with an ``(N,)`` boolean inlier mask. See
        :func:`ds_msp.ops.solve_pnp_ransac`. Use the plain :meth:`solve_pnp` when the
        correspondences are already clean."""
        from .ops import solve_pnp_ransac as _ransac
        return _ransac(self, points_3d, points_2d, thresh_px=thresh_px, max_iters=max_iters,
                       confidence=confidence, seed=seed, refine=refine)

    # ========================================================================
    # Visualization
    # ========================================================================
    
    def draw_axes(self, img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                  axis_length: float = 0.1, K: Optional[np.ndarray] = None
                 ) -> np.ndarray:
        """
        Draw 3D coordinate axes on image.
        
        Parameters
        ----------
        img : array
            Image to draw on
        rvec, tvec : (3,) arrays
            Pose (rotation and translation vectors)
        axis_length : float
            Length of axes in meters
        K : (3, 3) array, optional
            If None, draws on distorted image. Otherwise, draws on undistorted.
            
        Returns
        -------
        img_out : array
            Image with drawn axes
        """
        img_out = img.copy()
        
        # Define axes in 3D
        axes_3d = np.array([
            [0, 0, 0],
            [axis_length, 0, 0],  # X: Red
            [0, axis_length, 0],  # Y: Green
            [0, 0, axis_length]   # Z: Blue
        ])
        
        # Transform to camera coordinates
        R, _ = cv2.Rodrigues(rvec)
        axes_cam = (R @ axes_3d.T).T + tvec
        
        # Project
        if K is None:
            # Distorted image
            axes_2d, valid = self.project(axes_cam)
        else:
            # Undistorted image
            axes_2d_hom = (K @ axes_cam.T).T
            axes_2d = axes_2d_hom[:, :2] / axes_2d_hom[:, 2:3]
            valid = axes_cam[:, 2] > 0
        
        if not valid.all():
            return img_out
        
        axes_2d_int = axes_2d.astype(np.int32)
        origin = tuple(axes_2d_int[0])
        
        # Draw axes
        cv2.arrowedLine(img_out, origin, tuple(axes_2d_int[1]), (0, 0, 255), 3, tipLength=0.3)  # X: Red
        cv2.arrowedLine(img_out, origin, tuple(axes_2d_int[2]), (0, 255, 0), 3, tipLength=0.3)  # Y: Green
        cv2.arrowedLine(img_out, origin, tuple(axes_2d_int[3]), (255, 0, 0), 3, tipLength=0.3)  # Z: Blue
        cv2.circle(img_out, origin, 5, (255, 255, 255), -1)
        
        return img_out


# ============================================================================
# Convenience Functions
# ============================================================================

def undistort_fisheye(img: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                      xi: float, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Quick function to undistort a fisheye image."""
    h, w = img.shape[:2]
    cam = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha, w, h)
    return cam.undistort_image(img)


def solve_pnp_fisheye(points_3d: np.ndarray, points_2d: np.ndarray,
                      fx: float, fy: float, cx: float, cy: float,
                      xi: float, alpha: float
                     ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """Quick function to solve PnP for fisheye camera."""
    cam = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha)
    return cam.solve_pnp(points_3d, points_2d)


# ============================================================================
# Standalone Core Functions (moved to ds_msp.models.ds_math)
# ============================================================================
# The pure Double Sphere math now lives in the dependency-free math layer.
# Re-exported here so existing imports (`from ds_msp.model import ds_project`,
# `ds_project_jacobian`, `ds_unproject`, `balanced_pinhole_K`) keep working.

from .core.pinhole import balanced_pinhole_K  # noqa: E402  (re-export)
from .models.ds_math import (  # noqa: E402  (re-export for backward compatibility)
    ds_project,
    ds_project_jacobian,  # noqa: F401  (re-exported for `from ds_msp.model import ...`)
    ds_unproject,
)
