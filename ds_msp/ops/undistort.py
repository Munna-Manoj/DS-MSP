"""
Image / point undistortion that works on ANY camera model.

The stateful map cache lives HERE (in the service), not on the model — keeping
models as pure value objects. Depends only on the contract + core pinhole helper.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from ..core.contracts import CameraModel
from ..core.pinhole import balanced_pinhole_K


class Undistorter:
    """Undistort images/points from any model into a pinhole view.

    Model-agnostic counterpart to :meth:`ds_msp.model.DoubleSphereCamera.undistort_image`:
    works with any :class:`~ds_msp.core.contracts.CameraModel`. Caches the resampling map by
    output ``K_new`` (recomputed only when a different ``K_new`` is requested), so repeated
    calls with the same target intrinsics reuse the cached ``(mapx, mapy)``.

    Parameters
    ----------
    model : CameraModel
        The calibrated camera to undistort from.
    width, height : int
        Output (rectified) image size, pixels.

    Examples
    --------
    >>> import numpy as np
    >>> from ds_msp.models import DoubleSphereModel
    >>> model = DoubleSphereModel.sample()
    >>> img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    >>> und = Undistorter(model, 1920, 1080)
    >>> img_rect, K_new = und.undistort_image(img)        # works with any CameraModel
    >>> img_rect.shape, K_new.shape
    ((1080, 1920, 3), (3, 3))
    """

    def __init__(self, model: CameraModel, width: int, height: int) -> None:
        self.model = model
        self.width = int(width)
        self.height = int(height)
        self._mapx = None
        self._mapy = None
        self._K_new = None

    def new_K(self, balance: float = 0.5) -> np.ndarray:
        """Build a balanced pinhole intrinsic matrix for the output image.

        Parameters
        ----------
        balance : float, default=0.5
            Field-of-view/border trade-off in ``[0, 1]``: ``0.0`` keeps the
            widest field of view (more black border), ``1.0`` crops tightest
            (least border). See :func:`ds_msp.core.pinhole.balanced_pinhole_K`.

        Returns
        -------
        ndarray of shape (3, 3)
            The new pinhole ``K``, principal point at the output image center.
        """
        K = self.model.K
        return balanced_pinhole_K(K[0, 0], K[1, 1], self.width, self.height, balance)

    def maps(self, K_new: Optional[np.ndarray] = None
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build (or return the cached) ``cv2.remap`` lookup for a target ``K_new``.

        Parameters
        ----------
        K_new : ndarray of shape (3, 3), optional
            Target pinhole intrinsics for the rectified output. Defaults to
            :meth:`new_K` at ``balance=0.5`` when not given.

        Returns
        -------
        mapx : ndarray of shape (height, width), float32
            Source-image ``u`` for each output pixel, ``-1`` where the model
            has no valid projection.
        mapy : ndarray of shape (height, width), float32
            Source-image ``v`` for each output pixel, ``-1`` where invalid.
        K_new : ndarray of shape (3, 3)
            The intrinsics the maps were built for (echoes the input, or the
            computed default).
        """
        if K_new is None:
            K_new = self.new_K()
        if self._mapx is not None and self._K_new is not None \
                and np.array_equal(K_new, self._K_new):
            return self._mapx, self._mapy, self._K_new

        fx_n, fy_n = K_new[0, 0], K_new[1, 1]
        cx_n, cy_n = K_new[0, 2], K_new[1, 2]
        xg, yg = np.meshgrid(np.arange(self.width, dtype=np.float64),
                             np.arange(self.height, dtype=np.float64), indexing="xy")
        rays = np.stack([(xg - cx_n) / fx_n, (yg - cy_n) / fy_n, np.ones_like(xg)], axis=-1)
        uv, valid = self.model.project(rays)
        mapx = uv[..., 0].astype(np.float32)
        mapy = uv[..., 1].astype(np.float32)
        mapx[~valid] = -1
        mapy[~valid] = -1
        self._mapx, self._mapy, self._K_new = mapx, mapy, K_new
        return mapx, mapy, K_new

    def undistort_image(self, img: np.ndarray, K_new: Optional[np.ndarray] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample a distorted ``img`` into a rectified pinhole view.

        Parameters
        ----------
        img : ndarray of shape (H_src, W_src, ...)
            Source (distorted) image, any ``cv2.remap``-compatible dtype.
        K_new : ndarray of shape (3, 3), optional
            Target pinhole intrinsics; see :meth:`maps`. Defaults to
            :meth:`new_K` at ``balance=0.5``.

        Returns
        -------
        out : ndarray of shape (height, width, ...)
            The rectified image, sized ``(height, width)`` from the
            constructor. Pixels with no valid source ray are zero
            (``cv2.BORDER_CONSTANT``).
        K_new : ndarray of shape (3, 3)
            The intrinsics actually used (echoes the input, or the computed
            default).
        """
        mapx, mapy, K_new = self.maps(K_new)
        out = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return out, K_new

    def undistort_points(self, points: np.ndarray, K_new: Optional[np.ndarray] = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """Map distorted pixels to rectified pinhole pixels (in the ``K_new`` frame).

        Closed-form counterpart to :meth:`undistort_image`: unprojects each point
        through ``self.model`` and reprojects the resulting ray with ``K_new``.
        Exact wherever the ray is recoverable — unlike inverting a displacement
        mesh, this has no periphery error (see :mod:`ds_msp.ldc` for the mesh
        alternative and its accuracy trade-off).

        Parameters
        ----------
        points : ndarray of shape (N, 2)
            Distorted pixels in the original (source) image.
        K_new : ndarray of shape (3, 3), optional
            Target pinhole intrinsics; see :meth:`maps`. Defaults to
            :meth:`new_K` at ``balance=0.5``.

        Returns
        -------
        uv : ndarray of shape (N, 2)
            Rectified pixel coordinates in the ``K_new`` frame.
        valid : ndarray of shape (N,), bool
            ``True`` where the model successfully unprojected the input pixel.
        """
        if K_new is None:
            K_new = self.new_K()
        rays, valid = self.model.unproject(np.asarray(points, dtype=np.float64))
        rays_n = rays / (rays[:, 2:3] + 1e-12)
        u = K_new[0, 0] * rays_n[:, 0] + K_new[0, 2]
        v = K_new[1, 1] * rays_n[:, 1] + K_new[1, 2]
        return np.stack([u, v], axis=-1), valid

    def distort_points(self, points: np.ndarray, K_new: Optional[np.ndarray] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse of :meth:`undistort_points`: map rectified pixels back to distorted ones.

        Parameters
        ----------
        points : ndarray of shape (N, 2)
            Rectified pinhole pixels, in the ``K_new`` frame.
        K_new : ndarray of shape (3, 3), optional
            The intrinsics ``points`` are expressed in; see :meth:`maps`.
            Defaults to :meth:`new_K` at ``balance=0.5``.

        Returns
        -------
        uv : ndarray of shape (N, 2)
            Distorted pixels in the original (source) image.
        valid : ndarray of shape (N,), bool
            ``True`` where ``self.model.project`` produced a valid pixel for
            the corresponding ray.
        """
        if K_new is None:
            K_new = self.new_K()
        pts = np.asarray(points, dtype=np.float64)
        mx = (pts[:, 0] - K_new[0, 2]) / K_new[0, 0]
        my = (pts[:, 1] - K_new[1, 2]) / K_new[1, 1]
        rays = np.stack([mx, my, np.ones_like(mx)], axis=-1)
        rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
        return self.model.project(rays)
