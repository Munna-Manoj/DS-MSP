"""Spherical epipolar rectification for a top-bottom rig (Tier-1).

On a fisheye the epipolar lines are *curves*, so no single homography rectifies them. But if the
two cameras are arranged with a **vertical baseline** (top-bottom) and we resample into an
equirectangular image whose **pole is the baseline**, every epipolar great circle passes through
the poles and projects to a **constant-longitude vertical meridian**. Correspondences then lie on
the same column, and disparity is a pure vertical (angular) offset — 1-D search, exactly like
rectified pinhole stereo (360SD-Net).

This module needs only ``cam.project`` and a chart object exposing ``pixel_to_ray`` /
``ray_to_pixel`` (e.g. ``ds_msp.ops.Equirectangular``), passed in by the caller — so the stereo
layer stays independent of the chart layer.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

_POLE = np.array([0.0, 1.0, 0.0])           # equirectangular pole (our up axis is -y, +y is a pole)


def rectifying_rotation(baseline_dir: np.ndarray) -> np.ndarray:
    """Rotation ``R`` mapping the (unit) baseline direction onto the equirectangular pole.

    After rectifying both cameras by ``R``, the baseline lies along the pole, so epipolar great
    circles become vertical meridians.
    """
    b = np.asarray(baseline_dir, float)
    b = b / np.linalg.norm(b)
    v = np.cross(b, _POLE)
    s = np.linalg.norm(v)
    c = float(b @ _POLE)
    if s < 1e-12:                            # already (anti)parallel to the pole
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def rectify_maps(cam, R_rect: np.ndarray, chart) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``cv2.remap`` lookups that resample ``cam`` into ``chart`` after applying ``R_rect``.

    For each chart pixel: ``chart.pixel_to_ray`` (rectified frame) → ``R_rectᵀ`` → camera frame →
    ``cam.project``. ``valid`` marks imageable pixels (their map entries are ``-1`` otherwise).
    """
    R_rect = np.asarray(R_rect, float)
    h, w = chart.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    rays = chart.pixel_to_ray(u, v).reshape(-1, 3) @ R_rect      # rect → camera frame
    pts, ok = cam.project(rays)
    valid = ok.reshape(h, w)
    mapx = pts[:, 0].reshape(h, w).astype(np.float32)
    mapy = pts[:, 1].reshape(h, w).astype(np.float32)
    mapx[~valid] = -1
    mapy[~valid] = -1
    return mapx, mapy, valid


def rectify_image(cam, img: np.ndarray, R_rect: np.ndarray, chart) -> np.ndarray:
    """Resample ``img`` into the rectified equirectangular ``chart``."""
    mapx, mapy, _ = rectify_maps(cam, R_rect, chart)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def rectified_longitude(R_rect: np.ndarray, chart, ray: np.ndarray) -> np.ndarray:
    """The rectified column (longitude pixel) a camera-frame ``ray`` lands on. After top-bottom
    rectification, a 3D point's two camera rays share this value — the property this rectification guarantees."""
    rr = np.atleast_2d(np.asarray(ray, float)) @ R_rect.T        # camera → rect frame
    uv, _ = chart.ray_to_pixel(rr)
    return uv[..., 0]
