"""Sphere-sweep stereo — dense depth straight on calibrated fisheye views (Tier-1).

The modern wide-FOV alternative to plane-sweep, which is invalid past 90° (Meuleman et al.,
CVPR 2021). Instead of rectifying — which on a fisheye warps the image and makes a single
disparity mean different distances in different places — we sweep **depth** directly:

  for each reference pixel (ray f) and each candidate depth d:
      X = d · f                                   # a 3D hypothesis in the reference frame
      for each source view j:
          project R_j X + t_j into camera j, sample it, accumulate photo-cost
  depth(pixel) = argmin_d cost                    # the depth that is photo-consistent

No rectification, no homography, no pinhole — only ``unproject`` (reference rays) and
``project`` (into each source). Sampling **inverse depth** uniformly keeps the candidates evenly
spaced in disparity.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def inverse_depth_samples(near: float, far: float, n: int) -> np.ndarray:
    """``n`` depth candidates evenly spaced in **inverse** depth over ``[near, far]``."""
    return 1.0 / np.linspace(1.0 / far, 1.0 / near, n)


def _as_gray_f32(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def sphere_sweep(
    ref_cam, ref_img: np.ndarray,
    sources: Sequence[Tuple[object, np.ndarray, np.ndarray, np.ndarray]],
    depths: np.ndarray, *,
    min_views: int = 1, invalid_cost: float = 1e9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense depth for the reference view by sweeping ``depths``.

    Parameters
    ----------
    ref_cam, ref_img :
        The reference camera (with ``unproject``/``project``) and its image. The reference frame
        is the world frame (its centre at the origin).
    sources : list of ``(cam, img, R, t)``
        Each source camera, its image, and the pose mapping a **reference-frame** point into the
        source: ``X_src = R @ X_ref + t``.
    depths : (K,) array
        Candidate depths along each reference ray (see ``inverse_depth_samples``).
    min_views :
        A pixel/depth hypothesis needs at least this many sources to see it, else its cost is
        ``invalid_cost``.

    Returns
    -------
    (depth_map, cost_volume, valid) : ``depth_map`` (H, W) is the per-pixel argmin depth;
    ``cost_volume`` (H, W, K) the mean absolute photo-error; ``valid`` (H, W) marks pixels with
    at least one usable hypothesis.
    """
    ref = _as_gray_f32(ref_img)
    h, w = ref.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    rays, _ = ref_cam.unproject(np.stack([u, v], axis=-1).reshape(-1, 2))
    rays = rays.reshape(h, w, 3)

    src = [(cam, _as_gray_f32(img), np.asarray(R, float), np.asarray(t, float).reshape(3))
           for cam, img, R, t in sources]

    K = len(depths)
    cost = np.full((h, w, K), invalid_cost, dtype=np.float32)
    for k, d in enumerate(depths):
        X = d * rays                                            # (H, W, 3) hypotheses
        acc = np.zeros((h, w), np.float32)
        cnt = np.zeros((h, w), np.float32)
        for cam, img, R, t in src:
            sh, sw = img.shape
            Xs = X @ R.T + t
            pts, ok = cam.project(Xs.reshape(-1, 3))
            mapx = pts[:, 0].reshape(h, w).astype(np.float32)
            mapy = pts[:, 1].reshape(h, w).astype(np.float32)
            warped = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            inb = ok.reshape(h, w) & (mapx >= 0) & (mapx <= sw - 1) & (mapy >= 0) & (mapy <= sh - 1)
            acc += np.where(inb, np.abs(ref - warped), 0.0)
            cnt += inb
        seen = cnt >= min_views
        cost[:, :, k] = np.where(seen, acc / np.maximum(cnt, 1.0), invalid_cost)

    valid = np.any(cost < invalid_cost, axis=2)
    depth_map = np.asarray(depths)[np.argmin(cost, axis=2)]
    return depth_map, cost, valid


def sweep_to_points(ref_cam, depth_map: np.ndarray, valid: Optional[np.ndarray] = None
                    ) -> np.ndarray:
    """Back-project a depth map to a 3D point cloud in the reference frame, ``(M, 3)``."""
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    rays, _ = ref_cam.unproject(np.stack([u, v], axis=-1).reshape(-1, 2))
    pts = (depth_map.reshape(-1, 1) * rays)
    if valid is not None:
        pts = pts[valid.reshape(-1)]
    return pts
