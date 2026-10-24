"""Ray-cast ChArUco rig renderer with exact ground truth, for image-level benchmarks.

Every (supersampled) output pixel is unprojected through the *true* camera model, intersected
with each board plane of the target, and sampled from the board's real ChArUco texture, so
the whole pipeline -- detector, pixel convention, reconstruction, bundle adjustment -- is
scored against known extrinsics, intrinsics and target geometry. Pixel convention: integer
coordinates are pixel centres; a supersampled pixel centre ``X`` maps to ``(X + 0.5)/ss - 0.5``.
"""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from ds_msp.detect.charuco import BoardSpec, make_detectors

TEX_PX = 48                          # texture pixels per square
TEX_MARGIN = TEX_PX // 2             # white quiet zone around each board texture


def board_textures(specs: List[BoardSpec]) -> List[np.ndarray]:
    """One texture per board, with MC-Calib's per-board marker-id offsets (legacy pattern)."""
    out = []
    for det, spec in zip(make_detectors(specs), specs):
        w, h = spec.n_x * TEX_PX + 2 * TEX_MARGIN, spec.n_y * TEX_PX + 2 * TEX_MARGIN
        out.append(det.getBoard().generateImage((w, h), marginSize=TEX_MARGIN, borderBits=1))
    return out


def _texture_coords(xy: np.ndarray, spec: BoardSpec) -> np.ndarray:
    """Board-local metric (x, y) -> texture pixel-centre coordinates. DS-MSP's board origin is
    the first interior corner, one square in from the board edge."""
    return (xy / spec.square_size + 1.0) * TEX_PX + TEX_MARGIN - 0.5


def _bilinear(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear texture lookup at pixel-centre coordinates (edge-clamped)."""
    h, w = tex.shape
    x = np.clip(uv[:, 0].astype(float), 0, w - 1)
    y = np.clip(uv[:, 1].astype(float), 0, h - 1)
    x0, y0 = np.minimum(x.astype(int), w - 2), np.minimum(y.astype(int), h - 2)
    fx, fy = x - x0, y - y0
    t = tex.astype(float)
    return ((1 - fx) * (1 - fy) * t[y0, x0] + fx * (1 - fy) * t[y0, x0 + 1]
            + (1 - fx) * fy * t[y0 + 1, x0] + fx * fy * t[y0 + 1, x0 + 1])


def render(model, size, T_c_b: Dict[int, np.ndarray], specs: List[BoardSpec],
           textures: List[np.ndarray], *, ss: int = 2, background: int = 150,
           noise: float = 0.0, rng=None) -> np.ndarray:
    """Render boards (board->camera poses ``T_c_b``) seen by ``model`` into a ``size`` image."""
    w, h = size
    X, Y = np.meshgrid(np.arange(w * ss, dtype=float), np.arange(h * ss, dtype=float))
    uv = np.stack([(X + 0.5) / ss - 0.5, (Y + 0.5) / ss - 0.5], -1).reshape(-1, 2)
    rays, valid = model.unproject(uv)
    rays = np.asarray(rays, float).reshape(-1, 3)
    valid = np.asarray(valid, bool).ravel() if valid is not None else np.ones(len(rays), bool)
    img = np.full(len(rays), float(background))
    depth = np.full(len(rays), np.inf)
    for b, T in T_c_b.items():
        R, t = T[:3, :3], T[:3, 3]
        n = R[:, 2]
        denom = rays @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (t @ n) / denom
        hit = valid & np.isfinite(s) & (s > 0) & (s < depth)
        P = rays[hit] * s[hit, None]
        xy = (P - t) @ R[:, :2]
        tex = textures[b]
        uvt = _texture_coords(xy, specs[b]).astype(np.float32)
        inside = ((uvt[:, 0] >= -0.5) & (uvt[:, 0] <= tex.shape[1] - 0.5)
                  & (uvt[:, 1] >= -0.5) & (uvt[:, 1] <= tex.shape[0] - 0.5))
        idx = np.flatnonzero(hit)[inside]
        vals = _bilinear(tex, uvt[inside])
        img[idx] = vals
        depth[idx] = s[hit][inside]
    img = img.reshape(h * ss, w * ss).astype(np.float32)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if noise:
        img = img + (rng or np.random.default_rng(0)).normal(0, noise, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)
