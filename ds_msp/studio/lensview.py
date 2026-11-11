"""Pictures of an LDC mesh and an Isaac LUT at work, for the export pages.

* **LDC:** what the camera sees, and the same image after the board's correction. The correction
  is the board's: the output pixel reads the input where the stored Q3 mesh, interpolated
  bilinearly between nodes, sends it. The input is a real photo, or a test room seen through the
  lens.
* **Isaac LUT:** the test room rendered through the LUT's ray directions, as Isaac Sim would.

The test room is a box around the camera with a grid on every wall: straight lines on a wall stay
straight after a correct correction, and the floor, ceiling and front wall have their own colours
so a flipped or turned image shows at once. Everything is drawn at display size: each displayed
pixel is computed directly, never enlarged from a smaller picture.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from ..ldc import Q3_SCALE

# Wall colours (BGR): front, back, right, left, floor, ceiling. Light, so black stays "not seen".
_WALLS = np.array([
    [235, 225, 205],   # front (+z): light blue-grey
    [205, 215, 230],   # back
    [205, 235, 215],   # right (+x): light green
    [225, 205, 235],   # left: light violet
    [175, 205, 230],   # floor (+y, image down): sand
    [245, 240, 225],   # ceiling: near white
], dtype=np.float64)


def room_colours(rays_cv: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """BGR colours of the test room along camera rays (OpenCV axes: x right, y down, z ahead)."""
    d = np.asarray(rays_cv, dtype=np.float64)
    shape = d.shape[:-1]
    d = d.reshape(-1, 3)
    ok = np.asarray(valid, dtype=bool).reshape(-1) & np.all(np.isfinite(d), axis=-1)
    ok &= np.linalg.norm(d, axis=-1) > 0
    out = np.zeros((d.shape[0], 3), dtype=np.float64)
    a = np.abs(d)
    axis = np.argmax(a, axis=1)
    m = a[np.arange(len(d)), axis]
    m[m == 0] = 1.0
    p = d / m[:, None]                       # point on the unit cube
    sign = np.take_along_axis(d, axis[:, None], axis=1)[:, 0] >= 0
    wall = np.select([(axis == 2) & sign, (axis == 2) & ~sign, (axis == 0) & sign,
                      (axis == 0) & ~sign, (axis == 1) & sign], [0, 1, 2, 3, 4], 5)
    # the two coordinates across each wall
    uv = np.where(axis[:, None] == 0, p[:, [1, 2]], np.where(axis[:, None] == 1, p[:, [0, 2]], p[:, [0, 1]]))
    cells = np.floor((uv + 1.0) * 4.0).astype(int)          # 8 x 8 cells per wall
    checker = (cells[:, 0] + cells[:, 1]) % 2 == 0
    base = _WALLS[wall] * np.where(checker, 1.0, 0.9)[:, None]
    # grid lines every quarter wall, and a heavy cross through the centre of the front wall
    f = (uv + 1.0) * 4.0
    near = np.minimum(np.abs(f - np.round(f)), 1.0).min(axis=1)
    line = near < 0.035
    cross = (wall == 0) & (np.abs(uv).min(axis=1) < 0.02)
    base[line] = (70, 70, 70)
    base[cross] = (40, 40, 200)
    out[ok] = base[ok]
    return np.clip(out, 0, 255).astype(np.uint8).reshape(*shape, 3)


def _grid(width: int, height: int, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Display-pixel centres, in the pixel coordinates of an image ``width`` x ``height``."""
    dw, dh = max(1, round(width * scale)), max(1, round(height * scale))
    u = (np.arange(dw) + 0.5) / scale - 0.5
    v = (np.arange(dh) + 0.5) / scale - 0.5
    return np.meshgrid(u, v)


def _jpeg(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("could not encode the picture")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _scale(width: int, display: int) -> float:
    return min(1.0, float(display) / float(width))


def camera_view(model, size: Tuple[int, int], display: int, photo: Optional[np.ndarray]) -> np.ndarray:
    """What the camera sees: the photo, or the test room through the lens, at display size."""
    w, h = size
    s = _scale(w, display)
    if photo is not None:
        dw, dh = max(1, round(w * s)), max(1, round(h * s))
        return cv2.resize(photo, (dw, dh), interpolation=cv2.INTER_AREA)
    gu, gv = _grid(w, h, s)
    rays, ok = model.unproject(np.stack([gu, gv], axis=-1).reshape(-1, 2))
    return room_colours(np.asarray(rays).reshape(*gu.shape, 3), np.asarray(ok).reshape(gu.shape))


def ldc_view(model, input_size: Tuple[int, int], mesh_q3: np.ndarray, K_new: np.ndarray,
             downsample_factor: int, output_size: Tuple[int, int], display: int,
             photo: Optional[np.ndarray], overlay: bool,
             stats: Optional[Dict[str, float]] = None) -> np.ndarray:
    """The image after the board's correction with this mesh, at display size.

    With ``overlay``, corrected pixels whose source the mesh misplaces by more than 0.5 px are
    tinted amber, and by more than 2 px red (the error ``ldc.mesh_accuracy`` measures).
    """
    out_w, out_h = output_size
    s = _scale(out_w, display)
    gu, gv = _grid(out_w, out_h, s)
    step = 2 ** int(downsample_factor)
    mesh = np.asarray(mesh_q3, dtype=np.float64) / Q3_SCALE
    mx, my = np.clip(gu, 0, None) / step, np.clip(gv, 0, None) / step
    x0 = np.minimum(np.floor(mx).astype(int), mesh.shape[1] - 2)
    y0 = np.minimum(np.floor(my).astype(int), mesh.shape[0] - 2)
    fx, fy = (mx - x0)[..., None], (my - y0)[..., None]
    delta = ((1 - fx) * (1 - fy) * mesh[y0, x0] + fx * (1 - fy) * mesh[y0, x0 + 1]
             + (1 - fx) * fy * mesh[y0 + 1, x0] + fx * fy * mesh[y0 + 1, x0 + 1])
    src = np.stack([gu, gv], axis=-1) + delta            # where the board reads the input
    in_w, in_h = input_size
    inside = (src[..., 0] >= -0.5) & (src[..., 0] <= in_w - 0.5) & (src[..., 1] >= -0.5) & (src[..., 1] <= in_h - 0.5)
    if photo is not None:
        img = cv2.remap(photo, src[..., 0].astype(np.float32), src[..., 1].astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        rays, ok = model.unproject(src.reshape(-1, 2))
        img = room_colours(np.asarray(rays).reshape(*gu.shape, 3), np.asarray(ok).reshape(gu.shape))
    img[~inside] = 0
    if overlay:
        K = np.asarray(K_new, dtype=np.float64)
        rays = np.stack([(gu - K[0, 2]) / K[0, 0], (gv - K[1, 2]) / K[1, 1], np.ones_like(gu)], axis=-1)
        exact, ok = model.project(rays.reshape(-1, 3))
        exact = np.asarray(exact, dtype=np.float64).reshape(src.shape)
        err = np.linalg.norm(src - exact, axis=-1)
        err[~np.asarray(ok, dtype=bool).reshape(gu.shape) | ~np.isfinite(err)] = 0.0
        if stats is not None:
            n = max(1, int(inside.sum()))
            stats.update(over_half_px=float(((err > 0.5) & inside).sum() / n),
                         over_2_px=float(((err > 2.0) & inside).sum() / n))
        for limit, colour in ((0.5, (0, 170, 255)), (2.0, (0, 0, 230))):
            hit = (err > limit) & inside
            img[hit] = (0.45 * img[hit] + 0.55 * np.array(colour)).astype(np.uint8)
    return img


def lut_view_from_model(model, size: Tuple[int, int], display: int) -> np.ndarray:
    """The test room through the ray directions an Isaac LUT of this camera stores."""
    from ..isaac_sim.lut import IsaacProjectionAdapter

    w, h = size
    s = _scale(w, display)
    dw, dh = max(1, round(w * s)), max(1, round(h * s))
    u, v = np.meshgrid((np.arange(dw) + 0.5) / dw, (np.arange(dh) + 0.5) / dh)
    rays, ok = IsaacProjectionAdapter(model, w, h).unproject(u, v)
    return _rtx_room(np.asarray(rays), np.asarray(ok, dtype=bool))


def lut_view_from_texture(enter_rgb: np.ndarray, display: int) -> np.ndarray:
    """The test room through a written ray-direction texture (``*_ray_enter_direction.exr``)."""
    th, tw = enter_rgb.shape[:2]
    dw = max(1, min(int(display), tw * 4))
    dh = max(1, round(th * dw / tw))
    rays = cv2.resize(np.asarray(enter_rgb, dtype=np.float32), (dw, dh), interpolation=cv2.INTER_NEAREST)
    sentinel = np.all(np.isclose(rays, (0.0, 0.0, 1.0)), axis=-1)   # RTX "no ray"
    return _rtx_room(rays.astype(np.float64), ~sentinel)


def _rtx_room(rays_rtx: np.ndarray, valid: np.ndarray) -> np.ndarray:
    # RTX camera axes (+x right, +y up, -z ahead) to OpenCV (+x right, +y down, +z ahead)
    cv = np.stack([rays_rtx[..., 0], -rays_rtx[..., 1], -rays_rtx[..., 2]], axis=-1)
    return room_colours(cv, valid)


def encode(images: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Pictures as JPEG data URLs, with their sizes."""
    return {k: {"src": _jpeg(v), "width": int(v.shape[1]), "height": int(v.shape[0])}
            for k, v in images.items()}
