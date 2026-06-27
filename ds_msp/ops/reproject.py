"""Reproject a calibrated camera into image-domain charts (Tier-1).

A central camera measures rays, so any image is just a *chart* of those rays. This module turns
the verified sphere/cylinder/pinhole maps (the ``examples/08`` deep-dive) into a library API and
adds **cubemap** and **tangent-image (gnomonic)** charts — the low-distortion perspective patches
that let off-the-shelf models run on wide-FOV content.

Every chart is a pair of pure maps on the unit sphere:
``pixel_to_ray(u, v) -> ray`` and ``ray_to_pixel(ray) -> (uv, valid)``. To *resample* an image,
`reproject_maps` runs ``pixel_to_ray -> cam.project`` to build a ``cv2.remap`` lookup. Nothing
stores a second copy of the intrinsics — a chart is only ever a function of ``cam`` — so a chart
can never drift from the calibration (the killed "no canonical chart" assumption, honoured).

Ray convention matches the library: x right, y down, z forward; azimuth ``λ = atan2(x, z)``,
elevation ``ψ = atan2(-y, hypot(x, z))``. Round-trips ``pixel -> ray -> pixel`` to ~1e-13 px.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class Chart:
    """Base class: a chart is an output grid plus a bijection with unit rays."""

    width: int
    height: int

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height, self.width

    def pixel_to_ray(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def ray_to_pixel(self, rays: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class Equirectangular(Chart):
    """Unit-sphere chart: column linear in azimuth, row linear in elevation (full sphere)."""

    def __init__(self, width: int, height: int, hfov_deg: float = 360.0):
        self.width, self.height = int(width), int(height)
        self.f = width / np.radians(hfov_deg)            # px / rad
        self.cx, self.cy = width / 2.0, height / 2.0

    def pixel_to_ray(self, u, v):
        lam = (u - self.cx) / self.f
        psi = (self.cy - v) / self.f
        c = np.cos(psi)
        return np.stack([c * np.sin(lam), -np.sin(psi), c * np.cos(lam)], axis=-1)

    def ray_to_pixel(self, rays):
        x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
        lam = np.arctan2(x, z)
        psi = np.arctan2(-y, np.hypot(x, z))
        uv = np.stack([self.cx + self.f * lam, self.cy - self.f * psi], axis=-1)
        return uv, np.ones(rays.shape[:-1], dtype=bool)


class Cylindrical(Chart):
    """Cylinder chart: same azimuth column as the sphere, row linear in tan(elevation)."""

    def __init__(self, width: int, height: int, hfov_deg: float = 360.0):
        self.width, self.height = int(width), int(height)
        self.f = width / np.radians(hfov_deg)
        self.cx, self.cy = width / 2.0, height / 2.0

    def pixel_to_ray(self, u, v):
        lam = (u - self.cx) / self.f
        h = (self.cy - v) / self.f                       # = tan(elevation)
        return _unit(np.stack([np.sin(lam), -h, np.cos(lam)], axis=-1))

    def ray_to_pixel(self, rays):
        x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
        lam = np.arctan2(x, z)
        h = -y / np.hypot(x, z)
        uv = np.stack([self.cx + self.f * lam, self.cy - self.f * h], axis=-1)
        valid = z > 0                                    # cylinder covers |azimuth| < 90° per branch
        return uv, valid


class Pinhole(Chart):
    """Rectilinear (gnomonic) chart — a virtual pinhole, valid only for the front hemisphere."""

    def __init__(self, width: int, height: int, hfov_deg: float = 90.0,
                 R: np.ndarray | None = None):
        if hfov_deg >= 180.0:
            raise ValueError("pinhole hfov must be < 180° (tan diverges at 90°)")
        self.width, self.height = int(width), int(height)
        self.fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        self.fy = self.fx
        self.cx, self.cy = width / 2.0, height / 2.0
        self.R = np.eye(3) if R is None else np.asarray(R, float)   # chart→camera rotation

    def pixel_to_ray(self, u, v):
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        local = _unit(np.stack([x, y, np.ones_like(x)], axis=-1))
        return local @ self.R.T

    def ray_to_pixel(self, rays):
        d = rays @ self.R                                # into chart frame
        x, y, z = d[..., 0], d[..., 1], d[..., 2]
        valid = z > 1e-9
        zc = np.where(valid, z, 1.0)
        uv = np.stack([self.cx + self.fx * x / zc, self.cy + self.fy * y / zc], axis=-1)
        return uv, valid


class TangentImage(Chart):
    """A gnomonic patch tangent to the sphere at ``center`` (a unit ray) — a low-distortion
    perspective view of one region. Tile these (cubemap / icosahedron) to cover a wide FOV with
    near-frontal, straight-edged content. Equivalent to a `Pinhole` oriented at ``center``."""

    def __init__(self, center: np.ndarray, fov_deg: float, size: int):
        n = _unit(np.asarray(center, float))
        up = np.array([0.0, -1.0, 0.0])                  # world "up" is -y
        if abs(n @ up) > 0.999:                          # looking near a pole → pick another up
            up = np.array([0.0, 0.0, 1.0])
        right = _unit(np.cross(up, n))
        true_up = np.cross(n, right)
        self.R = np.stack([right, true_up, n], axis=1)   # columns: chart axes in camera frame
        self._pin = Pinhole(size, size, hfov_deg=fov_deg, R=self.R)
        self.width = self.height = int(size)
        self.center = n

    def pixel_to_ray(self, u, v):
        return self._pin.pixel_to_ray(u, v)

    def ray_to_pixel(self, rays):
        return self._pin.ray_to_pixel(rays)


_CUBE_DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def cubemap_charts(face_size: int, overlap_deg: float = 1.0) -> List[TangentImage]:
    """The six cube faces (±X, ±Y, ±Z) as tangent images — less polar distortion than
    equirectangular, and each face is a plain perspective view.

    A bare 90° face leaves a measure-zero gap at the cube corners (and a half-pixel sliver from
    the principal-point convention), so each face is widened by ``overlap_deg`` on every side,
    giving a small guard band that tiles the sphere seamlessly — the standard cubemap practice.
    """
    fov = 90.0 + 2.0 * overlap_deg
    return [TangentImage(np.array(d, float), fov, face_size) for d in _CUBE_DIRS]


def reproject_maps(cam, chart: Chart) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(mapx, mapy, valid)`` resampling lookups for a chart from a calibrated camera.

    For every output pixel: chart ``pixel_to_ray`` → ``cam.project`` → source pixel. ``valid``
    marks output pixels whose ray both exists in the chart and projects inside the camera's
    model-valid cone; their map entries are set to ``-1`` so ``cv2.remap`` leaves them blank.
    """
    h, w = chart.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    rays = chart.pixel_to_ray(u, v).reshape(-1, 3)
    pts, ok = cam.project(rays)
    valid = ok.reshape(h, w)
    mapx = pts[:, 0].reshape(h, w).astype(np.float32)
    mapy = pts[:, 1].reshape(h, w).astype(np.float32)
    mapx[~valid] = -1
    mapy[~valid] = -1
    return mapx, mapy, valid


def reproject_image(cam, img: np.ndarray, chart: Chart,
                    interpolation: int = cv2.INTER_LINEAR
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a fisheye ``img`` into ``chart``. Returns ``(out, valid_mask)``."""
    mapx, mapy, valid = reproject_maps(cam, chart)
    out = cv2.remap(img, mapx, mapy, interpolation, borderMode=cv2.BORDER_CONSTANT)
    return out, valid
