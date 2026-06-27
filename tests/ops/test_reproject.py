"""Image-domain charts: round-trip precision, ray validity, and real-camera resampling."""

import numpy as np
import pytest

from ds_msp.models import DoubleSphereModel
from ds_msp.ops.reproject import (
    Chart,
    Cylindrical,
    Equirectangular,
    Pinhole,
    TangentImage,
    cubemap_charts,
    reproject_maps,
)


def _interior_grid(chart: Chart, margin: int = 5):
    u, v = np.meshgrid(np.linspace(margin, chart.width - margin, 25),
                       np.linspace(margin, chart.height - margin, 25))
    return u, v


@pytest.mark.parametrize("chart", [
    Equirectangular(800, 400, hfov_deg=300),
    Cylindrical(800, 400, hfov_deg=160),
    Pinhole(512, 512, hfov_deg=90),
    TangentImage(np.array([0.3, -0.2, 1.0]), fov_deg=70, size=256),
])
def test_pixel_ray_pixel_roundtrips_to_machine_precision(chart):
    u, v = _interior_grid(chart)
    rays = chart.pixel_to_ray(u, v)
    assert np.allclose(np.linalg.norm(rays, axis=-1), 1.0, atol=1e-12)   # unit rays
    uv, valid = chart.ray_to_pixel(rays)
    back = np.stack([u, v], axis=-1)
    err = np.abs(uv[valid] - back[valid])
    assert err.max() < 1e-9                                              # exact inverse


def test_pinhole_rejects_the_back_hemisphere():
    chart = Pinhole(256, 256, hfov_deg=90)
    rays = np.array([[0, 0, 1.0], [0, 0, -1.0], [1.0, 0, 0.0]])          # front, back, sideways
    _, valid = chart.ray_to_pixel(rays)
    assert valid.tolist() == [True, False, False]


def test_tangent_image_center_pixel_looks_along_its_axis():
    center = np.array([0.4, -0.3, 1.0]) / np.linalg.norm([0.4, -0.3, 1.0])
    chart = TangentImage(center, fov_deg=60, size=128)
    cx = chart.width / 2.0                                               # principal point
    ray = chart.pixel_to_ray(np.array(cx), np.array(cx))
    assert np.allclose(ray, center, atol=1e-9)


def test_cubemap_is_six_faces_pointing_along_the_axes():
    faces = cubemap_charts(128)
    assert len(faces) == 6
    axes = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for face, ax in zip(faces, axes):
        c = face.pixel_to_ray(np.array(64.0), np.array(64.0))
        assert np.allclose(c, ax, atol=1e-9)


def test_cubemap_faces_cover_the_whole_sphere():
    # every direction on the sphere must be visible in at least one 90° face
    faces = cubemap_charts(64)
    rng = np.random.default_rng(0)
    dirs = rng.standard_normal((2000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    seen = np.zeros(len(dirs), dtype=bool)
    for face in faces:
        uv, valid = face.ray_to_pixel(dirs)
        inside = valid & (uv[:, 0] >= -0.5) & (uv[:, 0] <= face.width + 0.5) \
            & (uv[:, 1] >= -0.5) & (uv[:, 1] <= face.height + 0.5)
        seen |= inside
    assert seen.all()


def test_reproject_maps_from_a_real_double_sphere_camera():
    cam = DoubleSphereModel(fx=300.0, fy=300.0, cx=320.0, cy=320.0, xi=0.3, alpha=0.6)
    cam.width, cam.height = 640, 640
    chart = Equirectangular(400, 200, hfov_deg=200)
    mapx, mapy, valid = reproject_maps(cam, chart)
    assert mapx.shape == (200, 400) and mapy.shape == (200, 400)
    assert valid.any()                                                  # some rays are imageable
    assert np.isfinite(mapx[valid]).all() and np.isfinite(mapy[valid]).all()
    if (~valid).any():
        assert (mapx[~valid] == -1).all()                              # masked entries flagged
    # the forward ray (centre of the panorama) lands on the camera's principal point
    cu, cv = chart.width // 2, chart.height // 2
    assert abs(mapx[cv, cu] - cam.cx) < 1.0 and abs(mapy[cv, cu] - cam.cy) < 1.0
