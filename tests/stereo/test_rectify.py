"""Spherical rectification: a vertical baseline makes epipolar lines vertical meridians."""

import numpy as np

from ds_msp.models import DoubleSphereModel
from ds_msp.ops import Equirectangular
from ds_msp.stereo.rectify import (
    rectified_longitude,
    rectify_maps,
    rectifying_rotation,
)


def test_rectifying_rotation_sends_baseline_to_the_pole():
    rng = np.random.default_rng(0)
    for _ in range(10):
        b = rng.standard_normal(3)
        b = b / np.linalg.norm(b)
        R = rectifying_rotation(b)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)        # is a rotation
        assert np.allclose(R @ b, [0, 1, 0], atol=1e-9)          # baseline → pole


def test_correspondences_share_a_column_after_rectification():
    """The defining property: a 3D point's two camera rays rectify to the same longitude."""
    chart = Equirectangular(720, 360, hfov_deg=360)
    rng = np.random.default_rng(1)
    c_bot = np.array([0.0, -1.0, 0.2])                           # bottom camera centre (top frame)
    R_rect = rectifying_rotation(c_bot)                          # pole along the baseline
    max_dcol = 0.0
    for _ in range(200):
        X = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(2, 6)])
        top_ray = X / np.linalg.norm(X)
        bot = X - c_bot
        bot_ray = bot / np.linalg.norm(bot)                     # bottom frame == top frame here
        lo_t = float(np.ravel(rectified_longitude(R_rect, chart, top_ray))[0])
        lo_b = float(np.ravel(rectified_longitude(R_rect, chart, bot_ray))[0])
        d = abs(lo_t - lo_b)
        d = min(d, chart.width - d)                             # tolerate ±180° wraparound
        max_dcol = max(max_dcol, d)
    assert max_dcol < 1e-6                                       # same column, to sub-pixel


def test_rectify_maps_runs_on_a_real_camera():
    cam = DoubleSphereModel(fx=160.0, fy=160.0, cx=160.0, cy=160.0, xi=0.3, alpha=0.55)
    cam.width, cam.height = 320, 320
    chart = Equirectangular(360, 180, hfov_deg=200)
    R_rect = rectifying_rotation(np.array([0.0, -1.0, 0.0]))
    mapx, mapy, valid = rectify_maps(cam, R_rect, chart)
    assert mapx.shape == (180, 360) and valid.any()
    assert np.isfinite(mapx[valid]).all()
