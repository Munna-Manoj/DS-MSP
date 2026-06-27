"""Sphere-sweep stereo — recovers known depth on a synthetic textured-plane fisheye pair.

We render two Double Sphere views of a textured plane at depth ``D`` (reference frame), with a
known baseline. At a pixel's *true* depth the reference and source sample the **same** plane
point, so the photo-cost is zero there — the sweep must recover that depth. No rectification.
"""

import numpy as np
import pytest

from ds_msp.models import DoubleSphereModel
from ds_msp.stereo import inverse_depth_samples, sphere_sweep, sweep_to_points

H = W = 120
PLANE_Z = 5.0
BASELINE = 0.6


def _texture(x, y):
    """A smooth, distinctive plane texture (so wrong-depth samples differ), in [0, 255]."""
    return (128 + 110 * np.sin(1.5 * x) * np.cos(1.7 * y)).astype(np.float32)


def _render(cam, center_ref):
    """Render the plane z=PLANE_Z seen by a camera whose centre (ref frame) is `center_ref`
    and whose rotation ref→cam is identity."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    g, ok = cam.unproject(np.stack([u, v], axis=-1).reshape(-1, 2))
    g = g.reshape(H, W, 3)
    gz = g[..., 2]
    s = np.where(gz > 1e-6, (PLANE_Z - center_ref[2]) / np.where(gz > 1e-6, gz, 1.0), np.nan)
    X = center_ref + s[..., None] * g                     # plane intersection, ref frame
    img = _texture(X[..., 0], X[..., 1])
    img[(gz <= 1e-6) | ~ok.reshape(H, W)] = 0
    return img


@pytest.fixture
def cam():
    return DoubleSphereModel(fx=130.0, fy=130.0, cx=W / 2, cy=H / 2, xi=0.3, alpha=0.55)


def test_sphere_sweep_recovers_planar_depth(cam):
    ref_img = _render(cam, np.zeros(3))                   # reference at origin
    src_img = _render(cam, np.array([-BASELINE, 0.0, 0.0]))   # source centre = -t
    R, t = np.eye(3), np.array([BASELINE, 0.0, 0.0])      # X_src = R X_ref + t

    depths = inverse_depth_samples(near=PLANE_Z * 0.8, far=PLANE_Z * 3.5, n=64)
    depth_map, cost, valid = sphere_sweep(cam, ref_img, [(cam, src_img, R, t)], depths)

    # ground-truth per-pixel depth along the reference ray: D / f_z
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    f, ok = cam.unproject(np.stack([u, v], axis=-1).reshape(-1, 2))
    fz = f.reshape(H, W, 3)[..., 2]
    true_depth = np.where(fz > 1e-6, PLANE_Z / np.where(fz > 1e-6, fz, 1.0), np.nan)

    # evaluate on a central window where both cameras see the plane and depth is in range
    cy0, cx0 = slice(35, 85), slice(35, 85)
    m = valid[cy0, cx0] & np.isfinite(true_depth[cy0, cx0]) & (true_depth[cy0, cx0] < PLANE_Z * 3.0)
    rel = np.abs(depth_map[cy0, cx0][m] - true_depth[cy0, cx0][m]) / true_depth[cy0, cx0][m]
    assert np.median(rel) < 0.05                          # within 5% of true depth
    assert (rel < 0.10).mean() > 0.85                     # the vast majority are close


def test_zero_baseline_is_degenerate_but_safe(cam):
    """With no baseline every depth is equally photo-consistent — the sweep must not crash and
    must still return a finite depth map (the classic 'no parallax' degeneracy)."""
    img = _render(cam, np.zeros(3))
    depths = inverse_depth_samples(4.0, 15.0, 16)
    depth_map, cost, valid = sphere_sweep(cam, img, [(cam, img, np.eye(3), np.zeros(3))], depths)
    assert depth_map.shape == (H, W)
    assert np.isfinite(depth_map).all()


def test_sweep_to_points_back_projects_consistently(cam):
    img = _render(cam, np.zeros(3))
    depths = inverse_depth_samples(4.0, 15.0, 8)
    depth_map, _, valid = sphere_sweep(cam, img, [(cam, img, np.eye(3), np.zeros(3))], depths)
    pts = sweep_to_points(cam, depth_map, valid)
    assert pts.shape[1] == 3 and pts.shape[0] == int(valid.sum())
    # each point sits at its depth along the corresponding ray (positive, finite)
    assert np.isfinite(pts).all()
    assert np.linalg.norm(pts, axis=1).min() > 0
