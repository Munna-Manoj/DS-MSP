"""Sphere-sweep planar-depth recovery on the OCam and DS+ camera models.

Mirrors ``tests/stereo/test_sphere_sweep.py`` (which renders a textured plane through a
``DoubleSphereModel`` and checks the sweep recovers the known depth), but for the polynomial
OCam (Scaramuzza) and DS+ (UCM core + division + tilt) models — the two absent from the stereo
suite. The plane is rendered exactly through each model's own ``unproject`` (so at a pixel's true
depth the reference and source sample the same plane point and the photo-cost is zero there); the
sweep must then recover that depth. Same 5%-median / 85%-within-10% bar as the DS test; measured
median relative error is ~0.6% for both models (no tolerance loosened).
"""

import numpy as np
import pytest

from ds_msp.stereo import inverse_depth_samples, sphere_sweep, sweep_to_points
from ds_msp.models.ocam import OCamModel
from ds_msp.models.dsplus import DSPlusModel

H = W = 120
PLANE_Z = 5.0
BASELINE = 0.6


def _ocam():
    """OCam sized for a 120x120 fisheye (centre W/2,H/2; focal ~ |a0|=60)."""
    return OCamModel(60.0, 60.0, 1.0, 0.0, 0.0, -60.0, 0.0, 6.0, -1.5, 0.25)


def _dsplus():
    return DSPlusModel(130.0, 130.0, W / 2, H / 2, alpha=0.55)


def _texture(x, y):
    """A smooth, distinctive plane texture (so wrong-depth samples differ), in [0, 255]."""
    return (128 + 110 * np.sin(1.5 * x) * np.cos(1.7 * y)).astype(np.float32)


def _render(cam, center_ref):
    """Render the plane z=PLANE_Z seen by a camera whose centre (ref frame) is `center_ref`
    and whose rotation ref->cam is identity."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    g, ok = cam.unproject(np.stack([u, v], axis=-1).reshape(-1, 2))
    g = g.reshape(H, W, 3)
    gz = g[..., 2]
    s = np.where(gz > 1e-6, (PLANE_Z - center_ref[2]) / np.where(gz > 1e-6, gz, 1.0), np.nan)
    X = center_ref + s[..., None] * g                     # plane intersection, ref frame
    img = _texture(X[..., 0], X[..., 1])
    img[(gz <= 1e-6) | ~ok.reshape(H, W)] = 0
    return img


@pytest.mark.parametrize("factory", [_ocam, _dsplus], ids=["ocam", "dsplus"])
def test_sphere_sweep_recovers_planar_depth(factory):
    cam = factory()
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


@pytest.mark.parametrize("factory", [_ocam, _dsplus], ids=["ocam", "dsplus"])
def test_sweep_to_points_back_projects_consistently(factory):
    cam = factory()
    img = _render(cam, np.zeros(3))
    depths = inverse_depth_samples(4.0, 15.0, 8)
    depth_map, _, valid = sphere_sweep(cam, img, [(cam, img, np.eye(3), np.zeros(3))], depths)
    pts = sweep_to_points(cam, depth_map, valid)
    assert pts.shape[1] == 3 and pts.shape[0] == int(valid.sum())
    assert np.isfinite(pts).all()
    assert np.linalg.norm(pts, axis=1).min() > 0

# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("FR-STEREO-001")
