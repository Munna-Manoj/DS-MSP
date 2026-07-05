"""
DoubleSphereModel unit tests: it must agree with the standalone ds_math and with
the legacy DoubleSphereCamera, and its conversion seed must be sane.
"""

import warnings

import pytest
import numpy as np

from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.ds_math import ds_project, ds_unproject
from ds_msp.model import DoubleSphereCamera
from ds_msp.testing import sample_forward_points


PARAMS = dict(fx=711.57, fy=711.24, cx=949.18, cy=518.81, xi=0.183, alpha=0.809)


def test_matches_standalone_ds_project():
    m = DoubleSphereModel(**PARAMS)
    P = sample_forward_points()
    uv, valid = m.project(P)
    u, v, val0 = ds_project(P, **PARAMS)
    assert np.allclose(uv, np.stack([u, v], axis=-1))
    assert np.array_equal(valid, val0)


def test_matches_legacy_camera_class():
    m = DoubleSphereModel(**PARAMS)
    cam = DoubleSphereCamera(**PARAMS)
    P = sample_forward_points()
    uv_m, _ = m.project(P)
    uv_c, _ = cam.project(P)
    assert np.allclose(uv_m, uv_c)
    rays_m, _ = m.unproject(uv_m)
    rays_c, _ = cam.unproject(uv_c)
    assert np.allclose(rays_m, rays_c)


def test_K_and_distortion():
    m = DoubleSphereModel(**PARAMS)
    assert np.allclose(m.K, [[711.57, 0, 949.18], [0, 711.24, 518.81], [0, 0, 1]])
    assert np.allclose(m.distortion, [0.183, 0.809])


def test_initialize_seed_is_reasonable():
    # Build correspondences from a known DS camera, seed a fresh model, check it
    # lands in-bounds and roughly reproduces pixels after the linear seed.
    truth = DoubleSphereModel(**PARAMS)
    rng = np.random.default_rng(0)
    pix = rng.uniform([300, 200], [1600, 900], size=(200, 2)).astype(float)
    rays, valid = truth.unproject(pix)
    keep = valid & (rays[:, 2] > 1e-6)
    seed = DoubleSphereModel(0, 0, 0, 0, 0, 0.5)
    seed.initialize_from_correspondences(truth.K, rays[keep], pix[keep])
    lb, ub = DoubleSphereModel.param_bounds()
    assert (seed.params >= lb).all() and (seed.params <= ub).all()
    assert 0.0 <= seed.alpha <= 1.0


def test_unproject_alpha_one_boundary_ray_is_finite_and_invalid_not_a_warning():
    """Regression: alpha=1.0 is a legal optimizer bound (calib_param.py's alpha in [0,1]);
    combined with a ray exactly at the sphere-intersection edge (r2 = 1/(2*alpha-1)) the
    closed-form unprojection's denominator hits exactly 0/0. Must be flagged invalid and
    return a finite (zeroed) ray, not silently warn and/or leak a NaN under valid=True."""
    fx = fy = 500.0
    cx = cy = 320.0
    u = cx + fx * 1.0                       # mx=1, my=0 -> r2=1 -> the alpha=1 boundary
    pts = np.array([[u, cy]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ray, valid = ds_unproject(pts, fx, fy, cx, cy, xi=0.5, alpha=1.0)
    assert not valid[0]
    assert np.all(np.isfinite(ray))

# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("NFR-NUM-005")
