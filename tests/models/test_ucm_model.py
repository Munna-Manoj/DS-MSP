"""UCM unit tests: UCM must equal Double Sphere with xi = 0."""

import warnings

import numpy as np

from ds_msp.models.ucm import UCMModel
from ds_msp.models.ucm_math import ucm_unproject
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.testing import sample_forward_points


def test_ucm_equals_ds_with_zero_xi():
    alpha = 0.62
    ucm = UCMModel(700.0, 700.0, 640.0, 360.0, alpha)
    ds = DoubleSphereModel(700.0, 700.0, 640.0, 360.0, 0.0, alpha)
    P = sample_forward_points()
    uv_u, vu = ucm.project(P)
    uv_d, vd = ds.project(P)
    assert np.allclose(uv_u, uv_d, atol=1e-9)
    assert np.array_equal(vu, vd)
    # unprojection agreement
    rays_u, _ = ucm.unproject(uv_u)
    rays_d, _ = ds.unproject(uv_d)
    assert np.allclose(rays_u, rays_d, atol=1e-9)


def test_ucm_roundtrip():
    m = UCMModel.sample()
    P = sample_forward_points()
    uv, v1 = m.project(P)
    rays, v2 = m.unproject(uv)
    ok = v1 & v2
    d = P[ok] / np.linalg.norm(P[ok], axis=1, keepdims=True)
    cos = np.sum(rays[ok] * d, axis=1)
    assert (cos > 1 - 1e-6).all()


def test_unproject_alpha_one_boundary_ray_is_finite_and_invalid_not_a_warning():
    """Regression: unlike DS, UCM's unproject had NO second validity check downstream of the
    mz-denominator division — an alpha=1.0 (a legal optimizer bound) boundary ray produced a
    NaN ray tagged valid=True, silently corrupting anything that trusted the mask."""
    fx = fy = 500.0
    cx = cy = 320.0
    pts = np.array([[cx + fx * 1.0, cy]])   # mx=1, my=0 -> r2=1 -> the alpha=1 boundary
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ray, valid = ucm_unproject(pts, fx, fy, cx, cy, alpha=1.0)
    assert not valid[0]
    assert np.all(np.isfinite(ray))
