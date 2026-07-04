"""EUCM unit tests: EUCM with beta=1 reduces to UCM; round-trip holds."""

import warnings

import numpy as np

from ds_msp.models.eucm import EUCMModel
from ds_msp.models.eucm_math import eucm_unproject
from ds_msp.models.ucm import UCMModel
from ds_msp.testing import sample_forward_points


def test_eucm_beta1_equals_ucm():
    alpha = 0.62
    eucm = EUCMModel(700.0, 700.0, 640.0, 360.0, alpha, beta=1.0)
    ucm = UCMModel(700.0, 700.0, 640.0, 360.0, alpha)
    P = sample_forward_points()
    uv_e, _ = eucm.project(P)
    uv_u, _ = ucm.project(P)
    assert np.allclose(uv_e, uv_u, atol=1e-9)


def test_eucm_roundtrip():
    m = EUCMModel.sample()
    P = sample_forward_points()
    uv, v1 = m.project(P)
    rays, v2 = m.unproject(uv)
    ok = v1 & v2
    d = P[ok] / np.linalg.norm(P[ok], axis=1, keepdims=True)
    cos = np.sum(rays[ok] * d, axis=1)
    assert (cos > 1 - 1e-6).all()


def test_unproject_alpha_one_boundary_ray_is_finite_and_invalid_not_a_warning():
    """Regression: same mz-denominator gap as UCM (identical closed form) — an alpha=1.0
    boundary ray previously produced a NaN ray tagged valid=True."""
    fx = fy = 500.0
    cx = cy = 320.0
    pts = np.array([[cx + fx * 1.0, cy]])   # mx=1, my=0 -> r2=1 -> the alpha=1 boundary
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ray, valid = eucm_unproject(pts, fx, fy, cx, cy, alpha=1.0, beta=1.0)
    assert not valid[0]
    assert np.all(np.isfinite(ray))
