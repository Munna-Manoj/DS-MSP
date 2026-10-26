"""DS+ unit tests: agrees with standalone dsplus_math, reduces to UCM when the
extra (division + tilt) DOF are zero, and its closed-form inverse round-trips."""

import warnings

import numpy as np
import pytest

from ds_msp.models.dsplus import DSPlusModel
from ds_msp.models.dsplus_math import _invert_division, dsplus_project, dsplus_unproject
from ds_msp.models.ucm import UCMModel
from ds_msp.testing import sample_forward_points


PARAMS = dict(fx=711.57, fy=711.24, cx=949.18, cy=518.81, alpha=0.62,
              lambda1=-0.10, lambda2=0.02, tau_x=0.001, tau_y=-0.001)


def test_matches_standalone_dsplus_project():
    m = DSPlusModel(**PARAMS)
    P = sample_forward_points()
    uv, valid = m.project(P)
    u, v, val0 = dsplus_project(P, **PARAMS)
    assert np.allclose(uv, np.stack([u, v], axis=-1))
    assert np.array_equal(valid, val0)


def test_reduces_to_ucm_when_extra_dof_zero():
    # lambda1=lambda2=tau_x=tau_y=0 collapses the division+tilt stages, leaving the
    # bare UCM(alpha) core (DS+ is DS with xi dropped).
    dsp = DSPlusModel(700.0, 700.0, 640.0, 360.0, alpha=0.62,
                      lambda1=0.0, lambda2=0.0, tau_x=0.0, tau_y=0.0)
    ucm = UCMModel(700.0, 700.0, 640.0, 360.0, 0.62)
    P = sample_forward_points()
    uv_d, _ = dsp.project(P)
    uv_u, _ = ucm.project(P)
    assert np.allclose(uv_d, uv_u, atol=1e-9)


def test_closed_form_roundtrip():
    m = DSPlusModel(**PARAMS)
    P = sample_forward_points()
    uv, v1 = m.project(P)
    rays, v2 = m.unproject(uv)
    ok = v1 & v2
    d = P[ok] / np.linalg.norm(P[ok], axis=1, keepdims=True)
    cos = np.sum(rays[ok] * d, axis=1)
    assert (cos > 1 - 1e-6).all()


def test_K_and_distortion():
    m = DSPlusModel(**PARAMS)
    assert np.allclose(m.K, [[711.57, 0, 949.18], [0, 711.24, 518.81], [0, 0, 1]])
    assert np.allclose(m.distortion, [0.62, -0.10, 0.02, 0.001, -0.001])


def test_initialize_seed_is_reasonable():
    # Correspondences from a known DS+ camera; a fresh model seeded from them must
    # land in-bounds (the UCM linear seed sets alpha, zeroes the extra DOF).
    truth = DSPlusModel(**PARAMS)
    rng = np.random.default_rng(0)
    pix = rng.uniform([300, 200], [1600, 900], size=(200, 2)).astype(float)
    rays, valid = truth.unproject(pix)
    keep = valid & (rays[:, 2] > 1e-6)
    seed = DSPlusModel(0, 0, 0, 0)
    seed.initialize_from_correspondences(truth.K, rays[keep], pix[keep])
    lb, ub = DSPlusModel.param_bounds()
    assert (seed.params >= lb).all() and (seed.params <= ub).all()
    assert seed.lambda1 == 0.0 and seed.lambda2 == 0.0


def test_unproject_alpha_one_boundary_ray_is_finite_and_invalid_not_a_warning():
    """Regression: same mz-denominator gap as UCM (identical closed form, DS+ adds tilt +
    division stages ahead of it) — an alpha=1.0 boundary ray previously produced a NaN ray
    tagged valid=True."""
    fx = fy = 500.0
    cx = cy = 320.0
    pts = np.array([[cx + fx * 1.0, cy]])   # mx=1, my=0 -> r2=1 -> the alpha=1 boundary
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ray, valid = dsplus_unproject(pts, fx, fy, cx, cy, alpha=1.0, lambda1=0.0,
                                      lambda2=0.0, tau_x=0.0, tau_y=0.0)
    assert not valid[0]
    assert np.all(np.isfinite(ray))


def test_invert_division_falls_back_when_quartic_has_no_positive_real_root():
    """Regression: the 2-term (lambda2 != 0) distortion inversion picks the smallest positive
    real root of a quartic; for some (lambda1, lambda2, s) no such root exists (a normalized
    radius beyond where the forward distortion map is invertible). Previously this silently
    returned +inf with no fallback, poisoning every downstream computation with inf/nan —
    unlike every other branch of this function, which already falls back to ``s``. Concrete
    repro found by random search over (lambda1, lambda2, s)."""
    lambda1, lambda2, s = 0.42654310306871945, 0.9179862439359936, 1.6313313494048032
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rho = _invert_division(np.array([s]), lambda1, lambda2)
    assert np.isfinite(rho[0])
    assert rho[0] == s                      # no valid root -> falls back to the identity


@pytest.mark.req("FR-MODEL-001")
def test_tilt_reproduces_a_camera_rotation_exactly_in_the_pinhole_limit():
    # Why DS+ is experimental (ADR-0027): with a perspective core (alpha=0, no
    # division), tau_y = tan(theta) plus a focal/centre change reproduces a camera
    # rotation by theta about x exactly, so a rig cannot separate tilt from rotation.
    theta = np.deg2rad(10.0)
    c, s, t = np.cos(theta), np.sin(theta), np.tan(theta)
    fx, fy, cx, cy = 700.0, 690.0, 640.0, 360.0
    P = sample_forward_points()
    P = P[P[:, 2] > 0.5 * np.linalg.norm(P, axis=1)]
    R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    rotated = DSPlusModel(fx, fy, cx, cy, alpha=0.0)
    tilted = DSPlusModel(fx / c, fy / c**2, cx, cy - fy * t, alpha=0.0, tau_y=t)
    uv_rot, ok_rot = rotated.project(P @ R.T)
    uv_tilt, ok_tilt = tilted.project(P)

    ok = ok_rot & ok_tilt
    assert ok.sum() > 0.5 * len(P)
    assert np.abs(uv_rot[ok] - uv_tilt[ok]).max() < 1e-9
