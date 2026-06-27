"""DS+ unit tests: agrees with standalone dsplus_math, reduces to UCM when the
extra (division + tilt) DOF are zero, and its closed-form inverse round-trips."""

import numpy as np

from ds_msp.models.dsplus import DSPlusModel
from ds_msp.models.dsplus_math import dsplus_project
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
