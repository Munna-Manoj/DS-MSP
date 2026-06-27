"""EUCM+ unit tests: agrees with standalone eucmplus_math, reduces to EUCM when the
extra (division + tilt) DOF are zero, and its sqrt-only inverse round-trips tightly."""

import numpy as np

from ds_msp.models.eucmplus import EUCMPlusModel
from ds_msp.models.eucmplus_math import eucmplus_project
from ds_msp.models.eucm import EUCMModel
from ds_msp.testing import sample_forward_points


PARAMS = dict(fx=711.57, fy=711.24, cx=949.18, cy=518.81, alpha=0.62, beta=1.10,
              lambda1=-0.10, tau_x=0.001, tau_y=-0.001)


def test_matches_standalone_eucmplus_project():
    m = EUCMPlusModel(**PARAMS)
    P = sample_forward_points()
    uv, valid = m.project(P)
    u, v, val0 = eucmplus_project(P, **PARAMS)
    assert np.allclose(uv, np.stack([u, v], axis=-1))
    assert np.array_equal(valid, val0)


def test_reduces_to_eucm_when_extra_dof_zero():
    # lambda1=tau_x=tau_y=0 collapses the division+tilt stages, leaving the EUCM core.
    eup = EUCMPlusModel(700.0, 700.0, 640.0, 360.0, alpha=0.62, beta=1.1,
                        lambda1=0.0, tau_x=0.0, tau_y=0.0)
    eucm = EUCMModel(700.0, 700.0, 640.0, 360.0, 0.62, 1.1)
    P = sample_forward_points()
    uv_e, _ = eup.project(P)
    uv_u, _ = eucm.project(P)
    assert np.allclose(uv_e, uv_u, atol=1e-9)


def test_sqrt_only_roundtrip():
    # EUCM+'s whole inverse is solvable with square roots alone (no cube root, no
    # iteration) — it should round-trip to near machine precision.
    m = EUCMPlusModel(**PARAMS)
    P = sample_forward_points()
    uv, v1 = m.project(P)
    rays, v2 = m.unproject(uv)
    ok = v1 & v2
    d = P[ok] / np.linalg.norm(P[ok], axis=1, keepdims=True)
    cos = np.sum(rays[ok] * d, axis=1)
    assert (cos > 1 - 1e-9).all()


def test_K_and_distortion():
    m = EUCMPlusModel(**PARAMS)
    assert np.allclose(m.K, [[711.57, 0, 949.18], [0, 711.24, 518.81], [0, 0, 1]])
    assert np.allclose(m.distortion, [0.62, 1.10, -0.10, 0.001, -0.001])


def test_initialize_seed_is_reasonable():
    truth = EUCMPlusModel(**PARAMS)
    rng = np.random.default_rng(0)
    pix = rng.uniform([300, 200], [1600, 900], size=(200, 2)).astype(float)
    rays, valid = truth.unproject(pix)
    keep = valid & (rays[:, 2] > 1e-6)
    seed = EUCMPlusModel(0, 0, 0, 0)
    seed.initialize_from_correspondences(truth.K, rays[keep], pix[keep])
    lb, ub = EUCMPlusModel.param_bounds()
    assert (seed.params >= lb).all() and (seed.params <= ub).all()
    assert seed.beta == 1.0 and seed.lambda1 == 0.0
