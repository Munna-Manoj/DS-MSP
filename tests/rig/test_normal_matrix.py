"""``jacobian.normal_matrix`` (per-observation block summation of ``JᵀWJ``) must equal the dense
product the observability audit used to form, for every Jacobian layout the BA builds."""

import numpy as np
import pytest

from ds_msp.rig import bundle
from ds_msp.rig.audit import audit_rig

from .test_bundle_jacobian import _build_small_rig

pytestmark = pytest.mark.req("NFR-PERF-006")

LAYOUTS = [
    dict(fix_intrinsics=True),
    dict(fix_intrinsics=False),
    dict(fix_intrinsics=True, fix_extrinsics=True),
    dict(fix_intrinsics=True, residual_mode="angular"),
]


def _dense(jacobian, state, w):
    J = np.asarray(jacobian(state), float)
    return (J * w[:, None]).T @ J


def _close(H, ref):
    scale = np.abs(ref).max()
    np.testing.assert_allclose(H, ref, rtol=0, atol=1e-12 * scale)


@pytest.mark.parametrize("layout", LAYOUTS, ids=lambda d: ",".join(f"{k}={v}" for k, v in d.items()))
def test_normal_matrix_equals_dense_product(layout):
    rig, obs = _build_small_rig()
    state0, residual, jacobian, _retract, K = bundle.build_problem(rig, obs, **layout)
    rows = residual(state0).size
    w = np.random.default_rng(0).uniform(0.1, 2.0, rows)
    _close(jacobian.normal_matrix(state0, w), _dense(jacobian, state0, w))
    _close(jacobian.normal_matrix(state0), _dense(jacobian, state0, np.ones(rows)))
    assert jacobian.normal_matrix(state0).shape == (K, K)


def test_normal_matrix_with_pixel_covariances():
    rig, obs = _build_small_rig()
    rng = np.random.default_rng(1)
    for o in obs[::2]:
        A = rng.normal(0, 0.3, (len(o.pts_2d), 2, 2))
        o.pixel_covariances = np.einsum("nij,nkj->nik", A, A) + 0.5 * np.eye(2)
    state0, residual, jacobian, _retract, _K = bundle.build_problem(rig, obs, fix_intrinsics=False)
    w = rng.uniform(0.1, 2.0, residual(state0).size)
    _close(jacobian.normal_matrix(state0, w), _dense(jacobian, state0, w))


def test_normal_matrix_rejects_wrong_weight_length():
    rig, obs = _build_small_rig()
    state0, residual, jacobian, _retract, _K = bundle.build_problem(rig, obs)
    with pytest.raises(ValueError, match="row_weights"):
        jacobian.normal_matrix(state0, np.ones(residual(state0).size + 2))


def test_audit_never_forms_the_dense_jacobian(monkeypatch):
    """The audit must take H from the blocked sum: calling the dense Jacobian is an error."""
    rig, obs = _build_small_rig()
    expected = audit_rig(rig, obs, fix_intrinsics=False)
    real = bundle.build_problem

    def no_dense(*a, **k):
        state0, residual, jacobian, retract, K = real(*a, **k)

        def forbidden(state):
            raise AssertionError("audit formed the dense Jacobian")
        forbidden.normal_matrix = jacobian.normal_matrix
        return state0, residual, forbidden, retract, K

    monkeypatch.setattr(bundle, "build_problem", no_dense)
    got = audit_rig(rig, obs, fix_intrinsics=False)
    assert got["n_weak"] == expected["n_weak"] and got["gauge_ok"] == expected["gauge_ok"]
    assert np.isclose(got["cond"], expected["cond"], rtol=1e-9)
