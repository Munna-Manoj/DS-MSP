"""``jacobian.normal_matrix`` (per-observation block summation of ``JᵀWJ``) must equal the dense
product the observability audit used to form, for every Jacobian layout the BA builds."""

import numpy as np
import pytest

from ds_msp.rig import bundle
from ds_msp.rig.audit import audit_rig

from .test_bundle_jacobian import _build_small_rig

pytestmark = pytest.mark.req("NFR-PERF-006", "NFR-PERF-007")

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


def _gauge_deficient_H(seed=0, n=60, null=3):
    """Symmetric PSD H with ``null`` near-null directions mixed over all coordinates."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    ev = np.r_[np.full(null, 1e-9), rng.uniform(1.0, 50.0, n - null)]
    H = (Q * ev) @ Q.T
    return 0.5 * (H + H.T)


def test_subset_eigenvectors_match_full_eigh():
    """The audit's subset solve (lowest_eigh) reports what the full eigh reports."""
    from ds_msp.core.observability import eigen_weakness
    from ds_msp.rig.audit import _lowest_eigenvectors

    H = _gauge_deficient_H()
    full = eigen_weakness(H.copy(), tau_rel=1e-3, corr_thresh=0.95)
    sub = eigen_weakness(H.copy(), tau_rel=1e-3, corr_thresh=0.95, overwrite=True,
                         lowest_eigh=_lowest_eigenvectors)
    assert full["n_weak"] == sub["n_weak"] == 3
    np.testing.assert_array_equal(full["scale"], sub["scale"])
    assert full["pairs"] == sub["pairs"]
    np.testing.assert_allclose(sub["eigvals"], full["eigvals"], rtol=0,
                               atol=1e-12 * full["eigvals"].max())
    for a, b in zip(full["weak"], sub["weak"]):
        assert a["index"] == b["index"]
        np.testing.assert_allclose(b["energy"].sum(), 1.0, atol=1e-12)
    # the 3 weak vectors span the same subspace (individual vectors may rotate inside it)
    Ea = np.stack([w["energy"] for w in full["weak"]]).sum(0)
    Eb = np.stack([w["energy"] for w in sub["weak"]]).sum(0)
    np.testing.assert_allclose(Ea, Eb, atol=1e-8)


def _truth_rig(n_cam=6, n_frame=30):
    """make_rig scene as a RigState at the true cameras, object poses from PnP at the true models."""
    from tests.rig._synth import make_rig
    from ds_msp.core.lie import so3_exp
    from ds_msp.ops.pose import solve_pnp_robust
    from ds_msp.rig.types import RigState
    obj, obs, img_size, gt_ext, models = make_rig(n_cam=n_cam, n_frame=n_frame, seed=3)
    poses = {}
    for o in obs:
        key = (o.object_id, o.frame_id)
        if key in poses or len(o.point_rows) < 8:
            continue
        ok, rv, tv, _ = solve_pnp_robust(models[o.cam_id], obj.pts_3d[o.point_rows], o.pts_2d)
        if ok:
            T = np.eye(4)
            T[:3, :3] = so3_exp(np.ravel(rv))
            T[:3, 3] = np.ravel(tv)
            poses[key] = np.linalg.inv(gt_ext[o.cam_id]) @ T
    rig = RigState(cameras=models, T_c_g=dict(gt_ext), ref_cam_id=0, object_poses=poses,
                   objects={0: obj}, img_size=img_size)
    return rig, obs


def test_schur_linearize_never_forms_the_dense_shared_jacobian():
    """The sparse BA's shared Jacobian is expanded one group at a time: linearize's peak
    allocation stays far below the dense (N, 2, shared_dim) array it used to build."""
    import tracemalloc
    rig, obs = _truth_rig(n_cam=12)          # shared_dim 174: dense A ~14 MB vs ~4 MB of the rest
    state, _res, linearize, _ret, k, ng = bundle.build_schur_problem(rig, obs, fix_intrinsics=False)
    n_pts = sum(len(o.point_rows) for o in obs if (o.object_id, o.frame_id) in rig.object_poses)
    dense_bytes = n_pts * 2 * k * 8
    tracemalloc.start()
    rs, As, Bs = linearize(state)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 0.5 * dense_bytes, (peak, dense_bytes)
    # the lazy sequence behaves like the list it replaces
    assert len(As) == len(rs) == len(Bs) == ng
    blocks = list(As)
    np.testing.assert_array_equal(As[-1], blocks[-1])
    for r, A, B in zip(rs, blocks, Bs):
        assert A.shape == (len(r), k) and B.shape == (len(r), 6)
    with pytest.raises(IndexError):
        As[ng]
