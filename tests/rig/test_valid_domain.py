"""The model's valid projection domain is a constraint in every BA.

A point outside the domain at the start of a solve is left out and counted. A point the
solve uses that leaves the domain at a trial state gives a NaN residual, so LM rejects the
step. A zero residual there would make leaving the domain look like a perfect fit.
"""

import copy

import numpy as np
import pytest

from ds_msp.core.lie import so3_exp
from ds_msp.geometry.calibrate_core import bundle_adjust
from ds_msp.models.radtan import RadTanModel
from ds_msp.rig import bundle

from .test_bundle_jacobian import _build_small_rig

pytestmark = pytest.mark.req("NFR-NUM-013")


def _cam_rows(obs, cam, point_dim):
    """Residual-row mask of camera ``cam`` in the dense ``build_problem`` order."""
    return np.concatenate([np.full(point_dim * len(o.point_rows), o.cam_id == cam)
                           for o in obs])


def _behind(rig, cam):
    """Copy of ``rig`` with camera ``cam`` moved 100 units back, so it sees nothing in front."""
    rig = copy.deepcopy(rig)
    rig.T_c_g[cam] = rig.T_c_g[cam].copy()
    rig.T_c_g[cam][2, 3] -= 100.0
    return rig


@pytest.mark.parametrize("sparse", [False, True])
def test_leaving_the_domain_gives_nan_residuals(sparse):
    rig, obs = _build_small_rig()
    if sparse:
        state0, residual, *_ = bundle.build_schur_problem(rig, obs, fix_intrinsics=False)
    else:
        state0, residual, *_ = bundle.build_problem(rig, obs, fix_intrinsics=False)
    assert residual.excluded == 0
    assert np.isfinite(residual(state0)).all()

    state = dict(state0)
    state["cam_t"] = dict(state0["cam_t"])
    state["cam_t"][1] = state0["cam_t"][1] + [0.0, 0.0, -100.0]
    bad = ~np.isfinite(residual(state))
    n1 = sum(len(o.point_rows) for o in obs if o.cam_id == 1)
    assert bad.sum() == 2 * n1                     # exactly camera 1's rows
    if not sparse:
        assert np.array_equal(bad, _cam_rows(obs, 1, 2))


def test_angular_residual_leaving_the_domain_gives_nan():
    rig, obs = _build_small_rig()
    state0, residual, *_ = bundle.build_problem(rig, obs, fix_intrinsics=True,
                                                residual_mode="angular")
    state = dict(state0)
    state["cam_t"] = dict(state0["cam_t"])
    state["cam_R"] = dict(state0["cam_R"])
    # Put camera 1's centre on the first point it sees: a zero-length ray has no bearing.
    o = next(o for o in obs if o.cam_id == 1)
    key = (o.object_id, o.frame_id)
    Xg = state0["obj_R"][key] @ rig.objects[o.object_id].pts_3d[o.point_rows[0]] \
        + state0["obj_t"][key]
    state["cam_t"][1] = -state0["cam_R"][1] @ Xg
    assert np.isnan(residual(state)).any()


@pytest.mark.parametrize("sparse", [False, True])
def test_points_outside_the_domain_at_the_start_are_excluded_and_reported(sparse):
    rig, obs = _build_small_rig()
    far = _behind(rig, 1)
    n1 = sum(len(o.point_rows) for o in obs if o.cam_id == 1)
    build = bundle.build_schur_problem if sparse else bundle.build_problem
    state0, residual, *_ = build(far, obs, fix_intrinsics=True)
    assert residual.excluded == n1
    r = residual(state0)
    assert np.isfinite(r).all()

    log = []
    bundle.refine(far, obs, fix_intrinsics=True, sparse=sparse, log=log)
    assert log[-1]["excluded"] == n1
    assert log[-1]["status"] != "NUMERICAL_FAILURE"


def _views(model, n=6, seed=0):
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(7) * 0.05, np.arange(5) * 0.05)
    X = np.c_[gx.ravel() - 0.15, gy.ravel() - 0.1, np.zeros(gx.size)]
    Rs, ts, uvs = [], [], []
    for _ in range(n):
        R = so3_exp(rng.uniform(-0.3, 0.3, 3))
        t = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 0.6])
        uv, valid = model.project(X @ R.T + t)
        assert np.all(valid)
        Rs.append(R)
        ts.append(t)
        uvs.append(uv)
    return X, np.array(Rs), np.array(ts), uvs


def test_single_camera_bundle_excludes_and_counts_a_point_behind_the_camera():
    truth = RadTanModel(600.0, 600.0, 640.0, 480.0, -0.05, 0.01, 0.0, 0.0, 0.0)
    X, R, t, uvs = _views(truth)
    Xs = [X] * len(uvs)
    vis = [np.ones(len(X), bool) for _ in uvs]
    # One extra point 10 m behind the camera in view 0, with a made-up pixel.
    Xs[0] = np.vstack([X, [0.0, 0.0, -10.0]])
    uvs[0] = np.vstack([uvs[0], [10.0, 10.0]])
    vis[0] = np.ones(len(X) + 1, bool)
    p0 = truth.params * np.r_[1.02, 0.98, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    params, _R, _t, out = bundle_adjust(RadTanModel, p0, R, t, Xs, uvs, vis)
    assert out.excluded == 1
    assert out.status == "CONVERGED"
    np.testing.assert_allclose(params[:4], truth.params[:4], rtol=1e-6)
