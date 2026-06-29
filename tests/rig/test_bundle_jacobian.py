"""The critical test: the analytic BA Jacobian must match finite differences.

Guards the board->object->camera->project chain (implementation doc §9.1). A regression
here is the classic source of a BA that "converges" to the wrong answer.
"""

import numpy as np

from ds_msp.rig import bundle
from ds_msp.rig.calibrate import _front_end_opencv
from ds_msp.rig.types import RigState
from ._synth import make_rig


def _build_small_rig():
    obj, obs, img_size, gt_ext, model = make_rig(n_cam=2, n_frame=4, seed=3)
    from collections import defaultdict
    by_cam = defaultdict(list)
    for o in obs:
        by_cam[o.cam_id].append(o)
    cams = _front_end_opencv(obj, by_cam, img_size)
    # build a RigState from GT extrinsics + per-frame object poses (rough init via T_c_o)
    object_poses = {}
    for o in obs:
        key = (o.object_id, o.frame_id)
        if key not in object_poses and o.cam_id == 0 and o.T_c_o is not None:
            object_poses[key] = o.T_c_o
    for o in obs:                                   # fill any frames cam0 missed
        key = (o.object_id, o.frame_id)
        if key not in object_poses and o.T_c_o is not None:
            object_poses[key] = np.linalg.inv(gt_ext[o.cam_id]) @ o.T_c_o
    return RigState(cameras=cams, T_c_g=dict(gt_ext), ref_cam_id=0,
                    object_poses=object_poses, objects={0: obj}, img_size=img_size), obs


def _check(fix_intrinsics, fix_extrinsics=False):
    rig, obs = _build_small_rig()
    state0, residual, jacobian, retract, K = bundle.build_problem(
        rig, obs, fix_intrinsics=fix_intrinsics, fix_extrinsics=fix_extrinsics)
    J = jacobian(state0)
    eps = 1e-6
    rng = np.random.default_rng(1)
    # check a random subset of tangent directions
    for j in rng.choice(K, size=min(K, 25), replace=False):
        d = np.zeros(K)
        d[j] = eps
        rp = residual(retract(state0, d))
        rm = residual(retract(state0, -d))
        fd = (rp - rm) / (2 * eps)
        assert np.allclose(J[:, j], fd, atol=1e-3, rtol=1e-3), \
            f"Jacobian column {j} mismatch (fix_intrinsics={fix_intrinsics}, " \
            f"fix_extrinsics={fix_extrinsics})"


def test_jacobian_poses_only():
    _check(fix_intrinsics=True)


def test_jacobian_with_intrinsics():
    _check(fix_intrinsics=False)


def test_jacobian_object_poses_only():
    # the per-object intermediate stage: cameras + intrinsics fixed, only object poses
    _check(fix_intrinsics=True, fix_extrinsics=True)


def test_jacobian_angular_bearing_residual():
    """The bearing (angular) residual's analytic Jacobian must match finite differences —
    same chain, with ∂r/∂Xc = E·(I-d dᵀ)/‖Xc‖ replacing the projection Jacobian."""
    rig, obs = _build_small_rig()
    state0, residual, jacobian, retract, K = bundle.build_problem(
        rig, obs, fix_intrinsics=True, residual_mode="angular")
    J = jacobian(state0)
    eps = 1e-6
    rng = np.random.default_rng(2)
    for j in rng.choice(K, size=min(K, 25), replace=False):
        d = np.zeros(K)
        d[j] = eps
        fd = (residual(retract(state0, d)) - residual(retract(state0, -d))) / (2 * eps)
        assert np.allclose(J[:, j], fd, atol=1e-3, rtol=1e-3), f"angular Jac col {j}"


def test_angular_refine_recovers_extrinsics():
    """Refining with the bearing residual pulls perturbed extrinsics back to ground truth."""
    import copy
    from ds_msp.core.lie import so3_exp
    rig, obs = _build_small_rig()
    pert = copy.copy(rig)
    pert.T_c_g = dict(rig.T_c_g)
    for c in list(pert.T_c_g):
        if c == pert.ref_cam_id:
            continue
        T = pert.T_c_g[c].copy()
        T[:3, :3] = T[:3, :3] @ so3_exp([0.012, -0.009, 0.007])
        T[:3, 3] += 0.012
        pert.T_c_g[c] = T
    before = bundle.reprojection_rms(pert, obs)
    out = bundle.refine(pert, obs, fix_intrinsics=True, residual_mode="angular", max_iter=60)
    after = bundle.reprojection_rms(out, obs)
    assert max(after.values()) < 0.2 * max(before.values()) + 1e-6


def test_refine_object_structure_reduces_reprojection():
    """Perturbing the fused object's non-reference points and refining structure (cameras +
    poses fixed) drives reprojection back down — MC-Calib's refineObject."""
    rig, obs = _build_small_rig()
    bad = bundle._rig_from_state(rig, bundle._state_from_rig(rig))      # deep-ish copy
    import copy
    new_obj = copy.copy(rig.objects[0])
    pts = rig.objects[0].pts_3d.copy()
    free = [i for i, (b, _c) in enumerate(rig.objects[0].pts_obj_2_board)
            if int(b) != rig.objects[0].ref_board_id]
    rng = np.random.default_rng(5)
    pts[free] += rng.normal(scale=0.01, size=(len(free), 3))    # corrupt non-ref structure
    new_obj.pts_3d = pts
    bad.objects = {0: new_obj}
    before = max(bundle.reprojection_rms(bad, obs).values())
    fixed = bundle.refine_object_structure(bad, obs, iters=15)
    after = max(bundle.reprojection_rms(fixed, obs).values())
    assert after < 0.5 * before
