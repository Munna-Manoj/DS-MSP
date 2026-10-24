"""Rigid multi-board target refinement (structure_mode="rigid", the default).

``build_objects`` seeds inter-board poses from averaged single-view PnP and no BA stage refines
them, so an error there stays frozen into the fused object unless stage (d) corrects it. These
tests perturb a non-reference board's pose in the object handed to ``calibrate_rig`` (the
observations are generated from the true target) and check that rigid refinement recovers it
while keeping every board's printed shape exactly.
"""
import copy
from dataclasses import replace

import numpy as np
import pytest

from ds_msp.core.lie import so3_exp
from ds_msp.rig import bundle
from ds_msp.rig.calib_param import load_config
from ds_msp.rig.calibrate import calibrate_rig, resolve_structure_mode
from ds_msp.rig.pipeline import make_fixed_intrinsic_front_end

from ._synth import make_rig

pytestmark = pytest.mark.req("FR-RIG-026")


def _rot_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def _perturbed(obj, board=1, rotvec=(0.0, 0.0175, 0.01), shift=(0.01, -0.006, 0.008)):
    bad = copy.deepcopy(obj)
    rows = [i for i, (b, _c) in enumerate(obj.pts_obj_2_board) if int(b) == board]
    c = obj.pts_3d[rows].mean(0)
    R = so3_exp(np.asarray(rotvec))
    bad.pts_3d[rows] = (obj.pts_3d[rows] - c) @ R.T + c + np.asarray(shift)
    D = np.eye(4)
    D[:3, :3] = R
    D[:3, 3] = c + np.asarray(shift) - R @ c
    bad.T_co_b[board] = D @ obj.T_co_b[board]           # keep board->object pose consistent
    return bad, rows


def _calibrate(obj, obs, img_size, gt_models, mode):
    return calibrate_rig(obj, copy.deepcopy(obs), img_size, fix_intrinsics=True,
                         front_end=make_fixed_intrinsic_front_end(gt_models),
                         structure_mode=mode)


def _ext_err(rig, gt_ext):
    rot = max(_rot_deg(rig.T_c_g[c][:3, :3], gt_ext[c][:3, :3]) for c in gt_ext)
    trans = max(np.linalg.norm(rig.T_c_g[c][:3, 3] - gt_ext[c][:3, 3]) for c in gt_ext)
    return rot, trans


def _pairwise(P):
    return np.linalg.norm(P[:, None] - P[None], axis=-1)


@pytest.fixture(scope="module")
def scene():
    obj, obs, img_size, gt_ext, gt_models = make_rig(n_cam=3, n_frame=30, noise_px=0.3, seed=4)
    bad, rows = _perturbed(obj)
    return obj, bad, rows, obs, img_size, gt_ext, gt_models


def test_rigid_recovers_interboard_pose_and_keeps_board_shape(scene):
    obj, bad, rows, obs, img_size, gt_ext, gt_models = scene
    rig = _calibrate(bad, obs, img_size, gt_models, "rigid")
    P = rig.objects[0].pts_3d
    # printed shape held exactly for every board
    for b in obj.board_ids:
        r = [i for i, (bb, _c) in enumerate(obj.pts_obj_2_board) if int(bb) == b]
        np.testing.assert_allclose(_pairwise(P[r]), _pairwise(obj.pts_3d[r]), atol=1e-9)
    # perturbed board returns to its true pose (gauge = reference board, held fixed)
    assert np.sqrt(np.mean(np.sum((P[rows] - obj.pts_3d[rows]) ** 2, 1))) < 1.5e-3
    # board->object pose stays consistent with the moved points
    T = rig.objects[0].T_co_b[1]
    nominal = (np.linalg.inv(obj.T_co_b[1]) @ np.c_[obj.pts_3d[rows], np.ones(len(rows))].T).T
    np.testing.assert_allclose((T @ nominal.T).T[:, :3], P[rows], atol=1e-9)


def test_rigid_beats_frozen_interboard_pose(scene):
    _obj, bad, _rows, obs, img_size, gt_ext, gt_models = scene
    off = _calibrate(bad, obs, img_size, gt_models, "off")
    rigid = _calibrate(bad, obs, img_size, gt_models, "rigid")
    rot_off, tr_off = _ext_err(off, gt_ext)
    rot_rigid, tr_rigid = _ext_err(rigid, gt_ext)
    assert rot_rigid < 0.05 and tr_rigid < 2e-3
    assert rot_rigid < 0.5 * rot_off or tr_rigid < 0.5 * tr_off
    rms_off = max(bundle.reprojection_rms(off, obs).values())
    rms_rigid = max(bundle.reprojection_rms(rigid, obs).values())
    assert rms_rigid < 0.45 < rms_off           # 0.3 px/axis noise = 0.42 px 2-D floor


def test_board_pose_jacobian_matches_finite_difference(scene):
    _obj, bad, _rows, obs, img_size, _gt_ext, gt_models = scene
    rig = _calibrate(bad, obs[:12], img_size, gt_models, "off")
    sub = [o for o in obs[:12] if (o.object_id, o.frame_id) in rig.object_poses]
    sub = [replace(o, pixel_covariances=np.tile([[2., .3], [.3, 1.]], (len(o.pts_2d), 1, 1)))
           for o in sub]
    state, residual, linearize, retract, k, ng = bundle.build_schur_problem(
        rig, sub, fix_intrinsics=False, refine_boards=True, bend_boards=True)
    n_intr = sum(len(type(m).param_names) for m in rig.cameras.values())
    assert k == 2 * 6 + n_intr + 2 * 6 + 3 * 3        # 2 non-ref cams, 2 movable boards, 3 bows
    state = retract(state, np.r_[np.zeros(k - 9), 1e-3 * np.arange(1, 10)], np.zeros((ng, 6)))
    _rs, As, _Bs = linearize(state)
    A = np.vstack(As)
    zero = np.zeros((ng, 6))
    eps = 1e-6
    numeric = np.column_stack([
        (residual(retract(state, np.eye(k)[i] * eps, zero))
         - residual(retract(state, -np.eye(k)[i] * eps, zero))) / (2 * eps) for i in range(k)])
    np.testing.assert_allclose(A, numeric, atol=2e-5, rtol=1e-4)


def test_structure_mode_resolution_and_config_default(tmp_path):
    assert resolve_structure_mode(None) == "rigid"
    assert resolve_structure_mode(None, refine_structure=True) == "free"
    assert resolve_structure_mode("OFF", refine_structure=True) == "off"
    with pytest.raises(ValueError):
        resolve_structure_mode("warp")
    cfg = tmp_path / "calib_param.yml"
    cfg.write_text("%YAML:1.0\n---\nnumber_camera: 1\nnumber_x_square: 5\nnumber_y_square: 5\n"
                   "root_path: \"\"\nsave_path: \"\"\n")
    assert load_config(str(cfg)).structure_mode == "rigid"
    assert load_config(str(cfg), {"structure_mode": "free"}).structure_mode == "free"
    assert load_config(str(cfg), {"refine_structure": "true"}).structure_mode == "free"


@pytest.mark.req("FR-RIG-025")
def test_structure_prior_is_weighted_by_pixel_noise(scene):
    """``prior_sigma`` is a target length: the MAP weight against reprojection is
    ``(σ_px/σ_target)²``, so (σ_target=s, σ_px=p) equals the unit-noise prior s/p."""
    _obj, bad, _rows, obs, img_size, _gt_ext, gt_models = scene
    rig = _calibrate(bad, obs, img_size, gt_models, "off")
    a = bundle.refine_object_structure(rig, obs, prior_sigma=0.002, pixel_sigma=0.25)
    b = bundle.refine_object_structure(rig, obs, prior_sigma=0.008, pixel_sigma=1.0)
    np.testing.assert_allclose(a.objects[0].pts_3d, b.objects[0].pts_3d, atol=1e-12)


def test_residual_sigma_recovers_injected_noise():
    obj, obs, img_size, gt_ext, gt_models = make_rig(n_cam=2, n_frame=20, noise_px=0.4, seed=2)
    rig = _calibrate(obj, obs, img_size, gt_models, "off")
    blocks = [(o.cam_id, (o.object_id, o.frame_id), np.asarray(o.point_rows),
               np.asarray(o.point_rows), o.pts_2d, None)
              for o in obs if (o.object_id, o.frame_id) in rig.object_poses]
    sigma = bundle._residual_sigma(rig, blocks, rig.objects[0].pts_3d)
    assert 0.33 < sigma < 0.47


def test_bend_recovers_a_bowed_board_that_rigid_cannot(monkeypatch):
    """A physically bowed board (quadratic sag) biases rigid refinement; bend models it."""
    from . import _synth
    flat_maker = _synth.make_object

    def bowed_maker(board_poses, **kw):
        bowed = flat_maker(board_poses, **kw)
        rows = [i for i, (b, _c) in enumerate(bowed.pts_obj_2_board) if int(b) == 1]
        q = bowed.pts_3d[rows] - bowed.pts_3d[rows].mean(0)
        _u, _s, Vt = np.linalg.svd(q, full_matrices=False)
        u = q @ Vt[0]
        sag = 0.012 * (u / np.abs(u).max()) ** 2                     # 12 mm over the board
        bowed.pts_3d[rows] += np.outer(sag - sag.mean(), Vt[2])
        return bowed

    monkeypatch.setattr(_synth, "make_object", bowed_maker)
    truth, obs, img_size, _gt_ext, gt_models = _synth.make_rig(n_cam=3, n_frame=30, seed=6)
    monkeypatch.undo()
    flat = _synth.make_object(dict(truth.T_co_b))                    # nominal printed target
    rows = [i for i, (b, _c) in enumerate(flat.pts_obj_2_board) if int(b) == 1]
    rigid = _calibrate(flat, obs, img_size, gt_models, "rigid")
    bend = _calibrate(flat, obs, img_size, gt_models, "bend")
    err = {m: np.sqrt(np.mean(np.sum((r.objects[0].pts_3d[rows] - truth.pts_3d[rows]) ** 2, 1)))
           for m, r in (("rigid", rigid), ("bend", bend))}
    assert err["bend"] < 2e-4 and err["rigid"] > 1e-3
    assert max(bundle.reprojection_rms(bend, obs).values()) < 0.02
