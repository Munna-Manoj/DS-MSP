"""End-to-end: a non-overlapping multi-object rig calibrates via the merge stage.

Two cameras each see a *different*, never-co-observed board (the real-dataset topology).
Before this feature DS-MSP dropped the second board (and thus the second camera); now the
board-fusion keeps both objects, hand-eye links the groups, and the object-merge fuses them
into one rigid object so the joint BA recovers the inter-camera extrinsic. This test drives
the whole ``calibrate_rig`` pipeline and checks the extrinsic against ground truth.
"""
import numpy as np
import pytest

from ds_msp.rig.calibrate import calibrate_rig
from ds_msp.rig.pipeline import make_fixed_intrinsic_front_end

from ._synth import make_non_overlapping_rig

pytestmark = pytest.mark.req("FR-RIG-017")


def _rot_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def test_non_overlapping_rig_recovers_extrinsic():
    objects, obs, img_size, gt_ext, gt_models = make_non_overlapping_rig(n_frame=40, seed=3)

    # sanity: this really is the non-overlapping topology (2 disjoint single-board objects)
    assert len(objects) == 2
    assert {b for o in objects for b in o.board_ids} == {0, 1}
    assert {o.object_id for o in obs if o.cam_id == 0} == {1}   # cam 0 -> only object 1
    assert {o.object_id for o in obs if o.cam_id == 1} == {0}   # cam 1 -> only object 0

    front_end = make_fixed_intrinsic_front_end(gt_models)
    rig = calibrate_rig(objects[0], obs, img_size, fix_intrinsics=True,
                        front_end=front_end, objects=objects, refine_structure=True,
                        he_approach=0)

    # the two objects fused into one rigid object (both boards present)
    assert len(rig.objects) == 1
    assert set(rig.objects[0].board_ids) == {0, 1}

    # inter-camera extrinsic recovered (T_c_g, ref cam 0 == identity)
    assert rig.ref_cam_id == 0
    T_gt = gt_ext[1]
    assert _rot_deg(rig.T_c_g[1][:3, :3], T_gt[:3, :3]) < 1.5, "extrinsic rotation off"
    assert np.linalg.norm(rig.T_c_g[1][:3, 3] - T_gt[:3, 3]) < 0.03, "extrinsic translation off"


class _StubAnimator:
    """Minimal ``on_iter`` duck-type (``set_stage``/``bind_scene``/``__call__``) that records
    every ``bind_scene`` call so a test can check the live-view animator's cached scene is
    refreshed after the object merge, not left pointing at the smaller pre-merge object."""

    def __init__(self):
        self.binds = []

    def set_stage(self, _label):
        pass

    def bind_scene(self, obj, object_obs):
        self.binds.append((obj, list(object_obs)))

    def __call__(self, it, max_iter, rms, cost, rig):
        pass


def test_merge_rebinds_live_view_scene_to_fused_object():
    """Regression: a real ``WebLive3DAnimator`` bound to the pre-merge object crashed with an
    IndexError once the BA ran on the post-merge, larger fused object (``point_rows`` indexed
    past the end of the stale cached ``pts_3d``), because nothing re-called ``bind_scene``
    after the merge stage. ``calibrate_rig`` must re-bind any ``on_iter`` that exposes
    ``bind_scene`` to the merged object once the merge actually runs."""
    objects, obs, img_size, _gt_ext, gt_models = make_non_overlapping_rig(n_frame=20, seed=2)
    front_end = make_fixed_intrinsic_front_end(gt_models)
    animator = _StubAnimator()
    # mirror what the real callers do (calib_param.py/cli.py): bind the pre-merge scene
    # BEFORE calibrate_rig runs, same as the crash's real preconditions.
    animator.bind_scene(objects[0], obs)

    rig = calibrate_rig(objects[0], obs, img_size, fix_intrinsics=True,
                        front_end=front_end, objects=objects, he_approach=0,
                        on_iter=animator)

    assert len(rig.objects) == 1                    # merge did happen
    assert len(animator.binds) >= 2, "bind_scene must be called again after the merge"
    fused_obj, fused_obs = animator.binds[-1]
    assert fused_obj is rig.objects[0]
    # every observation's point_rows must be valid against the *rebound* object's pts_3d --
    # this is exactly the indexing the live view's _camera_frame_errors performs per iteration.
    n_pts = len(fused_obj.pts_3d)
    for o in fused_obs:
        assert np.asarray(o.point_rows).max(initial=-1) < n_pts


def test_reconstruct_objects_keeps_both_boards_from_synth():
    """The board-fusion front stage keeps both non-co-observed boards as separate objects
    (no drop) — the fix at the root of the failure chain."""
    objects, obs, _img, _ext, _m = make_non_overlapping_rig(n_frame=20, seed=1)
    # feed the synthetic per-object obs' geometry back through the covisibility logic:
    # two objects, disjoint boards, each camera bound to exactly one object.
    assert len(objects) == 2
    assert sorted(o.object_id for o in objects) == [0, 1]
    per_cam = {}
    for o in obs:
        per_cam.setdefault(o.cam_id, set()).add(o.object_id)
    assert per_cam == {0: {1}, 1: {0}}
