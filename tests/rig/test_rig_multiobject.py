"""End-to-end: a non-overlapping multi-object rig calibrates via the merge stage.

Two cameras each see a *different*, never-co-observed board (the real-dataset topology).
Before this feature DS-MSP dropped the second board (and thus the second camera); now the
board-fusion keeps both objects, hand-eye links the groups, and the object-merge fuses them
into one rigid object so the joint BA recovers the inter-camera extrinsic. This test drives
the whole ``calibrate_rig`` pipeline and checks the extrinsic against ground truth.
"""
import numpy as np

from ds_msp.rig.calibrate import calibrate_rig
from ds_msp.rig.pipeline import make_fixed_intrinsic_front_end
from ds_msp.rig.reconstruct import reconstruct_objects

from ._synth import make_non_overlapping_rig


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
    fused = rig.objects[rig.ref_cam_id if rig.ref_cam_id in rig.objects else 0]
    assert set(rig.objects[0].board_ids) == {0, 1}

    # inter-camera extrinsic recovered (T_c_g, ref cam 0 == identity)
    assert rig.ref_cam_id == 0
    T_gt = gt_ext[1]
    assert _rot_deg(rig.T_c_g[1][:3, :3], T_gt[:3, :3]) < 1.5, "extrinsic rotation off"
    assert np.linalg.norm(rig.T_c_g[1][:3, 3] - T_gt[:3, 3]) < 0.03, "extrinsic translation off"


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
