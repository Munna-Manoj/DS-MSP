"""Frame recovery must explain the entire frame without changing the fixed rig."""
import copy

import numpy as np
import pytest

from ds_msp.core.lie import so3_exp
from ds_msp.models.kb import KannalaBrandtModel
from ds_msp.models.radtan import RadTanModel
from ds_msp.rig.pose_recovery import frame_support, recover_frame_poses
from ds_msp.rig.types import ObjectObs, RigState

from ._synth import make_object

pytestmark = pytest.mark.req("FR-RIG-001", "FR-RIG-029")


def _scene(model_cls=RadTanModel, *, corruption=0.0):
    rng = np.random.default_rng(721)
    side = np.eye(4)
    side[:3, :3] = so3_exp([0, .6, 0])
    side[:3, 3] = [.35, 0, .2]
    obj = make_object({0: np.eye(4), 1: side})
    cams = {i: model_cls(500, 510, 640, 480) for i in range(2)}
    ext = {0: np.eye(4), 1: np.eye(4)}
    ext[1][:3, :3] = so3_exp([0, -.12, .03])
    ext[1][:3, 3] = [.3, -.1, .05]
    poses, obs = {}, []
    for frame in range(3):
        T = np.eye(4)
        T[:3, :3] = so3_exp([.12 * frame, -.14, .08])
        T[:3, 3] = [-.2 + frame * .1, -.12, 2.0 + frame * .2]
        poses[(0, frame)] = T
        for cam in cams:
            C = ext[cam] @ T
            uv, valid = cams[cam].project(obj.pts_3d @ C[:3, :3].T + C[:3, 3])
            assert valid.all()
            uv += rng.normal(0, .03, uv.shape)
            if corruption and frame == 1:
                uv[:int(len(uv) * corruption)] += rng.uniform(80, 140, (int(len(uv) * corruption), 2))
            obs.append(ObjectObs(cam, frame, 0, np.arange(len(uv)), uv))
    rig = RigState(cameras=cams, T_c_g=ext, ref_cam_id=0, object_poses=poses,
                   objects={0: obj})
    return rig, obs


def _damage(rig, *, behind=False):
    rig = copy.deepcopy(rig)
    T = rig.object_poses[(0, 1)]
    T[:3, :3] = so3_exp([.8, -.5, .4])
    T[:3, 3] += [.7, -.5, -5 if behind else -.7]
    return rig


@pytest.mark.parametrize("model_cls", [RadTanModel, KannalaBrandtModel])
def test_recovers_bad_frame_with_outliers_and_preserves_fixed_rig(model_cls):
    truth, observations = _scene(model_cls, corruption=.15)
    damaged = _damage(truth)
    before = copy.deepcopy(damaged)
    recovered, records = recover_frame_poses(damaged, observations)
    assert len(records) == 1 and records[0]["accepted"]
    assert records[0]["after"]["support_fraction"] > .8
    assert records[0]["after"]["truncated_cost"] < records[0]["before"]["truncated_cost"]
    np.testing.assert_allclose(recovered.object_poses[(0, 1)], truth.object_poses[(0, 1)], atol=.002)
    for key, pose in before.object_poses.items():
        np.testing.assert_array_equal(damaged.object_poses[key], pose)
        if key != (0, 1):
            np.testing.assert_array_equal(recovered.object_poses[key], pose)
    for cam in before.cameras:
        np.testing.assert_array_equal(recovered.cameras[cam].params, before.cameras[cam].params)
        np.testing.assert_array_equal(recovered.T_c_g[cam], before.T_c_g[cam])
    np.testing.assert_array_equal(recovered.objects[0].pts_3d, before.objects[0].pts_3d)
    assert sum(len(obs.point_rows) for obs in observations) == 192


def test_invalid_old_projections_are_counted_and_can_be_recovered():
    truth, observations = _scene()
    recovered, records = recover_frame_poses(_damage(truth, behind=True), observations)
    record = records[0]
    assert record["before"]["n_valid"] == 0
    assert record["before"]["n_observed"] == 64
    assert record["before"]["truncated_cost"] == 64 * 9
    assert record["accepted"] and record["after"]["n_valid"] == 64
    np.testing.assert_allclose(recovered.object_poses[(0, 1)], truth.object_poses[(0, 1)], atol=.002)


def test_final_support_audit_counts_every_frame_and_missing_pose_without_mutation():
    truth, observations = _scene()
    damaged = _damage(truth, behind=True)
    del damaged.object_poses[(0, 2)]
    before = copy.deepcopy(damaged)
    records = frame_support(damaged, observations)
    assert [r["frame_id"] for r in records] == [0, 1, 2]
    assert [r["supported"] for r in records] == [True, False, False]
    assert [r["n_observed"] for r in records] == [64, 64, 64]
    assert records[1]["n_valid"] == records[2]["n_valid"] == 0
    assert records[1]["truncated_cost"] == records[2]["truncated_cost"] == 64 * 9
    for key in before.object_poses:
        np.testing.assert_array_equal(damaged.object_poses[key], before.object_poses[key])
    assert (0, 2) not in damaged.object_poses


def test_final_support_audit_counts_corners_from_a_missing_camera():
    rig, observations = _scene()
    del rig.cameras[1]
    records = frame_support(rig, observations)
    assert len(records) == 3
    assert all(r["n_observed"] == 64 and r["n_valid"] == 32 for r in records)
    assert all(r["support_fraction"] == .5 and not r["supported"] for r in records)


def test_one_camera_consensus_cannot_replace_a_frame_that_disagrees_with_the_other():
    truth, observations = _scene()
    for obs in observations:
        if obs.cam_id == 1 and obs.frame_id == 1:
            obs.pts_2d += [500, -400]
    damaged = _damage(truth)
    recovered, records = recover_frame_poses(damaged, observations)
    assert len(records) == 1 and not records[0]["accepted"]
    assert records[0]["reason"] == "no_supported_candidate"
    np.testing.assert_array_equal(recovered.object_poses[(0, 1)], damaged.object_poses[(0, 1)])


def test_missing_camera_cannot_make_recovery_claim_full_frame_support():
    truth, observations = _scene()
    damaged = _damage(truth)
    del damaged.cameras[1]
    recovered, records = recover_frame_poses(damaged, observations)
    record = next(record for record in records if record["frame_id"] == 1)
    assert record["before"]["n_observed"] == 64
    assert not record["accepted"]
    np.testing.assert_array_equal(recovered.object_poses[(0, 1)], damaged.object_poses[(0, 1)])


def test_missing_frame_pose_is_resected_without_dropping_its_observations():
    truth, observations = _scene()
    missing = copy.deepcopy(truth)
    del missing.object_poses[(0, 1)]
    recovered, records = recover_frame_poses(missing, observations)
    record = next(record for record in records if record["frame_id"] == 1)
    assert record["before"]["n_observed"] == 64
    assert record["before"]["n_valid"] == record["before"]["n_supported"] == 0
    assert record["accepted"] and record["after"]["n_supported"] == 64
    assert (0, 1) not in missing.object_poses
    np.testing.assert_allclose(recovered.object_poses[(0, 1)], truth.object_poses[(0, 1)], atol=.002)


def test_well_supported_frames_do_not_run_resection(monkeypatch):
    from ds_msp.rig import pose_recovery
    rig, observations = _scene()
    def forbidden(*args, **kwargs):
        raise AssertionError("a supported frame should not need PnP")
    monkeypatch.setattr(pose_recovery, "solve_pnp", forbidden)
    monkeypatch.setattr(pose_recovery, "solve_pnp_robust", forbidden)
    recovered, records = recover_frame_poses(rig, observations)
    assert records == []
    for key in rig.object_poses:
        np.testing.assert_array_equal(recovered.object_poses[key], rig.object_poses[key])


@pytest.mark.parametrize("kw", [{"threshold_px": 0}, {"threshold_px": float("nan")},
                               {"min_support": 0}, {"min_support": 1.01},
                               {"max_iter": 0}])
def test_invalid_policy_is_rejected(kw):
    rig, observations = _scene()
    with pytest.raises(ValueError):
        recover_frame_poses(rig, observations, **kw)


@pytest.mark.integration
@pytest.mark.parametrize("injection_stage", ["joint", "boards"])
def test_pipeline_recovers_before_rejection_and_records_late_repairs(monkeypatch, injection_stage):
    """Run real stages/resection; introduce one bad pose only at the specified boundary."""
    from ds_msp.rig import bundle
    from ds_msp.rig.calibrate import calibrate_rig
    from ds_msp.rig.pipeline import make_fixed_intrinsic_front_end, solver_acceptance
    truth, observations = _scene()
    native_refine = bundle.refine
    injected = []

    def with_one_bad_pose(*args, **kwargs):
        result = native_refine(*args, **kwargs)
        is_pipeline = kwargs.get("log") is not None
        is_board = kwargs.get("refine_boards", False)
        if (not injected and is_pipeline and kwargs.get("robust_kernel") == "cauchy"
                and is_board == (injection_stage == "boards")):
            result = _damage(result)
            injected.append(True)
        return result

    monkeypatch.setattr(bundle, "refine", with_one_bad_pose)
    result = calibrate_rig(
        truth.objects[0], copy.deepcopy(observations), {0: (1280, 960), 1: (1280, 960)},
        front_end=make_fixed_intrinsic_front_end(truth.cameras), fix_intrinsics=True,
        structure_mode="rigid" if injection_stage == "boards" else "off",
        reproj_gate_px=3.03)
    assert injected == [True]
    assert result.rejected == []
    ledger = solver_acceptance(result)["pose_recovery"]
    expected = "c4_pose_recovery" if injection_stage == "joint" else "e_pose_recovery"
    assert any(r["accepted"] and r["frame_id"] == 1 and r["stage"] == expected for r in ledger)
    assert solver_acceptance(result)["final_status"] == "CONVERGED"
    for cam in truth.cameras:
        np.testing.assert_array_equal(result.cameras[cam].params, truth.cameras[cam].params)
    from ds_msp.rig.pose_recovery import _frame_errors
    for frame in range(3):
        subset = [obs for obs in observations if obs.frame_id == frame]
        errors, valid = _frame_errors(result, subset, result.object_poses[(0, frame)])
        assert valid.all() and np.percentile(errors, 95) < .15


@pytest.mark.integration
def test_final_support_audit_catches_regression_after_recovery_polish(monkeypatch):
    """The recovery ledger alone must not certify the subsequent global solve."""
    from ds_msp.rig import bundle
    from ds_msp.rig.calibrate import calibrate_rig
    from ds_msp.rig.pipeline import (
        _gate_solver, make_fixed_intrinsic_front_end, solver_acceptance,
    )
    truth, observations = _scene()
    native_refine = bundle.refine
    injected = []

    def damage_board_refinement_and_final_polish(*args, **kwargs):
        result = native_refine(*args, **kwargs)
        if kwargs.get("log") is not None and kwargs.get("refine_boards"):
            result = _damage(result)
            injected.append(True)
        return result

    monkeypatch.setattr(bundle, "refine", damage_board_refinement_and_final_polish)
    result = calibrate_rig(
        truth.objects[0], copy.deepcopy(observations), {0: (1280, 960), 1: (1280, 960)},
        front_end=make_fixed_intrinsic_front_end(truth.cameras), fix_intrinsics=True,
        structure_mode="rigid")
    assert len(injected) == 2
    report = solver_acceptance(result)
    assert any(r["accepted"] and r["stage"] == "e_pose_recovery"
               for r in report["pose_recovery"])
    final = {r["frame_id"]: r for r in report["frame_support"]}
    assert not final[1]["supported"]
    assert final[0]["supported"] and final[2]["supported"]
    with pytest.warns(RuntimeWarning, match="support"):
        _gate_solver(report, convergence_gate="warn")
