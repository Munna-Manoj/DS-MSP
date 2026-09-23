"""FR-RIG-024: full pairwise SE(3) and exact task-space truth metrics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ds_msp.core.lie import so3_exp
from ds_msp.rig.accuracy import (
    ACCURACY_SCHEMA,
    AccuracyProfile,
    accuracy_metrics,
    accuracy_profile_from_dict,
    camera_to_world_as_world_to_camera,
    compare_world_to_camera_poses,
    load_accuracy_profile,
)

pytestmark = pytest.mark.req("FR-RIG-024")


def _pose(*, center=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0)):
    """World-to-camera transform from camera center and axis-angle rotation."""
    T = np.eye(4)
    T[:3, :3] = so3_exp(np.asarray(rotation, float))
    T[:3, 3] = -T[:3, :3] @ np.asarray(center, float)
    return T


def _profile(**updates) -> AccuracyProfile:
    value = {
        "schema": "ds-msp-rig-accuracy-profile",
        "schema_version": 1,
        "name": "ten-metre-volume",
        "length_unit": "m",
        "roi": {"frame": "pair_source_camera", "min": [-1.0, -1.0, 2.0],
                "max": [1.0, 1.0, 10.0], "grid_shape": [3, 3, 5]},
        "required_pairs": [],
        "limits": {"task_space_p95_max": 0.01},
    }
    value.update(updates)
    return accuracy_profile_from_dict(value)


def test_exact_all_pair_metrics_are_zero_and_schema_versioned():
    poses = {0: _pose(), 1: _pose(center=(1.0, 0.0, 0.0)),
             2: _pose(center=(0.0, 2.0, 0.0), rotation=(0.0, 0.1, 0.0))}
    result = accuracy_metrics(
        poses, ground_truth_world_to_camera=poses,
        profile=_profile())

    assert result["schema"] == ACCURACY_SCHEMA
    assert result["profile"]["length_unit"] == "m"
    comparison = result["ground_truth"]
    assert comparison["status"] == "EVALUATED"
    assert comparison["summary"]["n_pairs"] == 3
    assert comparison["summary"]["rotation_error_deg_max"] == pytest.approx(0.0)
    assert comparison["summary"]["camera_center_error_max"] == pytest.approx(0.0)
    assert comparison["summary"]["relative_translation_error_max"] == pytest.approx(0.0)
    assert comparison["summary"]["task_space"]["max"] == pytest.approx(0.0)
    assert comparison["limit_checks"]["all_pass"] is True


def test_equal_baseline_with_wrong_direction_does_not_score_as_accurate():
    reference = {0: _pose(), 1: _pose(center=(1.0, 0.0, 0.0))}
    estimated = {0: _pose(), 1: _pose(center=(0.0, 1.0, 0.0))}

    pair = compare_world_to_camera_poses(
        estimated, reference, profile=_profile())["pairs"][0]

    assert pair["baseline_length_error"] == pytest.approx(0.0)
    assert pair["baseline_length_error_pct"] == pytest.approx(0.0)
    assert pair["camera_center_error"] == pytest.approx(np.sqrt(2.0))
    assert pair["translation_direction_error_deg"] == pytest.approx(90.0)
    assert pair["task_space"]["max"] == pytest.approx(np.sqrt(2.0))


def test_exact_task_space_detects_rotation_at_range_with_exact_translation_center():
    theta = np.deg2rad(2.0)
    reference = {0: _pose(), 1: _pose(center=(1.0, 0.0, 0.0))}
    estimated = {0: _pose(), 1: _pose(center=(1.0, 0.0, 0.0), rotation=(0.0, 0.0, theta))}
    profile = _profile(roi={"frame": "pair_source_camera", "points": [[1.0, 10.0, 0.0]]})

    pair = compare_world_to_camera_poses(estimated, reference, profile=profile)["pairs"][0]

    assert pair["camera_center_error"] == pytest.approx(0.0, abs=1e-12)
    assert pair["relative_translation_error"] == pytest.approx(
        2.0 * np.sin(theta / 2.0), abs=1e-12)
    assert pair["baseline_length_error"] == pytest.approx(0.0, abs=1e-12)
    assert pair["rotation_error_deg"] == pytest.approx(2.0)
    assert pair["task_space"]["max"] == pytest.approx(2.0 * 10.0 * np.sin(theta / 2.0))


def test_pair_metrics_are_invariant_to_a_common_world_frame_change():
    reference = {0: _pose(), 1: _pose(center=(1.0, 0.2, 0.1)),
                 2: _pose(center=(-0.4, 0.8, 0.2), rotation=(0.1, -0.2, 0.05))}
    estimated = dict(reference)
    estimated[2] = _pose(center=(-0.39, 0.82, 0.2), rotation=(0.11, -0.2, 0.05))
    before = compare_world_to_camera_poses(estimated, reference, profile=_profile())

    world_change = _pose(center=(3.0, -2.0, 1.0), rotation=(0.2, -0.1, 0.3))
    estimated_changed = {c: T @ world_change for c, T in estimated.items()}
    reference_changed = {c: T @ world_change for c, T in reference.items()}
    after = compare_world_to_camera_poses(
        estimated_changed, reference_changed, profile=_profile())

    for key in ("rotation_error_deg_max", "camera_center_error_max",
                "translation_direction_error_deg_max"):
        assert after["summary"][key] == pytest.approx(before["summary"][key], abs=1e-10)
    assert after["summary"]["task_space"]["max"] == pytest.approx(
        before["summary"]["task_space"]["max"], abs=1e-10)


def test_camera_to_world_normalization_is_an_exact_inverse():
    world_to_camera = {0: _pose(center=(0.1, -0.2, 0.3), rotation=(0.2, 0.1, -0.3))}
    normalized = camera_to_world_as_world_to_camera(
        {0: np.linalg.inv(world_to_camera[0])})
    assert normalized[0] == pytest.approx(world_to_camera[0], abs=1e-12)


def test_blender_gt_and_mccalib_output_conventions_are_both_normalized():
    poses = {0: _pose(), 1: _pose(center=(0.7, 0.0, 0.0))}
    result = accuracy_metrics(
        poses,
        ground_truth_world_to_camera=poses,
        mccalib_camera_to_world={c: np.linalg.inv(T) for c, T in poses.items()},
        profile=_profile())
    assert result["ground_truth"]["summary"]["camera_center_error_max"] == 0.0
    assert result["mccalib_reference"]["summary"]["camera_center_error_max"] == 0.0


def test_missing_required_pair_fails_closed_as_incomplete():
    result = compare_world_to_camera_poses(
        {0: _pose(), 1: _pose(center=(1.0, 0.0, 0.0))},
        {0: _pose(), 1: _pose(center=(1.0, 0.0, 0.0))},
        profile=_profile(required_pairs=[[0, 2]]))
    assert result["status"] == "INCOMPLETE"
    assert result["missing_pairs"] == [[0, 2]]
    assert result["summary"]["n_pairs"] == 0


def test_without_profile_pose_is_evaluated_but_task_space_is_not():
    poses = {0: _pose(), 1: _pose(center=(1.0, 0.0, 0.0))}
    result = compare_world_to_camera_poses(poses, poses)
    assert result["status"] == "EVALUATED"
    assert result["summary"]["task_space"]["status"] == "NOT_EVALUATED"
    assert result["limit_checks"]["status"] == "NOT_CONFIGURED"


def test_profile_file_is_strict_versioned_and_hash_stable(tmp_path):
    raw = {
        "schema": "ds-msp-rig-accuracy-profile",
        "schema_version": 1,
        "name": "explicit-points",
        "length_unit": "mm",
        "roi": {"frame": "pair_source_camera", "points": [[0, 0, 1000], [100, 0, 2000]]},
        "required_pairs": [[1, 0]],
        "limits": {"task_space_p95_max": 10.0, "rotation_error_deg_max": 0.1},
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    first = load_accuracy_profile(path)
    second = load_accuracy_profile(path)
    assert first.required_pairs == ((1, 0),)  # order is semantic: ROI is in the source frame
    assert first.roi_points.shape == (2, 3)
    assert first.sha256 == second.sha256 and len(first.sha256) == 64

    raw["limits"]["made_up_metric"] = 1.0
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown accuracy profile limit"):
        load_accuracy_profile(path)

    raw["limits"].pop("made_up_metric")
    raw["length_units"] = "mm"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown accuracy profile field"):
        load_accuracy_profile(path)
