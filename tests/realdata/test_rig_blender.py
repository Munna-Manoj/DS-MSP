"""Real-data rig validation gate (FR-RIG-001, FR-RIG-024) — MC-Calib Blender datasets.

This is the real-data sign-off behind shipping ``ds_msp.rig``: it drives the full
config-based rig calibration on the MC-Calib Blender scenarios and asserts the recovered
full all-pair pose/task-space metrics are reported against independent truth at sub-pixel
training fit, both with the authors' intrinsics held fixed and with DS+ estimated from scratch.

Dataset-gated: set ``DSMSP_BLENDER_DIR`` to a ``Blender_Images`` directory, or place one at
the repo root. Skipped (not failed) when the imagery/keypoints are absent, like the other
``realdata`` tests — but required green for a release-gated rig release.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ds_msp.rig.calib_param import calibrate_from_config

# board geometry is identical across the set (5x5 squares); (number_board, number_camera) vary.
_SCN = {"Scenario_1": dict(nb=3, ncam=2), "Scenario_2": dict(nb=1, ncam=5)}

_CFG = """%YAML:1.0
number_x_square: 5
number_y_square: 5
length_square: 0.04
length_marker: 0.03
number_board: {nb}
square_size: 0.192
number_camera: {ncam}
camera_models: ["{model}"]
fix_intrinsic: "{fix}"
cam_params_path: "{cam_params}"
root_path: "None"
cam_prefix: "Cam_"
keypoints_path: "{kp}"
object_path: "{object_path}"
ransac_threshold: 10
number_iterations: 1000
he_approach: 0
save_path: "None"
verbose: "false"
webviewer: "false"
accuracy_profile_path: "{accuracy_profile}"
"""

_PROFILE = {
    "schema": "ds-msp-rig-accuracy-profile", "schema_version": 1,
    "name": "blender-release-1-to-10m", "length_unit": "m",
    "roi": {"frame": "pair_source_camera", "min": [-2.0, -1.0, 1.0],
            "max": [2.0, 1.0, 10.0], "grid_shape": [5, 3, 10]},
    "required_pairs": [],
    "limits": {"rotation_error_deg_max": 0.1, "camera_center_error_max": 0.01,
               "translation_direction_error_deg_max": 0.5,
               "task_space_p95_max": 0.01, "task_space_max": 0.02},
}


def _root():
    env = os.environ.get("DSMSP_BLENDER_DIR")
    if env and os.path.isdir(env):
        return env
    repo = Path(__file__).resolve().parents[2] / "Blender_Images"
    return str(repo) if repo.is_dir() else None


pytestmark = [pytest.mark.realdata, pytest.mark.req("FR-RIG-001", "FR-RIG-024")]


@pytest.mark.parametrize("scn,mode", [
    ("Scenario_1", "given"),    # authors' intrinsics held fixed (radtan), extrinsics-only
    ("Scenario_1", "dsplus"),   # DS+ estimated from scratch
    ("Scenario_2", "dsplus"),   # 5-camera, non-globally-overlapping
])
def test_rig_full_pose_and_roi_accuracy_vs_gt(scn, mode, tmp_path):
    root = _root()
    if root is None:
        pytest.skip("Blender_Images not present (set DSMSP_BLENDER_DIR)")
    res_dir = os.path.join(root, scn, "Results")
    kp = os.path.join(res_dir, "detected_keypoints_data.yml")
    if not os.path.exists(kp):
        pytest.skip(f"{scn} detected keypoints not present under {res_dir}")

    meta = _SCN[scn]
    given = mode == "given"
    profile_path = tmp_path / "accuracy_profile.json"
    profile_path.write_text(json.dumps(_PROFILE))
    cfg_text = _CFG.format(
        nb=meta["nb"], ncam=meta["ncam"],
        model="radtan" if given else "dsplus",
        fix="true" if given else "false",
        cam_params=os.path.join(res_dir, "calibrated_cameras_data.yml") if given else "None",
        kp=kp,
        object_path=Path(kp).resolve().parent / "calibrated_objects_data.yml",
        accuracy_profile=profile_path)
    cfg_path = tmp_path / f"{scn}.{mode}.yml"
    cfg_path.write_text(cfg_text)

    res = calibrate_from_config(str(cfg_path))
    m = res["metrics"]
    accuracy = m["accuracy"]["ground_truth"]
    assert accuracy["status"] == "EVALUATED", accuracy
    assert accuracy["summary"]["n_pairs"] == meta["ncam"] * (meta["ncam"] - 1) // 2
    assert accuracy["summary"]["task_space"]["status"] == "EVALUATED"
    assert isinstance(accuracy["limit_checks"]["all_pass"], bool)
    if given:
        assert accuracy["limit_checks"]["all_pass"] is True, accuracy["limit_checks"]
    elif scn == "Scenario_1":
        # Regression for the exact failure the old baseline-only claim concealed: scratch DS+
        # keeps baseline length within 1%, yet its angular error creates >10 cm displacement
        # over the 1-10 m ROI. A truthful evaluator must fail the declared 10 mm profile.
        assert m["worst_baseline_pct_vs_gt"] < 1.0
        assert accuracy["limit_checks"]["all_pass"] is False
        assert accuracy["summary"]["task_space"]["p95"] > 0.1
    assert m["max_rms_px"] < 1.0, f"{scn}/{mode}: reprojection {m['max_rms_px']:.3f}px (>=1px)"
