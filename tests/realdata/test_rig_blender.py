"""Real-data rig validation gate (FR-RIG-001) — MC-Calib Blender datasets.

This is the real-data sign-off behind shipping ``ds_msp.rig``: it drives the full
config-based rig calibration on the MC-Calib Blender scenarios and asserts the recovered
extrinsics land within 1% of ground truth at sub-pixel reprojection, both with the authors'
intrinsics held fixed (``fix_intrinsic=1``) and with DS+ estimated from scratch.

Dataset-gated: set ``DSMSP_BLENDER_DIR`` to a ``Blender_Images`` directory, or place one at
the repo root. Skipped (not failed) when the imagery/keypoints are absent, like the other
``realdata`` tests — but required green for a release-gated rig release.
"""
from __future__ import annotations

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
fix_intrinsic: {fix}
cam_params_path: "{cam_params}"
root_path: "None"
cam_prefix: "Cam_"
keypoints_path: "{kp}"
ransac_threshold: 10
number_iterations: 1000
he_approach: 0
save_path: "None"
"""


def _root():
    env = os.environ.get("DSMSP_BLENDER_DIR")
    if env and os.path.isdir(env):
        return env
    repo = Path(__file__).resolve().parents[2] / "Blender_Images"
    return str(repo) if repo.is_dir() else None


pytestmark = [pytest.mark.realdata, pytest.mark.req("FR-RIG-001")]


@pytest.mark.parametrize("scn,mode", [
    ("Scenario_1", "given"),    # authors' intrinsics held fixed (radtan), extrinsics-only
    ("Scenario_1", "dsplus"),   # DS+ estimated from scratch
    ("Scenario_2", "dsplus"),   # 5-camera, non-globally-overlapping
])
def test_rig_extrinsics_within_1pct_of_gt(scn, mode, tmp_path):
    root = _root()
    if root is None:
        pytest.skip("Blender_Images not present (set DSMSP_BLENDER_DIR)")
    res_dir = os.path.join(root, scn, "Results")
    kp = os.path.join(res_dir, "detected_keypoints_data.yml")
    if not os.path.exists(kp):
        pytest.skip(f"{scn} detected keypoints not present under {res_dir}")

    meta = _SCN[scn]
    given = mode == "given"
    cfg_text = _CFG.format(
        nb=meta["nb"], ncam=meta["ncam"],
        model="radtan" if given else "dsplus",
        fix=1 if given else 0,
        cam_params=os.path.join(res_dir, "calibrated_cameras_data.yml") if given else "None",
        kp=kp)
    cfg_path = tmp_path / f"{scn}.{mode}.yml"
    cfg_path.write_text(cfg_text)

    res = calibrate_from_config(str(cfg_path))
    m = res["metrics"]
    assert m["worst_baseline_pct_vs_gt"] is not None, "no GroundTruth.yml found for the scenario"
    assert m["worst_baseline_pct_vs_gt"] < 1.0, \
        f"{scn}/{mode}: extrinsics {m['worst_baseline_pct_vs_gt']:.3f}% off GT (>1%)"
    assert m["max_rms_px"] < 1.0, f"{scn}/{mode}: reprojection {m['max_rms_px']:.3f}px (>=1px)"
