"""Real-data validation gate for the non-overlapping multi-object rig fix (FR-RIG-017,
ADR-0011).

Two RealSense cameras, each looking at a *different* face of a double-sided ChArUco target,
never co-observed in any single frame (MC-Calib topology 5). Before ADR-0011 this silently
calibrated as a 1-camera "rig" (the board camera 0 alone observes was dropped as the
minority covisibility component); this asserts both cameras calibrate, with the recovered
extrinsic close to the previously-measured real-data acceptance numbers (1.192 m baseline,
178.6 deg rotation -- the cameras face each other across the target).

Dataset-gated: set ``DSMSP_TWO_CAMERA_DIR`` to a directory containing ``calib_param_gaze.yml`` +
imagery, or place it at ``data/`` in the repo root (gitignored). Skipped (not failed) when
absent, like the other ``realdata`` tests -- but
required green for a release-gated rig release (FR-RIG-001 covers this code path).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from ds_msp.rig.calib_param import calibrate_from_config


def _root():
    env = os.environ.get("DSMSP_TWO_CAMERA_DIR")
    if env and os.path.isdir(env):
        return env
    repo = Path(__file__).resolve().parents[2] / "data"
    return str(repo) if repo.is_dir() else None


def _rel_extrinsic(rig):
    T0, T1 = rig.T_c_g[0], rig.T_c_g[1]
    T_rel = T1 @ np.linalg.inv(T0)
    baseline_m = float(np.linalg.norm(T_rel[:3, 3])) * 1e-3   # square_size units are mm here
    ang_deg = float(np.degrees(np.arccos(
        np.clip((np.trace(T_rel[:3, :3]) - 1) / 2, -1, 1))))
    return baseline_m, ang_deg


pytestmark = [pytest.mark.realdata, pytest.mark.req("FR-RIG-001", "FR-RIG-017", "FR-RIG-029")]


def test_non_overlapping_two_camera_rig_calibrates_both_cameras(tmp_path):
    root = _root()
    if root is None:
        pytest.skip("Two-camera rig imagery not present (set DSMSP_TWO_CAMERA_DIR)")
    cfg_path = os.path.join(root, "calib_param_gaze.yml")
    if not os.path.exists(cfg_path):
        pytest.skip(f"calib_param_gaze.yml not present under {root}")

    res = calibrate_from_config(cfg_path, overrides={
        "root_path": root,
        "cam_params_path": os.path.join(root, "calibrated_cameras_data_initial_params.yml"),
        "save_path": str(tmp_path),
        "webviewer": False,
    })

    rig = res["rig"]
    assert sorted(rig.T_c_g.keys()) == [0, 1], \
        "both cameras must calibrate -- the pre-fix bug silently dropped camera 0"
    assert res["metrics"]["max_rms_px"] < 1.5, \
        f"reprojection {res['metrics']['max_rms_px']:.3f}px higher than expected (~0.8px)"

    baseline_m, ang_deg = _rel_extrinsic(rig)
    assert abs(baseline_m - 1.192) < 0.10, f"baseline {baseline_m:.3f}m far from ~1.192m"
    assert abs(ang_deg - 178.6) < 3.0, f"rotation {ang_deg:.1f}deg far from ~178.6deg"

    # The final support ledger must describe the actual returned real-pixel rig,
    # even when no recovery was necessary on this acquisition.
    from ds_msp.rig.pose_recovery import frame_support
    recorded = rig.solver_report["frame_support"]
    assert recorded
    recomputed = frame_support(rig, res["scenario"].object_obs, threshold_px=recorded[0]["threshold_px"])
    assert recorded == recomputed

    # Exercise recovery on real detected pixels with one deliberately damaged
    # initialization. This tests resection, not independent physical pose truth.
    import copy
    from ds_msp.rig.pose_recovery import recover_frame_poses
    selected = max((r for r in recorded if r["supported"]), key=lambda r: r["n_observed"])
    key = (selected["object_id"], selected["frame_id"])
    damaged = copy.deepcopy(rig)
    damaged.object_poses[key][:3, 3] += [500, 500, 500]  # scene units: mm
    observations = res["scenario"].object_obs
    before = [(o.point_rows.copy(), o.pts_2d.copy()) for o in observations]
    recovered, attempts = recover_frame_poses(
        damaged, observations, threshold_px=selected["threshold_px"])
    attempt = next(r for r in attempts if (r["object_id"], r["frame_id"]) == key)
    assert attempt["accepted"]
    assert attempt["after"]["support_fraction"] >= .75
    assert attempt["after"]["n_observed"] == selected["n_observed"]
    for obs, (rows, pixels) in zip(observations, before):
        np.testing.assert_array_equal(obs.point_rows, rows)
        np.testing.assert_array_equal(obs.pts_2d, pixels)
    for camera in rig.cameras:
        np.testing.assert_array_equal(recovered.cameras[camera].params, rig.cameras[camera].params)
        np.testing.assert_array_equal(recovered.T_c_g[camera], rig.T_c_g[camera])
