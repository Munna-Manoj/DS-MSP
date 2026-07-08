"""Real-data validation gate for the non-overlapping multi-object rig fix (FR-RIG-017,
ADR-0011).

Two RealSense cameras, each looking at a *different* face of a double-sided ChArUco target,
never co-observed in any single frame (MC-Calib topology 5). Before ADR-0011 this silently
calibrated as a 1-camera "rig" (the board camera 0 alone observes was dropped as the
minority covisibility component); this asserts both cameras calibrate, with the recovered
extrinsic close to the previously-measured real-data acceptance numbers (1.192 m baseline,
178.6 deg rotation -- the cameras face each other across the target).

Dataset-gated: set ``DSMSP_SELTOS_DIR`` to a directory containing ``calib_param_gaze.yml`` +
imagery, or place ``seltos_cameras_rig/seltos_cams/`` at the repo root (gitignored, local-only
real data). Skipped (not failed) when absent, like the other ``realdata`` tests -- but
required green for a release-gated rig release (FR-RIG-001 covers this code path).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from ds_msp.rig.calib_param import calibrate_from_config


def _root():
    env = os.environ.get("DSMSP_SELTOS_DIR")
    if env and os.path.isdir(env):
        return env
    repo = Path(__file__).resolve().parents[2] / "seltos_cameras_rig" / "seltos_cams"
    return str(repo) if repo.is_dir() else None


def _rel_extrinsic(rig):
    T0, T1 = rig.T_c_g[0], rig.T_c_g[1]
    T_rel = T1 @ np.linalg.inv(T0)
    baseline_m = float(np.linalg.norm(T_rel[:3, 3])) * 1e-3   # square_size units are mm here
    ang_deg = float(np.degrees(np.arccos(
        np.clip((np.trace(T_rel[:3, :3]) - 1) / 2, -1, 1))))
    return baseline_m, ang_deg


pytestmark = [pytest.mark.realdata, pytest.mark.req("FR-RIG-017")]


def test_non_overlapping_seltos_rig_calibrates_both_cameras(tmp_path):
    root = _root()
    if root is None:
        pytest.skip("Seltos rig imagery not present (set DSMSP_SELTOS_DIR)")
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
