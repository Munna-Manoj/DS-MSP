"""Legacy baseline-length parity against MC-Calib's Blender benchmark.

This does not establish full extrinsic accuracy; FR-RIG-024 covers all-pair SE(3) and ROI
displacement. Skipped when the dataset isn't present locally.
"""

import os

import numpy as np
import pytest

from ds_msp.io.mccalib import load_scenario
from ds_msp.rig import calibrate_rig

_ROOT = os.environ.get("DSMSP_BLENDER_ROOT", "")  # set locally; tests skip when unset/absent
_SCENARIOS = ["Scenario_1", "Scenario_2", "Scenario_3", "Scenario_4", "Scenario_5"]


def _rel(Tref, Ti):
    return Ti @ np.linalg.inv(Tref)


@pytest.mark.parametrize("name", _SCENARIOS)
def test_blender_within_2pct(name):
    scn_dir = os.path.join(_ROOT, name)
    if not os.path.isdir(scn_dir):
        pytest.skip(f"Blender dataset not present: {scn_dir}")
    scn = load_scenario(scn_dir)
    rig = calibrate_rig(scn.object, scn.object_obs, scn.img_size, fix_intrinsics=False)
    ref = rig.ref_cam_id
    worst = 0.0
    for c in sorted(rig.T_c_g):
        if c == ref or c not in scn.gt:
            continue
        base = np.linalg.norm(_rel(rig.T_c_g[ref], rig.T_c_g[c])[:3, 3])
        gtb = np.linalg.norm(_rel(scn.gt[ref].pose, scn.gt[c].pose)[:3, 3])
        worst = max(worst, abs(base - gtb) / gtb)
    assert worst < 0.02, f"{name}: worst translation error {worst*100:.2f}% exceeds 2%"
