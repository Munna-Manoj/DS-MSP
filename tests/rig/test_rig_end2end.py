"""End-to-end synthetic rig: calibrate_rig must recover known extrinsics."""

import pytest
import numpy as np

from ds_msp.rig import bundle, calibrate_rig
from ._synth import make_rig


def _rel(Tref, Ti):
    return Ti @ np.linalg.inv(Tref)


def _ang(A, B):
    return np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1)))


def test_recovers_extrinsics_noiseless():
    obj, obs, img_size, gt, _ = make_rig(n_cam=3, n_frame=40, noise_px=0.0, seed=0)
    rig = calibrate_rig(obj, obs, img_size, fix_intrinsics=False)
    ref = rig.ref_cam_id
    for c in sorted(rig.T_c_g):
        if c == ref:
            continue
        # internal T_c_g is world->cam; GT here is also world->cam (ref->cam)
        T_mine = _rel(rig.T_c_g[ref], rig.T_c_g[c])
        T_gt = _rel(gt[ref], gt[c])
        base_mine = np.linalg.norm(T_mine[:3, 3])
        base_gt = np.linalg.norm(T_gt[:3, 3])
        assert abs(base_mine - base_gt) / base_gt < 0.02      # <2% translation
        assert _ang(T_mine[:3, :3], T_gt[:3, :3]) < 0.5       # <0.5 deg rotation


def test_recovers_extrinsics_with_noise():
    obj, obs, img_size, gt, _ = make_rig(n_cam=3, n_frame=60, noise_px=0.3, seed=1)
    rig = calibrate_rig(obj, obs, img_size, fix_intrinsics=False)
    rms = bundle.reprojection_rms(rig, obs)
    assert max(rms.values()) < 1.0                            # sub-pixel at 0.3px noise
    ref = rig.ref_cam_id
    worst = max(
        abs(np.linalg.norm(_rel(rig.T_c_g[ref], rig.T_c_g[c])[:3, 3])
            - np.linalg.norm(_rel(gt[ref], gt[c])[:3, 3]))
        / np.linalg.norm(_rel(gt[ref], gt[c])[:3, 3])
        for c in rig.T_c_g if c != ref)
    assert worst < 0.02

# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("FR-RIG-001")
