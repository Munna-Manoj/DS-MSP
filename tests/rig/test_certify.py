"""Rotation-backbone optimality certificate (rig/certify.py, FR-RIG-022).

Assertions locked to the measured characterization (characterization runs, 2026-07-18): zero-noise
BA==truth certifies with d(BA,chordal)=0; noise keeps d tracking the residual RMS ~1:1; a
planted 60-deg wrong camera yields d~57 deg against ~1 deg residuals — the wrong-basin
detection contract.
"""
import numpy as np
import pytest

from ds_msp.core.lie import so3_exp
from ds_msp.rig.certify import certify_rotations
from ds_msp.rig.types import RigState

from ._synth import make_rig


def _rig_with_measurements(noise_deg=0.0, seed=0):
    """GT rig whose per-view PnP rotations are exact (plus optional rotational noise) —
    the certificate reads only ObjectObs.T_c_o, never pixels."""
    obj, obs, img_size, gt_ext, gt_models = make_rig(n_cam=3, n_frame=20, noise_px=0.0,
                                                     seed=seed)
    rng = np.random.default_rng(seed + 7)
    poses = {}
    for o in obs:
        poses.setdefault((o.object_id, o.frame_id), None)
    for key in sorted(poses):
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        T = np.eye(4)
        T[:3, :3] = so3_exp(ax * rng.uniform(-0.5, 0.5))
        T[:3, 3] = [0, 0, 2.0]
        poses[key] = T
    for o in obs:
        Tco = gt_ext[o.cam_id] @ poses[(o.object_id, o.frame_id)]
        if noise_deg > 0:
            ax = rng.normal(size=3)
            ax /= np.linalg.norm(ax)
            Tco = Tco.copy()
            Tco[:3, :3] = so3_exp(ax * np.radians(noise_deg)) @ Tco[:3, :3]
        o.T_c_o = Tco
    rig = RigState(cameras=gt_models, T_c_g=dict(gt_ext), ref_cam_id=0,
                   object_poses=poses, objects={0: obj}, img_size=img_size)
    return rig, obs


@pytest.mark.req("FR-RIG-022")
def test_noise_free_solution_certifies_with_zero_distance():
    rig, obs = _rig_with_measurements(noise_deg=0.0)
    c = certify_rotations(rig, obs)
    assert c["certified"] is True and c["ba_consistent"] is True
    assert c["eta"] >= -1e-10
    assert c["d_cam_deg"] < 1e-4 and c["d_frame_deg"] < 1e-4
    assert c["grad_norm"] < 1e-8
    assert c["n_components"] == 1
    assert c["n_outlier_edges"] == 0


@pytest.mark.req("FR-RIG-022")
def test_noisy_solution_certifies_and_distance_tracks_noise():
    rig, obs = _rig_with_measurements(noise_deg=1.0)
    c = certify_rotations(rig, obs)
    assert c["certified"] is True and c["ba_consistent"] is True
    assert c["d_cam_deg"] < 3.0 * max(c["resid_med_deg"], 0.5)


@pytest.mark.req("FR-RIG-022")
def test_planted_wrong_basin_camera_is_positively_detected():
    """The flagship behavior: rotate one camera's calibrated extrinsic 60 deg away from
    what its own PnP measurements support — the certificate finds and certifies the true
    optimum and reports the calibrated CAMERA rotations as far from it (wrong-basin
    warning). The planted camera's own measurement edges stay inliers (they are consistent
    with each other), so the trim must NOT eat the evidence."""
    rig, obs = _rig_with_measurements(noise_deg=1.0)
    rig.T_c_g[2] = rig.T_c_g[2].copy()
    rig.T_c_g[2][:3, :3] = so3_exp(np.array([0.0, np.radians(60.0), 0.0])) \
        @ rig.T_c_g[2][:3, :3]
    c = certify_rotations(rig, obs)
    assert c["certified"] is True
    assert c["ba_consistent"] is False
    assert c["d_cam_deg"] > 30.0
    assert "WRONG-BASIN" in c["message"]


@pytest.mark.req("FR-RIG-022")
def test_gross_outlier_measurement_is_trimmed_reported_and_does_not_flag_cameras():
    """Real-data scenario (measured on Seltos 2026-07-18: one 92-deg flipped PnP pose):
    a single antipodal-flipped measurement must be detected and excluded, the calibration
    must still certify as consistent (cameras were never wrong), and the outlier must be
    named with its camera/frame identity."""
    rig, obs = _rig_with_measurements(noise_deg=1.0)
    victim = next(o for o in obs if o.cam_id == 1)
    victim.T_c_o = victim.T_c_o.copy()
    victim.T_c_o[:3, :3] = so3_exp(np.array([0.0, np.radians(92.0), 0.0])) \
        @ victim.T_c_o[:3, :3]
    c = certify_rotations(rig, obs)
    assert c["n_outlier_edges"] == 1
    cam_id, key, resid = c["outlier_edges"][0]
    assert cam_id == 1 and key == (victim.object_id, victim.frame_id)
    assert resid > 45.0
    assert c["certified"] is True and c["ba_consistent"] is True
    assert c["d_cam_deg"] < 3.0 * max(c["resid_med_deg"], 0.5)
    assert "outlier" in c["message"]


@pytest.mark.req("FR-RIG-022")
def test_no_measurements_skips_gracefully():
    rig, obs = _rig_with_measurements(noise_deg=0.0)
    for o in obs:
        o.T_c_o = None
    c = certify_rotations(rig, obs)
    assert c["certified"] is None
    assert "skipped" in c["message"]
