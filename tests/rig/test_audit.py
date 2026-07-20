"""Observability audit (rig/audit.py, FR-RIG-021).

Assertions are locked to the measured characterization (characterization runs 2026-07-18):
structural degeneracies sit at equilibrated-eigenvalue ratios <= 1e-10, the softest healthy
directions at >= 1e-5 — the two-tier thresholds (1e-6 critical / 1e-3 soft) sit inside that
measured gap, not at hand-tuned values.
"""
import dataclasses

import numpy as np
import pytest

from ds_msp.core.lie import so3_exp
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.rig.audit import audit_rig
from ds_msp.rig.types import ObjectObs, RigState

from ._synth import make_object, make_rig


def _planar_ds_rig(*, tilt: bool, centered: bool, n_frame=25, seed=3):
    """Single DS camera watching one planar board: the canonical degenerate captures."""
    rng = np.random.default_rng(seed)
    w, h = 1280, 960
    model = DoubleSphereModel(450.0, 450.0, w / 2, h / 2, 0.6, 0.55)
    obj = make_object({0: np.eye(4)}, nx=6, ny=6, pitch=0.08)
    obs, poses = [], {}
    for fr in range(n_frame):
        if tilt:
            ax = rng.normal(size=3)
            ax /= np.linalg.norm(ax)
            Rg = so3_exp(ax * rng.uniform(-0.5, 0.5))
        else:
            Rg = np.eye(3)
        if centered:
            tg = np.array([rng.uniform(-0.03, 0.03) - 0.2,
                           rng.uniform(-0.03, 0.03) - 0.2, rng.uniform(1.6, 1.9)])
        else:
            tg = np.array([rng.uniform(-0.45, 0.45) - 0.2,
                           rng.uniform(-0.35, 0.35) - 0.2, rng.uniform(0.9, 1.6)])
        T = np.eye(4)
        T[:3, :3] = Rg
        T[:3, 3] = tg
        Xc = (Rg @ obj.pts_3d.T).T + tg
        uv, valid = model.project(Xc)
        inb = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        rows = np.where(inb)[0]
        if len(rows) < 8:
            continue
        pts = uv[rows] + rng.normal(scale=0.05, size=(len(rows), 2))
        obs.append(ObjectObs(cam_id=0, frame_id=fr, object_id=0,
                             point_rows=rows, pts_2d=pts))
        poses[(0, fr)] = T
    rig = RigState(cameras={0: model}, T_c_g={0: np.eye(4)}, ref_cam_id=0,
                   object_poses=poses, objects={0: obj}, img_size={0: (w, h)})
    return rig, obs


def _conditioned_rig(seed=1):
    """3-camera, genuinely-3D multi-board rig at its ground-truth state."""
    obj, obs, img_size, gt_ext, gt_models = make_rig(n_cam=3, n_frame=25, noise_px=0.05,
                                                     seed=seed)
    from ds_msp.rig.pose_init import estimate_pose_ransac
    poses = {}
    for o in obs:
        key = (o.object_id, o.frame_id)
        if key not in poses and o.cam_id == 0:
            T = estimate_pose_ransac(gt_models[0], obj.pts_3d[o.point_rows], o.pts_2d)
            if isinstance(T, tuple):
                T = T[0]
            poses[key] = np.linalg.inv(gt_ext[0]) @ T
    obs = [o for o in obs if (o.object_id, o.frame_id) in poses]
    rig = RigState(cameras=gt_models, T_c_g=gt_ext, ref_cam_id=0,
                   object_poses=poses, objects={0: obj}, img_size=img_size)
    return rig, obs


@pytest.mark.req("FR-RIG-021")
def test_well_conditioned_capture_is_silent_and_scaling_is_necessary():
    """(c)+(f): no findings on a healthy rig; the measured eigen-gap the thresholds rely on
    is real; and cond(H) >> cond(Ĥ) proves the units artefact the equilibration removes."""
    rig, obs = _conditioned_rig()
    a = audit_rig(rig, obs)
    assert a["n_weak"] == 0 and a["gauge_ok"]
    assert not a["findings"]
    assert a["cond"] < 1e8
    assert a["cond_raw"] > 100 * a["cond"]


@pytest.mark.req("FR-RIG-021")
def test_gauge_positive_control_fires_exactly_six_global_gauge_findings():
    """(d): the shipped layout pins the datum (silent above); artificially unfixing the
    reference camera must surface exactly the 6 global-gauge modes, all named as such."""
    rig, obs = _conditioned_rig()
    a = audit_rig(dataclasses.replace(rig, ref_cam_id=-1), obs)
    assert not a["gauge_ok"]
    gauge = [f for f in a["findings"] if f["kind"] == "global_gauge"]
    assert len(gauge) == 6
    assert all(f["ratio"] < 1e-10 for f in gauge)


@pytest.mark.req("FR-RIG-021")
def test_no_tilt_planar_capture_flags_focal_distortion_coupling():
    """(a): frontoparallel-only planar DS capture must name the focal<->xi coupling."""
    rig, obs = _planar_ds_rig(tilt=False, centered=False)
    a = audit_rig(rig, obs)
    fdc = [f for f in a["findings"] if f["kind"] == "focal_distortion_coupling"]
    assert fdc, f"expected focal_distortion_coupling, got {a['findings']}"
    assert fdc[0]["cam"] == 0
    assert "xi" in fdc[0]["params"]
    assert fdc[0]["ratio"] < 1e-10                      # structural, not soft
    assert "tilt" in fdc[0]["message"]


@pytest.mark.req("FR-RIG-021")
def test_centered_board_capture_flags_periphery_and_coverage_corroborates():
    """(b): a centered-only capture leaves the outer FOV empty; the audit must both flag a
    structural finding for cam0 and report near-zero periphery occupancy."""
    rig, obs = _planar_ds_rig(tilt=True, centered=True)
    a = audit_rig(rig, obs)
    assert a["n_weak"] >= 1
    assert a["coverage"][0]["periphery_frac"] < 0.02
    assert any(f["cam"] == 0 for f in a["findings"] if f["cam"] is not None)


@pytest.mark.req("FR-RIG-021")
def test_naming_is_unit_invariant_only_under_equilibration():
    """(f2): the naming must not depend on the dataset's arbitrary length unit. Re-express
    the translation parameters in millimeters (columns x1e-3): the equilibrated bottom
    eigenvector names the same {fx, fy, xi} set, while the raw-H bottom eigenvector is
    captured by the rescaled translation columns — the units artefact the design predicts.
    (On THIS capture the raw and scaled namings happen to coincide in meters — measured
    2026-07-18 — so unit-invariance, not meter-basis disagreement, is the honest check.)"""
    from ds_msp.core.observability import equilibrate
    from ds_msp.rig import bundle
    from ds_msp.rig.audit import _column_labels

    rig, obs = _planar_ds_rig(tilt=False, centered=False)
    state0, _res, jacobian, _re, _K = bundle.build_problem(rig, obs, fix_intrinsics=False)
    J = np.asarray(jacobian(state0), float)
    labels = _column_labels(rig, fix_intrinsics=False)
    t_cols = [j for j, lab in enumerate(labels)
              if lab[0] == "obj_pose" and lab[2].startswith("t")]

    def top_intr(V):
        return {labels[j][2] for j in np.argsort(V[:, 0] ** 2)[::-1][:3]
                if labels[j][0] == "intr"}

    def bottom_vec(H, scaled):
        M = equilibrate(H)[0] if scaled else 0.5 * (H + H.T)
        return np.linalg.eigh(M)[1]

    J_mm = J.copy()
    J_mm[:, t_cols] *= 1e-3                       # translations now in millimeters
    H_m, H_mm = J.T @ J, J_mm.T @ J_mm

    named_m = top_intr(bottom_vec(H_m, scaled=True))
    named_mm = top_intr(bottom_vec(H_mm, scaled=True))
    assert "xi" in named_m and ("fx" in named_m or "fy" in named_m)
    assert named_mm == named_m                    # equilibrated: unit-invariant

    raw_m = top_intr(bottom_vec(H_m, scaled=False))
    raw_mm = top_intr(bottom_vec(H_mm, scaled=False))
    assert raw_mm != raw_m, "raw-H naming unexpectedly survived the unit change"
