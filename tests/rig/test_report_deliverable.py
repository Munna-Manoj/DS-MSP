"""Full calibration deliverable (rig/report.py full_report_data/render_full_report/
write_report_files, FR-RIG-023): the terminal report and the persisted
calibration_report.txt/.json render from ONE data dict, so their content is identical by
construction — these tests pin that contract, plus the informativeness of each section
(intrinsics with 1-sigma, extrinsics vs reference, trust layer, verdict).
"""
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest

from ds_msp.core.lie import so3_exp
from ds_msp.models.radtan import RadTanModel
from ds_msp.rig import bundle
from ds_msp.rig import report as rpt
from ds_msp.rig.audit import audit_rig
from ds_msp.rig.certify import certify_rotations
from ds_msp.rig.cli import _report_and_exit_code
from ds_msp.rig.types import ObjectObs, RigState

from ._synth import make_object

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _consistent_rig(n_cam=2, n_frame=6):
    """A rig whose observations, poses, and extrinsics are exactly consistent (zero
    reprojection error), so every report number is meaningful and deterministic."""
    w, h, f = 640, 480, 520.0
    cams = {c: RadTanModel(f, f, w / 2, h / 2, -0.05, 0.01, 0.0, 0.0, 0.0)
            for c in range(n_cam)}
    T1 = np.eye(4)
    T1[:3, :3] = so3_exp(np.array([0.0, 0.6, 0.0]))
    T1[:3, 3] = [0.45, 0.0, 0.25]
    obj = make_object({0: np.eye(4), 1: T1})   # genuinely 3D target: focals observable
    gt_ext = {0: np.eye(4)}
    for c in range(1, n_cam):
        T = np.eye(4)
        T[:3, :3] = so3_exp(np.array([0.0, np.deg2rad(8.0 * c), 0.0]))
        T[:3, 3] = [0.15 * c, 0.0, 0.0]
        gt_ext[c] = T
    rng = np.random.default_rng(0)
    poses, obs = {}, []
    for fr in range(n_frame):
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        T_g_o = np.eye(4)
        T_g_o[:3, :3] = so3_exp(ax * rng.uniform(-0.4, 0.4))
        T_g_o[:3, 3] = [rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2),
                        rng.uniform(1.8, 2.4)]
        poses[(0, fr)] = T_g_o
        Xg = (T_g_o[:3, :3] @ obj.pts_3d.T).T + T_g_o[:3, 3]
        for c in range(n_cam):
            Xc = (gt_ext[c][:3, :3] @ Xg.T).T + gt_ext[c][:3, 3]
            uv, valid = cams[c].project(Xc)
            inb = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            rows = np.where(inb)[0]
            if len(rows) < 6:
                continue
            obs.append(ObjectObs(cam_id=c, frame_id=fr, object_id=0, point_rows=rows,
                                 pts_2d=uv[rows], T_c_o=gt_ext[c] @ T_g_o))
    rig = RigState(cameras=cams, T_c_g=gt_ext, ref_cam_id=0, object_poses=poses,
                   objects={0: obj}, img_size={c: (w, h) for c in range(n_cam)})
    return rig, SimpleNamespace(name="synthetic-report", object_obs=obs)


def _full_data(rig, scn, *, with_cov=True):
    per_cam, overall = rpt.camera_and_overall_stats(rig, scn.object_obs)
    level, message = rpt.verdict(overall)
    models = {c: rig.cameras[c].name for c in rig.cameras}
    cov = bundle.parameter_covariance(rig, scn.object_obs, fix_intrinsics=False) \
        if with_cov else None
    return rpt.full_report_data(
        rig, scn, models, per_cam, overall, level, message,
        metrics={"max_rms_px": overall.rms}, audit=audit_rig(rig, scn.object_obs),
        certificate=certify_rotations(rig, scn.object_obs), covariance=cov,
        output_files={"mccalib_output_dir": "/out"})


@pytest.mark.req("FR-RIG-023")
def test_saved_txt_is_verbatim_the_rendered_report_and_json_roundtrips(tmp_path):
    rig, scn = _consistent_rig()
    data = _full_data(rig, scn)
    paths = rpt.write_report_files(str(tmp_path), data)
    with open(paths["report_txt"]) as f:
        assert f.read() == rpt.render_full_report(data, color=False) + "\n"
    with open(paths["report_json"]) as f:
        assert json.load(f) == data     # jsonify is exact: load-back equals the source dict


@pytest.mark.req("FR-RIG-023")
def test_report_sections_are_informative_and_numbers_correct():
    rig, scn = _consistent_rig()
    data = _full_data(rig, scn)
    text = rpt.render_full_report(data, color=False)

    # verdict + zero-error distribution on an exactly-consistent rig
    assert "verdict:" in text and " PASS " in text
    assert data["overall_errors_px"]["median"] < 1e-9

    # intrinsics: every model parameter present, with a 1-sigma bound from the clustered cov
    for p in ("fx", "fy", "cx", "cy", "k1"):
        assert f"{p}=" in text
    assert "±" in text and "frame-clustered" in text
    fx = next(p for e in data["intrinsics"] if e["cam_id"] == 0
              for p in e["params"] if p["name"] == "fx")
    assert fx["value"] == pytest.approx(520.0) and fx["sigma"] is not None

    # extrinsics: cam 1's center in the ref frame reproduces the GT 0.15-unit baseline
    e1 = next(e for e in data["extrinsics"] if e["cam_id"] == 1)
    assert e1["baseline"] == pytest.approx(0.15, abs=1e-9)
    assert e1["rot_deg"] == pytest.approx(8.0, abs=1e-9)
    assert e1["rot_sigma_deg_rss"] is not None
    assert "extrinsics" in text and "(reference)" in text

    # trust layer: audit line + certificate verdict, capture summary counts
    assert "observability:" in text and "certificate:" in text and "CERTIFIED" in text
    assert data["capture"]["n_cameras"] == 2
    assert data["capture"]["n_corners"] == sum(len(o.point_rows) for o in scn.object_obs)


@pytest.mark.req("FR-RIG-023")
def test_color_render_differs_only_by_ansi_codes():
    rig, scn = _consistent_rig()
    data = _full_data(rig, scn, with_cov=False)
    assert _ANSI.sub("", rpt.render_full_report(data, color=True)) == \
        rpt.render_full_report(data, color=False)
    # without covariance the report says how to get uncertainties instead of showing stale ones
    assert "report_covariance" in rpt.render_full_report(data, color=False)


@pytest.mark.req("FR-RIG-023")
def test_cli_tail_prints_the_same_content_it_persists(tmp_path, capsys):
    rig, scn = _consistent_rig()
    models = {c: rig.cameras[c].name for c in rig.cameras}
    code = _report_and_exit_code(
        rig, scn, models, str(tmp_path), pass_px=1.0, warn_px=3.0, html_path=None,
        audit=audit_rig(rig, scn.object_obs),
        certificate=certify_rotations(rig, scn.object_obs),
        covariance=bundle.parameter_covariance(rig, scn.object_obs, fix_intrinsics=False),
        metrics={"max_rms_px": 0.0})
    assert code == 0
    out = _ANSI.sub("", capsys.readouterr().out)
    with open(tmp_path / "calibration_report.txt") as f:
        saved = f.read().rstrip("\n")
    assert saved in out                 # terminal shows verbatim what the file contains
    assert (tmp_path / "calibration_report.json").exists()
