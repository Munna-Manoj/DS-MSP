"""Per-stage solver status in the rig calibration and its acceptance gates (NFR-NUM-011), and
visible fallbacks in the front-end, hand-eye linking and covariance reporting (NFR-NUM-012)."""
import json
import warnings

import numpy as np
import pytest

import ds_msp.rig.calibrate as rc
from ds_msp.core.optimize import CONVERGED, NO_CONVERGENCE, NUMERICAL_FAILURE
from ds_msp.io.mccalib import Scenario
from ds_msp.models.radtan import RadTanModel
from ds_msp.rig import bundle, pipeline
from ds_msp.rig.calib_param import load_config
from ds_msp.rig.calibrate import calibrate_rig, make_bundle_front_end
from ds_msp.rig.handeye import _tsai_solve, link_groups

from ._synth import make_rig


def _scenario():
    obj, obs, img_size, _ext, models = make_rig(n_cam=2, n_frame=14, noise_px=0.3, seed=2)
    scn = Scenario(name="solver", object=obj, object_obs=obs, cam_ids=[0, 1], img_size=img_size,
                   gt={}, mccalib={}, mccalib_rms={})
    return scn, models


@pytest.mark.req("NFR-NUM-011")
def test_every_stage_is_recorded_and_labelled():
    obj, obs, img_size, _ext, _m = make_rig(n_cam=2, n_frame=14, noise_px=0.3, seed=2)
    fe = make_bundle_front_end("radtan", n_jobs=1)
    rig = calibrate_rig(obj, obs, img_size, front_end=fe)
    rep = rig.solver_report
    labels = [s["stage"] for s in rep["stages"]]
    assert labels[:3] == ["a_object_poses", "b_group_extrinsics", "c_global_joint"]
    for s in rep["stages"]:
        assert s["status"] in (CONVERGED, NO_CONVERGENCE, NUMERICAL_FAILURE)
        assert s["iterations"] >= 1 and s["stop_reason"]
    assert rep["stages"][-1]["status"] == CONVERGED
    assert set(rep["front_end"]) == {"0", "1"}
    assert all(f["fallback"] is None and f["status"] for f in rep["front_end"].values())
    json.dumps(rep, allow_nan=False)                     # strict-JSON safe


@pytest.mark.req("NFR-NUM-011")
def test_refine_log_reports_a_truncated_solve():
    obj, obs, img_size, _ext, models = make_rig(n_cam=2, n_frame=14, noise_px=0.3, seed=2)
    rig = calibrate_rig(obj, obs, img_size,
                        front_end=pipeline.make_fixed_intrinsic_front_end(models),
                        fix_intrinsics=True)
    for T in rig.T_c_g.values():                          # knock the rig off its optimum
        T[:3, 3] += 0.05
    log = []
    bundle.refine(rig, obs, max_iter=1, log=log)
    assert len(log) == 1 and log[0]["stop_reason"] == "max_iter"
    assert log[0]["status"] == NO_CONVERGENCE and log[0]["gradient_ratio"] > 1e-6


def _with_final_status(monkeypatch, status, stage=-1):
    real = pipeline.calibrate_rig

    def patched(*a, **k):
        rig = real(*a, **k)
        rig.solver_report["stages"][stage]["status"] = status
        return rig
    monkeypatch.setattr(pipeline, "calibrate_rig", patched)


@pytest.mark.req("NFR-NUM-011")
def test_acceptance_file_records_the_solver_summary(tmp_path):
    scn, models = _scenario()
    res = pipeline.calibrate_scenario(scn, "radtan", fix_intrinsics=True, init_cameras=models,
                                      save_dir=str(tmp_path))
    acc = json.loads((tmp_path / "calibration_acceptance.json").read_text())
    assert acc["convergence_gate"] == "warn"
    solver = acc["solver"]
    assert solver["final_status"] == CONVERGED and solver["numerical_failures"] == []
    assert solver["final_stage"] == solver["stages"][-1]["stage"]
    assert solver["front_end"] is None                   # fixed intrinsics: no front-end fit
    assert res["acceptance"]["solver"] == solver


@pytest.mark.req("NFR-NUM-011")
def test_non_converged_final_stage_warns_by_default_and_refuses_when_asked(tmp_path, monkeypatch):
    _with_final_status(monkeypatch, NO_CONVERGENCE)
    scn, models = _scenario()
    with pytest.warns(RuntimeWarning, match="final BA stage .* ended NO_CONVERGENCE"):
        pipeline.calibrate_scenario(scn, "radtan", fix_intrinsics=True, init_cameras=models)
    scn, models = _scenario()
    with pytest.raises(RuntimeError, match="convergence gate refused"):
        pipeline.calibrate_scenario(scn, "radtan", fix_intrinsics=True, init_cameras=models,
                                    save_dir=str(tmp_path / "refused"), convergence_gate="refuse")
    assert not (tmp_path / "refused").exists()
    scn, models = _scenario()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        pipeline.calibrate_scenario(scn, "radtan", fix_intrinsics=True, init_cameras=models,
                                    convergence_gate="off")


@pytest.mark.req("NFR-NUM-011")
def test_an_intermediate_non_convergence_is_not_gated(monkeypatch):
    _with_final_status(monkeypatch, NO_CONVERGENCE, stage=0)    # warm-up stage only
    scn, models = _scenario()
    res = pipeline.calibrate_scenario(scn, "radtan", fix_intrinsics=True, init_cameras=models,
                                      convergence_gate="refuse")
    assert res["acceptance"]["solver"]["stages"][0]["status"] == NO_CONVERGENCE


@pytest.mark.req("NFR-NUM-011")
def test_numerical_failure_in_any_stage_always_fails(tmp_path, monkeypatch):
    _with_final_status(monkeypatch, NUMERICAL_FAILURE, stage=0)
    scn, models = _scenario()
    with pytest.raises(RuntimeError, match="non-finite cost .* a_object_poses"):
        pipeline.calibrate_scenario(scn, "radtan", fix_intrinsics=True, init_cameras=models,
                                    save_dir=str(tmp_path / "x"), convergence_gate="off")
    assert not (tmp_path / "x").exists()


@pytest.mark.req("NFR-NUM-011")
def test_production_profile_refuses_non_convergence(tmp_path):
    base = ("%YAML:1.0\n---\nnumber_camera: 2\nnumber_x_square: 5\nnumber_y_square: 4\n"
            "root_path: \"\"\nsave_path: \"\"\nmax_rotation_sigma_deg: 0.1\n"
            "max_centre_sigma: 1.0\n")
    p = tmp_path / "calib_param.yml"
    p.write_text(base + "profile: \"production\"\n")
    assert load_config(str(p)).convergence_gate == "refuse"
    p.write_text(base + "profile: \"production\"\nconvergence_gate: \"warn\"\n")
    assert load_config(str(p)).convergence_gate == "warn"
    p.write_text(base + "convergence_gate: \"maybe\"\n")
    with pytest.raises(ValueError, match="convergence_gate"):
        load_config(str(p))


@pytest.mark.req("NFR-NUM-012")
def test_front_end_focal_collapse_is_recorded(monkeypatch):
    obj, obs, img_size, _ext, _m = make_rig(n_cam=2, n_frame=14, noise_px=0.3, seed=2)
    real = rc._calibrate_single

    def nan_fit(*a, **k):                                  # every candidate fit diverges
        return dict(real(*a, **k), rms_px=float("nan"))
    monkeypatch.setattr(rc, "_calibrate_single", nan_fit)
    cam0 = [o for o in obs if o.cam_id == 0]
    w, h = img_size[0]
    r = rc._fit_one_camera(RadTanModel, {obj.object_id: obj}, cam0, w, h, None,
                           "soft_l1", 1.0, 30)
    assert r["fit"] == {"status": None, "rms_px": None, "fallback": "focal_collapse"}


@pytest.mark.req("NFR-NUM-012")
def test_unlinkable_group_is_reported_and_warns():
    eye = np.eye(4)
    T = np.eye(4)
    T[:3, 3] = [0.3, 0.0, 0.0]
    extr = {0: eye, 1: eye}

    class _O:
        def __init__(self, cam, frame):
            self.cam_id, self.frame_id, self.T_c_o = cam, frame, T

    obs = [_O(0, 0), _O(0, 1), _O(1, 1), _O(1, 2)]        # one shared frame only
    report = []
    with pytest.warns(RuntimeWarning, match="could not link camera group"):
        out = link_groups([[0], [1]], extr, obs, report=report)
    assert report == [{"group": [1], "common_frames": 1, "method": "bootstrap", "linked": False}]
    np.testing.assert_array_equal(out[1], eye)


@pytest.mark.req("NFR-NUM-012")
def test_tsai_without_rotation_returns_none_not_identity():
    M = np.eye(4)
    M[:3, 3] = [0.1, 0.2, 0.0]                             # pure translation: no rotation axis
    assert _tsai_solve([M, M, M], [M, M, M]) is None


@pytest.mark.req("NFR-NUM-012")
def test_negative_variance_is_unbounded_not_zero():
    cov = np.diag([4.0, 1.0, -1e-17, -1e-3])
    std = bundle._std_from_cov(cov)
    np.testing.assert_array_equal(std, [2.0, 1.0, 0.0, np.inf])
    std = bundle._std_from_cov(np.diag([1.0, np.nan]))
    assert std[0] == 1.0 and std[1] == np.inf
