"""A solve must not report success on a state it lost or diverged from (NFR-NUM-015)."""
import warnings

import cv2
import numpy as np
import pytest

from ds_msp.core.optimize import (
    CONVERGED,
    GRADIENT_RATIO_TOL,
    NO_CONVERGENCE,
    _status,
    gnc_tls_solve,
)
from ds_msp.io.mccalib import Scenario
from ds_msp.rig import bundle, pipeline
from ds_msp.rig.calib_param import load_config
from ds_msp.rig.calibrate import calibrate_rig

from ._synth import make_rig


def _solved():
    obj, obs, img_size, _ext, models = make_rig(n_cam=2, n_frame=14, noise_px=0.3, seed=2)
    rig = calibrate_rig(obj, obs, img_size,
                        front_end=pipeline.make_fixed_intrinsic_front_end(models),
                        fix_intrinsics=True)
    return rig, obs


def _scenario():
    obj, obs, img_size, _ext, models = make_rig(n_cam=2, n_frame=14, noise_px=0.3, seed=2)
    scn = Scenario(name="coverage", object=obj, object_obs=obs, cam_ids=[0, 1], img_size=img_size,
                   gt={}, mccalib={}, mccalib_rms={})
    return scn, models


def _flip_camera_after_solve(monkeypatch, cam):
    real = pipeline.calibrate_rig

    def patched(*a, **k):
        rig = real(*a, **k)
        rig.T_c_g[cam] = rig.T_c_g[cam].copy()
        rig.T_c_g[cam][:3, :3] = cv2.Rodrigues(np.array([0.0, np.pi, 0.0]))[0] @ rig.T_c_g[cam][:3, :3]
        return rig
    monkeypatch.setattr(pipeline, "calibrate_rig", patched)


@pytest.mark.req("NFR-NUM-015")
def test_function_test_needs_a_small_gradient():
    assert _status("function_tol", GRADIENT_RATIO_TOL / 2, 1.0) == CONVERGED
    assert _status("function_tol", 0.2, 1.0) == NO_CONVERGENCE
    assert _status("function_tol", float("nan"), 1.0) == CONVERGED
    assert _status("step_tol", 0.2, 1.0) == CONVERGED


@pytest.mark.req("NFR-NUM-015")
def test_gnc_that_rejects_every_block_is_not_converged():
    x = np.linspace(0.0, 1.0, 40)
    y = 2.0 * x + 1.0
    r = lambda s: s[0] * x + s[1] - y                                      # noqa: E731
    J = lambda s: np.c_[x, np.ones_like(x)]                                # noqa: E731
    ret = lambda s, d: s + d                                               # noqa: E731
    good = gnc_tls_solve(np.zeros(2), r, J, ret, noise_bound=0.05, block=1)
    assert good.status == CONVERGED and good.stop_reason in ("binary", "cost_stable")
    bad = gnc_tls_solve(np.zeros(2), r, J, ret, noise_bound=0.05, block=1, inner_max_iter=1,
                        weights=np.zeros_like(x))
    assert bad.status == NO_CONVERGENCE or np.any(bad.weights > 0)
    from ds_msp.core.optimize import _gnc_result, lm_solve
    last = lm_solve(np.zeros(2), r, J, ret, block=1, max_iter=1)
    res = _gnc_result(last.state, np.ones_like(x), np.zeros_like(x), 3, last, "binary")
    assert res.status == NO_CONVERGENCE and res.stop_reason == "no_inliers" and not res.success


@pytest.mark.req("NFR-NUM-015")
def test_coverage_is_complete_on_a_good_rig():
    rig, obs = _solved()
    cov = bundle.camera_coverage(rig, obs, 3.03)
    assert set(cov) == {0, 1}
    for v in cov.values():
        assert v["coverage"] == pytest.approx(1.0)
        assert v["n_within"] == v["n_valid"] == v["n_observed"] > 0


@pytest.mark.req("NFR-NUM-015")
def test_coverage_counts_invalid_missing_and_rejected_corners_as_unexplained():
    rig, obs = _solved()
    n1 = sum(len(o.point_rows) for o in obs if o.cam_id == 1)
    assert bundle.camera_coverage(rig, obs, 1e-9)[1]["n_within"] == 0        # outside the band
    flipped = cv2.Rodrigues(np.array([0.0, np.pi, 0.0]))[0]
    rig.T_c_g[1] = rig.T_c_g[1].copy()
    saved = rig.T_c_g[1][:3, :3].copy()
    rig.T_c_g[1][:3, :3] = flipped @ saved
    cov = bundle.camera_coverage(rig, obs, 3.03)
    assert cov[1]["coverage"] == 0.0 and cov[1]["n_observed"] == n1          # behind the camera
    assert cov[0]["coverage"] == pytest.approx(1.0)                          # others unaffected
    rig.T_c_g[1][:3, :3] = saved
    gone = next(k for k in rig.object_poses)
    n_gone = sum(len(o.point_rows) for o in obs if o.cam_id == 1 and (o.object_id, o.frame_id) == gone)
    del rig.object_poses[gone]
    assert bundle.camera_coverage(rig, obs, 3.03)[1]["n_within"] == n1 - n_gone   # no pose
    rec = {"cam_id": 1, "corners": 10}
    cov = bundle.camera_coverage(rig, obs, 3.03, rejected=[rec])
    assert cov[1]["n_observed"] == n1 + 10 and cov[1]["n_within"] == n1 - n_gone  # ledger counted


@pytest.mark.req("NFR-NUM-015")
def test_good_rig_passes_and_acceptance_records_coverage():
    scn, models = _scenario()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="too few observed corners")
        res = pipeline.calibrate_scenario(scn, "radtan", fix_intrinsics=True, init_cameras=models,
                                          coverage_gate="refuse")
    acc = res["acceptance"]
    assert acc["coverage_gate"] == "refuse" and acc["min_camera_coverage"] == 0.75
    assert acc["coverage"]["1"]["coverage"] == pytest.approx(1.0)


@pytest.mark.req("NFR-NUM-015")
def test_lost_camera_warns_by_default_and_refuses_when_asked(tmp_path, monkeypatch):
    _flip_camera_after_solve(monkeypatch, 1)
    scn, models = _scenario()
    with pytest.warns(RuntimeWarning, match=r"camera 1 0\.0%"):
        pipeline.calibrate_scenario(scn, "radtan", fix_intrinsics=True, init_cameras=models)
    scn, models = _scenario()
    with pytest.raises(RuntimeError, match="coverage gate refused"):
        pipeline.calibrate_scenario(scn, "radtan", fix_intrinsics=True, init_cameras=models,
                                    save_dir=str(tmp_path / "refused"), coverage_gate="refuse")
    assert not (tmp_path / "refused").exists()
    scn, models = _scenario()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="too few observed corners")
        pipeline.calibrate_scenario(scn, "radtan", fix_intrinsics=True, init_cameras=models,
                                    coverage_gate="off")


@pytest.mark.req("NFR-NUM-015")
def test_bad_gate_arguments_are_refused():
    scn, models = _scenario()
    with pytest.raises(ValueError, match="coverage_gate"):
        pipeline.calibrate_scenario(scn, "radtan", fix_intrinsics=True, init_cameras=models,
                                    coverage_gate="maybe")
    with pytest.raises(ValueError, match="min_camera_coverage"):
        pipeline.calibrate_scenario(scn, "radtan", fix_intrinsics=True, init_cameras=models,
                                    min_camera_coverage=1.5)


@pytest.mark.req("NFR-NUM-015")
def test_config_defaults_and_production_profile(tmp_path):
    p = tmp_path / "c.yml"
    base = "%YAML:1.0\n---\nnumber_camera: 2\nnumber_board: 1\n"
    p.write_text(base)
    cfg = load_config(str(p))
    assert cfg.coverage_gate == "warn" and cfg.min_camera_coverage == 0.75
    p.write_text(base + 'profile: "production"\nmax_rotation_sigma_deg: 1.0\nmax_centre_sigma: 1.0\n')
    assert load_config(str(p)).coverage_gate == "warn"                  # limit not yet set for all FOVs
    p.write_text(base + 'coverage_gate: "off"\nmin_camera_coverage: 0.5\n')
    cfg = load_config(str(p))
    assert cfg.coverage_gate == "off" and cfg.min_camera_coverage == 0.5
    p.write_text(base + "min_camera_coverage: 2.0\n")
    with pytest.raises(ValueError, match="min_camera_coverage"):
        load_config(str(p))


@pytest.mark.req("NFR-NUM-015")
def test_noise_free_fit_at_rounding_level_is_converged_whatever_stopped_it():
    # Noise-free data, seen on numpy 2.2: f 6e-25, gradient ratio 1.1 is rounding over rounding.
    assert _status("function_tol", 1.1, 6e-25, rms=1e-13) == CONVERGED
    assert _status("function_tol", 0.2, 1e15, rms=5e5) == NO_CONVERGENCE
    assert _status("function_tol", 0.2, 1.0, rms=1e-3) == NO_CONVERGENCE
