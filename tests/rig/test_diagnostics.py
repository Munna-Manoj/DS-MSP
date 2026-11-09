"""Diagnostic exports of a solved rig: signed per-corner residuals, outlier shares, lens sigma
grids at infinity and at the working distance, board flatness, and the files the pipeline
writes for them (FR-RIG-031)."""
import copy
import json
import warnings

import numpy as np
import pytest

from ds_msp.io.mccalib import Scenario
from ds_msp.rig import bundle, diagnostics as dg, lens_check as lc
from ds_msp.rig.pipeline import calibrate_scenario

from ._synth import make_rig

pytestmark = pytest.mark.req("FR-RIG-031")


def _scenario(seed=4, corrupt=None):
    obj, obs, img_size, _ext, models = make_rig(n_cam=3, n_frame=30, noise_px=0.05, seed=seed)
    if corrupt is not None:                         # shift a few corners of one view by 25 px
        obs = list(obs)
        i, n = corrupt
        o = copy.copy(obs[i])
        p = np.array(o.pts_2d, float)
        p[:n] += 25.0
        o.pts_2d = p
        obs[i] = o
    scn = Scenario(name="diag", object=obj, object_obs=obs, cam_ids=sorted(models),
                   img_size=img_size, gt={}, mccalib={}, mccalib_rms={})
    return scn, models


def _run(scn, models, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return calibrate_scenario(scn, {c: m.name for c, m in models.items()}, noise_bound=1.0,
                                  uncertainty_gate="off", **kw)


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    scn, models = _scenario(corrupt=(5, 4))
    out = tmp_path_factory.mktemp("diag")
    res = _run(scn, models, save_dir=str(out))
    return scn, res, out


def test_residuals_are_signed_prediction_minus_detection(solved):
    scn, res, _out = solved
    rig = res["rig"]
    r = dg.corner_residuals(rig, scn.object_obs, 3.03)
    assert set(r) == set(dg.RESIDUAL_COLUMNS)
    assert len(r["cam"]) == sum(len(o.point_rows) for o in scn.object_obs)
    o = scn.object_obs[0]
    T = rig.T_c_g[o.cam_id] @ rig.object_poses[(o.object_id, o.frame_id)]
    Xc = rig.objects[o.object_id].pts_3d[o.point_rows] @ T[:3, :3].T + T[:3, 3]
    uv, _ok = rig.cameras[o.cam_id].project(Xc)
    k = len(o.point_rows)
    np.testing.assert_allclose(np.c_[r["dx"][:k], r["dy"][:k]], uv - o.pts_2d)
    np.testing.assert_array_equal(r["row"][:k], o.point_rows)
    assert np.isnan(r["dx"][~r["valid"]]).all()
    norm = np.hypot(r["dx"], r["dy"])
    np.testing.assert_array_equal(r["within_band"], r["valid"] & (norm <= 3.03))


def test_corrupted_corners_show_as_outliers_of_their_camera(solved):
    scn, res, _out = solved
    r = dg.corner_residuals(res["rig"], scn.object_obs, 3.03)
    bad = scn.object_obs[5]
    sel = (r["cam"] == bad.cam_id) & (r["frame"] == bad.frame_id)
    assert not r["within_band"][sel][:4].any()                 # the 4 shifted corners
    summ = dg.outlier_summary(r, 3.03)
    assert summ[str(bad.cam_id)]["n_valid"] - summ[str(bad.cam_id)]["n_within"] >= 4
    assert summ[str(bad.cam_id)]["outlier_frac"] > 0.0
    clean = [c for c in summ if c != str(bad.cam_id)]
    assert all(summ[c]["outlier_frac"] == 0.0 for c in clean)


def test_grid_reproduces_the_lens_check_median(solved):
    scn, res, _out = solved
    unc = bundle.shared_covariance(res["rig"], scn.object_obs, fix_intrinsics=False,
                                   refine_boards=True)
    lens = lc.lens_uncertainty(res["rig"], scn.object_obs, unc, limit_px=1.0)
    grids = lc.lens_sigma_grids(res["rig"], scn.object_obs, unc)
    assert set(grids) == set(res["rig"].cameras)
    for c, g in grids.items():
        assert g["sigma_px"].shape == (lc.GRID[0] * lc.GRID[1],)
        assert np.nanmedian(g["sigma_px"]) == lens["cameras"][str(c)]["sigma_median_px"]
        assert g["observed"].any() and (g["sigma_px"][np.isfinite(g["sigma_px"])] > 0).all()


def test_board_flatness_is_zero_for_a_rigid_target_and_measures_a_bow(solved):
    _scn, res, _out = solved
    flat = dg.board_flatness(res["rig"])
    assert all(b["sag_ptp"] < 1e-9 for per in flat.values() for b in per.values())
    rig = copy.copy(res["rig"])
    obj = copy.copy(next(iter(rig.objects.values())))
    P = np.array(obj.pts_3d, float)
    board = np.asarray(obj.pts_obj_2_board, int)[:, 0]
    sel = board == board[0]
    q = P[sel] - P[sel].mean(0)
    _u, _s, Vt = np.linalg.svd(q, full_matrices=False)
    x = q @ Vt[0]
    x = x / np.abs(x).max()
    amp = 0.01 * np.ptp(q @ Vt[0])                             # 1% of the board length
    w = amp * (x ** 2 - np.mean(x ** 2))
    P[sel] += np.outer(w, Vt[2])
    obj.pts_3d = P
    rig.objects = {obj.object_id: obj}
    b = dg.board_flatness(rig)[str(obj.object_id)][str(board[0])]
    A = np.c_[np.ones(len(q)), q @ Vt[0], q @ Vt[1]]          # what a tilted plane absorbs
    rest = w - A @ np.linalg.lstsq(A, w, rcond=None)[0]
    assert b["sag_ptp"] == pytest.approx(np.ptp(rest), rel=1e-6)
    assert b["sag_ptp"] > 0.5 * amp


def test_pipeline_writes_the_files_and_records_them(solved):
    scn, res, out = solved
    acc = json.loads((out / "calibration_acceptance.json").read_text())
    d = acc["diagnostics"]
    assert d["files"] == {"residuals": "diagnostics/residuals.npz",
                          "lens_sigma": "diagnostics/lens_sigma.npz"}
    assert set(d["outliers"]) == {str(c) for c in res["rig"].cameras}
    r = np.load(out / d["files"]["residuals"])
    assert set(r.files) == set(dg.RESIDUAL_COLUMNS)
    assert len(r["cam"]) == sum(len(o.point_rows) for o in scn.object_obs)
    g = np.load(out / d["files"]["lens_sigma"])
    for c in res["rig"].cameras:
        assert g[f"cam{c}_sigma_px"].shape == (lc.GRID[0] * lc.GRID[1],)


def test_fixed_intrinsics_writes_residuals_without_lens_grids(tmp_path):
    scn, models = _scenario()
    res = _run(scn, models, fix_intrinsics=True, init_cameras=models, save_dir=str(tmp_path))
    assert res["diagnostics"]["files"] == {"residuals": "diagnostics/residuals.npz"}
    assert not (tmp_path / "diagnostics" / "lens_sigma.npz").exists()
