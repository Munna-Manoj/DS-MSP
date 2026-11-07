"""Per-camera lens checks: square-root lens sigma, optimizer bounds, split cross-validation,
and their gates and acceptance records (FR-RIG-030)."""
import json
import warnings

import numpy as np
import pytest

from ds_msp.io.mccalib import Scenario
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.radtan import RadTanModel
from ds_msp.rig import bundle, lens_check as lc
from ds_msp.rig.pipeline import calibrate_scenario

from ._synth import make_rig

pytestmark = pytest.mark.req("FR-RIG-030")


def _scenario(n_frame=30, noise_px=0.05, seed=4, starve=None, keep=1):
    obj, obs, img_size, _ext, models = make_rig(n_cam=3, n_frame=n_frame, noise_px=noise_px,
                                                seed=seed)
    if starve is not None:
        frames = sorted({o.frame_id for o in obs if o.cam_id == starve})[:keep]
        obs = [o for o in obs if o.cam_id != starve or o.frame_id in frames]
    scn = Scenario(name="lens", object=obj, object_obs=obs, cam_ids=sorted(models),
                   img_size=img_size, gt={}, mccalib={}, mccalib_rms={})
    return scn, models


def _run(scn, models, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = calibrate_scenario(scn, {c: m.name for c, m in models.items()}, noise_bound=1.0,
                                 uncertainty_gate="off", **kw)
    return res, [str(x.message) for x in w if issubclass(x.category, RuntimeWarning)]


@pytest.fixture(scope="module")
def healthy():
    scn, models = _scenario()
    res, warns = _run(scn, models)
    return scn, models, res, warns


def test_t_widening_matches_student_t():
    assert lc.t_widening(1) is None and lc.t_widening(0) is None
    z = 1.959964                                        # t-table values, two-sided 95%
    assert lc.t_widening(2) == pytest.approx(12.7062 / z, rel=1e-4)
    assert lc.t_widening(3) == pytest.approx(4.3027 / z, rel=1e-4)
    assert lc.t_widening(31) == pytest.approx(2.0423 / z, rel=1e-4)


def test_square_root_factor_reproduces_the_bread_inverse(healthy):
    """L Lᵀ = S⁻¹ on a well-conditioned rig, so ‖G L‖² is G S⁻¹ Gᵀ without forming S⁻¹."""
    scn, _m, res, _w = healthy
    unc = bundle.shared_covariance(res["rig"], scn.object_obs, fix_intrinsics=False,
                                   refine_boards=True)
    assert unc["gauges"] == [] or all(n == 0 for _c, n in unc["gauges"])
    fac = bundle.bread_factor(unc)
    assert fac["N"].shape[1] == 0
    L, S = fac["L"], unc["bread"]
    np.testing.assert_allclose(S @ (L @ L.T), np.eye(len(S)), atol=1e-6)
    assert 0.0 < unc["sigma_px"] < 0.2                     # 0.05 px noise, robust MAD estimate


def test_a_null_direction_is_isolated_not_fatal():
    """A singular bread leaves only the quantities along its null direction unbounded."""
    S = np.diag([4.0, 1.0, 0.0])
    fac = bundle.bread_factor({"bread": S, "bread_G": np.eye(3), "gauges": []})
    assert fac["N"].shape[1] == 1 and abs(fac["N"][2, 0]) > 0.99
    np.testing.assert_allclose(fac["L"] @ fac["L"].T, np.diag([0.25, 1.0, 0.0]), atol=1e-12)
    assert lc._null_share(np.array([[[1.0, 0.0, 0.0]]]), fac["N"], fac["BD"]) < 1e-12
    assert lc._null_share(np.array([[[0.0, 0.0, 1.0]]]), fac["N"], fac["BD"]) > 0.99
    assert bundle.bread_factor({"bread": np.diag([1.0, np.nan, 1.0]), "bread_G": np.eye(3),
                                "gauges": []}) is None


def test_realignment_removes_what_a_rotation_undoes():
    """A lens change equal to a small camera rotation moves no ray against the others."""
    m = RadTanModel(800.0, 800.0, 640.0, 480.0, -0.05, 0.01, 0.0, 0.0, 0.0)
    X, ok = lc._rays(m, lc._grid(1280, 960))
    _q, Jx, _G, val = m.project_jacobian(X[ok])
    X, Jx = X[ok][val], np.asarray(Jx)[val]
    x, y, z = X.T
    skew = np.zeros((len(X), 3, 3))
    skew[:, 0, 1], skew[:, 0, 2], skew[:, 1, 2] = -z, y, -x
    skew[:, 1, 0], skew[:, 2, 0], skew[:, 2, 1] = z, -y, x
    R = -np.einsum("nij,njk->nik", Jx, skew)
    out = lc._realigned(R.copy(), Jx, X, np.ones(len(X), bool))
    assert np.abs(out).max() < 1e-9 * np.abs(R).max()


def test_projection_diff_is_zero_for_a_lens_and_itself():
    m = RadTanModel(800.0, 800.0, 640.0, 480.0, -0.05, 0.01, 0.0, 0.0, 0.0)
    uv = lc._grid(1280, 960)
    d = lc.projection_diff(m, m, uv, np.ones(len(uv), bool))
    assert np.nanmax(d) < 1e-6


def test_parameters_at_a_bound_are_listed():
    lo, hi = DoubleSphereModel.param_bounds()
    names = list(DoubleSphereModel.param_names)
    m = DoubleSphereModel(400.0, 400.0, 640.0, 480.0, float(hi[names.index("xi")]), 0.5)
    assert lc.params_at_bound(m) == ["xi"]
    assert lc.params_at_bound(DoubleSphereModel(400.0, 400.0, 640.0, 480.0, 0.2, 0.5)) == []


def test_well_observed_rig_passes_both_checks(healthy):
    _scn, _m, res, warns = healthy
    lens, cv = res["lens"], res["crossval"]
    assert lens["failed"] == [] and cv["applicable"] and cv["failed"] == []
    for v in lens["cameras"].values():
        assert v["frames"] >= 20 and v["sigma_median_widened_px"] < 1.0
    for v in cv["cameras"].values():
        assert v["applicable"] and v["error_estimate_px"] < 1.0
    assert not any("lens" in w for w in warns)


def test_single_frame_camera_is_unbounded_and_warns():
    scn, models = _scenario(starve=1, keep=1)
    res, warns = _run(scn, models)
    v = res["lens"]["cameras"]["1"]
    assert v["frames"] == 1 and v["t_factor"] is None and v["sigma_median_widened_px"] is None
    assert not v["ok"] and "sigma" in v["reasons"] and res["lens"]["failed"] == ["1"]
    assert res["crossval"]["cameras"]["1"]["applicable"] is False
    assert any("lens not determined" in w and "camera 1" in w for w in warns)


def test_lens_gate_refuse_blocks_a_starved_camera():
    scn, models = _scenario(starve=1, keep=1)
    with pytest.raises(RuntimeError, match="lens gate refused"):
        _run(scn, models, lens_gate="refuse", crossval_gate="off")


def test_checks_are_recorded_as_strict_json(tmp_path, healthy):
    scn, models, _res, _w = healthy
    res, _warns = _run(scn, models, save_dir=str(tmp_path))
    acc = json.loads((tmp_path / "calibration_acceptance.json").read_text())
    assert acc["lens_gate"] == "warn" and acc["crossval_gate"] == "warn"
    assert set(acc["lens"]["cameras"]) == {"0", "1", "2"}
    assert acc["crossval"]["applicable"] is True
    assert acc["observability"]["n_weak"] == res["audit"]["n_weak"]


def test_fixed_intrinsics_skip_the_lens_checks(healthy):
    scn, models, res, _w = healthy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = calibrate_scenario(scn, {c: m.name for c, m in models.items()},
                                 fix_intrinsics=True, init_cameras=res["rig"].cameras,
                                 uncertainty_gate="off")
    assert "lens" not in out and "crossval" not in out
    assert out["acceptance"]["lens"] is None and out["acceptance"]["crossval"] is None
