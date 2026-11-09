"""Studio verdict, actions and per-camera evidence (FR-UI-004, FR-UI-005)."""
import copy
import json

import pytest

from ds_msp.studio import runs
from ds_msp.studio.verdict import MAX_ACTIONS, OUTLIER_CHECK_FRAC, run_verdict

pytestmark = pytest.mark.req("FR-UI-004")

REPORT = {"capture": {"n_cameras": 3}}


def _acc(**kw):
    a = {"solver": {"final_status": "CONVERGED", "numerical_failures": [], "frame_support": []},
         "lens": {"limit_px": 1.0, "cameras": {c: {"frames": 20, "sigma_median_widened_px": 0.1,
                                                     "sigma_p95_px": 0.3, "params_at_bound": [],
                                                     "ok": True, "reasons": []} for c in "012"}},
         "coverage": {c: {"coverage": 0.99} for c in "012"}, "min_camera_coverage": 0.5,
         "observability": {"n_weak": 0, "gauge_ok": True, "findings": []},
         "extrinsic_uncertainty": {"pairs": {"0-1": {"identifiable": True}, "0-2": {"identifiable": True}},
                                   "within_limits": True, "limits": {}},
         "diagnostics": {"outliers": {c: {"outlier_frac": 0.0} for c in "012"}}}
    a.update(kw)
    return a


def test_a_clean_run_is_accepted_with_no_actions():
    v = run_verdict(REPORT, _acc())
    assert v["level"] == "accept" and v["actions"] == []
    assert all(e["level"] == "accept" for e in v["cameras"].values())


def test_starved_lens_rejects_and_names_the_camera():
    a = _acc()
    a["lens"]["cameras"]["2"].update(ok=False, reasons=["sigma"], sigma_median_widened_px=None, frames=1)
    v = run_verdict(REPORT, a)
    assert v["level"] == "reject" and v["cameras"]["2"]["level"] == "reject"
    assert v["cameras"]["0"]["level"] == "accept"
    assert v["actions"][0]["cams"] == ["2"] and v["actions"][0]["evidence"] == "lens"
    assert "camera 2" in v["actions"][0]["text"]


def test_uncertain_lens_and_weak_link_improve_worst_first():
    a = _acc()
    a["lens"]["cameras"]["1"].update(ok=False, reasons=["sigma"], sigma_median_widened_px=2.5)
    a["observability"]["findings"] = [{"kind": "extrinsic_chain_weak", "cam": None, "cams": [2],
                                       "message": "camera 2 ...", "hint": "Add shared views"}]
    a["coverage"]["0"] = {"coverage": 0.2}
    v = run_verdict(REPORT, a)
    assert v["level"] == "reject"                                     # coverage 20% < 50%
    assert [x["level"] for x in v["actions"]] == ["reject", "improve", "improve"]
    assert v["cameras"]["1"]["level"] == "improve" and v["cameras"]["2"]["level"] == "improve"
    assert v["max_shown"] == MAX_ACTIONS


def test_outlier_share_only_raises_check():
    a = _acc()
    a["diagnostics"]["outliers"]["0"]["outlier_frac"] = OUTLIER_CHECK_FRAC + 0.01
    v = run_verdict(REPORT, a)
    assert v["level"] == "check" and v["cameras"]["0"]["level"] == "check"
    assert v["actions"][0]["evidence"] == "frames"


def test_unidentifiable_pair_and_unconverged_solver():
    a = _acc()
    a["extrinsic_uncertainty"]["pairs"]["0-2"]["identifiable"] = False
    a["solver"]["final_status"] = "MAX_ITERATIONS"
    v = run_verdict(REPORT, a)
    assert v["level"] == "reject"
    assert {"0", "2"} <= {c for x in v["actions"] for c in x["cams"]}
    assert any(r["check"] == "solver" for r in v["rig_reasons"])


def test_old_results_without_new_fields_still_get_a_verdict():
    v = run_verdict(REPORT, {"solver": {"final_status": "CONVERGED"}})
    assert v["level"] == "accept"
    assert run_verdict(REPORT, None)["level"] == "accept"


@pytest.mark.req("FR-UI-005")
def test_run_doc_carries_the_verdict_and_diagnostics_load(results_root):
    folder = results_root / "lab" / "run_a"
    doc = runs.load_run(folder)
    assert doc["has_diagnostics"] is True
    assert doc["verdict"]["level"] in ("accept", "check", "improve", "reject")
    d = runs.load_diagnostics(folder, 0)
    r = d["residuals"]
    assert r["n_shown"] == len(r["u"]) == len(r["dx"]) <= runs.MAX_ARROWS
    assert r["n_valid"] <= r["n_total"] and r["rms_px"] > 0
    assert 0 < len(d["worst_frames"]) <= runs.N_WORST_FRAMES
    rms = [f["rms_px"] for f in d["worst_frames"]]
    assert rms == sorted(rms, reverse=True)
    lens = d["lens"]
    assert lens["nx"] * lens["ny"] == len(lens["sigma_px"]) == len(lens["observed"])
    assert d["width"] and d["height"] and d["outliers"]["n_observed"] == r["n_total"]
    json.dumps(d, allow_nan=False)


@pytest.mark.req("FR-UI-005")
def test_subsampling_keeps_every_outlier(results_root, monkeypatch):
    folder = results_root / "lab" / "run_a"
    monkeypatch.setattr(runs, "MAX_ARROWS", 50)
    d = runs.load_diagnostics(folder, 1)
    assert d["residuals"]["n_shown"] <= max(50, d["residuals"]["within"].count(False))


@pytest.mark.req("FR-UI-005")
def test_diagnostics_files_outside_the_folder_are_ignored(results_root, tmp_path):
    src = results_root / "lab" / "run_a"
    dst = tmp_path / "run"
    dst.mkdir()
    for p in src.iterdir():
        if p.is_file():
            (dst / p.name).write_bytes(p.read_bytes())
    acc = json.loads((dst / "calibration_acceptance.json").read_text())
    acc = copy.deepcopy(acc)
    acc["diagnostics"]["files"] = {"residuals": "../../etc/passwd", "lens_sigma": "../x.npz"}
    (dst / "calibration_acceptance.json").write_text(json.dumps(acc))
    d = runs.load_diagnostics(dst, 0)
    assert d["residuals"] is None and d["lens"] is None
