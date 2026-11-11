"""How far a result is from the precision someone needs, and what more photos would do."""
import json
import math

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from ds_msp.studio.precision import MAX_ROWS, _t_factor, frames_for, precision_gap  # noqa: E402
from ds_msp.studio.server import create_app  # noqa: E402

pytestmark = pytest.mark.req("FR-UI-022")


def _pair(rot, ctr, frames=33, known=True):
    t = _t_factor(frames)
    return {"rotation_sigma_deg": rot, "centre_sigma": ctr, "rotation_sigma_deg_t": rot * t,
            "centre_sigma_t": ctr * t, "identifiable": known}


def _unc(pairs, frames=33, limits=None):
    return {"n_frame_clusters": frames, "pairs": pairs, "limits": limits or {}}


def test_the_estimate_meets_the_limit_with_the_fewest_frames():
    need = frames_for(0.31, 33, 0.05)
    widened = lambda n: 0.31 * math.sqrt(33 / n) * _t_factor(n)  # noqa: E731
    assert widened(need) <= 0.05 and need == pytest.approx(33 * (0.31 / 0.05) ** 2, rel=0.01)
    assert widened(need - 8) > 0.05


def test_a_result_within_its_limits_needs_no_more_photos():
    doc = precision_gap(_unc({"0-1": _pair(0.01, 0.1)}), 0.05, 1.0)
    assert doc["n_over"] == 0 and doc["frames_needed"] is None and doc["times"] is None


def test_the_tightest_limit_of_the_worst_pair_sets_the_estimate():
    doc = precision_gap(_unc({"0-1": _pair(0.1, 1.0), "0-2": _pair(0.2, 1.0)}), 0.05, 0.25)
    assert doc["n_over"] == 2 and doc["worst_rotation"]["pair"] == "0-2"
    assert doc["frames_needed"] == max(frames_for(0.2, 33, 0.05), frames_for(1.0, 33, 0.25))


def test_a_limit_can_be_given_alone():
    doc = precision_gap(_unc({"0-1": _pair(0.2, 50.0)}), 0.1, None)
    assert doc["limits"] == {"rotation": 0.1, "centre": None}
    assert doc["frames_needed"] == frames_for(0.2, 33, 0.1)


def test_the_limits_of_the_run_are_used_when_none_are_typed():
    limits = {"max_rotation_sigma_deg": 0.1, "max_centre_sigma": 5.0}
    doc = precision_gap(_unc({"0-1": _pair(0.2, 1.0)}, limits=limits))
    assert doc["from_run"] and doc["limits"] == {"rotation": 0.1, "centre": 5.0} and doc["n_over"] == 1


def test_without_any_limit_nothing_is_over():
    doc = precision_gap(_unc({"0-1": _pair(0.2, 1.0)}))
    assert doc["n_over"] == 0 and doc["frames_needed"] is None and not doc["from_run"]


def test_more_photos_are_not_promised_for_an_undetermined_pair():
    bad = {"rotation_sigma_deg": None, "centre_sigma": None, "rotation_sigma_deg_t": None,
           "centre_sigma_t": None, "identifiable": False}
    doc = precision_gap(_unc({"0-1": _pair(0.2, 1.0), "0-2": bad}), 0.1, 0.5)
    assert doc["n_undetermined"] == 1 and doc["frames_needed"] is None and doc["rows"][0]["pair"] == "0-2"


def test_no_uncertainty_is_said():
    assert precision_gap(None)["ok"] is False and precision_gap({"pairs": {}})["ok"] is False


def test_rows_are_capped_and_counts_are_not():
    pairs = {f"0-{k}": _pair(0.1 + k / 100, 1.0) for k in range(1, MAX_ROWS + 6)}
    doc = precision_gap(_unc(pairs), 0.05, None)
    assert len(doc["rows"]) == MAX_ROWS and doc["n_pairs"] == doc["n_over"] == MAX_ROWS + 5
    assert doc["rows"][0]["pair"] == f"0-{MAX_ROWS + 5}"


def test_over_the_api(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "calibration_report.json").write_text(json.dumps({"schema": "x"}))
    (run / "calibration_acceptance.json").write_text(json.dumps({"extrinsic_uncertainty": _unc({"0-1": _pair(0.2, 1.0)})}))
    c = TestClient(create_app([str(tmp_path)]))
    doc = c.get("/api/run/precision", params={"id": "0:run", "rotation": 0.1}).json()
    assert doc["n_over"] == 1 and doc["frames_needed"] == frames_for(0.2, 33, 0.1)
    assert c.get("/api/run/precision", params={"id": "0:run", "rotation": 0}).status_code == 422
