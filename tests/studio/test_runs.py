"""Reading result folders for the Studio viewer."""
import json

import pytest

from ds_msp.studio import runs

pytestmark = pytest.mark.req("FR-UI-001")


def test_finds_result_folders_and_skips_the_rest(results_root):
    found = list(runs.iter_run_dirs(results_root))
    assert [p.relative_to(results_root).as_posix() for p in found] == ["lab/run_a"]
    assert list(runs.iter_run_dirs(results_root, max_depth=1)) == []


def test_run_document_carries_the_report_and_checks(results_root):
    doc = runs.load_run(results_root / "lab" / "run_a")
    assert doc["schema"] == runs.RUN_SCHEMA and doc["schema_version"] == 1
    report = json.loads((results_root / "lab" / "run_a" / runs.REPORT_FILE).read_text())
    assert doc["report"]["capture"] == report["capture"]
    assert doc["report"]["per_camera_errors_px"] == report["per_camera_errors_px"]
    assert set(doc["acceptance"]["lens"]["cameras"]) == {"0", "1", "2"}
    assert doc["acceptance"]["crossval"]["applicable"] is True
    assert "frame_support" not in doc["acceptance"]["solver"]
    assert doc["manifest"]["run_id"]
    names = [f["name"] for f in doc["files"]]
    assert names[:3] == [runs.REPORT_FILE, runs.ACCEPTANCE_FILE, runs.MANIFEST_FILE]
    json.dumps(doc, allow_nan=False)                          # strict JSON


def test_summary_fields(results_root):
    s = runs.summarize_run(results_root / "lab" / "run_a")
    assert s["n_cameras"] == 3 and s["fit_level"] == "PASS"
    assert s["lens_failed"] == [] and s["solver_status"] == "CONVERGED"
    assert 0 < s["median_px"] < 0.2


def test_non_finite_numbers_become_null(tmp_path):
    (tmp_path / runs.REPORT_FILE).write_text('{"overall_errors_px": {"median": NaN, "max": Infinity}}')
    doc = runs.load_run(tmp_path)
    assert doc["report"]["overall_errors_px"] == {"median": None, "max": None}
    assert doc["acceptance"] is None and doc["manifest"] is None
    json.dumps(doc, allow_nan=False)


def test_unreadable_and_missing_reports(tmp_path):
    with pytest.raises(runs.RunError):
        runs.load_run(tmp_path)
    (tmp_path / runs.REPORT_FILE).write_text("{not json")
    with pytest.raises(runs.RunError):
        runs.load_run(tmp_path)
