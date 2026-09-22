from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from ds_msp.rig.provenance import sha256_file

pytestmark = pytest.mark.req("NFR-REPRO-002")


def _module():
    path = Path(__file__).resolve().parents[2] / "tools" / "replay_rig_manifest.py"
    spec = importlib.util.spec_from_file_location("replay_rig_manifest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_replay_rewrites_only_destination_and_display_arguments(tmp_path):
    output = str(tmp_path / "new")
    args = _module().replay_args([
        "--config", "/data/calib.yml", "--set", "save_path=/old", "--set", "noise_bound=1",
        "--report", "/old/report.html", "--pass-px", "0.8",
    ], output)
    assert args[:2] == ["--config", "/data/calib.yml"]
    assert ["--set", "noise_bound=1"] == args[2:4]
    assert "save_path=/old" not in args
    assert f"save_path={output}" in args
    assert args.count("--report") == 1
    assert "--quiet" in args and "--no-webviewer" in args
    assert args[args.index("--pass-px") + 1] == "0.8"


def test_input_verification_detects_content_change_and_respects_recorded_absence(tmp_path):
    present = tmp_path / "present.yml"
    absent = tmp_path / "absent.yml"
    present.write_text("value: 1\n", encoding="utf-8")
    manifest = {"inputs": {"files": [
        {"path": str(present), "exists": True, "sha256": sha256_file(present)},
        {"path": str(absent), "exists": False, "sha256": None},
    ]}}
    replay = _module()
    assert replay.verify_inputs(manifest) == []
    present.write_text("value: 2\n", encoding="utf-8")
    assert "changed" in replay.verify_inputs(manifest)[0]
    present.write_text("value: 1\n", encoding="utf-8")
    absent.write_text("now present\n", encoding="utf-8")
    assert "previously absent" in replay.verify_inputs(manifest)[0]


def test_comparison_ignores_dynamic_reports_but_checks_calibration_artifacts():
    original = {"inputs": {"observations": {"sha256": "obs"},
                           "target_geometry": {"sha256": "target"}},
                "outputs": {"cameras": {"sha256": "same"},
                            "report_json": {"sha256": "old-run-id"}}}
    replayed = {"inputs": {"observations": {"sha256": "obs"},
                           "target_geometry": {"sha256": "target"}},
                "outputs": {"cameras": {"sha256": "same"},
                            "report_json": {"sha256": "new-run-id"}}}
    replay = _module()
    assert replay.compare_manifests(original, replayed) == []
    replayed["outputs"]["cameras"]["sha256"] = "different"
    assert "cameras" in replay.compare_manifests(original, replayed)[0]


def test_missing_manifest_has_actionable_error_without_traceback(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "RESULTS" / "calibration_manifest.json"
    monkeypatch.setattr(sys, "argv", ["replay_rig_manifest.py", str(missing), "--verify-only"])

    assert _module().main() == 2

    stderr = capsys.readouterr().err
    assert f"REPLAY REFUSED: manifest not found: {missing}" in stderr
    assert "Pass the actual calibration_manifest.json" in stderr
    assert "Traceback" not in stderr
