from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.req("NFR-REPRO-002")


def _module():
    path = Path(__file__).resolve().parents[2] / "tools" / "require_tests_executed.py"
    spec = importlib.util.spec_from_file_location("require_tests_executed", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_counts_executed_and_skipped_junit_cases(tmp_path):
    path = tmp_path / "result.xml"
    path.write_text(
        '<testsuites><testsuite><testcase name="run"/>'
        '<testcase name="skip"><skipped/></testcase></testsuite></testsuites>',
        encoding="utf-8")
    assert _module().executed_count(str(path)) == (1, 1)


def test_all_skipped_is_detectable_as_no_evidence(tmp_path):
    path = tmp_path / "result.xml"
    path.write_text(
        '<testsuite><testcase name="skip"><skipped/></testcase></testsuite>',
        encoding="utf-8")
    assert _module().executed_count(str(path)) == (0, 1)
