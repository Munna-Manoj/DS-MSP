"""Mirrored test for docs_src/how_to/convert_between_models/quality_report.py.

Asserts the exact values shown on docs/how-to/convert_between_models.md's
"Read the quality report" section.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.how_to.convert_between_models import quality_report

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    quality_report.main()
    out = capsys.readouterr().out
    assert "median_px=0.00018" in out
    assert "max_px=0.00099" in out
    assert "fov_covered_deg=179.9" in out
    assert "source_model='ds'" in out
    assert "target_model='kb'" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.how_to.convert_between_models.quality_report"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "fov_covered_deg=179.9" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
