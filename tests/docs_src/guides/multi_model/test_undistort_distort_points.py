"""Mirrored test for docs_src/guides/multi_model/undistort_distort_points.py.

Asserts the exact values shown on docs/MULTI_MODEL.md's
"Undistort / distort points" section.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.guides.multi_model import undistort_distort_points

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    undistort_distort_points.main()
    out = capsys.readouterr().out
    assert "[715.819, 509.335]" in out
    assert "[640.0, 480.0], [900.0, 300.0]" in out
    assert "round-trip max error: 2.89e-10 px" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.guides.multi_model.undistort_distort_points"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "round-trip max error" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
