"""Mirrored test for docs_src/how_to/undistort_images/estimate_and_undistort.py.

Asserts the exact values shown on docs/how-to/undistort_images.md's
"Undistort an image in three calls" section.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.how_to.undistort_images import estimate_and_undistort

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    estimate_and_undistort.main()
    out = capsys.readouterr().out
    assert "D = [0.183, 0.809]" in out
    assert "(1080, 1920, 3)" in out
    assert "284.56" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.how_to.undistort_images.estimate_and_undistort"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "284.56" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
