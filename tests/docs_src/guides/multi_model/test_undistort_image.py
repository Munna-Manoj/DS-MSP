"""Mirrored test for docs_src/guides/multi_model/undistort_image.py.

Asserts the exact values shown on docs/MULTI_MODEL.md's
"Undistort an image to a pinhole view" section.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.guides.multi_model import undistort_image

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    undistort_image.main()
    out = capsys.readouterr().out
    assert "(1080, 1920, 3)" in out
    assert "426.84" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.guides.multi_model.undistort_image"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "426.84" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
