"""Mirrored test for docs_src/guides/multi_model/opencv_interop.py.

Asserts the exact values shown on docs/MULTI_MODEL.md's
"Direct OpenCV interop" section.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.guides.multi_model import opencv_interop

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    opencv_interop.main()
    out = capsys.readouterr().out
    assert "(1080, 1920, 3)" in out
    assert "[610.26, 480.33]" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.guides.multi_model.opencv_interop"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "(1080, 1920, 3)" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
