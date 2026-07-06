"""Mirrored test for docs_src/how_to/undistort_images/object_api.py.

Asserts the exact values shown on docs/how-to/undistort_images.md's
"Or use the object API" section.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.how_to.undistort_images import object_api

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    object_api.main()
    out = capsys.readouterr().out
    assert "(1080, 1920, 3)" in out
    assert "426.84" in out
    assert "TypeError: DoubleSphereCamera.undistort_image() got an unexpected keyword " \
           "argument 'balance'" in out
    assert "569.12" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.how_to.undistort_images.object_api"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "569.12" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
