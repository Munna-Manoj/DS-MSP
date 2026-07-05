"""Mirrored test for docs_src/guides/rig_calibration_guide/load_camera.py.

Asserts the exact values shown on docs/RIG_CALIBRATION_GUIDE.md's "Loading a
camera back into a ready instance" section.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.guides.rig_calibration_guide import load_camera

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    load_camera.main()
    out = capsys.readouterr().out
    assert ("KannalaBrandtModel(fx=900.000, fy=900.000, cx=960.000, cy=540.000, "
            "k=[0.01000, -0.02000, 0.00000, 0.00000])") in out
    assert "uv[0] = (960.000, 540.000)   valid=True" in out   # on-axis point -> principal point
    assert "uv[1] = (1136.736, 422.176)   valid=True" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.guides.rig_calibration_guide.load_camera"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "KannalaBrandtModel" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
