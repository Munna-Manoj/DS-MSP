"""Mirrored test for docs_src/how_to/solve_pnp_on_fisheye/pinhole_contrast.py.

Asserts the exact values shown on docs/how-to/solve_pnp_on_fisheye.md's
"Contrast: pinhole PnP on the same points" section.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.how_to.solve_pnp_on_fisheye import pinhole_contrast

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    pinhole_contrast.main()
    out = capsys.readouterr().out
    assert "cv2 rotation error: 0.57 deg" in out
    assert "cv2 translation error: 1.37 m" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.how_to.solve_pnp_on_fisheye.pinhole_contrast"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "cv2 rotation error" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
