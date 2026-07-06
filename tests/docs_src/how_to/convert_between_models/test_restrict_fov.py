"""Mirrored test for docs_src/how_to/convert_between_models/restrict_fov.py.

Asserts the exact values shown on docs/how-to/convert_between_models.md's
"Restrict the FOV for narrow targets" section.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.how_to.convert_between_models import restrict_fov

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    restrict_fov.main()
    out = capsys.readouterr().out
    assert "rms_px=0.768" in out
    assert "fov_covered_deg=119.9" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.how_to.convert_between_models.restrict_fov"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "rms_px=0.768" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
