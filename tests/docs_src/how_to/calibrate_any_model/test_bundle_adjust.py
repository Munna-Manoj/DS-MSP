"""Mirrored test for docs_src/how_to/calibrate_any_model/bundle_adjust.py.

Asserts the exact values shown on docs/how-to/calibrate_any_model.md's
"Bundle-adjust from a seed model" section.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.how_to.calibrate_any_model import bundle_adjust

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    bundle_adjust.main()
    out = capsys.readouterr().out
    assert "True" in out
    assert "0.5768" in out
    assert "DoubleSphereModel(fx=737.994, fy=737.123, cx=951.364, cy=517.716, " \
           "xi=0.2350, alpha=0.8203)" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.how_to.calibrate_any_model.bundle_adjust"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "0.5768" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
