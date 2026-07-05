"""Mirrored test for docs_src/how_to/convert_between_models/convert_in_one_call.py.

Asserts the exact values shown on docs/how-to/convert_between_models.md's
"Convert in one call" section.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.how_to.convert_between_models import convert_in_one_call

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    convert_in_one_call.main()
    out = capsys.readouterr().out
    assert "rms_px=0.00021" in out
    assert "converged=True" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.how_to.convert_between_models.convert_in_one_call"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "rms_px=0.00021" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
