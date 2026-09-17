"""Mirrored test for docs_src/how_to/export_ldc_mesh/any_model.py.

Asserts the exact values shown in docs/how-to/export_ldc_mesh.md's "any camera model"
section: a Kannala-Brandt and an OCam camera through the same generator.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.how_to.export_ldc_mesh import any_model

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    any_model.main()
    out = capsys.readouterr().out
    assert "kb (31, 41, 2) int16 True" in out
    assert "192.3 kb" in out
    assert "[[320.0, 240.0], [441.83, 179.28]] [True, True]" in out
    assert "ocam (31, 41, 2) int16 True" in out
    assert "132.0 ocam" in out
    assert "[[320.0, 240.0], [436.57, 181.71]] [True, True]" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.how_to.export_ldc_mesh.any_model"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ocam (31, 41, 2) int16 True" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
