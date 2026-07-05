"""Mirrored test for docs_src/learn/two_view_geometry_on_rays/eight_point_residual.py.

Asserts the exact values shown on docs/learn/08_two_view_geometry_on_rays.md's section 3.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.learn.two_view_geometry_on_rays import eight_point_residual

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    eight_point_residual.main()
    out = capsys.readouterr().out
    assert "max epipolar residual: 5.69e-16" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.learn.two_view_geometry_on_rays.eight_point_residual"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "max epipolar residual" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
