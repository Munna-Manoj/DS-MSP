"""Mirrored test for docs_src/learn/two_view_geometry_on_rays/double_sphere_roundtrip.py.

Asserts the exact values shown on docs/learn/08_two_view_geometry_on_rays.md's section 4.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.learn.two_view_geometry_on_rays import double_sphere_roundtrip

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    double_sphere_roundtrip.main()
    out = capsys.readouterr().out
    assert "valid pairs          : 60" in out
    assert "rotation error       : 1.21e-06 deg" in out
    assert "translation-dir error: 0.00e+00 deg" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.learn.two_view_geometry_on_rays.double_sphere_roundtrip"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "valid pairs" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
