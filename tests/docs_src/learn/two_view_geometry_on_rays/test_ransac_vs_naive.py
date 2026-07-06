"""Mirrored test for docs_src/learn/two_view_geometry_on_rays/ransac_vs_naive.py.

Asserts the exact values shown on docs/learn/08_two_view_geometry_on_rays.md's section 5.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.learn.two_view_geometry_on_rays import ransac_vs_naive

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    ransac_vs_naive.main()
    out = capsys.readouterr().out
    assert "naive  rotation error : 26.78 deg" in out
    assert "RANSAC rotation error : 0.107 deg" in out
    assert "RANSAC trans-dir error: 0.274 deg" in out
    assert "inlier precision/recall: 0.989 / 1.000  (92/120)" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.learn.two_view_geometry_on_rays.ransac_vs_naive"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "RANSAC rotation error" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
