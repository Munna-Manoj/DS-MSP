"""Mirrored test for docs_src/learn/two_view_geometry_on_rays/recover_pose_basic.py.

Asserts the exact values shown on docs/learn/08_two_view_geometry_on_rays.md's section 1.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.learn.two_view_geometry_on_rays import recover_pose_basic

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    recover_pose_basic.main()
    out = capsys.readouterr().out
    assert "rotation error       : 0.00e+00 deg" in out
    assert "translation-dir error: 0.00e+00 deg" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.learn.two_view_geometry_on_rays.recover_pose_basic"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "rotation error" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
