"""Mirrored test for docs_src/learn/double_sphere_model/forward_projection.py.

Asserts the exact values shown on docs/learn/02_double_sphere_model.md's section 3.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.learn.double_sphere_model import forward_projection

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    forward_projection.main()
    out = capsys.readouterr().out
    assert "x=-1.2837 y=-0.5700 z=1.8213" in out
    assert "d1=2.3000  z1=2.2427  d2=2.6462  den=2.5690" in out
    assert "u=593.6100  v=361.0100  (matches the original detected corner 593.61, 361.01)" in out
    assert "ds_project() agrees:  u=593.6100  v=361.0100  valid=True" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.learn.double_sphere_model.forward_projection"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "d1=2.3000  z1=2.2427  d2=2.6462  den=2.5690" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
