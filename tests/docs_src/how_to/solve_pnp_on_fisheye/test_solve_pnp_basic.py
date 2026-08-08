"""Mirrored test for docs_src/how_to/solve_pnp_on_fisheye/solve_pnp_basic.py.

Asserts the exact values shown on docs/how-to/solve_pnp_on_fisheye.md's
"Verify it on a synthetic scene" section.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.how_to.solve_pnp_on_fisheye import solve_pnp_basic

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    solve_pnp_basic.main()
    out = capsys.readouterr().out
    assert "True 40" in out
    assert "rotation error: 0.00e+00 deg" in out
    assert "translation error: 0.00e+00 m" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.how_to.solve_pnp_on_fisheye.solve_pnp_basic"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "rotation error" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
