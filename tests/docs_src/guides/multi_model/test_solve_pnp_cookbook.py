"""Mirrored test for docs_src/guides/multi_model/solve_pnp_cookbook.py.

Asserts the exact values shown on docs/MULTI_MODEL.md's
"Pose estimation (PnP)" section.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.guides.multi_model import solve_pnp_cookbook

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    solve_pnp_cookbook.main()
    out = capsys.readouterr().out
    assert "ok=True" in out
    assert "rvec=[-0.4809, -0.1674, -0.127]" in out
    assert "tvec=[-0.2892, -0.0329, 0.4515]" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.guides.multi_model.solve_pnp_cookbook"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok=True" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
