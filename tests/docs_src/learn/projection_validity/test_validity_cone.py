"""Mirrored test for docs_src/learn/projection_validity/validity_cone.py.

Asserts the exact values shown on docs/learn/03_projection_validity.md's section 2.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.learn.projection_validity import validity_cone

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    validity_cone.main()
    out = capsys.readouterr().out
    assert "xi=0.1832, alpha=0.8086" in out
    assert "w2 = 0.3967, theta_max = 113.4 deg" in out
    assert "numeric check: 113.3 deg" in out
    assert "total field of view: 227 deg" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.learn.projection_validity.validity_cone"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "theta_max = 113.4 deg" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
