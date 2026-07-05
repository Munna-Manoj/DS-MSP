"""Mirrored test for docs_src/learn/projection_validity/balance_sweep.py.

Asserts the exact values shown on docs/learn/03_projection_validity.md's section 4.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.learn.projection_validity import balance_sweep

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    balance_sweep.main()
    out = capsys.readouterr().out
    assert "balance=0.00  hfov=147.0 deg  filled=92.5%" in out
    assert "balance=0.25  hfov=139.3 deg  filled=97.8%" in out
    assert "balance=0.50  hfov=132.1 deg  filled=99.9%" in out
    assert "balance=0.75  hfov=125.2 deg  filled=100.0%" in out
    assert "balance=1.00  hfov=118.7 deg  filled=100.0%" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.learn.projection_validity.balance_sweep"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "balance=1.00  hfov=118.7 deg  filled=100.0%" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
