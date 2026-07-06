"""Mirrored test for docs_src/how_to/undistort_images/balance_tradeoff.py.

Asserts the exact values shown on docs/how-to/undistort_images.md's
"Control the FOV-vs-border trade-off with balance" section.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.how_to.undistort_images import balance_tradeoff

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    balance_tradeoff.main()
    out = capsys.readouterr().out
    assert "balance=0.0  fx_new=284.56 px  black_fraction=0.075" in out
    assert "balance=0.5  fx_new=426.84 px  black_fraction=0.001" in out
    assert "balance=1.0  fx_new=569.12 px  black_fraction=0.000" in out
    assert "fx_new(1.0) / fx_new(0.0) = 2.00" in out
    assert "midpoint(0.0, 1.0) = 426.84  (balance=0.5 gives 426.84)" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.how_to.undistort_images.balance_tradeoff"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "balance=1.0  fx_new=569.12 px  black_fraction=0.000" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
