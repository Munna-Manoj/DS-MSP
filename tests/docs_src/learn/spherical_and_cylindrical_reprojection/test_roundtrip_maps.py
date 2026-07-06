"""Mirrored test for docs_src/learn/spherical_and_cylindrical_reprojection/roundtrip_maps.py.

Asserts the exact values shown on docs/learn/spherical_and_cylindrical_reprojection.md's
"number that proves they're inverses" section.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.learn.spherical_and_cylindrical_reprojection import roundtrip_maps

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_residuals(capsys):
    roundtrip_maps.main()
    out = capsys.readouterr().out
    # Both residuals are float64 round-off (~1e-13 px); the exact digits vary by platform's
    # BLAS/LAPACK backend -- check magnitude, not digits (see the docs page's own caveat).
    m_cyl = re.search(r"sphere -> cylinder -> sphere: max residual = ([\d.eE+-]+) px", out)
    m_pin = re.search(r"sphere -> pinhole  -> sphere: max residual = ([\d.eE+-]+) px", out)
    assert m_cyl is not None and m_pin is not None, out
    assert float(m_cyl.group(1)) < 1e-9
    assert float(m_pin.group(1)) < 1e-9
    assert "(front hemisphere only)" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.learn.spherical_and_cylindrical_reprojection.roundtrip_maps"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "sphere -> cylinder -> sphere" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
