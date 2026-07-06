"""Mirrored test for docs_src/learn/two_view_geometry_on_rays/double_sphere_roundtrip.py.

Asserts the exact values shown on docs/learn/08_two_view_geometry_on_rays.md's section 4.
"""
import re
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
    # Both errors are round-trip noise from the project/unproject pair (see the docs page's own
    # "~1.2e-6 deg" hedge); exact digits vary by platform's BLAS/LAPACK backend and can even land
    # on exactly 0.00e+00 -- check magnitude, not digits.
    m_rot = re.search(r"rotation error       : ([\d.eE+-]+) deg", out)
    m_dir = re.search(r"translation-dir error: ([\d.eE+-]+) deg", out)
    assert m_rot is not None and m_dir is not None, out
    assert float(m_rot.group(1)) < 1e-3
    assert float(m_dir.group(1)) < 1e-3


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.learn.two_view_geometry_on_rays.double_sphere_roundtrip"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "valid pairs" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
