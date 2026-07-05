"""Mirrored test for docs_src/learn/double_sphere_model/closed_form_unprojection.py.

Asserts the exact values shown on docs/learn/02_double_sphere_model.md's section 4.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.learn.double_sphere_model import closed_form_unprojection

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    closed_form_unprojection.main()
    out = capsys.readouterr().out
    assert "mx=-0.4997  my=-0.2219  r2=0.2989  mz=0.8730" in out
    assert "pixels tested: 30 / 30" in out
    assert "project(unproject(u)) round-trip: mean=1.89e-14px  max=1.14e-13px" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.learn.double_sphere_model.closed_form_unprojection"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "pixels tested: 30 / 30" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
