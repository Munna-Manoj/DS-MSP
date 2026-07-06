"""Mirrored test for docs_src/explain/are_two_models_the_same_camera/paraxial_focal_check.py.

Asserts the exact values shown on docs/explain/are_two_models_the_same_camera.md's
section 2 (the Double Sphere paraxial-focal derivation, checked on the bundled fixture).
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.explain.are_two_models_the_same_camera import paraxial_focal_check

ROOT = Path(__file__).resolve().parents[4]


def test_paraxial_focal_check_main(capsys):
    paraxial_focal_check.main()
    out = capsys.readouterr().out

    assert "fx=711.5745  xi=0.1832" in out
    assert "closed form   fx/(1+xi)        = 601.392276" in out
    assert "finite diff   radius(h)/h      = 601.392276" in out


def test_paraxial_focal_check_smoke():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.explain.are_two_models_the_same_camera.paraxial_focal_check"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "closed form" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
