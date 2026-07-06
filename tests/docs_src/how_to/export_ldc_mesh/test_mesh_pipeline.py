"""Mirrored test for docs_src/how_to/export_ldc_mesh/mesh_pipeline.py.

Asserts the exact values shown across docs/how-to/export_ldc_mesh.md's continuation chain
(cam -> gen -> res -> mesh_lut -> downsample sweep -> keypoint undistortion).
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.how_to.export_ldc_mesh import mesh_pipeline

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    mesh_pipeline.main()
    out = capsys.readouterr().out
    assert "(69, 121, 2) int16" in out
    assert "426.84" in out
    assert "['mesh_lut', 'mesh_lut_float', 'K_new', 'config']" in out
    assert "(69, 121, 2)" in out
    assert "4" in out
    assert "[ -87 -156]" in out
    assert "[-10.875 -19.5  ]" in out or "[-10.875 -19.5]" in out
    assert "-3046 2873" in out
    assert "3 8 (136, 241, 2)" in out
    assert "4 16 (69, 121, 2)" in out
    assert "5 32 (35, 61, 2)" in out
    assert "[[ 967.68  555.05]" in out
    assert "[1427.98  832.03]" in out
    assert "[ 657.04  350.07]]" in out
    assert "[ True  True  True]" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.how_to.export_ldc_mesh.mesh_pipeline"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "426.84" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
