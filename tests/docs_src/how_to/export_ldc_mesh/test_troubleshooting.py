"""Mirrored test for docs_src/how_to/export_ldc_mesh/troubleshooting.py.

Asserts the exact values shown in docs/how-to/export_ldc_mesh.md's Troubleshooting section.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.how_to.export_ldc_mesh import troubleshooting

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    troubleshooting.main()
    out = capsys.readouterr().out
    assert "(69, 121, 2)" in out
    assert "ValueError: compute_K_new requires image dimensions" in out
    assert "426.84" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.how_to.export_ldc_mesh.troubleshooting"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ValueError: compute_K_new requires image dimensions" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
