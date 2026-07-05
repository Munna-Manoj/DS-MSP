"""Mirrored test for docs_src/how_to/read_write_kalibr/write_kalibr_yaml.py.

Asserts the exact YAML shown on docs/how-to/read_write_kalibr.md's write section.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.how_to.read_write_kalibr import write_kalibr_yaml

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_yaml(capsys):
    write_kalibr_yaml.main()
    out = capsys.readouterr().out
    assert "camera_model: ds" in out
    assert "intrinsics: [0.183, 0.809, 711.57, 711.24, 949.18, 518.81]" in out
    assert "distortion_model: none" in out
    assert "distortion_coeffs: []" in out
    assert "resolution: [1920, 1080]" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.how_to.read_write_kalibr.write_kalibr_yaml"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "camera_model: ds" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
