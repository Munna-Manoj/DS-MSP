"""Mirrored test for docs_src/how_to/read_write_kalibr/roundtrip_kalibr_yaml.py.

Asserts the exact values shown on docs/how-to/read_write_kalibr.md's round-trip section.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.how_to.read_write_kalibr import roundtrip_kalibr_yaml

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    roundtrip_kalibr_yaml.main()
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "True"
    assert out[1] == "True"
    assert out[2] == "0.0"


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.how_to.read_write_kalibr.roundtrip_kalibr_yaml"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["True", "True", "0.0"]


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
