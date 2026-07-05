"""Mirrored test for docs_src/how_to/convert_between_models/pick_target_model.py.

Asserts the exact value shown on docs/how-to/convert_between_models.md's
"Pick a target model" section.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.how_to.convert_between_models import pick_target_model

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    pick_target_model.main()
    out = capsys.readouterr().out
    assert "rms_px=0.014" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.how_to.convert_between_models.pick_target_model"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "rms_px=0.014" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
