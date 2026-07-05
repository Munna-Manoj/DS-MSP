"""Mirrored test for docs_src/how_to/convert_between_models/use_everywhere.py.

Asserts the exact value shown on docs/how-to/convert_between_models.md's
"Use the converted model everywhere" section.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.how_to.convert_between_models import use_everywhere

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    use_everywhere.main()
    out = capsys.readouterr().out
    assert "ok=True" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.how_to.convert_between_models.use_everywhere"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok=True" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
