"""Mirrored test for docs_src/guides/multi_model/project_unproject.py.

Asserts the exact values shown on docs/MULTI_MODEL.md's
"Project / unproject" section.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.guides.multi_model import project_unproject

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    project_unproject.main()
    out = capsys.readouterr().out
    assert "[979.227, 518.81]" in out
    assert "[True, True]" in out
    assert "[1.0, 1.0]" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.guides.multi_model.project_unproject"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "[979.227, 518.81]" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
