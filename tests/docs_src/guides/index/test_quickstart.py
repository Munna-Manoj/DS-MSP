"""Mirrored test for docs_src/guides/index/quickstart.py.

Asserts the exact values shown on docs/index.md's quickstart snippet.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from docs_src.guides.index import quickstart

ROOT = Path(__file__).resolve().parents[4]


def test_main_prints_expected_values(capsys):
    quickstart.main()
    out = capsys.readouterr().out
    assert "500/500 points valid through the round trip" in out
    assert "max round-trip error: 2.27e-13 px" in out


def test_module_runs_as_script():
    result = subprocess.run(
        [sys.executable, "-m", "docs_src.guides.index.quickstart"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "points valid through the round trip" in result.stdout


# Traceability: links this suite to the requirement it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-002")
