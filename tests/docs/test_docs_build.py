"""Build-time verification for the published MkDocs Material site.

``mkdocs build --strict`` fails on any broken internal link, missing nav reference, or
mkdocstrings import error -- the mechanical backstop against exactly the "40 relative
whole-file links that 404 on the real site" class of drift a docs/learn/ audit found by hand.
Skipped (not failed) when the optional ``docs`` dependency group isn't installed or
``mkdocs.yml`` doesn't exist yet, matching this repo's dataset-gated test convention
(``tests/realdata/``) -- this becomes a real, enforced gate once the docs site infra lands.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _mkdocs_ready() -> bool:
    try:
        import mkdocs  # noqa: F401
    except ImportError:
        return False
    return (ROOT / "mkdocs.yml").exists()


@pytest.mark.skipif(not _mkdocs_ready(),
                    reason="mkdocs not installed (pip install ds-msp[docs]) or mkdocs.yml "
                          "doesn't exist yet")
@pytest.mark.xfail(reason="known: docs/learn/*.md and the top-level guides still have ~54 "
                         "relative ../../ds_msp/... and ../../assets/... links that don't "
                         "resolve inside the built site -- the docs-overhaul plan's Phase "
                         "2-P3 converts these to absolute, line-anchored GitHub links. Remove "
                         "this xfail marker once that phase lands; a green run here again "
                         "means the link-fix regressed.", strict=True)
def test_mkdocs_build_strict(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict",
         "--site-dir", str(tmp_path / "site")],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"mkdocs build --strict failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("NFR-DOCS-001")
