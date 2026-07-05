#!/usr/bin/env python3
"""docs_src/ completeness gate (pure stdlib, CI-safe).

``docs_src/`` holds one small, standalone, unconditionally-runnable Python file per code
example shown on the published docs site, included into a page via mkdocs's
``{* docs_src/<path> hl[...] *}`` macro syntax (``mdx-include`` + ``markdown-include-variants``).
The whole point of the mechanism is that every sample shown to a reader is real, tested code —
that guarantee only holds if every file is both referenced by a page and covered by a test, and
every reference resolves to a real file. This script enforces both directions:

  * every ``{* docs_src/... *}`` reference found in ``docs/**/*.md`` resolves to a real file
    under ``docs_src/`` (catches a page pointing at a renamed/deleted example); and
  * every ``docs_src/**/*.py`` file (excluding ``__init__.py``) is referenced by at least one
    page AND has a mirrored test at ``tests/docs_src/<same-relative-path>/test_<name>.py``
    (catches an orphaned example nobody includes, or one with no regression coverage).

Run in CI on every PR:  python tools/check_docs_src_coverage.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_SRC = ROOT / "docs_src"
DOCS = ROOT / "docs"
TESTS_DOCS_SRC = ROOT / "tests" / "docs_src"

# Same set mkdocs.yml's `exclude_docs:` keeps out of the published site -- contributor-facing,
# not reader-facing, so a `{* ... *}` example there (e.g. WRITING_GUIDE.md's own convention
# writeup) is prose about the mechanism, not a real page reference.
EXCLUDED_DOC_PATHS = (DOCS / "process", DOCS / "research", DOCS / "readme-options",
                      DOCS / "WRITING_GUIDE.md")

# {* docs_src/learn/foo/bar.py hl[1,2] *}  or with a line range, per markdown-include-variants
INCLUDE_RE = re.compile(r"\{\*\s*(docs_src/[^\s*]+?\.py)(?:\s+[^*]*)?\*\}")


def _is_excluded(md: Path) -> bool:
    return any(md == p or p in md.parents for p in EXCLUDED_DOC_PATHS)


def _referenced_paths() -> dict[str, list[Path]]:
    """Map ``docs_src/...`` path string -> list of published markdown pages referencing it."""
    refs: dict[str, list[Path]] = {}
    for md in DOCS.rglob("*.md"):
        if _is_excluded(md):
            continue
        text = md.read_text()
        for match in INCLUDE_RE.finditer(text):
            refs.setdefault(match.group(1), []).append(md)
    return refs


def _real_docs_src_files() -> list[Path]:
    if not DOCS_SRC.is_dir():
        return []
    return [p for p in DOCS_SRC.rglob("*.py") if p.name != "__init__.py"]


def _mirrored_test_path(src_file: Path) -> Path:
    rel = src_file.relative_to(DOCS_SRC)  # e.g. learn/projection_validity/validity_cone.py
    return TESTS_DOCS_SRC / rel.parent / f"test_{rel.name}"


def main() -> int:
    refs = _referenced_paths()
    real_files = _real_docs_src_files()
    real_rel = {str(p.relative_to(ROOT)) for p in real_files}
    errors: list[str] = []

    # (1) every {* docs_src/... *} reference must resolve to a real file
    for ref_path, pages in sorted(refs.items()):
        if ref_path not in real_rel:
            for page in pages:
                errors.append(
                    f"{page.relative_to(ROOT)} references '{{* {ref_path} *}}' "
                    f"but no such file exists")

    # (2) every real docs_src file must be referenced by >=1 page and have a mirrored test
    for src_file in sorted(real_files):
        rel = str(src_file.relative_to(ROOT))
        if rel not in refs:
            errors.append(f"{rel} is not included by any docs page (orphaned example)")
        test_path = _mirrored_test_path(src_file)
        if not test_path.is_file():
            errors.append(
                f"{rel} has no mirrored test at {test_path.relative_to(ROOT)}")

    if errors:
        print("DOCS_SRC COVERAGE: FAIL")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"DOCS_SRC COVERAGE: OK ({len(real_files)} examples, "
          f"{sum(len(v) for v in refs.values())} page references)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
