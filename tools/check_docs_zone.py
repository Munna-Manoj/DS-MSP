#!/usr/bin/env python3
"""docs/ top-level zoning gate (pure stdlib, CI-safe).

Enforces a positive allowlist for anything living directly under ``docs/``, instead of trying
to recognize unwanted content after the fact. A content pattern scan can only catch phrasing
someone already thought to check for — a file with different wording slips through undetected.
An allowlist cannot: any new top-level ``docs/`` entry not already on the manifest fails closed
regardless of its name or content, until a human deliberately adds it (a reviewable one-line
diff, not a silent drop).

Scope is deliberately narrow: only *top-level* entries directly under ``docs/``. Nested pages
under the Diataxis directories (explain/, how-to/, learn/, reference/, process/) are already
covered by the documentation workflow's own review step, ``mkdocs build --strict``, and
``docs_src`` coverage; this check targets specifically an ad hoc top-level file landing outside
all of that.

Run in CI on every PR:  python tools/check_docs_zone.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MANIFEST = ROOT / "docs" / "process" / "governance" / "DOCS_TOP_LEVEL_MANIFEST.txt"


def _load_manifest() -> set[str]:
    if not MANIFEST.is_file():
        print(f"DOCS ZONE: ERROR — manifest not found at {MANIFEST}", file=sys.stderr)
        sys.exit(2)
    entries: set[str] = set()
    for line in MANIFEST.read_text().splitlines():
        name = line.split("#", 1)[0].strip()
        if name:
            entries.add(name)
    return entries


def main() -> int:
    allowed = _load_manifest()
    if not DOCS.is_dir():
        print("DOCS ZONE: OK (no docs/ directory)")
        return 0

    actual = {p.name for p in DOCS.iterdir()}
    unlisted = sorted(actual - allowed)

    if unlisted:
        print("DOCS ZONE: FAIL — new top-level docs/ entries not on the manifest:")
        for name in unlisted:
            print(f"  - docs/{name}")
        print(
            f"\nEvery top-level docs/ file or directory must be explicitly listed in "
            f"{MANIFEST.relative_to(ROOT)}. This is a positive allowlist, not a content scan: it "
            "does not matter what the new entry is named or contains — if it's genuinely "
            "public-facing documentation, add one line to the manifest in this PR (a reviewable, "
            "deliberate act); otherwise remove it."
        )
        return 1

    print(f"DOCS ZONE: OK ({len(actual)} top-level docs/ entries, all on the manifest)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
