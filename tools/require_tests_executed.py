#!/usr/bin/env python3
"""Fail when a JUnit result contains no executed (non-skipped) tests.

Pytest exits successfully when every dataset-gated test skips.  That is useful for an optional
nightly but is not release evidence.  This small stdlib guard gives a required validation dispatch
different semantics without coupling tests to GitHub Actions.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET


def executed_count(path: str) -> tuple[int, int]:
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    skipped = sum(case.find("skipped") is not None for case in cases)
    return len(cases) - skipped, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("junit_xml")
    args = ap.parse_args()
    try:
        executed, skipped = executed_count(args.junit_xml)
    except (OSError, ET.ParseError) as exc:
        print(f"REALDATA EVIDENCE: FAIL — cannot read JUnit result: {exc}", file=sys.stderr)
        return 2
    if executed == 0:
        print(f"REALDATA EVIDENCE: FAIL — 0 tests executed ({skipped} skipped)", file=sys.stderr)
        return 1
    print(f"REALDATA EVIDENCE: OK ({executed} executed, {skipped} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
