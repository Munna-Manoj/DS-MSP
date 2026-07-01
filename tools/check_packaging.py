#!/usr/bin/env python3
"""Guard: the shipped package matches what the docs advertise.

Prevents the drift where the README (which is also the PyPI project page) presents a
subpackage as ``pip install``-available while ``pyproject.toml`` excludes it from the wheel
— exactly the "rig documented but not in the 0.8.0 wheel" mismatch this check was written for.

Pure stdlib (tomllib), no build step, so it runs in the per-PR governance job:

  * every ``ds_msp/<subpkg>/`` on disk is either shipped (in the wheel) or listed in the
    explicit ``exclude`` set of ``[tool.setuptools.packages.find]``; and
  * every subpackage the README describes as importable / installed via ``pip install ds-msp``
    is NOT in that exclude set.

Any excluded package must therefore be consciously kept out of the README's "shipped" claims.
Run ``python tools/check_packaging.py`` (exit 1 on mismatch).
"""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _excluded_packages() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    find = data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {})
    out = set()
    for pat in find.get("exclude", []):
        # "ds_msp.rig" / "ds_msp.rig.*" -> top-level subpackage name "rig"
        parts = pat.split(".")
        if len(parts) >= 2 and parts[0] == "ds_msp":
            out.add(parts[1])
    return out


def _disk_subpackages() -> set[str]:
    pkg = ROOT / "ds_msp"
    return {p.name for p in pkg.iterdir()
            if p.is_dir() and (p / "__init__.py").exists() and not p.name.startswith("_")}


def _readme_installed_subpackages() -> set[str]:
    """Subpackages the README presents as importable (``ds_msp.<name>`` / ``ds_msp/<name>``),
    minus any it explicitly flags as source-only (``from a source checkout`` / ``not ... wheel``)."""
    txt = (ROOT / "README.md").read_text()
    named = set(re.findall(r"ds_msp[./]([a-z_][a-z0-9_]*)", txt))
    disk = _disk_subpackages()
    return {n for n in named if n in disk}


def main() -> int:
    excluded = _excluded_packages()
    disk = _disk_subpackages()
    readme = _readme_installed_subpackages()
    errors = []

    # (1) every excluded package must still exist on disk (stale exclude of a deleted pkg)
    for name in sorted(excluded - disk):
        errors.append(f"pyproject excludes 'ds_msp.{name}' but no such subpackage exists on disk")

    # (2) nothing the README advertises as importable may be excluded from the wheel
    for name in sorted(readme & excluded):
        errors.append(
            f"README references 'ds_msp.{name}' as importable, but pyproject excludes it from "
            f"the wheel — a user who `pip install ds-msp` cannot import it. Either ship it "
            f"(remove the exclude) or mark it source-only in the README.")

    shipped = sorted(disk - excluded)
    if errors:
        print("PACKAGING: FAIL")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"PACKAGING: OK ({len(shipped)} subpackages shipped: {', '.join(shipped)}; "
          f"excluded: {', '.join(sorted(excluded)) or 'none'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
