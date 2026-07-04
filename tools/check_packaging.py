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
  * every ``[project.scripts]`` console entry (e.g. ``ds-msp-calibrate-rig``) points at a
    real, importable ``module:function`` -- catches the entry point rotting after a rename/
    move (this is exactly how ``ds_msp.rig.cli`` replaced ``scripts/calibrate_rig.py``, which
    is NOT shipped, so any console command MUST live inside ``ds_msp*`` or pip-only users get
    a broken command with no repo to fall back on).
  * every ``[tool.setuptools.package-data]`` glob resolves to at least one real file on disk
    -- catches a stale/typo'd pattern that would silently ship an empty data set (e.g.
    ``ds-msp-calibrate-rig --init-config`` needs its templates to actually be in the wheel).

Any excluded package must therefore be consciously kept out of the README's "shipped" claims.
Run ``python tools/check_packaging.py`` (exit 1 on mismatch).
"""
from __future__ import annotations

import glob
import importlib
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


def _console_script_errors() -> list[str]:
    """Every ``[project.scripts]`` entry must resolve to a real, importable ``module:function``
    living inside a package that actually ships (``ds_msp*`` -- ``scripts/`` does not)."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    scripts = data.get("project", {}).get("scripts", {})
    excluded = _excluded_packages()
    errors = []
    for name, target in scripts.items():
        module, _, func = target.partition(":")
        top = module.split(".")[0]
        if top != "ds_msp":
            errors.append(f"console script '{name}' points at '{target}', which is not inside "
                          f"the ds_msp package -- only ds_msp* ships, so this command would be "
                          f"broken for anyone who `pip install`s rather than clones the repo")
            continue
        parts = module.split(".")
        if len(parts) >= 2 and parts[1] in excluded:
            errors.append(f"console script '{name}' points at '{module}', but "
                          f"'ds_msp.{parts[1]}' is excluded from the wheel")
            continue
        try:
            mod = importlib.import_module(module)
        except ImportError as e:
            errors.append(f"console script '{name}' -> '{module}' failed to import: {e}")
            continue
        if func and not hasattr(mod, func):
            errors.append(f"console script '{name}' -> '{target}': '{module}' has no "
                          f"attribute '{func}'")
    return errors


def _package_data_errors() -> list[str]:
    """Every ``[tool.setuptools.package-data]`` glob must match at least one real file, so a
    stale/typo'd pattern doesn't silently ship an empty data set (caught the way it would bite
    a user: e.g. ``ds-msp-calibrate-rig --init-config`` needs its templates in the wheel)."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pkg_data = data.get("tool", {}).get("setuptools", {}).get("package-data", {})
    errors = []
    for package, patterns in pkg_data.items():
        pkg_dir = ROOT / Path(*package.split("."))
        if not pkg_dir.is_dir():
            errors.append(f"package-data declares '{package}' but no such directory exists "
                          f"at {pkg_dir.relative_to(ROOT)}")
            continue
        for pat in patterns:
            if not glob.glob(str(pkg_dir / pat)):
                errors.append(f"package-data pattern '{package}': '{pat}' matches no files "
                              f"under {pkg_dir.relative_to(ROOT)}/")
    return errors


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

    # (3) console scripts resolve to real, shipped, importable targets
    errors.extend(_console_script_errors())

    # (4) package-data globs actually match real files
    errors.extend(_package_data_errors())

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
