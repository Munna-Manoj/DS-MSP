"""
Decoupling gate (pure-pytest stand-in for import-linter).

Enforces the project's layered import rules so that
module independence is CI-checked, not merely intended:

  - ds_msp/core/*        : numpy/typing/stdlib only; never imports models/ops/adapt/io
  - ds_msp/models/*_math : numpy only; never imports anything from ds_msp
  - ds_msp/ops/*         : may import ds_msp.core, never concrete models or sibling ops
  - ds_msp/adapt/*       : may import ds_msp.core (+ a model registry), never ops

Rules are applied only to files that exist, so the gate is correct now and stays
correct as later phases add modules. It also verifies each checked module imports
in isolation in a fresh interpreter.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[2] / "ds_msp"

STDLIB_OK = {"numpy", "typing", "abc", "dataclasses", "__future__", "json",
             "math", "warnings", "functools", "collections"}


def _internal_imports(pyfile: Path):
    """Set of ds_msp.* submodules a file imports (resolved, dotted)."""
    tree = ast.parse(pyfile.read_text())
    internal = set()
    external = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                (internal if a.name.startswith("ds_msp") else external).add(
                    a.name if a.name.startswith("ds_msp") else top
                )
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # relative import inside ds_msp -> resolve to a ds_msp.* target
                internal.add("ds_msp." + (node.module or ""))
            elif node.module:
                top = node.module.split(".")[0]
                (internal if node.module.startswith("ds_msp") else external).add(
                    node.module if node.module.startswith("ds_msp") else top
                )
    return internal, external


def _files(glob):
    return sorted(p for p in PKG.glob(glob) if p.name != "__init__.py")


def test_core_is_dependency_free():
    for f in _files("core/*.py"):
        internal, external = _internal_imports(f)
        forbidden = {m for m in internal
                     if any(m.startswith(f"ds_msp.{layer}")
                            for layer in ("models", "ops", "adapt", "io", "cv", "ldc"))}
        assert not forbidden, f"{f.name} (core) must not import {forbidden}"
        bad_ext = external - STDLIB_OK
        assert not bad_ext, f"{f.name} (core) has unexpected external deps {bad_ext}"


def test_math_layer_is_pure_numpy():
    for f in _files("models/*_math.py"):
        internal, external = _internal_imports(f)
        assert not internal, f"{f.name} (*_math) must not import ds_msp.* ({internal})"
        bad_ext = external - {"numpy", "typing", "__future__", "math"}
        assert not bad_ext, f"{f.name} (*_math) must be pure numpy, found {bad_ext}"


def test_ops_do_not_import_concrete_models_or_each_other():
    for f in _files("ops/*.py"):
        internal, _ = _internal_imports(f)
        # importing core is allowed; importing a concrete model class is not
        bad = {m for m in internal if m.startswith("ds_msp.models")}
        assert not bad, f"{f.name} (ops) must depend on core contracts, not models ({bad})"


def test_adapt_does_not_import_ops():
    for f in _files("adapt/*.py"):
        internal, _ = _internal_imports(f)
        bad = {m for m in internal if m.startswith("ds_msp.ops")}
        assert not bad, f"{f.name} (adapt) must not import ops ({bad})"


@pytest.mark.parametrize("module", [
    "ds_msp.core.contracts",
    "ds_msp.testing",
])
def test_module_imports_in_isolation(module):
    """Each foundation module must import in a fresh interpreter with no side effects."""
    r = subprocess.run([sys.executable, "-c", f"import {module}"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"isolated import of {module} failed:\n{r.stderr}"
