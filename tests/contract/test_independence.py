"""
Decoupling gate (pure-pytest stand-in for import-linter).

Enforces the project's layered import rules so that module independence is CI-checked,
not merely intended. The layers (low → high):

  - core/        : numpy/typing/stdlib only; never imports any higher layer
  - data/        : neutral containers; imports core only
  - geometry/    : shared geometry primitives (one PnP/BA/averaging); imports core + data
  - models/*_math: numpy only; never imports anything from ds_msp
  - capability services {ops, adapt, calib, mvg, stereo}: mutually independent
  - pipeline services {rig, vo}: COMPOSE capabilities (rig->calib, vo->mvg) but stay
    independent of each other; capabilities never import a pipeline (acyclic)
  - the math foundation {core, data, geometry, models} is cv2/scipy-free

Rules are applied only to files that exist, so the gate is correct now and stays correct
as later phases add modules. It also verifies each checked module imports in isolation in
a fresh interpreter. Mirrors the import-linter contracts in pyproject.toml.
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


def _module_name(pyfile: Path) -> str:
    """Dotted module name of a file under the package, e.g. ``ds_msp.data.dataset``."""
    rel = pyfile.relative_to(PKG.parent).with_suffix("")
    return ".".join(rel.parts)


def _resolved_internal_imports(pyfile: Path):
    """Like :func:`_internal_imports` but resolves *relative* imports against the file's
    own package, so ``from .observations import X`` in ``ds_msp/data/dataset.py`` resolves
    to ``ds_msp.data.observations`` (not the mis-rooted ``ds_msp.observations``)."""
    pkgparts = _module_name(pyfile).split(".")[:-1]      # package of this module
    tree = ast.parse(pyfile.read_text())
    internal, external = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("ds_msp"):
                    internal.add(a.name)
                else:
                    external.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                base = pkgparts[:len(pkgparts) - (node.level - 1)]
                tgt = ".".join(base + ([node.module] if node.module else []))
                internal.add(tgt)
            elif node.module:
                if node.module.startswith("ds_msp"):
                    internal.add(node.module)
                else:
                    external.add(node.module.split(".")[0])
    return internal, external


def _files(glob):
    return sorted(p for p in PKG.glob(glob) if p.name != "__init__.py")


#: Service packages split into two tiers (an acyclic, PyTorch-like composition):
#:  * CAPABILITIES — single-purpose, mutually independent building blocks.
#:  * PIPELINES — higher-level orchestrators that COMPOSE capabilities (rig builds on
#:    calib; vo builds on mvg). Pipelines may import capabilities (downward), never each
#:    other; capabilities never import a pipeline (no upward edge → no cycles).
CAPABILITIES = ["ops", "adapt", "calib", "mvg", "stereo"]
PIPELINES = ["rig", "vo"]


def test_core_is_dependency_free():
    for f in _files("core/*.py"):
        internal, external = _internal_imports(f)
        forbidden = {m for m in internal
                     if any(m.startswith(f"ds_msp.{layer}")
                            for layer in ("models", "data", "geometry", "detect", "ops",
                                          "adapt", "io", "calib", "mvg", "stereo", "rig",
                                          "vo", "cv", "ldc", "isaac_sim"))}
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


def test_data_layer_depends_only_on_core():
    """ds_msp/data/* is the neutral container layer — it may import core (+ itself) only."""
    for f in _files("data/*.py"):
        internal, _ = _resolved_internal_imports(f)
        bad = {m for m in internal if m.startswith("ds_msp.")
               and not m.startswith("ds_msp.core")
               and not m.startswith("ds_msp.data")}
        assert not bad, f"data/{f.name} may import only core, found {bad}"


def test_geometry_layer_depends_only_on_core_and_data():
    """ds_msp/geometry/* is the shared geometry primitives layer — core + data only."""
    for f in _files("geometry/*.py"):
        internal, _ = _resolved_internal_imports(f)
        bad = {m for m in internal if m.startswith("ds_msp.")
               and not m.startswith(("ds_msp.core", "ds_msp.data", "ds_msp.geometry"))}
        assert not bad, f"geometry/{f.name} may import only core+data, found {bad}"


def _imports_service(internal, other):
    return {m for m in internal if m.startswith(f"ds_msp.{other}.") or m == f"ds_msp.{other}"}


def test_capabilities_are_mutually_independent_and_pipeline_free():
    """Capability services import no other service — not a sibling capability, not a pipeline
    (which would create an upward edge / cycle)."""
    for svc in CAPABILITIES:
        for f in _files(f"{svc}/*.py"):
            internal, _ = _resolved_internal_imports(f)
            for other in CAPABILITIES + PIPELINES:
                if other == svc:
                    continue
                hits = _imports_service(internal, other)
                assert not hits, f"capability {svc}/{f.name} must not import '{other}': {hits}"


def test_pipelines_do_not_import_each_other():
    """Pipelines may compose capabilities (rig->calib, vo->mvg) but must stay independent
    of one another."""
    for svc in PIPELINES:
        for f in _files(f"{svc}/*.py"):
            internal, _ = _resolved_internal_imports(f)
            for other in PIPELINES:
                if other == svc:
                    continue
                hits = _imports_service(internal, other)
                assert not hits, f"pipeline {svc}/{f.name} must not import pipeline '{other}': {hits}"


def test_math_foundation_is_cv2_and_scipy_free():
    """The math foundation (core/data/geometry/models) never touches OpenCV or SciPy —
    cv2 stays confined to detection/IO/image services, keeping the solver path portable."""
    for sub in ("core", "data", "geometry", "models"):
        for f in _files(f"{sub}/*.py"):
            _, external = _resolved_internal_imports(f)
            bad = external & {"cv2", "scipy"}
            assert not bad, f"{sub}/{f.name} (math foundation) must be cv2/scipy-free, found {bad}"


@pytest.mark.parametrize("module", [
    "ds_msp.core.contracts",
    "ds_msp.testing",
    "ds_msp.data.observations",
    "ds_msp.data.dataset",
    "ds_msp.geometry.resection",
    "ds_msp.geometry.calibrate_core",
])
def test_module_imports_in_isolation(module):
    """Each foundation module must import in a fresh interpreter with no side effects."""
    r = subprocess.run([sys.executable, "-c", f"import {module}"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"isolated import of {module} failed:\n{r.stderr}"

# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("NFR-ARCH-001", "NFR-ARCH-002")
