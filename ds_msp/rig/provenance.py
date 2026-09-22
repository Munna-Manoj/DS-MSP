"""Reproducible run manifests for the rig-calibration CLI.

The manifest deliberately records the *effective estimator inputs* (observations and target
geometry) in addition to source-file hashes.  A source image can be decoded differently by a
future OpenCV build; conversely, a cached keypoint file can reproduce the numerical calibration
without the raw images.  Keeping both levels makes that distinction explicit instead of claiming
that a command line alone is sufficient provenance.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.metadata
import importlib.resources
import io
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np

SCHEMA = "ds-msp-rig-run-manifest"
SCHEMA_VERSION = 1
MANIFEST_NAME = "calibration_manifest.json"
_DEPENDENCIES = ("ds_msp", "numpy", "scipy", "opencv-python", "opencv-python-headless",
                 "PyYAML")
_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256_file(path: os.PathLike | str) -> str:
    """Return a streaming SHA-256 digest without loading a potentially large image at once."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {k: _jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}
    return repr(value)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_array(h: "hashlib._Hash", name: str, value: Any, dtype) -> None:
    a = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    h.update(name.encode("utf-8") + b"\0")
    h.update(str(a.shape).encode("ascii") + b"\0")
    h.update(a.tobytes(order="C"))


def observation_digest(scn) -> Dict[str, Any]:
    """Hash the exact ordered 2D observations consumed by bundle adjustment."""
    obs = sorted(getattr(scn, "object_obs", []),
                 key=lambda o: (o.object_id, o.frame_id, o.cam_id))
    h = hashlib.sha256()
    n_corners = 0
    for i, o in enumerate(obs):
        h.update(f"obs:{i}:{o.object_id}:{o.frame_id}:{o.cam_id}".encode("ascii") + b"\0")
        _hash_array(h, "point_rows", o.point_rows, "<i8")
        _hash_array(h, "pts_2d", o.pts_2d, "<f8")
        n_corners += len(o.point_rows)
    for cam_id, size in sorted(getattr(scn, "img_size", {}).items()):
        h.update(f"image-size:{cam_id}:{int(size[0])}:{int(size[1])}".encode("ascii") + b"\0")
    return {"sha256": h.hexdigest(), "n_observations": len(obs),
            "n_corners": int(n_corners)}


def target_geometry_digest(objects: Any) -> Dict[str, Any]:
    """Hash target point geometry and board-to-object mappings before result serialization."""
    if objects is None:
        seq = []
    elif isinstance(objects, Mapping):
        seq = [objects[k] for k in sorted(objects)]
    elif isinstance(objects, (list, tuple)):
        seq = sorted(objects, key=lambda o: o.object_id)
    else:
        seq = [objects]
    h = hashlib.sha256()
    n_points = 0
    for obj in seq:
        h.update(f"object:{int(obj.object_id)}:{int(obj.ref_board_id)}".encode("ascii") + b"\0")
        _hash_array(h, "pts_3d", obj.pts_3d, "<f8")
        _hash_array(h, "pts_obj_2_board", obj.pts_obj_2_board, "<i8")
        for board_id, T in sorted(obj.T_co_b.items()):
            h.update(f"board:{int(board_id)}".encode("ascii") + b"\0")
            _hash_array(h, "T_co_b", T, "<f8")
        n_points += len(obj.pts_3d)
    return {"sha256": h.hexdigest(), "n_objects": len(seq), "n_points": int(n_points)}


def _git_state(start: Optional[os.PathLike | str] = None) -> Dict[str, Any]:
    cwd = os.fspath(start or Path.cwd())

    def run(*args: str) -> Optional[str]:
        try:
            p = subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True,
                               timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return p.stdout.strip() if p.returncode == 0 else None

    root = run("rev-parse", "--show-toplevel")
    if root is None:
        return {"available": False, "commit": None, "dirty": None}
    # Git pathspecs such as ``ls-files --others`` are relative to ``-C``.  Rebase at the
    # discovered root so an installed/editable module under ``ds_msp/rig`` still sees untracked
    # files anywhere in the worktree, not only below its own package directory.
    cwd = root
    commit = run("rev-parse", "HEAD")
    if commit is None:
        return {"available": False, "commit": None, "dirty": None}
    tracked_status = run("status", "--porcelain=v1", "--untracked-files=no")
    untracked_status = run("ls-files", "--others", "--exclude-standard")
    branch = run("branch", "--show-current")
    tracked_lines = tracked_status.splitlines() if tracked_status else []
    untracked_lines = untracked_status.splitlines() if untracked_status else []
    return {"available": True, "root": root, "commit": commit, "branch": branch or None,
            "dirty": bool(tracked_lines), "tracked_status_sha256": hashlib.sha256(
                (tracked_status or "").encode("utf-8")).hexdigest(),
            "n_changed_paths": len(tracked_lines), "n_untracked_paths": len(untracked_lines),
            "untracked_paths_sha256": hashlib.sha256(
                (untracked_status or "").encode("utf-8")).hexdigest()}


def load_reference_environment() -> Dict[str, Any]:
    ref = importlib.resources.files("ds_msp.rig") / "configs" / "reference_environment.json"
    with importlib.resources.as_file(ref) as path:
        return json.loads(path.read_text(encoding="utf-8"))


def _package_versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name in _DEPENDENCIES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def runtime_environment() -> Dict[str, Any]:
    """Capture the numerical runtime and whether it matches the shipped reference profile."""
    packages = _package_versions()
    reference = load_reference_environment()
    required = reference["packages"]
    package_match = all(packages.get(name) == version for name, version in required.items())
    forbidden_present = [name for name in reference.get("forbidden_packages", [])
                         if packages.get(name) is not None]
    python_match = platform.python_version().startswith(reference["python"] + ".")

    numpy_cfg = io.StringIO()
    with contextlib.redirect_stdout(numpy_cfg):
        np.show_config()
    opencv_build = cv2.getBuildInformation()
    return {
        "python": {"version": platform.python_version(), "implementation":
                   platform.python_implementation(), "executable": sys.executable},
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine(), "libc": list(platform.libc_ver()),
                     "cpu_count": os.cpu_count()},
        "packages": packages,
        "numerical_backend": {
            "numpy_config_sha256": hashlib.sha256(
                numpy_cfg.getvalue().encode("utf-8")).hexdigest(),
            "opencv_build_sha256": hashlib.sha256(opencv_build.encode("utf-8")).hexdigest(),
            "thread_environment": {name: os.environ.get(name) for name in _THREAD_ENV},
        },
        "reference": {"profile": reference["name"], "python_match": python_match,
                      "package_match": package_match,
                      "forbidden_packages_present": forbidden_present,
                      "matches": bool(python_match and package_match and not forbidden_present)},
    }


def image_files_under(root: Optional[os.PathLike | str], *,
                      exclude: Optional[os.PathLike | str] = None) -> list[str]:
    """List raw calibration images deterministically; returns an empty list for no root."""
    if not root or not os.path.isdir(root):
        return []
    excluded = Path(exclude).resolve() if exclude else None
    return [str(p.resolve()) for p in sorted(Path(root).rglob("*"))
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
            and not (excluded and (p.resolve() == excluded or excluded in p.resolve().parents))]


def input_file_records(paths: Iterable[Optional[os.PathLike | str]]) -> list[Dict[str, Any]]:
    """Hash unique regular input files; absent declared paths stay visible in the manifest."""
    normalized = sorted({os.path.abspath(os.fspath(p)) for p in paths if p})
    records = []
    for path in normalized:
        if os.path.isfile(path):
            records.append({"path": path, "exists": True, "size_bytes": os.path.getsize(path),
                            "sha256": sha256_file(path)})
        else:
            records.append({"path": path, "exists": False, "size_bytes": None, "sha256": None})
    return records


def build_manifest(scn, *, configuration: Optional[Mapping[str, Any]] = None,
                   source_paths: Sequence[os.PathLike | str] = (),
                   replay_argv: Sequence[str] = (), seed: Optional[int] = None,
                   input_objects: Any = None, repo_path: Optional[os.PathLike | str] = None
                   ) -> Dict[str, Any]:
    """Build the immutable part of a run manifest; output hashes are attached after writing."""
    created = datetime.now(timezone.utc).isoformat()
    observations = observation_digest(scn)
    target = target_geometry_digest(input_objects if input_objects is not None
                                    else getattr(scn, "object", None))
    config = _jsonable(configuration or {})
    files = input_file_records(source_paths)
    input_summary = {"observations": observations, "target_geometry": target, "files": files}
    identity = {"created_utc": created, "pid": os.getpid(), "inputs": input_summary,
                "configuration_sha256": _sha256_json(config)}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_utc": created,
        "invocation": {"argv": list(sys.argv), "cwd": os.getcwd(),
                       "replay_argv": list(replay_argv), "seed": seed},
        "software": {"git": _git_state(repo_path or Path(__file__).resolve().parent)},
        "environment": runtime_environment(),
        "configuration": config,
        "configuration_sha256": _sha256_json(config),
        "inputs": input_summary,
        "outputs": {},
    }


def summary(manifest: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Small non-circular subset embedded in ``calibration_report.json``."""
    if not manifest:
        return None
    git = manifest.get("software", {}).get("git", {})
    env = manifest.get("environment", {}).get("reference", {})
    obs = manifest.get("inputs", {}).get("observations", {})
    return {"schema_version": manifest.get("schema_version"),
            "run_id": manifest.get("run_id"), "git_commit": git.get("commit"),
            "git_dirty": git.get("dirty"), "reference_environment": env.get("profile"),
            "reference_environment_match": env.get("matches"),
            "observations_sha256": obs.get("sha256")}


def write_manifest(path: os.PathLike | str, manifest: Mapping[str, Any],
                   output_files: Mapping[str, Any]) -> str:
    """Finalize output hashes and atomically persist a strict-JSON manifest.

    The manifest never hashes itself. Reports can therefore point to the manifest while the
    manifest hashes those reports without creating a circular digest.
    """
    out = copy.deepcopy(_jsonable(manifest))
    records = {}
    manifest_abs = os.path.abspath(os.fspath(path))
    for key, raw in sorted(output_files.items()):
        if not isinstance(raw, (str, os.PathLike)):
            continue
        value = os.path.abspath(os.fspath(raw))
        if value == manifest_abs:
            continue
        if os.path.isfile(value):
            records[str(key)] = {"path": value, "type": "file",
                                 "size_bytes": os.path.getsize(value),
                                 "sha256": sha256_file(value)}
        elif os.path.isdir(value):
            records[str(key)] = {"path": value, "type": "directory", "sha256": None}
        else:
            records[str(key)] = {"path": os.fspath(raw), "type": "unresolved", "sha256": None}
    out["outputs"] = records
    os.makedirs(os.path.dirname(manifest_abs) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".calibration_manifest.", suffix=".tmp",
                               dir=os.path.dirname(manifest_abs) or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        os.replace(tmp, manifest_abs)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return manifest_abs
