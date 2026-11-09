"""Find rig calibration result folders and read one into a single JSON document.

A result folder is any folder holding ``calibration_report.json``, as written by
``ds-msp-calibrate-rig`` and :func:`ds_msp.rig.pipeline.calibrate_scenario`. Nothing here
writes to a result folder; the viewer is read-only.

:func:`load_run` returns ``{"schema": "ds-msp-studio-run", "schema_version": 1, ...}``. It
passes the report through unchanged and trims only bulk the viewer does not draw (per-frame
solver support and per-pair covariance matrices), so a 25-camera run is a few hundred kB.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

RUN_SCHEMA = "ds-msp-studio-run"
RUN_SCHEMA_VERSION = 1

REPORT_FILE = "calibration_report.json"
ACCEPTANCE_FILE = "calibration_acceptance.json"
MANIFEST_FILE = "calibration_manifest.json"

#: Files of a result folder the viewer may list and serve, in display order.
KNOWN_FILES = (
    REPORT_FILE, ACCEPTANCE_FILE, MANIFEST_FILE, "calibration_report.txt", "report.html",
    "calibrated_cameras_data.yml", "calibrated_objects_data.yml",
    "calibrated_objects_pose_data.yml", "detected_keypoints_data.yml",
    "reprojection_error_data.yml",
)

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "Detection", "Reprojection"}


class RunError(ValueError):
    """A folder is not a readable result folder."""


def _finite(o: Any) -> Any:
    """Replace NaN and infinities, which strict JSON cannot carry, by ``None``."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    return o


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise RunError(f"{path.name} is not valid JSON: {exc}") from exc
    return data if isinstance(data, dict) else None


def is_run_dir(folder: os.PathLike | str) -> bool:
    """True when ``folder`` holds a calibration report."""
    return (Path(folder) / REPORT_FILE).is_file()


def iter_run_dirs(root: os.PathLike | str, max_depth: int = 6) -> Iterator[Path]:
    """Result folders under ``root`` (``root`` included), at most ``max_depth`` levels down.
    Hidden folders and image output folders are skipped; symlinks are not followed."""
    root = Path(root)
    stack = [(root, 0)]
    while stack:
        folder, depth = stack.pop()
        if is_run_dir(folder):
            yield folder
        if depth >= max_depth:
            continue
        try:
            children = sorted((p for p in folder.iterdir()
                               if p.is_dir() and not p.is_symlink()
                               and not p.name.startswith(".") and p.name not in _SKIP_DIRS),
                              reverse=True)
        except OSError:
            continue
        stack.extend((c, depth + 1) for c in children)


def summarize_run(folder: os.PathLike | str) -> Dict[str, Any]:
    """The few fields a run list shows. Reads the report and, when present, the manifest."""
    folder = Path(folder)
    report = _read_json(folder / REPORT_FILE) or {}
    manifest = _read_json(folder / MANIFEST_FILE) or {}
    acceptance = _read_json(folder / ACCEPTANCE_FILE) or {}
    capture = report.get("capture") or {}
    fit = report.get("fit") or report.get("verdict") or {}
    overall = report.get("overall_errors_px") or {}
    solver = acceptance.get("solver") or {}
    lens = acceptance.get("lens") or report.get("lens") or {}
    observability = acceptance.get("observability") or report.get("audit") or {}
    return _finite({
        "name": folder.name,
        "run_id": manifest.get("run_id"),
        "created_utc": manifest.get("created_utc"),
        "n_cameras": capture.get("n_cameras"),
        "n_frames": capture.get("n_frames"),
        "models": sorted(set((report.get("models") or {}).values())),
        "fit_level": fit.get("level"),
        "median_px": overall.get("median"),
        "p95_px": overall.get("p95"),
        "solver_status": solver.get("final_status"),
        "n_weak": observability.get("n_weak"),
        "lens_failed": lens.get("failed"),
        "accuracy_status": acceptance.get("accuracy_status"),
    })


def _trim_acceptance(acc: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(acc)
    solver = out.get("solver")
    if isinstance(solver, dict) and "frame_support" in solver:
        solver = dict(solver)
        fs = solver.pop("frame_support")
        solver["frame_support_count"] = len(fs) if isinstance(fs, list) else None
        out["solver"] = solver
    unc = out.get("extrinsic_uncertainty")
    if isinstance(unc, dict) and isinstance(unc.get("pairs"), dict):
        unc = dict(unc)
        unc["pairs"] = {k: {kk: vv for kk, vv in v.items() if kk != "covariance"}
                        if isinstance(v, dict) else v for k, v in unc["pairs"].items()}
        out["extrinsic_uncertainty"] = unc
    return out


def _manifest_summary(man: Dict[str, Any]) -> Dict[str, Any]:
    keep = ("schema", "schema_version", "run_id", "created_utc", "software", "environment",
            "configuration_sha256")
    out = {k: man.get(k) for k in keep if k in man}
    conf = man.get("configuration")
    if isinstance(conf, dict):
        out["configuration"] = {k: conf.get(k) for k in ("entrypoint", "overrides", "resolved")
                                if k in conf}
    return out


def load_run(folder: os.PathLike | str) -> Dict[str, Any]:
    """One result folder as a single JSON-safe document for the viewer."""
    folder = Path(folder)
    report = _read_json(folder / REPORT_FILE)
    if report is None:
        raise RunError(f"no {REPORT_FILE} in this folder")
    acceptance = _read_json(folder / ACCEPTANCE_FILE)
    manifest = _read_json(folder / MANIFEST_FILE)
    files: List[Dict[str, Any]] = []
    for name in KNOWN_FILES:
        p = folder / name
        if p.is_file():
            files.append({"name": name, "bytes": p.stat().st_size})
    from .verdict import run_verdict
    diag = (acceptance or {}).get("diagnostics") or {}
    return _finite({
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        "name": folder.name,
        "summary": summarize_run(folder),
        "report": report,
        "acceptance": _trim_acceptance(acceptance) if acceptance else None,
        "manifest": _manifest_summary(manifest) if manifest else None,
        "verdict": run_verdict(report, acceptance),
        "has_diagnostics": bool(diag.get("files")),
        "files": files,
    })


#: At most this many residual arrows go to the browser per camera; every corner outside the
#: inlier band is always included.
MAX_ARROWS = 4000
N_WORST_FRAMES = 10


def _diag_file(folder: Path, acceptance: Dict[str, Any], key: str) -> Optional[Path]:
    rel = ((acceptance.get("diagnostics") or {}).get("files") or {}).get(key)
    if not rel:
        return None
    p = (folder / rel).resolve()
    if folder.resolve() not in p.parents or not p.is_file():
        return None
    return p


def load_diagnostics(folder: os.PathLike | str, cam: int) -> Dict[str, Any]:
    """Evidence for one camera, from the files FR-RIG-031 writes: residual arrows (subsampled
    to :data:`MAX_ARROWS`, outliers always kept), the worst frames by residual RMS, and the lens
    sigma grid. Missing files give ``None`` entries."""
    import numpy as np

    folder = Path(folder)
    acc = _read_json(folder / ACCEPTANCE_FILE) or {}
    report = _read_json(folder / REPORT_FILE) or {}
    intr = next((i for i in report.get("intrinsics") or [] if i.get("cam_id") == cam), {})
    out: Dict[str, Any] = {"cam": cam, "width": intr.get("width"), "height": intr.get("height"),
                           "band_px": acc.get("coverage_band_px"), "residuals": None,
                           "worst_frames": None, "lens": None,
                           "outliers": ((acc.get("diagnostics") or {}).get("outliers") or {}).get(str(cam))}
    rp = _diag_file(folder, acc, "residuals")
    if rp is not None:
        with np.load(rp) as z:
            m = z["cam"] == cam
            u, v, dx, dy = z["u"][m], z["v"][m], z["dx"][m], z["dy"][m]
            valid, within, frame = z["valid"][m], z["within_band"][m], z["frame"][m]
        keep = np.flatnonzero(valid)
        out_idx = keep[~within[keep]]
        in_idx = keep[within[keep]]
        room = max(MAX_ARROWS - len(out_idx), 0)
        if len(in_idx) > room:
            in_idx = in_idx[np.linspace(0, len(in_idx) - 1, room).astype(int)] if room else in_idx[:0]
        sel = np.sort(np.r_[out_idx, in_idx])
        r = np.hypot(dx, dy)
        out["residuals"] = {"n_total": int(m.sum()), "n_valid": int(valid.sum()),
                            "n_shown": int(len(sel)),
                            "u": np.round(u[sel], 2).tolist(), "v": np.round(v[sel], 2).tolist(),
                            "dx": np.round(dx[sel], 4).tolist(), "dy": np.round(dy[sel], 4).tolist(),
                            "within": within[sel].tolist(),
                            "rms_px": float(np.sqrt(np.mean(r[valid] ** 2))) if valid.any() else None}
        frames = []
        for f in np.unique(frame[valid]):
            fm = valid & (frame == f)
            rf = r[fm]
            frames.append({"frame": int(f), "n": int(fm.sum()), "rms_px": float(np.sqrt(np.mean(rf ** 2))),
                           "max_px": float(rf.max()), "n_outside": int((fm & ~within).sum())})
        frames.sort(key=lambda x: -x["rms_px"])
        out["worst_frames"] = frames[:N_WORST_FRAMES]
        out["n_frames"] = len(frames)
        if out["width"] and out["height"]:
            from .evidence import error_pattern
            out["pattern"] = error_pattern(u[valid], v[valid], dx[valid], dy[valid],
                                           out["width"], out["height"])
    gp = _diag_file(folder, acc, "lens_sigma")
    if gp is not None:
        with np.load(gp) as z:
            if f"cam{cam}_sigma_px" in z.files:
                s = z[f"cam{cam}_sigma_px"]
                out["lens"] = {"nx": 24, "ny": 18, "u": z[f"cam{cam}_u"].tolist(),
                               "v": z[f"cam{cam}_v"].tolist(),
                               "sigma_px": [None if not np.isfinite(x) else float(x) for x in s],
                               "unbounded": bool(np.isinf(s).all()),
                               "observed": z[f"cam{cam}_observed"].tolist(),
                               "limit_px": (acc.get("lens") or {}).get("limit_px")}
                if out["width"] and out["height"]:
                    from .evidence import lens_zones
                    out["lens"]["zones"] = lens_zones(out["lens"], out["width"], out["height"])
    return _finite(out)
