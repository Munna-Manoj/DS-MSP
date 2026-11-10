"""A photo folder as a capture with a history.

Studio saves every calibration in its own folder, each with a manifest that names the photos it
used. So for a photo folder it can list the calibrations already made from it, and say whether
the corner detections of one of them can be used again. Detection is the slow step and depends
only on the photos, the board layout and the detector; the lens types, the square size and
every solver setting can change without invalidating it.

Reuse never touches an earlier result: its detections file is copied into the new result's
folder.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .runs import iter_run_dirs

MANIFEST_FILE = "calibration_manifest.json"
REPORT_FILE = "calibration_report.json"
KEYPOINTS_FILE = "detected_keypoints_data.yml"
REUSED_DIR = "reused_detections"
_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _same_folder(a: Optional[str], b: Path) -> bool:
    try:
        return a is not None and Path(a).expanduser().resolve() == b
    except OSError:
        return False


def detection_key(resolved: Dict[str, Any]) -> Tuple:
    """What a detection depends on in a resolved configuration: the board layout (squares,
    marker geometry, marker-id offsets), the camera folders and the frame-number rule. The
    metric square size is not part of it: it scales the target, not the corners found."""
    boards = tuple((b.get("n_x"), b.get("n_y"), b.get("length_square"), b.get("length_marker"),
                    b.get("id_offset")) for b in resolved.get("boards") or [])
    return (boards, resolved.get("number_camera"), resolved.get("cam_prefix") or "Cam_",
            resolved.get("frame_pattern") or None)


def key_from_values(values: Dict[str, Any]) -> Tuple:
    """:func:`detection_key` of form values, through the loader's own board rules."""
    from .photos import board_specs

    boards = tuple((s.n_x, s.n_y, s.length_square, s.length_marker, s.id_offset)
                   for s in board_specs(values))
    return (boards, values.get("number_camera"), values.get("cam_prefix") or "Cam_",
            values.get("frame_pattern") or None)


def _photos_unchanged(root: Path, manifest: Dict[str, Any]) -> Tuple[bool, str]:
    """Quick check that the photos are the ones a run used: the same files, each the same size
    and not modified after the run. (A run's exact checksums are checked by "Repeat exactly".)"""
    made = manifest.get("created_utc")
    try:
        made_ts = datetime.fromisoformat(str(made)).timestamp()
    except (TypeError, ValueError):
        return False, "the earlier run has no date"
    recorded = {}
    for rec in (manifest.get("inputs") or {}).get("files") or []:
        path = str(rec.get("path") or "")
        if path.lower().endswith(_IMAGE_EXT) and _same_folder(os.path.dirname(os.path.dirname(path)), root):
            recorded[path] = rec.get("size_bytes")
    if not recorded:
        return False, "the earlier run did not record its photos"
    now = {}
    for folder in root.iterdir():
        if folder.is_dir():
            for f in folder.iterdir():
                if f.is_file() and f.name.lower().endswith(_IMAGE_EXT):
                    now[str(f)] = f.stat()
    if set(now) != set(recorded):
        added, gone = len(set(now) - set(recorded)), len(set(recorded) - set(now))
        return False, f"the photos changed since ({added} added, {gone} removed)"
    for path, st in now.items():
        if recorded[path] is not None and st.st_size != recorded[path]:
            return False, f"{os.path.basename(path)} changed size since"
        if st.st_mtime > made_ts + 1.0:
            return False, f"{os.path.basename(path)} was modified after that run"
    return True, f"{len(now)} photos unchanged"


def history(root_path: str, roots: Sequence[Path], values: Optional[Dict[str, Any]] = None,
            run_id=None, max_depth: int = 6) -> Dict[str, Any]:
    """The calibrations already made from the photos in ``root_path``, newest first, and the
    newest one whose detections ``values`` could reuse.

    Each run: ``id``, ``name``, ``created_utc``, cameras, lens ``models``, ``median_px``, and
    ``reuse`` (``{"ok", "why"}``): whether its detections fit ``values``. ``reusable`` is the
    id of the newest run with ``reuse.ok``."""
    import cv2

    root = Path(root_path).expanduser().resolve()
    want = key_from_values(values) if values else None
    runs: List[Dict[str, Any]] = []
    for k, base in enumerate(roots):
        for folder in iter_run_dirs(base, max_depth=max_depth):
            m = _json(folder / MANIFEST_FILE)
            resolved = (m.get("configuration") or {}).get("resolved") or {}
            if not _same_folder(resolved.get("root_path"), root):
                continue
            report = _json(folder / REPORT_FILE)
            rel = folder.relative_to(base).as_posix()
            row: Dict[str, Any] = {
                "id": f"{k}:{'' if rel == '.' else rel}" if run_id is None else run_id(k, folder),
                "name": folder.name, "created_utc": m.get("created_utc"),
                "n_cameras": (report.get("capture") or {}).get("n_cameras"),
                "models": sorted(set((report.get("models") or {}).values())),
                "median_px": (report.get("overall_errors_px") or {}).get("median"),
                "has_config": (folder / "calib_param.yml").is_file(),
            }
            if want is not None:
                row["reuse"] = _reuse(folder, m, resolved, root, want, cv2.__version__)
            runs.append(row)
    runs.sort(key=lambda r: r.get("created_utc") or "", reverse=True)
    reusable = next((r["id"] for r in runs if (r.get("reuse") or {}).get("ok")), None)
    return {"root_path": str(root), "runs": runs, "reusable": reusable}


def _reuse(folder: Path, manifest: Dict[str, Any], resolved: Dict[str, Any], root: Path,
           want: Tuple, opencv: str) -> Dict[str, Any]:
    if not (folder / KEYPOINTS_FILE).is_file():
        return {"ok": False, "why": "it saved no detections"}
    if resolved.get("keypoints_path"):
        return {"ok": False, "why": "it used saved detections itself"}
    if detection_key(resolved) != want:
        return {"ok": False, "why": "the board layout or camera folders differ"}
    packages = (manifest.get("environment") or {}).get("packages") or {}
    was = packages.get("opencv-python") or packages.get("opencv-python-headless")
    if was and str(was).split(".")[:2] != str(opencv).split(".")[:2]:
        return {"ok": False, "why": f"it was detected with OpenCV {was}, this is {opencv}"}
    ok, why = _photos_unchanged(root, manifest)
    return {"ok": ok, "why": why}


def copy_detections(source: Path, target: Path) -> Path:
    """Copy a result's detections file into ``target`` (a new result folder), in a subfolder
    of its own, so that nothing else from the earlier result comes along. Returns the copy."""
    dest = Path(target) / REUSED_DIR
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(source) / KEYPOINTS_FILE, dest / KEYPOINTS_FILE)
    return dest / KEYPOINTS_FILE
