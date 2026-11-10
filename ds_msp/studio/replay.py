"""Repeating a calibration from its fingerprint.

Every saved calibration carries ``calibration_manifest.json``: the configuration it ran with,
a SHA-256 of every input file, digests of the detections and of the target it used, the
software and its environment, and a SHA-256 of every file it wrote. :func:`fingerprint` reads
that record and :func:`check_inputs` re-hashes the inputs, so a page can say whether anything
changed since. :func:`replay_command` rebuilds the run's own command for a new folder, and
:func:`compare` says whether a repeat gave the same calibration.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..rig.config_schema import PROPERTIES

MANIFEST_FILE = "calibration_manifest.json"
#: Outputs that differ between two identical calibrations: they carry times, paths or the
#: manifest itself.
_DYNAMIC = {"calibration_manifest_json", "calibration_report_json", "calibration_report_txt",
            "report_json", "report_txt", "interactive_html", "mccalib_output_dir",
            "detection_images", "reprojection_images", "acceptance", "diagnostics_residuals",
            "diagnostics_lens_sigma"}
_MAX_LISTED = 40


def _manifest(folder: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads((Path(folder) / MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def fingerprint(folder: Path) -> Dict[str, Any]:
    """The record a result keeps of how it was made, summarised: ``{"ok": False, "reason"}``
    when the folder has no manifest or it was not made with ``--config``."""
    m = _manifest(folder)
    if m is None:
        return {"ok": False, "reason": "This result has no calibration_manifest.json, so there "
                                       "is no record to repeat it from."}
    cfg = m.get("configuration") or {}
    inv = m.get("invocation") or {}
    files = (m.get("inputs") or {}).get("files") or []
    soft, env = m.get("software") or {}, m.get("environment") or {}
    git = soft.get("git") or {}
    doc: Dict[str, Any] = {
        "ok": True, "run_id": m.get("run_id"), "created_utc": m.get("created_utc"),
        "configuration_sha256": m.get("configuration_sha256"),
        "entrypoint": cfg.get("entrypoint"), "config_path": cfg.get("config_path"),
        "overrides": list(cfg.get("overrides") or []),
        "n_inputs": len(files),
        "input_bytes": int(sum(int(f.get("size_bytes") or 0) for f in files)),
        "observations": (m.get("inputs") or {}).get("observations"),
        "target_geometry": (m.get("inputs") or {}).get("target_geometry"),
        "software": {"version": ((env.get("packages") or {}).get("ds-msp")
                                 or (env.get("packages") or {}).get("ds_msp")),
                     "commit": git.get("commit"), "dirty": git.get("dirty")},
        "python": (env.get("python") or {}).get("version") if isinstance(env.get("python"), dict)
        else env.get("python"),
        "packages": env.get("packages"),
        "can_repeat": cfg.get("entrypoint") == "config" and bool(cfg.get("config_path")),
        "replay_argv": list(inv.get("replay_argv") or []),
    }
    if not doc["can_repeat"]:
        doc["reason"] = ("This result was not made from a calib_param.yml, so Studio cannot "
                         "repeat it. Use tools/replay_rig_manifest.py from a source checkout.")
    return doc


def check_inputs(folder: Path) -> Dict[str, Any]:
    """Re-hash every input file the manifest lists. Returns counts and, for at most 40 files,
    which are ``missing`` or ``changed``. ``same`` is true only when every file still has its
    recorded SHA-256."""
    from ..rig.provenance import sha256_file

    m = _manifest(folder)
    if m is None:
        return {"ok": False, "reason": "no manifest"}
    missing: List[str] = []
    changed: List[str] = []
    appeared: List[str] = []
    checked = 0
    for rec in (m.get("inputs") or {}).get("files") or []:
        path, expected = rec.get("path"), rec.get("sha256")
        if not path:
            continue
        checked += 1
        p = Path(path)
        if rec.get("exists") is False:
            if p.exists():
                appeared.append(path)
        elif not p.is_file():
            missing.append(path)
        elif not expected or sha256_file(p) != expected:
            changed.append(path)
    return {"ok": True, "checked": checked, "n_missing": len(missing), "n_changed": len(changed),
            "n_appeared": len(appeared), "missing": missing[:_MAX_LISTED],
            "changed": changed[:_MAX_LISTED], "appeared": appeared[:_MAX_LISTED],
            "same": not (missing or changed or appeared)}


def replay_command(folder: Path, output: Path, inside: Sequence[Path]) -> List[str]:
    """Arguments for the rig CLI that repeat ``folder``'s calibration into ``output``: the
    recorded config file and overrides, with only the output folder replaced. Every piece is
    checked, not passed through: the config must lie under ``inside``, an override must set a
    ``calib_param.yml`` key, and the two report thresholds must be numbers. Raises
    ``ValueError`` with the reason otherwise."""
    fp = fingerprint(folder)
    if not fp.get("ok") or not fp.get("can_repeat"):
        raise ValueError(fp.get("reason") or "this result cannot be repeated")
    config = Path(str(fp["config_path"])).resolve()
    if not any(root == config or root in config.parents for root in inside):
        raise ValueError(f"its config file, {config}, is outside the folders Studio was started with")
    if not config.is_file():
        raise ValueError(f"its config file, {config}, no longer exists")
    args = ["--config", str(config)]
    recorded = fp["replay_argv"]
    i = 0
    while i < len(recorded):
        arg = recorded[i]
        nxt = recorded[i + 1] if i + 1 < len(recorded) else None
        if arg == "--config":
            i += 2
        elif arg == "--set" and nxt is not None:
            key = str(nxt).split("=", 1)[0].strip()
            if key not in PROPERTIES:
                raise ValueError(f"the recorded run sets {key!r}, which is not a config key")
            if key not in ("save_path", "webviewer", "verbose"):
                args.extend(["--set", str(nxt)])
            i += 2
        elif arg in ("--pass-px", "--warn-px") and nxt is not None:
            args.extend([arg, str(float(nxt))])
            i += 2
        elif arg == "--report":
            i += 2
        else:
            i += 1
    args.extend(["--set", f"save_path={output}/"])
    return args


def compare(original: Path, repeat: Path) -> Dict[str, Any]:
    """Whether a repeat reproduced the original: the digests of the detections and of the
    target each run used, and the SHA-256 of every calibration file both wrote (files that
    carry times or paths are left out). ``same`` is true when all of them agree."""
    a, b = _manifest(original), _manifest(repeat)
    if a is None or b is None:
        return {"ok": False, "reason": "one of the two results has no manifest"}
    rows: List[Dict[str, Any]] = []
    for key, label in (("observations", "corner detections used"), ("target_geometry", "target used")):
        x = ((a.get("inputs") or {}).get(key) or {}).get("sha256")
        y = ((b.get("inputs") or {}).get(key) or {}).get("sha256")
        rows.append({"what": label, "same": bool(x) and x == y})
    oa, ob = a.get("outputs") or {}, b.get("outputs") or {}
    for key in sorted((set(oa) & set(ob)) - _DYNAMIC):
        x, y = (oa[key] or {}).get("sha256"), (ob[key] or {}).get("sha256")
        if x and y:
            name = Path(str((oa[key] or {}).get("path") or key)).name
            rows.append({"what": name, "same": x == y})
    return {"ok": True, "rows": rows, "same": all(r["same"] for r in rows) and bool(rows)}
