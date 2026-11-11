"""Reading, checking and writing ``calib_param.yml`` for the Studio forms.

The forms hold *values*: a dict of the keys the user set, named as in ``calib_param.yml`` and
described by :func:`ds_msp.rig.config_schema.config_schema`. A key that is not in the dict is
left out of the file, so the loader applies its default and a ``profile`` can still raise it;
writing every key would mark each as set by the user and switch the profile off for it.

:func:`check_values` checks a dict against the schema, :func:`write_config` writes the file the
CLI reads, and :func:`read_values` reads a file back into a dict. A file written here and read
by :func:`ds_msp.rig.calib_param.load_config` gives the same configuration as the same keys
passed on the command line.
"""
from __future__ import annotations

import math
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..rig.config_schema import PROPERTIES

CONFIG_NAME = "calib_param.yml"
__all__ = ["CONFIG_NAME", "PATH_KEYS", "PROPERTIES", "check_values", "cli_command", "load_check",
           "path_problems", "read_values", "render_config", "write_config"]

#: Keys whose value is a file or folder on the computer that runs Studio.
PATH_KEYS = ("root_path", "keypoints_path", "cam_params_path", "object_path",
             "accuracy_profile_path", "save_path")

_HEADER = ("%YAML:1.0\n"
           "# DS-MSP rig calibration config, written by DS-MSP Studio.\n"
           "# Only the keys that were set are listed; every other key keeps its default.\n"
           "# Run it from a terminal with: ds-msp-calibrate-rig --config <this file>\n")


def _types(prop: Dict[str, Any]) -> Tuple[str, ...]:
    t = prop["type"]
    return tuple(t) if isinstance(t, list) else (t,)


def _coerce(key: str, prop: Dict[str, Any], value: Any) -> Any:
    """``value`` as the type the schema gives ``key``; raises ``ValueError`` with the reason."""
    types = _types(prop)
    if value is None or (isinstance(value, str) and value.strip() == "" and "null" in types):
        if "null" in types:
            return None
        raise ValueError("needs a value")
    kind = next(t for t in types if t != "null")
    if kind == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in ("true", "false", "0", "1"):
            return value.strip().lower() in ("true", "1")
        raise ValueError("must be true or false")
    if kind in ("integer", "number"):
        if isinstance(value, bool):
            raise ValueError("must be a number")
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise ValueError("must be a number") from None
        if not math.isfinite(num):
            raise ValueError("must be a finite number")
        if kind == "integer":
            if num != int(num):
                raise ValueError("must be a whole number")
            num = int(num)
        if "minimum" in prop and num < prop["minimum"]:
            raise ValueError(f"must be at least {prop['minimum']}")
        return num
    if kind == "string":
        if not isinstance(value, str):
            raise ValueError("must be text")
        text = value.strip()
        if "enum" in prop and text not in prop["enum"]:
            raise ValueError("must be one of " + ", ".join(map(str, prop["enum"])))
        if "\n" in text or "\r" in text:
            raise ValueError("must be one line")
        return text
    if kind == "array":
        if isinstance(value, str):
            value = [s for s in (p.strip() for p in value.replace(";", ",").split(",")) if s]
        if not isinstance(value, (list, tuple)):
            raise ValueError("must be a list")
        item = dict(prop.get("items") or {"type": "string"})
        out = []
        for k, v in enumerate(value):
            try:
                out.append(_coerce(key, item, v))
            except ValueError as exc:
                raise ValueError(f"entry {k + 1} {exc}") from None
        return out
    raise ValueError(f"unsupported type {kind}")


def check_values(values: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Check ``values`` against the schema. Returns the cleaned dict (typed, with empty
    optional entries and entries equal to nothing removed) and a list of
    ``{"key", "message"}`` problems. A key that is not a ``calib_param.yml`` key is a problem,
    never passed through."""
    clean: Dict[str, Any] = {}
    errors: List[Dict[str, str]] = []
    for key, value in values.items():
        prop = PROPERTIES.get(key)
        if prop is None:
            errors.append({"key": str(key), "message": "is not a calib_param.yml key"})
            continue
        try:
            got = _coerce(key, prop, value)
        except ValueError as exc:
            errors.append({"key": key, "message": str(exc)})
            continue
        if got is not None:
            clean[key] = got
    if "number_camera" not in clean and not any(e["key"] == "number_camera" for e in errors):
        errors.append({"key": "number_camera", "message": "needs a value"})
    models = clean.get("camera_models")
    n_cam = clean.get("number_camera")
    if models and n_cam and len(models) not in (1, n_cam):
        errors.append({"key": "camera_models",
                       "message": f"lists {len(models)} lens models for {n_cam} cameras"})
    return clean, errors


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return '"true"' if value else '"false"'       # quoted: see the config template's note
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def render_config(values: Dict[str, Any]) -> str:
    """The text of a ``calib_param.yml`` that sets exactly ``values`` (already checked), keys in
    the schema's order."""
    lines = [_HEADER]
    group = None
    for key, prop in PROPERTIES.items():
        if key not in values or values[key] is None:
            continue
        if prop["x-group"] != group:
            group = prop["x-group"]
            lines.append(f"\n# {group}\n")
        v = values[key]
        text = ("[ " + ", ".join(_scalar(x) for x in v) + " ]" if v else "[]") \
            if isinstance(v, (list, tuple)) else _scalar(v)
        lines.append(f"{key}: {text}\n")
    return "".join(lines)


def write_config(path: Path, values: Dict[str, Any]) -> None:
    Path(path).write_text(render_config(values), encoding="utf-8")


def _node_value(node) -> Any:
    if node.isSeq():
        return [_node_value(node.at(i)) for i in range(node.size())]
    if node.isString():
        return node.string()
    if node.isInt():
        return int(node.real())
    if node.isReal():
        return float(node.real())
    return None


def read_values(text: str) -> Dict[str, Any]:
    """The schema keys set in a ``calib_param.yml`` given as text, typed as the schema says.

    Returns ``{"values", "unknown", "problems"}``: ``unknown`` lists keys in the file that are
    not ``calib_param.yml`` keys (MC-Calib keys DS-MSP ignores among them), ``problems`` the
    keys whose value could not be read as the schema's type. ``"None"`` paths, the file's way
    of saying "not set", are left out."""
    import cv2

    if not text.lstrip().startswith("%YAML"):
        text = "%YAML:1.0\n" + text
    try:
        fs = cv2.FileStorage(text, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    except (cv2.error, SystemError):                  # a parse error surfaces as either
        raise ValueError("not a readable calib_param.yml: check that every line is "
                         "'key: value'") from None
    if not fs.isOpened():
        raise ValueError("not a readable calib_param.yml")
    root = fs.root()
    values: Dict[str, Any] = {}
    unknown: List[str] = []
    problems: List[Dict[str, str]] = []
    for key in root.keys():
        prop = PROPERTIES.get(key)
        if prop is None:
            unknown.append(key)
            continue
        raw = _node_value(root.getNode(key))
        if isinstance(raw, str):
            raw = raw.split("#", 1)[0].strip() if "boolean" in _types(prop) else raw
            if raw.strip().lower() in ("none", "") and "null" in _types(prop):
                continue
        try:
            got = _coerce(key, prop, raw)
        except ValueError as exc:
            problems.append({"key": key, "message": str(exc)})
            continue
        if got is not None:
            values[key] = got
    fs.release()
    return {"values": values, "unknown": sorted(unknown), "problems": problems}


def cli_command(config_path: str) -> str:
    """The terminal command that runs the same calibration."""
    return "ds-msp-calibrate-rig --config " + shlex.quote(str(config_path))


def load_check(config_path: Path) -> Dict[str, Any]:
    """Parse a written config with the loader the CLI uses. Returns ``{"ok", "error",
    "warnings"}``: ``error`` is the loader's own message when it refuses the file."""
    import warnings

    from ..rig.calib_param import load_config

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            cfg = load_config(str(config_path))
        except (ValueError, OSError, KeyError) as exc:
            return {"ok": False, "error": str(exc), "warnings": []}
    notes = [str(w.message) for w in caught]
    notes.extend(str(w) for w in getattr(cfg, "config_warnings", []) or [])
    return {"ok": True, "error": None, "warnings": list(dict.fromkeys(notes))}


def path_problems(values: Dict[str, Any], inside: Optional[Sequence[Path]],
                  skip: Sequence[str] = ()) -> List[Dict[str, str]]:
    """Path keys that point outside every folder in ``inside`` or at something that does not
    exist. ``inside`` is ``None`` for a Studio that may read anywhere on this computer."""
    out: List[Dict[str, str]] = []
    for key in PATH_KEYS:
        if key in skip or not values.get(key):
            continue
        target = Path(str(values[key])).expanduser()
        if not target.is_absolute():
            out.append({"key": key, "message": "must be a full path"})
            continue
        target = target.resolve()
        if inside is not None and not any(target == root or root in target.parents for root in inside):
            out.append({"key": key, "message": "is outside the folders Studio was started with"})
        elif not target.exists():
            out.append({"key": key, "message": "does not exist"})
    return out
