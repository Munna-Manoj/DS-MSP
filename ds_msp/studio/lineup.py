"""Which photo folder is which camera, and lens values a person types in.

The calibration reads camera folders named ``<prefix>001``, ``002``, ... A photo folder laid out
any other way is not renamed: the person confirms which subfolder is which camera, and Studio
makes a folder of links with the expected names. Lens values typed into the setup page become a
lens file in the format the calibration reads. Both live in a hidden folder under a served
folder, named after their content, so the same choice always gives the same path and an
earlier result made from it is recognised.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence

#: Hidden folder, under a served folder, that holds linked camera folders and typed lens files.
WORK_DIR = ".camera_lineups"
#: What each linked folder stands for, next to the links and copied into the result.
LINEUP_FILE = "camera_lineup.json"
#: Name of a lens file made from typed values.
LENS_FILE = "entered_lens_values.yml"
PREFIX = "Cam_"


def natural_key(name: str) -> List[Any]:
    """Sorts ``cam2`` before ``cam10``."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _digest(doc: Any) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def stage(root: Path, folders: Sequence[str], under: Path) -> Path:
    """Link the subfolders ``folders`` of ``root``, in that order, as ``Cam_001``, ``Cam_002``,
    ... in a folder under ``under``, and return that folder. ``root`` is not changed."""
    from .photos import _images

    root = Path(root).resolve()
    if not folders:
        raise ValueError("choose at least one folder as a camera")
    if len(set(folders)) != len(folders):
        raise ValueError("a folder is used for two cameras")
    for name in folders:
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"{name!r} is not a folder directly inside this one")
        if not _images(root / name):
            raise ValueError(f"{name} holds no photos")
    dest = Path(under) / WORK_DIR / _digest([str(root), list(folders)])
    dest.mkdir(parents=True, exist_ok=True)
    cameras = []
    for k, name in enumerate(folders):
        camera = f"{PREFIX}{k + 1:03d}"
        link = dest / camera
        if link.is_symlink():
            link.unlink()
        try:
            os.symlink(root / name, link, target_is_directory=True)
        except OSError as exc:
            raise ValueError(f"could not link {name} as {camera}: {exc}") from None
        cameras.append({"camera": camera, "folder": name})
    (dest / LINEUP_FILE).write_text(
        json.dumps({"root": str(root), "cameras": cameras}, indent=1), encoding="utf-8")
    return dest


def origin(folder: Path) -> Optional[Dict[str, Any]]:
    """The folder and subfolders a folder of links stands for, or ``None`` for an ordinary one."""
    try:
        doc = json.loads((Path(folder) / LINEUP_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and "root" in doc and "cameras" in doc else None


def in_work_dir(path: Path) -> bool:
    """Whether ``path`` was made by this module."""
    return WORK_DIR in Path(path).parts


def lens_fields(model: str) -> List[str]:
    """The values a lens of type ``model`` needs, in the order the page asks for them."""
    from ..io.mccalib import _DISTORTION_LAYOUT

    if model not in _DISTORTION_LAYOUT:
        raise ValueError(f"unknown lens type {model!r}")
    return ["fx", "fy", "cx", "cy", *_DISTORTION_LAYOUT[model]]


def _number(camera: int, key: str, value: Any) -> float:
    try:
        got = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"camera {camera}: {key} needs a number") from None
    if not math.isfinite(got):
        raise ValueError(f"camera {camera}: {key} needs a finite number")
    return got


def write_lens_values(cameras: Sequence[Mapping[str, Any]], under: Path) -> Path:
    """Write typed lens values as a lens file under ``under`` and return its path.

    Each entry has ``model``, ``width``, ``height`` and the values of :func:`lens_fields`, one
    entry per camera in camera order. Whether the values fit the photos is a separate check
    (:func:`ds_msp.studio.advisor.lens_file_health`); here they only have to be complete and
    possible."""
    from ..io.mccalib import save_mccalib_cameras
    from ..models.registry import model_class

    if not cameras:
        raise ValueError("no camera has lens values")
    models: Dict[int, Any] = {}
    sizes: Dict[int, Any] = {}
    for k, cam in enumerate(cameras):
        model = str(cam.get("model") or "")
        params = {key: _number(k, key, cam.get(key)) for key in lens_fields(model)}
        if params["fx"] <= 0 or params["fy"] <= 0:
            raise ValueError(f"camera {k}: fx and fy must be above 0")
        width, height = _number(k, "width", cam.get("width")), _number(k, "height", cam.get("height"))
        if width < 1 or height < 1 or width != int(width) or height != int(height):
            raise ValueError(f"camera {k}: the image size must be whole pixels")
        try:
            models[k] = model_class(model)(**params)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"camera {k}: {exc}") from None
        sizes[k] = (int(width), int(height))
    import numpy as np

    content = [{key: cam.get(key) for key in ("model", "width", "height", *lens_fields(str(cam["model"])))}
               for cam in cameras]
    dest = Path(under) / WORK_DIR / "lens" / _digest(content)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / LENS_FILE
    rig = SimpleNamespace(cameras=models, img_size=sizes, T_c_g={k: np.eye(4) for k in models})
    save_mccalib_cameras(rig, str(path))
    return path


def run_problems(values: Mapping[str, Any]) -> List[Dict[str, str]]:
    """What stops these settings from calibrating, as ``{"key", "message"}``: the checks that
    otherwise only fail once the run has started, or never fail and give a wrong result."""
    from .photos import scan_photos
    from .sources import scan_intrinsics

    out: List[Dict[str, str]] = []
    if values.get("square_size") is None:
        out.append({"key": "square_size", "message": "needs the size of one square"})
    strict = values.get("profile") == "production"
    fixed = values.get("fix_intrinsic") is True
    lens = values.get("cam_params_path")
    if strict and not fixed:
        out.append({"key": "fix_intrinsic", "message": "strict checking keeps the lens values as "
                    "given: it does not work them out from these photos"})
    if (strict or fixed) and not lens:
        why = "strict checking needs" if strict else "keeping the lens values needs"
        out.append({"key": "cam_params_path",
                    "message": f"{why} lens values for every camera: choose a file or type them in"})
    n_cam = values.get("number_camera")
    if values.get("root_path"):
        doc = scan_photos(str(values["root_path"]), values.get("cam_prefix") or None,
                          values.get("frame_pattern") or None)
        if doc["problems"]:
            out.append({"key": "root_path", "message": " ".join(doc["problems"])})
        elif n_cam and doc["number_camera"] != n_cam:
            out.append({"key": "number_camera", "message": f"is {n_cam}, but the photo folder "
                        f"holds {doc['number_camera']} cameras"})
    if fixed and lens and n_cam and Path(str(lens)).is_file():
        try:
            listed = {c["index"]: c for c in scan_intrinsics(Path(str(lens)))["cameras"]}
        except ValueError as exc:
            return [*out, {"key": "cam_params_path", "message": str(exc)}]
        absent = [k for k in range(int(n_cam)) if k not in listed]
        if absent:
            out.append({"key": "cam_params_path", "message": "keeping the lens values needs every "
                        f"camera in the file; missing: {', '.join(map(str, absent))}"})
        chosen = values.get("camera_models") or []
        for k, cam in listed.items():
            want = chosen[0] if len(chosen) == 1 else chosen[k] if k < len(chosen) else None
            if k < int(n_cam) and want and cam["model"] and want != cam["model"]:
                out.append({"key": "camera_models", "message": f"camera {k} is set to {want}, but "
                            f"its lens values are for {cam['model']}"})
    return out
