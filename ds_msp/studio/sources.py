"""The files a calibration can start from other than a folder of photos: a ``calib_param.yml``
on the computer that runs Studio, a file of saved corner detections, and a file of lens values
already known. Each is read the way the calibration reads it and summarised for the form.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

from . import configio

OBJECT_FILE = "calibrated_objects_data.yml"


def config_file(path: Path) -> Dict[str, Any]:
    """The settings in a ``calib_param.yml`` on disk, with its relative paths made absolute
    against the file's own folder, as the CLI resolves them."""
    doc = configio.read_values(Path(path).read_text(encoding="utf-8", errors="replace"))
    base = Path(path).resolve().parent
    for key in configio.PATH_KEYS:
        value = doc["values"].get(key)
        if isinstance(value, str) and value and not os.path.isabs(os.path.expanduser(value)):
            doc["values"][key] = str((base / value).resolve())
    doc["path"] = str(Path(path).resolve())
    return doc


def settings_of(folder: Path, inside: Sequence[Path]) -> Dict[str, Any]:
    """The settings a result folder was calibrated with, to start a new run from them.

    A run started in Studio keeps its ``calib_param.yml`` in the folder. A run started from a
    terminal does not, so the config its manifest records is read instead, with the ``--set``
    values of that run applied; ``changed`` is true when that file no longer has the recorded
    SHA-256. ``save_path`` and ``webviewer`` are left out: the new run gets its own.

    Raises ``FileNotFoundError`` when there is no settings file under ``inside``."""
    from ..rig.provenance import sha256_file
    from .replay import fingerprint, manifest_file_sha

    folder = Path(folder)
    path = folder / configio.CONFIG_NAME
    overrides: List[str] = []
    changed = False
    if not path.is_file():
        fp = fingerprint(folder)
        recorded = Path(str(fp["config_path"])).resolve() if fp.get("config_path") else None
        if recorded is None or not recorded.is_file():
            raise FileNotFoundError("the settings file this result was made from is no longer there")
        if not any(r == recorded or r in recorded.parents for r in inside):
            raise FileNotFoundError("the settings file of this result is outside the folders "
                                    "Studio was started with")
        path, overrides = recorded, [str(o) for o in fp.get("overrides") or []]
        want = manifest_file_sha(folder, recorded)
        changed = want is not None and sha256_file(recorded) != want
    doc = config_file(path)
    for item in overrides:
        key, _sep, value = item.partition("=")
        key = key.strip()
        if key in configio.PROPERTIES and key != "number_camera":
            checked, _problems = configio.check_values({"number_camera": 1, key: value})
            if key in checked:
                doc["values"][key] = checked[key]
    doc["values"].pop("save_path", None)
    doc["values"].pop("webviewer", None)
    doc["changed"] = changed
    return doc


#: What a YAML file is, from the entries it holds: (pattern, what to call it).
_KINDS = (
    (r"^\s*(pts_2d|frame_idxs)\s*:", "a file of found corners (like detected_keypoints_data.yml)"),
    (r"^\s*(number_camera|number_x_square|root_path)\s*:", "a settings file (like calib_param.yml)"),
    (r"^\s*(object_\d+|nb_object)\s*:", "a target file (like calibrated_objects_data.yml)"),
    (r"^\s*camera_\d+\s*:", "a lens file (like calibrated_cameras_data.yml)"),
    (r"^\s*cam\d+\s*:", "a Kalibr camera file"),
    (r"^\s*(camera_matrix|distortion_coefficients)\s*:", "an OpenCV calibration of a single camera"),
)


def what_is(path: Path) -> str:
    """What kind of file ``path`` looks like, in words, or ``""`` when it cannot be told. Used
    to answer a wrong choice with what was chosen instead of only that it does not fit."""
    import re

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read(200_000)
    except OSError:
        return ""
    for pattern, name in _KINDS:
        if re.search(pattern, text, re.MULTILINE):
            return name
    return ""


def _wrong_kind(path: Path, wanted: str, hint: str) -> ValueError:
    """The error for a file chosen where ``wanted`` was expected."""
    name = Path(path).name
    kind = what_is(path).replace(f" (like {name})", "")       # no "like X" for a file named X
    if kind and not kind.startswith(wanted.split(" (")[0]):
        return ValueError(f"{name} is {kind}, not {wanted}. {hint}")
    return ValueError(f"{name} is not {wanted}. {hint}")


_LENS_HINT = ("A lens file lists every camera with its camera_matrix. Each DS-MSP or MC-Calib "
              "result folder holds one, named calibrated_cameras_data.yml; other formats are "
              "not read here.")
_CORNERS_HINT = ("A corners file lists, per camera, the frames and the pixels of the corners "
                 "found. Each result folder holds one, named detected_keypoints_data.yml.")


def _count(node) -> int:
    if node.empty():
        return 0
    if node.isSeq():
        return int(node.size())
    m = node.mat()
    return 0 if m is None else int(m.size)


def _ints(node) -> List[int]:
    if node.empty():
        return []
    if node.isSeq():
        return [int(node.at(i).real()) for i in range(node.size())]
    m = node.mat()
    return [] if m is None else [int(v) for v in m.ravel()]


def scan_keypoints(path: Path) -> Dict[str, Any]:
    """What a saved-detections file (``detected_keypoints_data.yml``) holds: per camera the
    image size, the photos with a board in them and the board sightings. ``reused_target`` names
    a ``calibrated_objects_data.yml`` beside it, which a multi-board calibration loads instead
    of rebuilding the target from the detections."""
    import cv2

    path = Path(path)
    wrong = _wrong_kind(path, "a file of found corners", _CORNERS_HINT)
    try:
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    except (cv2.error, SystemError):
        raise wrong from None
    if not fs.isOpened():
        raise wrong
    try:
        nb_node = fs.getNode("nb_camera")
        if nb_node.empty():
            raise wrong
        cameras = []
        for ci in range(int(nb_node.real())):
            cn = fs.getNode(f"camera_{ci}")
            if cn.empty():
                continue
            if cn.getNode("frame_idxs").empty() and cn.getNode("pts_2d").empty():
                raise wrong
            frames = _ints(cn.getNode("frame_idxs"))
            boards = _ints(cn.getNode("board_idxs"))
            cameras.append({"index": ci, "width": int(cn.getNode("img_width").real()),
                            "height": int(cn.getNode("img_height").real()),
                            "frames": len(set(frames)), "sightings": len(frames),
                            "boards": sorted(set(boards))})
        label = fs.getNode("pixel_convention")
        convention = label.string() if not label.empty() and label.isString() else None
    finally:
        fs.release()
    problems: List[str] = []
    if not cameras:
        problems.append("The file lists no camera with detections.")
    if convention is None:
        problems.append("The file does not say which pixel convention its corners use. Files "
                        "written by OpenCV 4 ChArUco detection need keypoints_pixel_convention "
                        "set to opencv4_charuco under More settings.")
    beside = path.parent / OBJECT_FILE
    n_boards = len({b for c in cameras for b in c["boards"]})
    return {"path": str(path.resolve()), "number_camera": len(cameras), "cameras": cameras,
            "number_board": n_boards, "pixel_convention": convention,
            "reused_target": str(beside) if beside.is_file() else None, "problems": problems}


def scan_intrinsics(path: Path) -> Dict[str, Any]:
    """The lenses in an intrinsics file (MC-Calib camera format): per camera its model, focal
    lengths and centre. A camera without a stated model is listed with ``model: None``."""
    import cv2

    from ..io.mccalib import _camera_model_field

    try:
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    except (cv2.error, SystemError):
        raise _wrong_kind(path, "a lens file", _LENS_HINT) from None
    if not fs.isOpened():
        raise _wrong_kind(path, "a lens file", _LENS_HINT)
    try:
        nb_node = fs.getNode("nb_camera")
        if nb_node.empty():
            raise _wrong_kind(path, "a lens file", _LENS_HINT)
        cameras = []
        for ci in range(int(nb_node.real())):
            cn = fs.getNode(f"camera_{ci}")
            if cn.empty():
                continue
            K = cn.getNode("camera_matrix").mat()
            if K is None or K.shape != (3, 3):
                continue
            w, h = cn.getNode("img_width"), cn.getNode("img_height")
            cameras.append({"index": ci, "model": _camera_model_field(cn),
                            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                            "width": None if w.empty() else int(w.real()),
                            "height": None if h.empty() else int(h.real())})
    finally:
        fs.release()
    if not cameras:
        raise _wrong_kind(path, "a lens file", _LENS_HINT)
    return {"path": str(Path(path).resolve()), "cameras": cameras}
