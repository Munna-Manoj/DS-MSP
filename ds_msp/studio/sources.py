"""The files a calibration can start from other than a folder of photos: a ``calib_param.yml``
on the computer that runs Studio, a file of saved corner detections, and a file of lens values
already known. Each is read the way the calibration reads it and summarised for the form.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

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
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise ValueError("not a readable detections file")
    try:
        nb_node = fs.getNode("nb_camera")
        if nb_node.empty():
            raise ValueError("this is not a detections file: it has no nb_camera entry")
        cameras = []
        for ci in range(int(nb_node.real())):
            cn = fs.getNode(f"camera_{ci}")
            if cn.empty():
                continue
            if cn.getNode("frame_idxs").empty() and cn.getNode("pts_2d").empty():
                raise ValueError("this is not a detections file: its cameras list no corners "
                                 "(it looks like a lens file)")
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

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise ValueError("not a readable lens file")
    try:
        nb_node = fs.getNode("nb_camera")
        if nb_node.empty():
            raise ValueError("this is not a lens file: it has no nb_camera entry")
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
        raise ValueError("this file holds no camera with a camera_matrix")
    return {"path": str(Path(path).resolve()), "cameras": cameras}
