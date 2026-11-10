"""Looking at a folder of calibration photos before a run: which cameras it holds, whether
their frames line up, and whether the configured board is found in a few sample photos.

These are the checks that catch a wrong folder layout or a wrong board description in
seconds, before a calibration is started. They use the detector and the file-name rules of the
calibration itself (:mod:`ds_msp.detect.charuco`), so what passes here is read the same way
there.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..detect.charuco import (
    _IMG_EXT, BoardSpec, _frame_id_from_name, _validate_frame_association, detect_image,
    make_detectors,
)

_CAM_FOLDER = re.compile(r"^(?P<prefix>.*?)(?P<number>\d{3})$")


def _images(folder: Path) -> List[Path]:
    try:
        return sorted(p for p in folder.iterdir()
                      if p.is_file() and p.name.lower().endswith(_IMG_EXT))
    except OSError:
        return []


def guess_prefix(root: Path) -> Optional[str]:
    """The camera-folder prefix used under ``root`` (``Cam_`` for ``Cam_001``), or ``None``
    when no subfolder ends in a three-digit number. The most common prefix wins."""
    counts: Dict[str, int] = {}
    try:
        children = [p for p in root.iterdir() if p.is_dir()]
    except OSError:
        return None
    for child in children:
        m = _CAM_FOLDER.match(child.name)
        if m:
            counts[m.group("prefix")] = counts.get(m.group("prefix"), 0) + 1
    return max(counts, key=lambda k: (counts[k], k == "Cam_")) if counts else None


def scan_photos(root_path: str, cam_prefix: Optional[str] = None,
                frame_pattern: Optional[str] = None) -> Dict[str, Any]:
    """Cameras found under ``root_path`` in the layout the calibration reads
    (``<root>/<cam_prefix>001``, ``002``, ...).

    Returns ``{"cam_prefix", "number_camera", "cameras", "shared_frames", "problems"}``. Each
    camera has ``index`` (0-based), ``folder``, ``images``, ``width``, ``height`` (of its first
    image; ``None`` if unreadable) and ``frames``. ``problems`` are sentences a person can act
    on; an empty list means the folder is ready to calibrate."""
    import cv2

    root = Path(root_path)
    problems: List[str] = []
    if not root.is_dir():
        return {"cam_prefix": cam_prefix, "number_camera": 0, "cameras": [],
                "shared_frames": 0, "problems": ["This folder does not exist."]}
    prefix = cam_prefix if cam_prefix else guess_prefix(root)
    if prefix is None:
        direct = len(_images(root))
        hint = (f"It holds {direct} images directly: put each camera's images in its own "
                f"subfolder, Cam_001, Cam_002 and so on." if direct else
                "Expected one subfolder per camera, named Cam_001, Cam_002 and so on.")
        return {"cam_prefix": None, "number_camera": 0, "cameras": [], "shared_frames": 0,
                "problems": ["No camera folders found here. " + hint]}

    cameras: List[Dict[str, Any]] = []
    frames_by_cam: Dict[int, set] = {}
    index = 0
    while True:
        folder = root / f"{prefix}{index + 1:03d}"
        if not folder.is_dir():
            break
        files = _images(folder)
        width = height = None
        if files:
            img = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                height, width = int(img.shape[0]), int(img.shape[1])
            else:
                problems.append(f"{folder.name}: {files[0].name} could not be read as an image.")
        else:
            problems.append(f"{folder.name} holds no images.")
        frames: set = set()
        bad_name = None
        for f in files:
            try:
                frame = _frame_id_from_name(str(f), frame_pattern)
            except ValueError as exc:
                bad_name = bad_name or str(exc)
                continue
            if frame in frames and bad_name is None:
                bad_name = f"{folder.name}: two images have frame number {frame}"
            frames.add(frame)
        if bad_name:
            problems.append(bad_name[0].upper() + bad_name[1:] + ".")
        frames_by_cam[index] = frames
        cameras.append({"index": index, "folder": folder.name, "images": len(files),
                        "width": width, "height": height, "frames": len(frames)})
        index += 1
    if not cameras:
        problems.append(f"No folder named {prefix}001 here. Camera folders are numbered from "
                        f"001 with no gaps.")
    extra = sorted(p.name for p in root.iterdir() if p.is_dir()
                   and p.name.startswith(prefix) and _CAM_FOLDER.match(p.name)
                   and int(_CAM_FOLDER.match(p.name).group("number")) > len(cameras))
    if cameras and extra:
        problems.append(f"{', '.join(extra[:4])} {'is' if len(extra) == 1 else 'are'} not "
                        f"used: the numbering stops at {cameras[-1]['folder']}.")
    shared = 0
    if len(cameras) > 1:
        try:
            _validate_frame_association(frames_by_cam)
        except ValueError as exc:
            problems.append(str(exc)[0].upper() + str(exc)[1:] + ".")
        seen: Dict[int, int] = {}
        for frames in frames_by_cam.values():
            for f in frames:
                seen[f] = seen.get(f, 0) + 1
        shared = sum(1 for n in seen.values() if n >= 2)
    return {"cam_prefix": prefix, "number_camera": len(cameras), "cameras": cameras,
            "shared_frames": shared, "problems": problems}


def board_specs(values: Dict[str, Any]) -> List[BoardSpec]:
    """The boards a config describes, from its form values (defaults as in the loader)."""
    n = int(values.get("number_board") or 1)
    nx, ny = int(values.get("number_x_square") or 5), int(values.get("number_y_square") or 5)
    ls, lm = float(values.get("length_square") or 0.04), float(values.get("length_marker") or 0.03)
    sq = float(values.get("square_size") or 1.0)
    nx_pb = list(values.get("number_x_square_per_board") or [])
    ny_pb = list(values.get("number_y_square_per_board") or [])
    specs = [BoardSpec(n_x=int(nx_pb[b]) if b < len(nx_pb) else nx,
                       n_y=int(ny_pb[b]) if b < len(ny_pb) else ny,
                       length_square=ls, length_marker=lm, square_size=sq) for b in range(n)]
    index = list(values.get("boards_index") or [])
    if len(index) == n:
        per_board = (nx * ny) // 2
        for spec, phys in zip(specs, index):
            spec.id_offset = int(phys) * per_board
    return specs


def check_board(root_path: str, values: Dict[str, Any], *, per_camera: int = 4,
                max_cameras: int = 12) -> Dict[str, Any]:
    """Detect the configured board in a few photos of each camera, spread over the capture.

    Returns ``{"cameras": [{"index", "folder", "tried", "found", "corners_max", "corners_of"}],
    "found", "tried", "message"}``. ``found`` counts photos where at least one board was
    detected; a board description that does not match the printed target finds none."""
    import cv2

    root = Path(root_path)
    prefix = values.get("cam_prefix") or guess_prefix(root) or "Cam_"
    specs = board_specs(values)
    detectors = make_detectors(specs)
    most = max(s.n_corners for s in specs)
    rows: List[Dict[str, Any]] = []
    total_found = total_tried = 0
    n_cam = int(values.get("number_camera") or 0) or max_cameras
    for index in range(min(n_cam, max_cameras)):
        folder = root / f"{prefix}{index + 1:03d}"
        files = _images(folder)
        if not files:
            if folder.is_dir() or index < int(values.get("number_camera") or 0):
                rows.append({"index": index, "folder": folder.name, "tried": 0, "found": 0,
                             "corners_max": 0, "corners_of": most})
            continue
        step = max(1, len(files) // per_camera)
        picks = files[step // 2::step][:per_camera]
        found = best = 0
        for f in picks:
            gray = cv2.imread(os.fspath(f), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            hits = detect_image(detectors, gray)
            if hits:
                found += 1
                best = max(best, max(len(ids) for _b, ids, _p in hits))
        rows.append({"index": index, "folder": folder.name, "tried": len(picks), "found": found,
                     "corners_max": best, "corners_of": most})
        total_found += found
        total_tried += len(picks)
    if total_tried == 0:
        message = "No photos to try."
    elif total_found == 0:
        message = ("The board was not found in any sample photo. Check the number of squares "
                   "across and down, and that the target is a ChArUco board with 6x6 markers.")
    elif total_found < total_tried:
        message = (f"The board was found in {total_found} of {total_tried} sample photos. Some "
                   f"photos without the board are normal.")
    else:
        message = f"The board was found in all {total_tried} sample photos."
    return {"cameras": rows, "found": total_found, "tried": total_tried, "message": message}
