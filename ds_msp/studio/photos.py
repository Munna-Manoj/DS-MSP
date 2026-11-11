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
    _DICT, _IMG_EXT, BoardSpec, _frame_id_from_name, _make_board, _validate_frame_association,
    detect_image, make_detectors,
)

#: Widest picture :func:`thumbnail` returns, in pixels.
THUMB_WIDTH = 640
#: cosine of 20 degrees: how well a marker's diagonal must line up with a neighbour to be a
#: corner-to-corner neighbour
_FILL_COS = 0.94
#: pairs of touching markers needed before the marker width is stated
_FILL_PAIRS = 20
#: widest middle half of the readings that still counts as one answer
_FILL_SPREAD = 0.05

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

    Returns ``{"cameras": [{"index", "folder", "tried", "found", "corners_max", "corners_of",
    "sample"}], "found", "tried", "message"}``. ``found`` counts photos where at least one
    board was detected; a board description that does not match the printed target finds none.
    ``sample`` is the tried photo with the most corners: its place in the folder listing
    (``photo``), its size, the pixel of every corner found, for drawing them on it, and the
    index of the board each corner belongs to (``boards``)."""
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
                             "corners_max": 0, "corners_of": most, "sample": None})
            continue
        step = max(1, len(files) // per_camera)
        picks = files[step // 2::step][:per_camera]
        found = best = 0
        sample: Optional[Dict[str, Any]] = None
        for f in picks:
            gray = cv2.imread(os.fspath(f), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            hits = detect_image(detectors, gray)
            if hits:
                found += 1
                n = sum(len(ids) for _b, ids, _p in hits)
                if sample is None or n > len(sample["corners"]):
                    # the photo that shows the most corners, with where they were found
                    sample = {"photo": files.index(f), "file": f.name,
                              "width": int(gray.shape[1]), "height": int(gray.shape[0]),
                              "corners": [[round(float(u), 1), round(float(v), 1)]
                                          for _b, _ids, pts in hits for u, v in pts],
                              "boards": [int(b) for b, ids, _pts in hits for _ in ids]}
                best = max(best, max(len(ids) for _b, ids, _p in hits))
        rows.append({"index": index, "folder": folder.name, "tried": len(picks), "found": found,
                     "corners_max": best, "corners_of": most, "sample": sample})
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


def thumbnail(root_path: str, cam: int, photo: int = 0, cam_prefix: Optional[str] = None,
              max_width: int = THUMB_WIDTH) -> Optional[bytes]:
    """Photo number ``photo`` (in file-name order) of camera ``cam`` as a JPEG at most
    ``max_width`` wide, or ``None`` when there is no such readable photo."""
    import cv2

    root = Path(root_path)
    prefix = cam_prefix or guess_prefix(root) or "Cam_"
    files = _images(root / f"{prefix}{cam + 1:03d}")
    if not 0 <= photo < len(files):
        return None
    img = cv2.imread(os.fspath(files[photo]), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, max(1, round(h * max_width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buf.tobytes() if ok else None


def _marker_cells(n_x: int, n_y: int) -> Dict[int, tuple]:
    """Board cell ``(column, row)`` of each marker of an ``n_x`` by ``n_y`` board, as the
    detector lays it out."""
    import cv2
    import numpy as np

    board = _make_board(BoardSpec(n_x, n_y, 1.0, 0.75, 1.0), cv2.aruco.getPredefinedDictionary(_DICT),
                        0, True)
    ids = np.asarray(board.getIds()).ravel()
    return {int(i): (int(np.floor(np.mean(c[:, 0]))), int(np.floor(np.mean(c[:, 1]))))
            for i, c in zip(ids, board.getObjPoints())}


def _neighbours(centres) -> List[tuple]:
    """Pairs ``(a, b)`` of markers that sit next to each other in a photo: up to four nearest,
    no further than 1.9 times the nearest one."""
    import numpy as np

    d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return [(a, int(b)) for a in range(len(centres)) for b in np.argsort(d[a])[:4]
            if d[a, b] <= 1.9 * d[a].min()]


def _groups(n: int, pairs: List[tuple]) -> List[List[int]]:
    """Markers joined by neighbour pairs: each group is one board in one photo."""
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        parent[find(a)] = find(b)
    out: Dict[int, List[int]] = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def _marker_fill(quads, centres) -> List[float]:
    """Marker width over square width, once per pair of markers that touch at a corner.

    Two such markers lie on one diagonal of the board: along it come the diagonal of one
    marker, a gap, and the diagonal of the other. The three lengths sit next to each other on
    one line, so the lens and the tilt of the board stretch them alike, and
    ``diagonal / (diagonal + gap)`` is the ratio the board was printed with."""
    import numpy as np

    out: List[float] = []
    d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    for a in range(len(centres)):
        for b in np.argsort(d[a])[:4]:
            b = int(b)
            if b < a or d[a, b] > 1.3 * d[a].min() or d[a, b] > 1.3 * d[b].min():
                continue
            ka = int(np.argmin(np.linalg.norm(quads[a] - centres[b], axis=1)))
            kb = int(np.argmin(np.linalg.norm(quads[b] - centres[a], axis=1)))
            pa, pb = quads[a, ka], quads[b, kb]
            along = (centres[b] - centres[a]) / d[a, b]
            da, db = quads[a, ka] - quads[a, (ka + 2) % 4], quads[b, (kb + 2) % 4] - quads[b, kb]
            la, lb = float(np.linalg.norm(da)), float(np.linalg.norm(db))
            # markers in one row or column are neighbours too; their diagonals point 45 degrees
            # away from the line that joins them
            if min(float(da @ along) / la, float(db @ along) / lb) < _FILL_COS:
                continue
            diagonal = 0.5 * (la + lb)
            out.append(diagonal / (diagonal + float((pb - pa) @ along)))
    return out


def _layout_score(n_x: int, n_y: int, views: List[tuple]) -> float:
    """Share of marker pairs that are neighbours in a photo and also neighbours on an ``n_x`` by
    ``n_y`` board. Uses only which markers are next to which, so lens distortion does not
    matter. Pairs of consecutive markers are left out (they are neighbours on a board of any
    width), and so are pairs from two different boards."""
    cells = _marker_cells(n_x, n_y)
    per_board = (n_x * n_y) // 2
    good = total = 0
    for ids, centres, _clear in views:
        for a, b in _neighbours(centres):
            if abs(int(ids[a]) - int(ids[b])) == 1 or ids[a] // per_board != ids[b] // per_board:
                continue
            total += 1
            ca, cb = cells[int(ids[a]) % per_board], cells[int(ids[b]) % per_board]
            good += abs(ca[0] - cb[0]) + abs(ca[1] - cb[1]) == 2
    return good / total if total >= 6 else 0.0


def find_board(root_path: str, values: Dict[str, Any], *, photos: int = 48,
               max_cameras: int = 8) -> Dict[str, Any]:
    """Work out the number of squares across and down from the markers in a few photos.

    A ChArUco board numbers its markers row by row. The markers that sit together in a photo
    are one board; when a photo shows a whole board, the run of numbers in that group is the
    marker count of a board. Which numbers sit next to each other then fixes how many squares
    a row has, and the marker count fixes the number of rows. Only which markers are neighbours
    is used, so the lens does not matter. Boards of one target are taken to be alike. About
    ``photos`` photos are read, spread over the first ``max_cameras`` cameras.

    Returns ``{"ok", "number_x_square", "number_y_square", "boards_seen", "markers", "photos",
    "marker_fill", "marker_fill_pairs", "message"}``; ``ok`` is false, with the reason in
    ``message``, when the photos do not settle it: the size is not guessed. ``marker_fill`` is
    the marker width over the square width measured in the same photos, whether or not the size
    was settled; it is ``None`` when too few markers touch or the readings disagree."""
    import cv2
    import numpy as np

    root = Path(root_path)
    prefix = values.get("cam_prefix") or guess_prefix(root) or "Cam_"
    params = cv2.aruco.DetectorParameters()
    # marker corners to a fraction of a pixel: the marker width is measured from them
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(_DICT), params)
    views: List[tuple] = []          # one per board per photo: ids, centres, clear of the border
    fills: List[float] = []
    seen: set = set()
    tried = 0
    n_cam = min(max_cameras, max(1, int(values.get("number_camera") or 1)))
    per_camera = max(6, photos // n_cam)
    for index in range(n_cam):
        files = _images(root / f"{prefix}{index + 1:03d}")
        step = max(1, len(files) // per_camera)
        for f in files[step // 2::step][:per_camera]:
            gray = cv2.imread(os.fspath(f), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            tried += 1
            corners, ids, _rejected = detector.detectMarkers(gray)
            if ids is None or len(ids) < 4:
                continue
            ids = np.asarray(ids, int).ravel()
            quads = np.array([np.asarray(c, float).reshape(4, 2) for c in corners])
            centres = quads.mean(axis=1)
            side = np.linalg.norm(quads[:, 0] - quads[:, 1], axis=1)
            h, w = gray.shape[:2]
            fills.extend(_marker_fill(quads, centres))
            for group in _groups(len(ids), _neighbours(centres)):
                if len(group) < 4:
                    continue                      # a stray reading, or too little of a board
                g = np.array(group)
                margin = 1.5 * float(np.median(side[g]))
                clear = bool((quads[g, :, 0].min() > margin) and (quads[g, :, 1].min() > margin)
                             and (quads[g, :, 0].max() < w - margin) and (quads[g, :, 1].max() < h - margin))
                views.append((ids[g], centres[g], clear))
                seen.update(int(i) for i in ids[g])
    out: Dict[str, Any] = {"ok": False, "number_x_square": None, "number_y_square": None,
                           "boards_seen": 0, "markers": len(seen), "photos": tried,
                           "marker_fill": None, "marker_fill_pairs": len(fills)}
    if len(fills) >= _FILL_PAIRS:
        low, mid, high = np.percentile(fills, [25, 50, 75])
        if high - low <= _FILL_SPREAD and 0.2 < mid < 0.98:
            out["marker_fill"] = round(float(mid), 3)
    if not views:
        out["message"] = ("No board markers were found in the sample photos. The target must be a "
                          "ChArUco board with 6x6 markers.")
        return out
    unsettled = ("The photos do not settle the board size: none of the sample photos shows a "
                 "whole board clear of the picture's edge. Count the squares across and down "
                 "and enter them.")
    # A group that holds every number in its range is a whole board, or a whole part of one.
    # The marker count of a board is the range seen that way at least twice, clear of the edge.
    runs: Dict[int, List[bool]] = {}
    for ids, _c, clear in views:
        span = int(ids.max() - ids.min() + 1)
        if span == len(ids):
            runs.setdefault(span, []).append(clear)
    supported = {span: len(flags) for span, flags in runs.items() if len(flags) >= 2 and any(flags)}
    if not supported:
        out["message"] = unsettled
        return out
    per_board = max(supported)
    if len({i // per_board for i in seen}) > 1:
        # several boards: the count seen most often is one board; larger runs are boards that
        # sit side by side, and must then be whole multiples of it
        per_board = max(supported, key=lambda span: (supported[span], span))
        if any(span > per_board and span % per_board for span in supported):
            out["message"] = unsettled
            return out
    best = sorted(((_layout_score(nx, ny, views), nx, ny) for nx in range(3, 25) for ny in range(2, 41)
                   if (nx * ny) // 2 == per_board), reverse=True)
    if not best or best[0][0] < 0.85 or (len(best) > 1 and best[1][0] > best[0][0] - 0.15):
        out["message"] = unsettled
        return out
    _score, nx, ny = best[0]
    boards = len({int(i) // per_board for i in seen})
    out.update(ok=True, number_x_square=nx, number_y_square=ny, boards_seen=boards,
               message=f"The board has {nx} squares across and {ny} down ({per_board} markers)."
               if boards == 1 else
               f"Each board has {nx} squares across and {ny} down ({per_board} markers). "
               f"{boards} different boards appear in the sample photos.")
    return out
