"""ChArUco detection from a raw image folder — MC-Calib's detection front-end in NumPy.

MC-Calib's ``apps/calibrate`` reads a folder of images and detects ChArUco corners with
``DICT_6X6_1000`` boards created by ``cv::aruco::CharucoBoard::create(n_x, n_y,
length_square, length_marker, dict)`` (``utilities.cpp:createCharucoBoards``), offsetting
each subsequent board's marker ids by the cumulative marker count. This module reproduces
that exactly with OpenCV's modern ``cv2.aruco.CharucoDetector`` API (OpenCV >= 4.7;
``interpolateCornersCharuco`` was removed at 4.13) so DS-MSP can ingest the *same raw
images* and produce the same per-(camera, frame, board) corner observations.

The 3-D object model is built directly at the metric ``square_size`` spacing (row-major
interior corners, matching MC-Calib's ``calibrated_objects_data.yml`` for a single board:
corner ``k`` at ``((k % ncx)·square_size, (k // ncx)·square_size, 0)``). Multi-board *fused*
object geometry (inter-board poses) is MC-Calib's board-group reconstruction and is out of
scope here — pass an existing :class:`Object3D` for multi-board rigs; this module builds the
single-board object and detects corners for any board count.
"""

from __future__ import annotations

import glob
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..data.observations import Object3D, ObjectObs

_DICT = cv2.aruco.DICT_6X6_1000


@dataclass
class BoardSpec:
    """The ChArUco geometry from MC-Calib's config (one entry per board)."""
    n_x: int
    n_y: int
    length_square: float          # marker-generation square length (e.g. 0.04)
    length_marker: float          # marker-generation marker length (e.g. 0.03)
    square_size: float            # metric spacing used for the 3-D points (e.g. 0.192)

    @property
    def n_corners(self) -> int:
        """Interior chessboard corner count, ``(n_x - 1) * (n_y - 1)``."""
        return (self.n_x - 1) * (self.n_y - 1)

    @property
    def n_markers(self) -> int:
        """ArUco marker count on this board, ``floor(n_x * n_y / 2)`` (markers fill the
        non-chessboard cells of the checker pattern)."""
        # DICT markers fill the non-chessboard cells: floor(n_x*n_y / 2).
        return (self.n_x * self.n_y) // 2


def _make_board(spec: BoardSpec, dictionary, id_offset: int, legacy: bool):
    ids = np.arange(spec.n_markers, dtype=np.int32) + id_offset
    board = cv2.aruco.CharucoBoard(
        (spec.n_x, spec.n_y), spec.length_square, spec.length_marker, dictionary, ids)
    # Blender benchmark images were rendered with the pre-4.7 ("legacy") corner pattern;
    # the modern constructor flips it. setLegacyPattern realigns marker/corner ids so the
    # detected charuco ids match MC-Calib's (and the legacy renderer's).
    if legacy and hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)
    return board


_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)


def _canonical_charuco_corners(
    corners: np.ndarray, *, opencv_version: str = cv2.__version__
) -> np.ndarray:
    """Convert detector output to DS-MSP's OpenCV-4/MC-Calib pixel convention.

    OpenCV 5.0.x reports ChArUco corners half a pixel lower in both coordinates than
    OpenCV 4.x for the same image.  DS-MSP supports 5.0.x but stores one canonical
    convention so detections remain interoperable across supported versions.  The
    dependency ceiling excludes later 5.x releases until their convention is measured.
    """
    try:
        major, minor = (int(part) for part in opencv_version.split(".", 2)[:2])
    except (TypeError, ValueError):
        return np.asarray(corners)
    out = np.asarray(corners)
    if (major, minor) == (5, 0):
        out = out + np.asarray(0.5, dtype=out.dtype)
    return out


def make_detectors(specs: List[BoardSpec], *, legacy: bool = True, tuned: bool = False):
    """One :class:`cv2.aruco.CharucoDetector` per board, with MC-Calib's id offsets.

    ``tuned=True`` enables the detection tricks that *recover more corners* a basic detector
    drops — ``tryRefineMarkers`` (re-find markers missed on the first pass via the board
    layout), ``minMarkers=1`` (accept a ChArUco corner adjacent to a single decoded marker,
    not two — the board-edge corners), and per-marker ``CORNER_REFINE_SUBPIX``. On the real
    8-camera rig it lifts the corner count ~24% (e.g. the 640px fisheyes 87/558 → 397/971).

    **It is OFF by default because, measured on that rig, the extra corners HURT accuracy**:
    the recovered corners are the strongly-distorted wide-fisheye *edge* corners, which are
    noisier — reprojection RMS rose 0.61→0.89 px. Turn it on only for a near-pinhole / mild-
    distortion rig, or when corner coverage (not accuracy) is the binding constraint."""
    dictionary = cv2.aruco.getPredefinedDictionary(_DICT)
    detectors, offset = [], 0
    for spec in specs:
        board = _make_board(spec, dictionary, offset, legacy)
        if tuned:
            cp = cv2.aruco.CharucoParameters()
            cp.tryRefineMarkers = True
            cp.minMarkers = 1
            dp = cv2.aruco.DetectorParameters()
            dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            detectors.append(cv2.aruco.CharucoDetector(board, cp, dp))
        else:
            detectors.append(cv2.aruco.CharucoDetector(board))
        offset += spec.n_markers
    return detectors


def board_object_points(spec: BoardSpec) -> np.ndarray:
    """``(n_corners, 3)`` interior-corner positions at ``square_size`` spacing, row-major
    — identical to MC-Calib's single-board ``calibrated_objects_data.yml``."""
    ncx, ncy = spec.n_x - 1, spec.n_y - 1
    k = np.arange(ncx * ncy)
    return np.c_[(k % ncx) * spec.square_size,
                 (k // ncx) * spec.square_size,
                 np.zeros(ncx * ncy)].astype(float)


def single_board_object(spec: BoardSpec, object_id: int = 0) -> Object3D:
    """Build an :class:`Object3D` for a single ChArUco board straight from the config."""
    xyz = board_object_points(spec)
    n = xyz.shape[0]
    board_ids = np.zeros(n, int)
    corner_ids = np.arange(n)
    b2o = {(0, int(c)): i for i, c in enumerate(corner_ids)}
    return Object3D(
        object_id=object_id, board_ids=[0], ref_board_id=0, T_co_b={0: np.eye(4)},
        pts_3d=xyz, pts_obj_2_board=np.c_[board_ids, corner_ids], pts_board_2_obj=b2o)


def _frame_id_from_name(path: str) -> int:
    """MC-Calib indexes frames by the digits in the filename (``..._000007.png`` -> 7)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


def detect_image(detectors, gray: np.ndarray, *, min_corners: int = 4, subpix: bool = False
                 ) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """Detect every board in one image. Returns ``[(board_id, corner_ids, pts_2d), ...]``
    with ``corner_ids`` shape ``(m,)`` and ``pts_2d`` shape ``(m, 2)``.

    ``subpix=True`` runs ``cv2.cornerSubPix`` on the interpolated ChArUco corners. OFF by
    default: ``CharucoDetector`` already sub-pixel-interpolates, and a second pass measurably
    *wandered* on this wide-fisheye rig (RMS 0.89→0.93 px) — the 5px window straddles
    neighbouring corners under heavy distortion. Useful on mild-distortion images."""
    out = []
    for b, det in enumerate(detectors):
        ch_corners, ch_ids, _, _ = det.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < min_corners:
            continue
        c = ch_corners.reshape(-1, 2).astype(np.float32)
        if subpix and len(c):
            c = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), _SUBPIX_CRITERIA)
        c = _canonical_charuco_corners(c)
        out.append((b, ch_ids.ravel().astype(int), c.astype(float)))
    return out


_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

_TLS = threading.local()


def _thread_detectors(specs: List[BoardSpec], legacy: bool):
    """One detector set per worker thread, built once and reused across that thread's images
    (``CharucoDetector`` construction is non-trivial; per-image rebuilds would dominate) —
    same idea as ``rig/reconstruct.py::_thread_detectors``, mirrored here rather than shared
    because that module builds board-level (object-free) detectors while this one detects
    straight onto a known fused :class:`Object3D`."""
    d = getattr(_TLS, "dets", None)
    if d is None or _TLS.key != (id(specs), legacy):
        d = make_detectors(specs, legacy=legacy)
        _TLS.dets, _TLS.key = d, (id(specs), legacy)
    return d


def _detect_one_image(cam_id: int, path: str, obj: Object3D, min_corners: int, detectors
                      ) -> Tuple[Optional[Tuple[int, int]], List[ObjectObs]]:
    """Detect every board in one image and map corners onto ``obj`` — the unit of work for
    both :func:`detect_folder` (sequential) and :func:`detect_rig` (thread-pooled, image-level
    load balancing across cameras of very different sizes)."""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None, []
    frame_id = _frame_id_from_name(path)
    out: List[ObjectObs] = []
    for board_id, corner_ids, pts in detect_image(detectors, gray, min_corners=min_corners):
        rows, uvs = [], []
        for cid, uv in zip(corner_ids, pts):
            key = (board_id, int(cid))
            if key in obj.pts_board_2_obj:
                rows.append(obj.pts_board_2_obj[key])
                uvs.append(uv)
        if len(rows) >= min_corners:
            out.append(ObjectObs(object_id=obj.object_id, cam_id=cam_id, frame_id=frame_id,
                                 point_rows=np.array(rows, int), pts_2d=np.array(uvs, float),
                                 T_c_o=None, image_path=path))
    return (gray.shape[1], gray.shape[0]), out


def detect_folder(image_dir: str, specs: List[BoardSpec], obj: Object3D, cam_id: int, *,
                  legacy: bool = True, min_corners: int = 4, pattern: str = "*",
                  progress_cb: Optional[Callable[[int, int, int, str], None]] = None
                  ) -> List[ObjectObs]:
    """Detect ChArUco corners over every image in ``image_dir`` for one camera, sequentially.

    Returns one :class:`ObjectObs` per (frame, board), with ``point_rows`` indexing
    ``obj.pts_3d`` via ``(board_id, corner_id)`` — the same observation objects the
    keypoint reader produces, so the rest of the pipeline is untouched. ``frame_id`` is the
    raw filename index; :func:`detect_rig` rebases it to MC-Calib's 0-indexed convention.
    ``progress_cb(cam_id, i, n, path)``, if given, fires before each image is detected — the
    per-image OpenCV detection pass is the part of a real calibration run that otherwise
    stays silent for minutes on a large image set. This is the single-camera building block;
    :func:`detect_rig` parallelizes across every camera's images at once rather than calling
    this once per camera.
    """
    detectors = make_detectors(specs, legacy=legacy)
    files = sorted(f for f in glob.glob(os.path.join(image_dir, pattern))
                   if f.lower().endswith(_IMG_EXT))
    n = len(files)
    obs: List[ObjectObs] = []
    for i, path in enumerate(files, 1):
        if progress_cb is not None:
            progress_cb(cam_id, i, n, path)
        _, o = _detect_one_image(cam_id, path, obj, min_corners, detectors)
        obs.extend(o)
    return obs


def detect_rig(root_path: str, cam_ids: List[int], specs: List[BoardSpec], obj: Object3D, *,
               cam_prefix: str = "Cam_", legacy: bool = True, min_corners: int = 4,
               progress_cb: Optional[Callable[[int, int, int, str], None]] = None,
               workers: Optional[int] = None
               ) -> Tuple[List[ObjectObs], Dict[int, Tuple[int, int]]]:
    """Detect over ``<root_path>/<cam_prefix><cam+1:03d>/`` for every camera (MC-Calib's
    1-indexed layout). Returns ``(object_obs, img_size_per_cam)``.

    Detection is **parallelised across cameras and images together**: one flat
    ``(cam_id, path)`` task list spanning every camera, run on a single
    :class:`~concurrent.futures.ThreadPoolExecutor` — the same pattern (and the same
    measured ~3.5x on an 8-camera rig) as ``rig/reconstruct.py::detect_board_obs_images``.
    OpenCV's C++ detection work releases the GIL, so threads scale near-linearly with cores
    without needing process-pool pickling of the (potentially large) fused ``Object3D``.
    Flat per-image tasks, not per-camera ones, keep every core busy despite very uneven
    per-camera image counts (e.g. a 2592px camera with 116 frames next to a 640px one with
    58 — a per-camera-only split would leave two workers doing 2x the work of the rest).
    ``workers=None`` (default) uses ``os.cpu_count()``; ``1`` forces the original serial
    per-camera loop (useful for debugging, or when there's only one camera / one image
    total, where pool start-up would only add overhead).

    ``progress_cb(cam_id, i, n, path)``, if given, still fires once per image, from whichever
    worker thread is currently detecting it — serialized by an internal lock so terminal
    output from concurrent workers doesn't interleave mid-line. ``i`` counts that camera's own
    images processed so far (as before). Tasks are **round-robin interleaved across cameras**
    (image-index-major, not camera-major): a naive flat list built camera-by-camera drains
    camera 0's entire directory before camera 1's first task is even queued, so the live
    progress readout would show a strictly sequential "cam 0 done, then cam 1 starts, ..."
    sweep even though the underlying work is genuinely concurrent — measured, not assumed
    (``rig/reconstruct.py::detect_board_obs_images`` had the exact same bug).
    """
    cam_dirs: Dict[int, str] = {}
    for c in cam_ids:
        d = os.path.join(root_path, f"{cam_prefix}{c + 1:03d}")
        if os.path.isdir(d):
            cam_dirs[c] = d

    files_by_cam: Dict[int, List[str]] = {}
    for c, d in cam_dirs.items():
        files = sorted(f for f in glob.glob(os.path.join(d, "*")) if f.lower().endswith(_IMG_EXT))
        if files:
            files_by_cam[c] = files
    tasks: List[Tuple[int, str]] = []          # round-robin (cam_id, path) list, not camera-grouped
    i = 0
    while True:
        row = [(c, files[i]) for c, files in files_by_cam.items() if i < len(files)]
        if not row:
            break
        tasks.extend(row)
        i += 1
    counts = Counter(c for c, _ in tasks)
    seen: Dict[int, int] = Counter()
    lock = threading.Lock()

    def _do(task: Tuple[int, str]):
        c, path = task
        if progress_cb is not None:
            with lock:                          # serialize both the counter and the write
                seen[c] += 1
                progress_cb(c, seen[c], counts[c], path)
        detectors = _thread_detectors(specs, legacy)
        return (c, *_detect_one_image(c, path, obj, min_corners, detectors))

    n_workers = (os.cpu_count() or 4) if workers is None else workers
    if n_workers and n_workers > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_do, tasks))
    else:
        results = [_do(t) for t in tasks]

    all_obs: List[ObjectObs] = []
    img_size: Dict[int, Tuple[int, int]] = {}
    for c, wh, obs in results:
        if wh is not None and c not in img_size:
            img_size[c] = wh
        all_obs.extend(obs)
    if all_obs:                       # rebase to MC-Calib's 0-indexed frames
        base = min(o.frame_id for o in all_obs)
        for o in all_obs:
            o.frame_id -= base
    return all_obs, img_size
