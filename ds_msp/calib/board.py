"""Board front ends — one protocol, three native implementations, one shared backend.

Every board type (checkerboard / ChArUco / AprilGrid) is just a different way to turn images
into 2D<->3D correspondences; the bundle-adjustment backend (:func:`ds_msp.calib.bundle.calibrate`)
that fits a camera model to those correspondences neither knows nor cares which board produced
them. :class:`Board` is that seam.

Each implementation below is native, not an adapter: it calls the low-level, per-image
detection primitives in :mod:`ds_msp.detect` directly and builds :class:`Observation`\\ s
inline, rather than wrapping some other function's pre-existing output shape after the fact.
None of the existing, tested detection code changes — ``detect_folder``/``detect_rig`` (used by
``ds_msp.rig``) and ``AprilGridTarget.build_correspondences`` (used by
``scripts/make_learn_gifs.py``) are untouched.

Single-camera calibration doesn't need boards to be rigidly related to each other the way
``ds_msp.rig``'s multi-board fused-object path does (many boards fixed to each other, seen by
many cameras, sharing one 3D point cloud) — it only needs many independent *views* of a *known*
planar pattern. That is why :class:`CharucoBoard` supports multiple simultaneous board
definitions with zero fusion machinery: each board actually seen in an image simply becomes its
own independent observation.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import cv2
import numpy as np

from ..data.observations import Observation
from ..detect.charuco import BoardSpec
from ..detect.charuco import board_object_points as charuco_object_points
from ..detect.charuco import detect_image, make_detectors
from ..detect.checkerboard import CheckerboardSpec
from ..detect.checkerboard import board_object_points as checkerboard_object_points
from ..detect.checkerboard import detect_corners
from ..detect.detect import detect_aprilgrid
from .targets import AprilGridTarget


ProgressCB = Callable[[int, int, str], None]


@runtime_checkable
class Board(Protocol):
    """The one seam between board-specific detection and the model-agnostic backend."""

    def detect(self, image_paths: Sequence[str],
              progress_cb: Optional[ProgressCB] = None) -> List[Observation]: ...


def to_correspondences(
    obs: List[Observation],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """The one place that unzips ``Observation``\\ s into ``bundle.calibrate``'s three lists."""
    return ([o.points_3d for o in obs], [o.pixels for o in obs], [o.visibility for o in obs])


class CheckerboardBoard:
    """Plain checkerboard, single board (v1 — see ``ds_msp.detect.checkerboard`` for why this
    is not recommended for extreme fisheye FOV; use :class:`CharucoBoard`/:class:`AprilGridBoard`
    instead beyond roughly 120 degrees)."""

    def __init__(self, spec: CheckerboardSpec):
        self.spec = spec

    def detect(self, image_paths: Sequence[str],
              progress_cb: Optional[ProgressCB] = None) -> List[Observation]:
        xyz = checkerboard_object_points(self.spec)
        vis = np.ones(len(xyz), dtype=bool)
        n = len(image_paths)
        obs: List[Observation] = []
        for i, path in enumerate(image_paths):
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                corners = detect_corners(gray, self.spec)
                if corners is not None:
                    obs.append(Observation(points_3d=xyz, pixels=corners, visibility=vis,
                                           frame_id=i))
            if progress_cb is not None:
                progress_cb(i + 1, n, path)
        return obs


class CharucoBoard:
    """ChArUco, with multi-board support: every board actually detected in an image becomes
    its own independent :class:`Observation` (its own 3D points in its own board-local frame,
    its own to-be-solved pose) — whether that board appears alone or alongside others in the
    same image is irrelevant to the bundle adjustment, which already solves one independent
    pose per observation while sharing one global intrinsics estimate across all of them."""

    def __init__(self, specs: Sequence[BoardSpec], *, legacy: bool = True, tuned: bool = False,
                min_corners: int = 4):
        self.specs = list(specs)
        self.min_corners = min_corners
        self._detectors = make_detectors(self.specs, legacy=legacy, tuned=tuned)
        self._xyz_by_board = [charuco_object_points(s) for s in self.specs]

    def detect(self, image_paths: Sequence[str],
              progress_cb: Optional[ProgressCB] = None) -> List[Observation]:
        n = len(image_paths)
        obs: List[Observation] = []
        for i, path in enumerate(image_paths):
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                for board_id, corner_ids, pts in detect_image(self._detectors, gray,
                                                               min_corners=self.min_corners):
                    xyz = self._xyz_by_board[board_id][corner_ids]
                    obs.append(Observation(points_3d=xyz, pixels=pts,
                                           visibility=np.ones(len(xyz), bool), frame_id=i))
            if progress_cb is not None:
                progress_cb(i + 1, n, path)
        return obs


class AprilGridBoard:
    """AprilGrid, single target (v1 — matches ``AprilGridTarget`` itself, which is inherently
    one grid). One grid sighting per image becomes one :class:`Observation`, concatenating
    every tag seen in it — unlike :class:`CharucoBoard`'s independent boards, tags *within* one
    AprilGrid genuinely are rigidly related by the grid's own known geometry, so they correctly
    share one pose per image, exactly as ``AprilGridTarget.build_correspondences`` already does
    internally (that existing method is untouched; this builds ``Observation``\\ s directly
    instead of going through its 3-list return).

    Calls :func:`ds_msp.detect.detect.detect_aprilgrid` one image at a time rather than on the
    whole batch: that function silently drops frames below ``min_tags``, so its return list is
    NOT aligned with ``image_paths`` — calling it per-image keeps ``frame_id`` a genuine image
    index (consistent with :class:`CharucoBoard`) without changing its detection behavior (it
    already loops over paths internally with no cross-image state)."""

    def __init__(self, target: AprilGridTarget, **detect_kwargs):
        self.target = target
        self.detect_kwargs = detect_kwargs

    def detect(self, image_paths: Sequence[str],
              progress_cb: Optional[ProgressCB] = None) -> List[Observation]:
        n = len(image_paths)
        obs: List[Observation] = []
        for i, path in enumerate(image_paths):
            found = detect_aprilgrid([path], target=self.target, **self.detect_kwargs)
            if found:
                det = found[0]
                objs, pix = [], []
                for tag_id, corners in det.items():
                    objs.append(self.target.object_points(int(tag_id)))
                    pix.append(np.asarray(corners, dtype=np.float64).reshape(-1, 2))
                xyz = np.concatenate(objs)
                uv = np.concatenate(pix)
                obs.append(Observation(points_3d=xyz, pixels=uv,
                                       visibility=np.ones(len(xyz), dtype=bool), frame_id=i))
            if progress_cb is not None:
                progress_cb(i + 1, n, path)
        return obs
