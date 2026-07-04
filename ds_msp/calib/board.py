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
              progress_cb: Optional[ProgressCB] = None) -> List[Observation]:
        """Detect this board's known target in a batch of images.

        Every concrete implementation (:class:`CheckerboardBoard`, :class:`CharucoBoard`,
        :class:`AprilGridBoard`) turns raw image files into
        :class:`~ds_msp.data.observations.Observation`\\ s carrying known 3D board-local
        points paired with their detected 2D pixels, so
        :func:`ds_msp.calib.single_camera.calibrate_camera` can fit any camera model
        without knowing which board produced the correspondences.

        Parameters
        ----------
        image_paths : Sequence[str]
            Paths to the images to search for the board, in any order.
        progress_cb : callable, optional
            ``progress_cb(i, n, path)`` called once per image (``1 <= i <= n``) as it is
            processed, for live progress reporting. ``None`` (default) disables it.

        Returns
        -------
        list of Observation
            One entry per detected board sighting. Images the board was not found in (or
            that failed to load) are silently skipped, so the result length is not
            guaranteed to equal ``len(image_paths)``.
        """
        ...


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
        """Detect the checkerboard in every readable image.

        One :class:`~ds_msp.data.observations.Observation` per image the board was found
        in, using the fixed board geometry from ``self.spec`` (see
        :func:`ds_msp.detect.checkerboard.board_object_points`); all corners are marked
        visible (``findChessboardCornersSB`` is all-or-nothing per image).

        Parameters
        ----------
        image_paths : Sequence[str]
            Paths to the images to search.
        progress_cb : callable, optional
            ``progress_cb(i, n, path)`` fired once per image; see :meth:`Board.detect`.

        Returns
        -------
        list of Observation
            One per image with a successful detection; unreadable images and images with
            no detected board are silently skipped.
        """
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
        """Detect every configured ChArUco board in every readable image.

        Each board actually found in an image (there may be zero, one, or several, when
        ``self.specs`` has more than one entry) becomes its own independent
        :class:`~ds_msp.data.observations.Observation`, sharing that image's ``frame_id``
        but carrying only that board's own corners — boards seen together in one image are
        not fused into a single pose. Boards with fewer than ``self.min_corners`` detected
        corners are dropped.

        Parameters
        ----------
        image_paths : Sequence[str]
            Paths to the images to search.
        progress_cb : callable, optional
            ``progress_cb(i, n, path)`` fired once per image; see :meth:`Board.detect`.

        Returns
        -------
        list of Observation
            Zero or more entries per image (one per board found in it); unreadable images
            contribute none.
        """
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
        """Detect the AprilGrid in every image, one image at a time.

        Every tag found in an image is concatenated into a single
        :class:`~ds_msp.data.observations.Observation` for that image (tags within one
        AprilGrid share one rigid pose, unlike :class:`CharucoBoard`'s independent
        boards). Calls :func:`ds_msp.detect.detect.detect_aprilgrid` per image, one at a
        time, so ``frame_id`` stays a genuine index into ``image_paths`` even though that
        function's own batch API silently drops frames below its ``min_tags`` threshold.

        Parameters
        ----------
        image_paths : Sequence[str]
            Paths to the images to search.
        progress_cb : callable, optional
            ``progress_cb(i, n, path)`` fired once per image; see :meth:`Board.detect`.

        Returns
        -------
        list of Observation
            One entry per image with at least one detected tag; images with no detection
            (below ``min_tags``, or too few recovered corners) are silently skipped.
        """
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
