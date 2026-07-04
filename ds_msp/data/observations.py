"""Neutral observation containers — the shared data layer.

These dataclasses are the correspondence records that *every* calibration subsystem
consumes: single-camera intrinsics (``calib``), multi-camera rig calibration (``rig``),
and the IO adapters. They live here (not inside any service package) so no service has
to import another just to name a shared type.

Dependency rule: this layer depends only on ``core`` (for the ``CameraModel`` protocol)
and NumPy — never on ``models``, detection, IO, or any service layer.

Pose convention (matches ``core.lie``): every pose is a 4x4 homogeneous matrix ``T``
mapping a point in the *child* frame into the *parent* frame, ``X_parent = T @ [X; 1]``:

* ``T_co_b`` — board -> object  ("board-in-object"); lives in :class:`Object3D`.
* ``T_g_o``  — object -> group-reference  ("object-in-group"); per (object, frame).
* ``T_c_g``  — group-reference -> camera  ("camera-in-group", the extrinsic solved).

A rig reprojection composes ``X_cam = T_c_g @ T_g_o @ T_co_b @ X_board``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.contracts import CameraModel


@dataclass
class Observation:
    """One view's 3D<->2D correspondences for a single camera in a single frame.

    The atomic unit both single-camera calibration and multi-camera rig calibration
    build on. ``points_3d`` are object/board-frame points; ``pixels`` the detected image
    points; ``visibility`` masks which rows are usable (e.g. decoded + in-bounds).
    """

    points_3d: np.ndarray        # (N, 3) object/board-frame points
    pixels: np.ndarray           # (N, 2) detected pixels
    visibility: np.ndarray       # (N,) bool
    cam_id: int = 0
    frame_id: int = 0

    def __post_init__(self) -> None:
        self.points_3d = np.asarray(self.points_3d, dtype=np.float64)
        self.pixels = np.asarray(self.pixels, dtype=np.float64)
        if self.visibility is None:
            self.visibility = np.ones(len(self.points_3d), dtype=bool)
        else:
            self.visibility = np.asarray(self.visibility, dtype=bool)
        n = len(self.points_3d)
        if self.points_3d.shape != (n, 3):
            raise ValueError(f"points_3d must be (N,3), got {self.points_3d.shape}")
        if self.pixels.shape != (n, 2):
            raise ValueError(f"pixels must be (N,2), got {self.pixels.shape}")
        if self.visibility.shape != (n,):
            raise ValueError(f"visibility must be (N,), got {self.visibility.shape}")


@dataclass
class BoardObs:
    """One planar board seen by one camera in one frame (cf. ``BoardObs.cpp``)."""

    cam_id: int
    frame_id: int
    board_id: int
    corner_ids: np.ndarray      # (K,) int  — board-local corner ids that were detected
    pts_2d: np.ndarray          # (K, 2)     — detected pixels
    T_c_b: Optional[np.ndarray] = None   # (4,4) board->camera from robust PnP
    valid: bool = True          # False if PnP inliers < 4 (BoardObs.cpp:149)
    image_path: Optional[str] = None     # source image this was detected in (for overlays)


@dataclass
class Object3D:
    """Several planar boards fused into one rigid 3D point cloud (cf. ``Object3D.cpp``)."""

    object_id: int
    board_ids: List[int]
    ref_board_id: int                                   # min(board_ids) (McCalib.cpp:898)
    T_co_b: Dict[int, np.ndarray]                       # board_id -> (4,4) board->object
    pts_3d: np.ndarray                                  # (P, 3) all corners in object frame
    pts_obj_2_board: np.ndarray                         # (P, 2) [board_id, corner_id]
    pts_board_2_obj: Dict[Tuple[int, int], int]         # (board_id, corner_id) -> row

    def row_of(self, board_id: int, corner_id: int) -> int:
        """Look up the row into :attr:`pts_3d` for a ``(board_id, corner_id)`` pair.

        Parameters
        ----------
        board_id : int
            One of :attr:`board_ids`.
        corner_id : int
            Board-local corner id (as detected by the board's own numbering).

        Returns
        -------
        int
            Row index into :attr:`pts_3d` / :attr:`pts_obj_2_board`.

        Raises
        ------
        KeyError
            If ``(board_id, corner_id)`` was never fused into this object.
        """
        return self.pts_board_2_obj[(int(board_id), int(corner_id))]


@dataclass
class ObjectObs:
    """One fused object seen by one camera in one frame (cf. ``Object3DObs.cpp``)."""

    cam_id: int
    frame_id: int
    object_id: int
    point_rows: np.ndarray      # (K,) int  — rows into Object3D.pts_3d
    pts_2d: np.ndarray          # (K, 2)
    T_c_o: Optional[np.ndarray] = None   # (4,4) object->camera from robust PnP
    image_path: Optional[str] = None     # source image this was detected in (for overlays)


@dataclass
class RigState:
    """The optimization variable mutated by the staged global BA (``rig.bundle``)."""

    cameras: Dict[int, CameraModel]                     # per-camera intrinsics
    T_c_g: Dict[int, np.ndarray]                        # camera-in-group; ref cam = identity
    ref_cam_id: int
    object_poses: Dict[Tuple[int, int], np.ndarray]     # (object_id, frame_id) -> T_g_o
    objects: Dict[int, Object3D]                        # holds T_co_b board poses
    img_size: Dict[int, Tuple[int, int]] = field(default_factory=dict)  # cam_id -> (w, h)
