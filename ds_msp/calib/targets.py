"""
Calibration target geometry — the 3D model of the board you photograph.

Pure NumPy, no OpenCV, no detector. An ``AprilGridTarget`` knows where every tag
corner sits in 3D board coordinates (metres), and turns per-image tag detections
into the ``(X_world, keypoints, visibility)`` correspondence lists that
``ds_msp.calib.bundle.calibrate`` consumes. Detection (which needs OpenCV + an
AprilTag backend) lives separately in ``detect.py``; this module stays dependency
-light so the board math can be imported and unit-tested on its own.

Imports: numpy only.
"""

from __future__ import annotations

from typing import List, Mapping, Tuple

import numpy as np


class AprilGridTarget:
    """A Kalibr-style AprilGrid: a ``rows x cols`` grid of AprilTags.

    Geometry follows Kalibr's convention exactly so detections from a board
    calibrated by Kalibr/Basalt line up:

    - Tag ids run row-major from the bottom-left corner: ``id = row * cols + col``.
    - Each tag's four corners are ordered **counter-clockwise starting bottom-left**
      ``(BL, BR, TR, TL)`` — the same order the AprilGrid detector returns them.
    - Tags are squares of side ``tag_size`` (metres) separated by a gap of
      ``tag_spacing * tag_size`` (``tag_spacing`` is Kalibr's gap/size ratio).

    ``tag_size`` only sets absolute scale, which affects the recovered *extrinsic*
    translations, not the intrinsics — so a slightly wrong board size still yields
    correct ``fx, fy, cx, cy`` and distortion.
    """

    def __init__(self, tag_rows: int = 6, tag_cols: int = 6,
                 tag_size: float = 0.088, tag_spacing: float = 0.3) -> None:
        self.tag_rows = int(tag_rows)
        self.tag_cols = int(tag_cols)
        self.tag_size = float(tag_size)
        self.tag_spacing = float(tag_spacing)

    @property
    def n_tags(self) -> int:
        """Total tag count, ``tag_rows * tag_cols``."""
        return self.tag_rows * self.tag_cols

    def object_points(self, tag_id: int) -> np.ndarray:
        """3D board coordinates (metres) of one tag's 4 corners, ``(4, 3)``.

        Order matches the detector: bottom-left, bottom-right, top-right, top-left.
        """
        if not 0 <= tag_id < self.n_tags:
            raise ValueError(f"tag_id {tag_id} out of range [0, {self.n_tags})")
        row, col = divmod(tag_id, self.tag_cols)
        pitch = self.tag_size * (1.0 + self.tag_spacing)
        s = self.tag_size
        x0, y0 = col * pitch, row * pitch
        return np.array([[x0,     y0,     0.0],
                         [x0 + s, y0,     0.0],
                         [x0 + s, y0 + s, 0.0],
                         [x0,     y0 + s, 0.0]], dtype=np.float64)

    def all_object_points(self) -> np.ndarray:
        """Every corner of the board, ``(n_tags * 4, 3)`` in tag-id order."""
        return np.concatenate([self.object_points(t) for t in range(self.n_tags)])

    def build_correspondences(
        self, detections_per_image: List[Mapping[int, np.ndarray]],
        *, min_corners: int = 8,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Turn per-image ``{tag_id: (4, 2) pixels}`` into calibration inputs.

        Returns ``(X_world_list, keypoints_list, visibility_list)`` ready for
        ``ds_msp.calib.bundle.calibrate``. Images with fewer than ``min_corners``
        detected corners are dropped (too few to constrain a pose).
        """
        X_world_list: List[np.ndarray] = []
        keypoints_list: List[np.ndarray] = []
        visibility_list: List[np.ndarray] = []
        for det in detections_per_image:
            obj, pix = [], []
            for tag_id, corners in det.items():
                corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
                if corners.shape[0] != 4:
                    continue
                obj.append(self.object_points(int(tag_id)))
                pix.append(corners)
            if not obj:
                continue
            X = np.concatenate(obj)
            uv = np.concatenate(pix)
            if len(X) < min_corners:
                continue
            X_world_list.append(X)
            keypoints_list.append(uv)
            visibility_list.append(np.ones(len(X), dtype=bool))
        return X_world_list, keypoints_list, visibility_list
