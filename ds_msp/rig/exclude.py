"""Leaving chosen views out of a calibration (``exclude_views``).

A view is one camera's photo of the target in one frame. An entry is ``"cam:frame"`` (the
whole view) or ``"cam:frame:board"`` (one board in it), with the camera number counted from 0
and the frame and board numbers as the result files give them
(``detected_keypoints_data.yml``; the first frame with a detection is frame 0). Excluded
corners are removed before the target is built and before any lens or pose is fitted, so they
influence nothing; the result records the list.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .types import BoardObs, Object3D, ObjectObs

Exclusion = Tuple[int, int, Optional[int]]


def parse_exclusions(items: Iterable) -> Set[Exclusion]:
    """``{(cam, frame, board or None)}`` from ``"cam:frame"`` / ``"cam:frame:board"`` entries.
    Raises ``ValueError`` naming an entry that is not of that form."""
    out: Set[Exclusion] = set()
    for item in items or []:
        text = str(item).strip()
        parts = text.split(":")
        if len(parts) not in (2, 3) or not all(p.strip().isdigit() for p in parts):
            raise ValueError(f"exclude_views entry {text!r} is not 'cam:frame' or "
                             f"'cam:frame:board' with whole numbers (camera counted from 0)")
        nums = [int(p) for p in parts]
        out.add((nums[0], nums[1], nums[2] if len(nums) == 3 else None))
    return out


def _hit(ex: Set[Exclusion], cam: int, frame: int, board: int) -> Optional[Exclusion]:
    for key in ((cam, frame, None), (cam, frame, board)):
        if key in ex:
            return key
    return None


def drop_board_obs(board_obs: Sequence[BoardObs], ex: Set[Exclusion]
                   ) -> Tuple[List[BoardObs], Set[Exclusion]]:
    """Board observations without the excluded ones, and the exclusions that matched."""
    if not ex:
        return list(board_obs), set()
    kept, used = [], set()
    for o in board_obs:
        key = _hit(ex, o.cam_id, o.frame_id, o.board_id)
        if key is None:
            kept.append(o)
        else:
            used.add(key)
    return kept, used


def drop_object_obs(object_obs: Sequence[ObjectObs], objects: Dict[int, Object3D],
                    ex: Set[Exclusion], min_corners: int = 4
                    ) -> Tuple[List[ObjectObs], Set[Exclusion]]:
    """Target observations without the excluded views and without the corners of excluded
    boards; an observation left with fewer than ``min_corners`` corners is dropped whole."""
    if not ex:
        return list(object_obs), set()
    kept, used = [], set()
    for o in object_obs:
        if (o.cam_id, o.frame_id, None) in ex:
            used.add((o.cam_id, o.frame_id, None))
            continue
        obj = objects.get(o.object_id)
        rows = np.asarray(o.point_rows, int)
        if obj is None or not len(rows):
            kept.append(o)
            continue
        boards = np.asarray(obj.pts_obj_2_board, int)[rows, 0]
        gone = np.zeros(len(rows), bool)
        for b in np.unique(boards):
            if (o.cam_id, o.frame_id, int(b)) in ex:
                gone |= boards == b
                used.add((o.cam_id, o.frame_id, int(b)))
        if not gone.any():
            kept.append(o)
        elif (~gone).sum() >= min_corners:
            o.point_rows = rows[~gone]
            o.pts_2d = np.asarray(o.pts_2d, float)[~gone]
            if o.pixel_covariances is not None:
                o.pixel_covariances = np.asarray(o.pixel_covariances)[~gone]
            kept.append(o)
    return kept, used


def describe(ex: Iterable[Exclusion]) -> List[str]:
    """Exclusions back as sorted ``"cam:frame[:board]"`` strings."""
    return [":".join(str(v) for v in key if v is not None)
            for key in sorted(ex, key=lambda k: (k[0], k[1], -1 if k[2] is None else k[2]))]
