"""Multi-board 3D-object fusion — stitch several planar boards into one rigid point
cloud (``computeBoardsPairPose`` -> ``initInterTransform`` -> ``initInterBoardsGraph``
-> ``init3DObjects``, McCalib.cpp:765-950).

Caveat reproduced exactly: board-in-object composition walks the shortest path
multiplying by the **inverse** of each averaged edge transform (McCalib.cpp:929),
because the edge stores ``T_next_current`` but we want current->object. The camera
graph (rig.extrinsics) composes *without* the inverse — opposite convention.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from .averaging import robust_average_transform
from .graph import connected_components, covis_weights, shortest_path
from .types import BoardObs, Object3D


def _pair_transforms(board_obs: List[BoardObs]):
    """For every image (same camera + frame) with >1 valid board, accumulate inter-board
    transforms ``T_b2_b1 = inv(T_c_b2) @ T_c_b1`` (McCalib.cpp:786) and co-observation
    counts. The grouping key is ``(cam_id, frame_id)``, not ``frame_id`` alone: MC-Calib's
    ``computeBoardsPairPose`` pairs boards seen in the *same image*. Two boards seen by
    *different* cameras in the same frame have ``T_c_b`` in different camera frames, so the
    camera does not cancel and the pair transform would be garbage."""
    by_image: Dict[Tuple[int, int], List[BoardObs]] = defaultdict(list)
    for o in board_obs:
        if o.valid and o.T_c_b is not None:
            by_image[(o.cam_id, o.frame_id)].append(o)

    samples: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for obs in by_image.values():
        for a in obs:
            for b in obs:
                if a.board_id == b.board_id:
                    continue
                T_pair = np.linalg.inv(b.T_c_b) @ a.T_c_b      # a expressed in b's frame
                samples[(a.board_id, b.board_id)].append(T_pair)
                if a.board_id < b.board_id:
                    counts[(a.board_id, b.board_id)] += 1
    return samples, counts


def build_objects(board_obs: List[BoardObs],
                  board_points: Dict[int, np.ndarray]) -> List[Object3D]:
    """Fuse boards into one :class:`Object3D` per connected covisibility component.

    ``board_points[board_id]`` is the ``(n_corners, 3)`` board-frame corner cloud
    (z=0 for a planar board). A single isolated board yields a trivial one-board object.
    """
    seen_boards = sorted({o.board_id for o in board_obs if o.valid})
    samples, counts = _pair_transforms(board_obs)
    # Pairs seen in few images include far, oblique boards whose planar pose can land on the other
    # branch, so select a consensus rather than averaging every sample (as the camera graph does).
    inter = {pair: robust_average_transform(Ts) for pair, Ts in samples.items()}
    weights = covis_weights(counts)
    comps = connected_components(seen_boards, weights)

    objects: List[Object3D] = []
    for obj_id, comp in enumerate(comps):
        ref = comp[0]                                          # min-id reference board
        T_co_b: Dict[int, np.ndarray] = {}
        for bid in comp:
            T = np.eye(4)
            if bid != ref:
                path = shortest_path(ref, bid, weights)
                for cur, nxt in zip(path[:-1], path[1:]):
                    T = T @ np.linalg.inv(inter[(cur, nxt)])   # cpp:929 — inverse of edge
            T_co_b[bid] = T

        pts, rows, b2o = [], [], {}
        for bid in comp:
            P_b = np.asarray(board_points[bid], float)
            P_h = np.c_[P_b, np.ones(len(P_b))]
            P_o = (T_co_b[bid] @ P_h.T).T[:, :3]
            for k, p in enumerate(P_o):
                b2o[(bid, k)] = len(pts)
                rows.append((bid, k))
                pts.append(p)
        objects.append(Object3D(
            object_id=obj_id, board_ids=list(comp), ref_board_id=ref,
            T_co_b=T_co_b, pts_3d=np.array(pts),
            pts_obj_2_board=np.array(rows, int), pts_board_2_obj=b2o,
        ))
    return objects
