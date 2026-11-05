"""Camera-group extrinsics initialization — co-visibility over objects, not boards
(``computeCamerasPairPose`` -> ``initInterTransform`` -> ``initInterCamerasGraph`` ->
``initCameraGroup``, McCalib.cpp:1054-1163).

Each directed edge (a, b) maps camera a coordinates into camera b coordinates.
Traversing from the group reference to a camera therefore left-multiplies each
successive edge to obtain the group-reference-to-camera transform.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from .averaging import robust_average_transform
from .graph import connected_components, covis_weights, shortest_path
from .types import ObjectObs


def _camera_pair_transforms(object_obs: List[ObjectObs]):
    """For every (frame, object) seen by >1 camera, accumulate inter-camera transforms
    ``T_c2_c1 = T_c2_o @ inv(T_c1_o)`` and co-observation counts."""
    by_fo: Dict[Tuple[int, int], List[ObjectObs]] = defaultdict(list)
    for o in object_obs:
        if o.T_c_o is not None:
            by_fo[(o.frame_id, o.object_id)].append(o)

    samples: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for obs in by_fo.values():
        for a in obs:
            for b in obs:
                if a.cam_id == b.cam_id:
                    continue
                T_pair = b.T_c_o @ np.linalg.inv(a.T_c_o)      # camera a -> camera b
                samples[(a.cam_id, b.cam_id)].append(T_pair)
                if a.cam_id < b.cam_id:
                    counts[(a.cam_id, b.cam_id)] += 1
    return samples, counts


def init_camera_groups(object_obs: List[ObjectObs], cam_ids: List[int]):
    """Partition cameras into overlapping groups and seed each camera's extrinsic.

    Returns ``(groups, extrinsics)`` where ``groups`` is a list of camera-id lists (one
    per connected component, each with ``group[0]`` as the reference camera) and
    ``extrinsics[cam_id]`` is the ``T_c_g`` (group-ref -> camera; ref cam = identity).
    """
    samples, counts = _camera_pair_transforms(object_obs)
    inter = {pair: robust_average_transform(Ts) for pair, Ts in samples.items()}
    weights = covis_weights(counts)
    comps = connected_components(sorted(cam_ids), weights)

    extrinsics: Dict[int, np.ndarray] = {}
    groups: List[List[int]] = []
    for comp in comps:
        ref = comp[0]
        groups.append(list(comp))
        for cam in comp:
            T = np.eye(4)
            if cam != ref:
                path = shortest_path(ref, cam, weights)
                for cur, nxt in zip(path[:-1], path[1:]):
                    T = inter[(cur, nxt)] @ T                 # T_nxt_cur @ T_cur_ref
            extrinsics[cam] = T
    return groups, extrinsics
