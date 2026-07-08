"""Multi-object fusion — stitch several rigid :class:`Object3D`s that are never
co-observed at the board level into one rigid point cloud (``merge3DObjects``,
McCalib.cpp).

This is the object-level analogue of :func:`ds_msp.rig.object3d.build_objects` (which
fuses *boards*). The two objects are relatable not because a camera sees both boards in
one image — by construction they never share an image — but because different cameras,
whose extrinsics are already known (all placed in one common "group" frame), see the two
objects *concurrently* in the same frame. Lifting each per-camera object pose into the
shared group frame (``T_g_o = inv(T_c_g) @ T_c_o``, cf.
``pose_init.average_object_pose_in_group``) makes the two objects comparable, and their
relative pose in that frame is the rigid inter-object transform.

Caveat reproduced exactly (mirrors object3d): object-in-merged-object composition walks
the shortest path multiplying by the **inverse** of each averaged edge transform
(McCalib.cpp:929), because the edge stores ``T_next_current`` but we want
current->merged-object. This is the board-graph convention, opposite to the camera graph
(rig.extrinsics), which composes *without* the inverse.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from .averaging import average_transform, robust_average_transform
from .graph import connected_components, covis_weights, shortest_path
from .types import Object3D, ObjectObs


def inter_object_transforms(object_obs: List[ObjectObs], extr: Dict[int, np.ndarray]):
    """For every frame in which two objects are seen (by any cameras), accumulate the
    inter-object transforms ``T_oj_oi = inv(T_g_oj) @ T_g_oi`` (object ``oi`` expressed in
    object ``oj``'s frame) and co-observation counts.

    Each object pose is first lifted into the common group frame:
    ``T_g_o = inv(extr[cam]) @ o.T_c_o`` (object->group). When the same (frame, object) is
    seen by several cameras its group poses are robust-averaged
    (:func:`robust_average_transform`) so a single outlier view cannot poison the pair.
    Then, per frame, every *ordered* pair of distinct objects contributes a sample; the
    covisibility count is bumped only for the canonical ``oi < oj`` direction (the graph
    layer symmetrises anyway). Analogous to ``object3d._pair_transforms``'s
    ``inv(T_c_b2) @ T_c_b1``, but with cameras cancelling via the known extrinsics rather
    than via a shared image.
    """
    # frame -> object_id -> list of group-frame poses (one per camera that saw it)
    by_frame: Dict[int, Dict[int, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for o in object_obs:
        if o.T_c_o is not None and o.cam_id in extr:
            T_g_o = np.linalg.inv(extr[o.cam_id]) @ o.T_c_o     # object->group
            by_frame[o.frame_id][o.object_id].append(T_g_o)

    samples: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for per_obj in by_frame.values():
        fused = {oid: robust_average_transform(Ts) for oid, Ts in per_obj.items()}
        for oi, T_g_oi in fused.items():
            for oj, T_g_oj in fused.items():
                if oi == oj:
                    continue
                T_pair = np.linalg.inv(T_g_oj) @ T_g_oi         # oi expressed in oj's frame
                samples[(oi, oj)].append(T_pair)
                if oi < oj:
                    counts[(oi, oj)] += 1
    return samples, counts


def merge_objects(objects: List[Object3D], object_obs: List[ObjectObs],
                  extr: Dict[int, np.ndarray], *, min_covis: int = 1
                  ) -> Tuple[List[Object3D], Dict[int, Tuple[int, int]]]:
    """Fuse rigidly-linked :class:`Object3D`s into one object per connected covisibility
    component — the object-level :func:`ds_msp.rig.object3d.build_objects`.

    Parameters
    ----------
    objects : list of Object3D
        The per-component objects from board fusion (each with its own point cloud).
    object_obs : list of ObjectObs
        Per (camera, frame) object detections carrying ``T_c_o`` (object->camera).
    extr : dict
        ``extr[cam_id]`` is ``T_c_g`` (group-ref -> camera; ref cam = identity).
    min_covis : int, optional
        Minimum co-observation count for an inter-object edge to be trusted.

    Returns
    -------
    merged_objects : list of Object3D
        One fused object per component, re-``object_id``'d ``0, 1, ...`` in order of
        descending corner count.
    remap : dict
        ``{old_object_id: (new_merged_object_id, row_offset)}`` — a point at row ``r`` in
        old object ``O`` lands at row ``r + row_offset`` in merged object
        ``new_merged_object_id``.
    """
    objects_by_id = {o.object_id: o for o in objects}
    samples, counts = inter_object_transforms(object_obs, extr)
    inter = {pair: average_transform(Ts) for pair, Ts in samples.items()}
    counts = {pair: n for pair, n in counts.items() if n >= min_covis}
    weights = covis_weights(counts)
    comps = connected_components(sorted(objects_by_id), weights)

    # Build each merged component; defer object_id assignment until we can sort by size.
    built: List[dict] = []
    for comp in comps:
        ref = comp[0]                                          # min-id reference object
        T_mo_o: Dict[int, np.ndarray] = {}
        for oid in comp:
            T = np.eye(4)
            if oid != ref:
                path = shortest_path(ref, oid, weights)
                for cur, nxt in zip(path[:-1], path[1:]):
                    T = T @ np.linalg.inv(inter[(cur, nxt)])   # cpp:929 — inverse of edge
            T_mo_o[oid] = T                                    # object oid -> merged frame

        pts: List[np.ndarray] = []
        rows: List[Tuple[int, int]] = []
        offsets: Dict[int, int] = {}
        T_co_b: Dict[int, np.ndarray] = {}
        board_ids: set = set()
        for oid in comp:
            obj = objects_by_id[oid]
            offsets[oid] = len(pts)                            # row offset within merged cloud
            P_o = np.asarray(obj.pts_3d, float)
            P_h = np.c_[P_o, np.ones(len(P_o))]
            P_mo = (T_mo_o[oid] @ P_h.T).T[:, :3]              # into merged-object frame
            for p in P_mo:
                pts.append(p)
            for r in obj.pts_obj_2_board:
                rows.append((int(r[0]), int(r[1])))
            for bid in obj.board_ids:
                board_ids.add(bid)
                T_co_b[bid] = T_mo_o[oid] @ obj.T_co_b[bid]    # board -> merged object
        b2o = {(bid, cid): k for k, (bid, cid) in enumerate(rows)}
        built.append(dict(
            comp=list(comp), ref=ref, offsets=offsets,
            board_ids=sorted(board_ids), ref_board_id=objects_by_id[ref].ref_board_id,
            T_co_b=T_co_b,
            pts_3d=np.array(pts, float).reshape(-1, 3),
            pts_obj_2_board=np.array(rows, int).reshape(-1, 2),
            pts_board_2_obj=b2o,
        ))

    # Reassign object_id by descending corner count (MC-Calib orders the biggest first).
    order = sorted(range(len(built)), key=lambda i: -len(built[i]["pts_3d"]))
    merged_objects: List[Object3D] = []
    remap: Dict[int, Tuple[int, int]] = {}
    for new_id, i in enumerate(order):
        b = built[i]
        merged_objects.append(Object3D(
            object_id=new_id, board_ids=b["board_ids"], ref_board_id=b["ref_board_id"],
            T_co_b=b["T_co_b"], pts_3d=b["pts_3d"],
            pts_obj_2_board=b["pts_obj_2_board"], pts_board_2_obj=b["pts_board_2_obj"],
        ))
        for oid, off in b["offsets"].items():
            remap[oid] = (new_id, off)
    return merged_objects, remap


def remap_object_obs(object_obs: List[ObjectObs], remap: Dict[int, Tuple[int, int]]
                     ) -> List[ObjectObs]:
    """Rewrite each :class:`ObjectObs` onto the merged objects: relabel its ``object_id``
    and shift its ``point_rows`` by the merged-cloud offset recorded in ``remap``."""
    out: List[ObjectObs] = []
    for o in object_obs:
        new_oid, offset = remap[o.object_id]
        out.append(ObjectObs(
            cam_id=o.cam_id, frame_id=o.frame_id, object_id=new_oid,
            point_rows=np.asarray(o.point_rows, int) + offset,
            pts_2d=o.pts_2d, T_c_o=o.T_c_o, image_path=o.image_path,
        ))
    return out
