"""Hand-eye linking for non-overlapping camera groups
(``handeyeBootstraptTranslationCalibration``, geometrytools.cpp:621).

When ``rig.extrinsics`` yields more than one connected component (camera groups that
never co-observe an object), the inter-group transform cannot come from direct
covisibility. It is recovered from the *motion* of the object as seen by a camera in
each group: cluster the motions, run repeated Tsai hand-eye solves on diverse motion
subsets, gate by rotational consistency (15 deg), and aggregate.
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from ..core.lie import so3_log
from .averaging import average_rotation, average_translation


def _rot_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def _tsai_solve(motions_a: List[np.ndarray], motions_b: List[np.ndarray],
                min_angle: float = 1e-3) -> np.ndarray:
    """Solve ``M_b = X · M_a · X⁻¹`` for the rigid transform ``X = T_b_a`` from paired
    relative motions (Tsai–Lenz). Rotation: ``log(R_b) = R_X · log(R_a)`` solved by SVD
    (Kabsch); translation: ``(R_{M_b} − I) t_X = R_X t_{M_a} − t_{M_b}`` by least squares.

    A direct solve avoids ``cv2.calibrateHandEye``'s absolute-pose convention (it rebuilds
    motions internally, so feeding it motions is wrong). Motions with near-zero rotation
    carry no constraint and are dropped (need rotational diversity).
    """
    A, B = [], []
    for Ma, Mb in zip(motions_a, motions_b):
        a = so3_log(Ma[:3, :3])
        if np.linalg.norm(a) < min_angle:
            continue
        A.append(a)
        B.append(so3_log(Mb[:3, :3]))
    if len(A) < 2:
        return np.eye(4)
    A, B = np.array(A), np.array(B)
    U, _, Vt = np.linalg.svd(B.T @ A)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    Rx = U @ D @ Vt
    C, d = [], []
    for Ma, Mb in zip(motions_a, motions_b):
        C.append(Mb[:3, :3] - np.eye(3))
        d.append(Rx @ Ma[:3, 3] - Mb[:3, 3])
    tx = np.linalg.lstsq(np.vstack(C), np.concatenate(d), rcond=None)[0]
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = Rx, tx
    return X


def handeye_bootstrap(poses_a: List[np.ndarray], poses_b: List[np.ndarray], *,
                      nb_cluster: int = 20, nb_pick: int = 6, nb_it: int = 200,
                      gate_deg: float = 15.0, seed: int = 0) -> np.ndarray:
    """Estimate ``T_b_a`` (group-a-ref -> group-b-ref) from paired absolute object poses
    ``poses_a[i]`` / ``poses_b[i]`` (object->camera 4x4) observed in the same frame.
    """
    n = len(poses_a)
    if n < 3:
        return _single_handeye(poses_a, poses_b)
    feats = np.array([np.r_[Ta[:3, 3], Tb[:3, 3]] for Ta, Tb in zip(poses_a, poses_b)],
                     np.float32)
    k = min(nb_cluster, n)
    _, labels, _ = cv2.kmeans(
        feats, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 0.01), 5,
        cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    by_cluster: Dict[int, List[int]] = {}
    for i, c in enumerate(labels):
        by_cluster.setdefault(int(c), []).append(i)

    rng = np.random.default_rng(seed)
    Rs, ts = [], []
    for _ in range(nb_it):
        clusters = rng.permutation(list(by_cluster))[:nb_pick]
        sel = [int(rng.choice(by_cluster[c])) for c in clusters]
        if len(sel) < 3:
            continue
        # relative motions between consecutive selected poses
        Ma_list = [poses_a[v] @ np.linalg.inv(poses_a[u]) for u, v in zip(sel[:-1], sel[1:])]
        Mb_list = [poses_b[v] @ np.linalg.inv(poses_b[u]) for u, v in zip(sel[:-1], sel[1:])]
        X = _tsai_solve(Ma_list, Mb_list)
        if np.allclose(X, np.eye(4)):
            continue
        # consistency: X must satisfy Mb @ X ~ X @ Ma for each motion
        worst = max((_rot_angle_deg((np.linalg.inv(Mb @ X) @ (X @ Ma))[:3, :3])
                     for Ma, Mb in zip(Ma_list, Mb_list)), default=180.0)
        if worst < gate_deg:
            Rs.append(X[:3, :3])
            ts.append(X[:3, 3])

    if len(Rs) > 3:
        R = average_rotation(Rs)
        t = average_translation(np.array(ts))
    else:
        return _single_handeye(poses_a, poses_b)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _single_handeye(poses_a, poses_b) -> np.ndarray:
    """Fallback solve over all consecutive motions (no RANSAC clustering)."""
    if len(poses_a) < 2:
        return np.eye(4)
    Ma = [poses_a[v] @ np.linalg.inv(poses_a[u])
          for u, v in zip(range(len(poses_a) - 1), range(1, len(poses_a)))]
    Mb = [poses_b[v] @ np.linalg.inv(poses_b[u])
          for u, v in zip(range(len(poses_b) - 1), range(1, len(poses_b)))]
    return _tsai_solve(Ma, Mb)


def link_groups(groups: List[List[int]], extr: Dict[int, np.ndarray],
                object_obs, *, he_approach: int = 0) -> Dict[int, np.ndarray]:
    """Merge non-overlapping camera groups into one frame.

    Each group's cameras keep their intra-group ``T_c_g`` (relative to the group's own
    reference). We estimate the transform from each non-base group's reference camera to
    the base group's reference camera via hand-eye on the object motion seen by both,
    then re-base every camera in that group. Returns a flat ``{cam: T_c_g0}``.

    ``he_approach`` mirrors MC-Calib's config key: ``0`` = the robust bootstrap
    (cluster + RANSAC Tsai, ``handeye_bootstrap``); ``1`` = the traditional single Tsai
    solve over all motions (``_single_handeye``).
    """
    solve = (handeye_bootstrap if int(he_approach) == 0 else
             (lambda pa, pb: _single_handeye(pa, pb)))
    base = groups[0]
    base_ref = base[0]
    out = {c: extr[c].copy() for c in base}

    # object pose per (group-ref camera, frame), needed to pair motions across groups
    def ref_poses(group):
        """``(ref_cam, {frame_id: T_c_o})`` for ``group``'s reference (first) camera."""
        ref = group[0]
        poses = {}
        for o in object_obs:
            if o.cam_id == ref and o.T_c_o is not None:
                poses[o.frame_id] = o.T_c_o
        return ref, poses

    base_ref, base_poses = ref_poses(base)
    for group in groups[1:]:
        gref, gposes = ref_poses(group)
        common = sorted(set(base_poses) & set(gposes))
        if len(common) >= 3:
            pa = [base_poses[f] for f in common]
            pb = [gposes[f] for f in common]
            T_gref_baseref = solve(pa, pb)                     # base-ref -> g-ref
        else:
            T_gref_baseref = np.eye(4)
        # re-base: camera in group has T_c_gref; want T_c_baseref = T_c_gref @ T_gref_baseref
        for c in group:
            out[c] = extr[c] @ T_gref_baseref
    return out
