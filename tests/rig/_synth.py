"""Synthetic multi-camera rig generator for rig-calibration tests.

Places N cameras at known extrinsics looking at a fused multi-board object, samples
random object poses over K frames, and projects with a RadTan model (+ optional pixel
noise). Ground-truth extrinsics are returned so every stage can be checked.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ds_msp.models.radtan import RadTanModel
from ds_msp.rig.types import Object3D, ObjectObs


def _grid_board(nx=4, ny=4, pitch=0.1) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(nx) * pitch, np.arange(ny) * pitch)
    return np.c_[xs.ravel(), ys.ravel(), np.zeros(nx * ny)]


def make_object(board_poses: Dict[int, np.ndarray], nx=4, ny=4, pitch=0.1) -> Object3D:
    """Fuse boards (given their board->object 4x4 poses) into one Object3D."""
    pts, rows, b2o = [], [], {}
    P_b = _grid_board(nx, ny, pitch)
    for bid, T in board_poses.items():
        P_o = (T @ np.c_[P_b, np.ones(len(P_b))].T).T[:, :3]
        for k, p in enumerate(P_o):
            b2o[(bid, k)] = len(pts)
            rows.append((bid, k))
            pts.append(p)
    return Object3D(object_id=0, board_ids=sorted(board_poses),
                    ref_board_id=min(board_poses),
                    T_co_b=dict(board_poses), pts_3d=np.array(pts),
                    pts_obj_2_board=np.array(rows, int), pts_board_2_obj=b2o)


def make_rig(n_cam=3, n_frame=40, noise_px=0.0, seed=0, w=1280, h=960,
             multi_board=True, model_factory=None, outlier_frac=0.0, outlier_px=40.0
             ) -> Tuple[Object3D, List[ObjectObs], Dict, Dict, Dict]:
    """Return ``(object, object_obs, img_size, gt_extrinsics, gt_models)``.

    ``gt_extrinsics[c]`` is the ground-truth ``T_c_g`` (group-ref -> camera, cam 0 = id).
    ``gt_models[c]`` is the ground-truth camera model used to project for camera ``c``.

    ``model_factory(cam_id, rng) -> CameraModel`` lets the caller represent cameras with
    any model (DS/UCM/EUCM/KB/...) to exercise model-agnosticism. Defaults to RadTan.
    ``outlier_frac`` corrupts that fraction of detections with a gross ``outlier_px`` shift
    (mis-decoded corners) to exercise robust weighting.
    """
    rng = np.random.default_rng(seed)
    f = 800.0
    if model_factory is None:
        def model_factory(cam_id, rng):
            return RadTanModel(f, f, w / 2, h / 2, -0.05, 0.01, 0.0, 0.0, 0.0)
    gt_models = {c: model_factory(c, rng) for c in range(n_cam)}

    from ds_msp.core.lie import so3_exp as _exp
    boards = {0: np.eye(4)}
    if multi_board:
        # Genuinely 3D target (like a calibration cube): tilt the extra boards and offset
        # them in depth so every camera — even an obliquely angled one — sees a non-planar
        # point cloud. A near-coplanar target would leave each camera's focal ambiguous.
        T1 = np.eye(4)
        T1[:3, :3] = _exp([0.0, 0.6, 0.0])
        T1[:3, 3] = [0.45, 0.0, 0.25]
        T2 = np.eye(4)
        T2[:3, :3] = _exp([-0.6, 0.0, 0.0])
        T2[:3, 3] = [0.0, 0.45, 0.25]
        boards[1] = T1
        boards[2] = T2
    obj = make_object(boards)

    # cameras: ref at origin, others mildly spread on an arc, all keeping the object
    # well inside the frame (so every camera is well-conditioned for intrinsics).
    gt_ext: Dict[int, np.ndarray] = {0: np.eye(4)}
    for c in range(1, n_cam):
        ang = np.deg2rad(8.0 * c)
        R = np.array([[np.cos(ang), 0, np.sin(ang)],
                      [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
        t = np.array([0.15 * c, 0.0, 0.0])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        gt_ext[c] = T

    object_obs: List[ObjectObs] = []
    for fr in range(n_frame):
        # random object pose in front of the rig
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        ang = rng.uniform(-0.55, 0.55)
        from ds_msp.core.lie import so3_exp
        Rg = so3_exp(axis * ang)
        # sweep the target over a good fraction of the image (so each model's distortion
        # is observable -> focal well constrained) while keeping most views full.
        tg = np.array([rng.uniform(-0.35, 0.35), rng.uniform(-0.3, 0.3),
                       rng.uniform(1.8, 2.6)])
        T_g_o = np.eye(4)
        T_g_o[:3, :3] = Rg
        T_g_o[:3, 3] = tg
        Xg = (T_g_o[:3, :3] @ obj.pts_3d.T).T + T_g_o[:3, 3]
        for c in range(n_cam):
            Xc = (gt_ext[c][:3, :3] @ Xg.T).T + gt_ext[c][:3, 3]
            uv, valid = gt_models[c].project(Xc)
            inb = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            rows = np.where(inb)[0]
            if len(rows) < 6:
                continue
            pts = uv[rows] + (rng.normal(scale=noise_px, size=(len(rows), 2))
                              if noise_px else 0.0)
            if outlier_frac:
                bad = rng.random(len(rows)) < outlier_frac
                pts[bad] += rng.uniform(-outlier_px, outlier_px, size=(int(bad.sum()), 2))
            object_obs.append(ObjectObs(cam_id=c, frame_id=fr, object_id=0,
                                        point_rows=rows, pts_2d=pts))
    img_size = {c: (w, h) for c in range(n_cam)}
    return obj, object_obs, img_size, gt_ext, gt_models


def _default_t_b0_b1() -> np.ndarray:
    """Ground-truth rigid transform of board 1 relative to board 0 (``T_b0_b1``).

    The two single-board objects in :func:`make_non_overlapping_rig` are one rigid
    body: a point on board 1 maps into board-0 (== object-0) coordinates by
    ``X_b0 = T_b0_b1 @ [X_b1; 1]``. A test can import this and compare it against the
    inter-board geometry recovered by the multi-object merge. The rotation is a
    non-trivial ~30 deg so both hand-eye and merge are genuinely exercised.
    """
    from ds_msp.core.lie import so3_exp
    T = np.eye(4)
    T[:3, :3] = so3_exp([0.15, -0.45, 0.30])   # ||.|| ~ 0.56 rad ~ 32 deg
    T[:3, 3] = [0.60, -0.10, 0.20]
    return T


def make_non_overlapping_rig(n_frame=40, noise_px=0.0, seed=0, w=1280, h=800,
                             model_factory=None, t_b0_b1=None
                             ) -> Tuple[List[Object3D], List[ObjectObs], Dict, Dict, Dict]:
    """Two-camera, non-overlapping rig — the multi-object-merge acceptance topology.

    Each camera sees a *different* single-board object that is never co-observed with
    the other, mirroring the real dataset where each camera looks at its own board.
    The two boards are one rigid body linked by a fixed known ``T_b0_b1`` (board 1 in
    board-0 coordinates), so the merge must recover that inter-board geometry from the
    two independent object motions (hand-eye across the two cameras).

    Returns ``(objects, object_obs, img_size, gt_ext, gt_models)`` where:

    * ``objects`` — two separate :class:`Object3D`: ``object_id=0`` wraps board 0
      (``board_ids=[0]``) and ``object_id=1`` wraps board 1 (``board_ids=[1]``); board
      ids are globally unique/disjoint.
    * ``object_obs`` — camera 0 observes ONLY object 1 (board 1); camera 1 observes
      ONLY object 0 (board 0). Each carries ``T_c_o`` (object -> camera).
    * ``img_size`` — ``{0: (w, h), 1: (w, h)}``.
    * ``gt_ext`` — ``{0: eye, 1: T_c1_g}`` (``T_c_g``, group-ref -> camera; cam 0 is the
      reference == identity, cam 1 a known extrinsic with meaningful rotation+translation).
    * ``gt_models`` — ``{0: RadTan, 1: RadTan}`` (mild distortion).

    Geometry (group-reference == world): per frame ``f`` the physical rig has object-0
    pose ``G(f)`` (a diverse random SE3 in front of the cameras) and object-1 pose
    ``G(f) @ T_b0_b1``. Hence ``T_c1_o0 = gt_ext[1] @ G(f)`` (cam 1 sees obj 0) and
    ``T_c0_o1 = gt_ext[0] @ (G(f) @ T_b0_b1)`` (cam 0 sees obj 1). Points are projected
    through the observing camera's RadTan model (+ optional ``noise_px`` gaussian) and
    filtered to the image, exactly as :func:`make_rig`.

    ``T_b0_b1`` is exposed two ways: pass ``t_b0_b1`` to override it, and when ``None``
    the fixed value returned by :func:`_default_t_b0_b1` is used — import that helper in a
    test to check the merge recovered the right inter-board transform.
    """
    from ds_msp.core.lie import so3_exp

    rng = np.random.default_rng(seed)
    f = 800.0
    if model_factory is None:
        def model_factory(cam_id, rng):
            return RadTanModel(f, f, w / 2, h / 2, -0.05, 0.01, 0.0, 0.0, 0.0)
    gt_models = {c: model_factory(c, rng) for c in range(2)}

    # Two single-board objects with globally-unique board ids (board 0 -> obj 0,
    # board 1 -> obj 1). make_object hard-codes object_id=0; retag object 1.
    obj0 = make_object({0: np.eye(4)})
    obj1 = make_object({1: np.eye(4)})
    obj1.object_id = 1
    objects = [obj0, obj1]

    if t_b0_b1 is None:
        t_b0_b1 = _default_t_b0_b1()
    T_b0_b1 = np.asarray(t_b0_b1, dtype=np.float64)

    # Two cameras: ref cam 0 = identity, cam 1 a known extrinsic (T_c_g) with a
    # meaningful ~22 deg rotation and a real baseline so hand-eye is well conditioned
    # yet every frame's board still lands in-frame for both cameras.
    T_c1_g = np.eye(4)
    T_c1_g[:3, :3] = so3_exp([0.08, 0.35, -0.12])   # ~0.38 rad ~ 22 deg
    T_c1_g[:3, 3] = [0.35, -0.05, 0.08]
    gt_ext: Dict[int, np.ndarray] = {0: np.eye(4), 1: T_c1_g}

    # cam -> (object it observes, that object's points).
    seen = {0: obj1, 1: obj0}

    object_obs: List[ObjectObs] = []
    for fr in range(n_frame):
        # G(f): diverse random rig motion in front of the cameras (axis-angle up to
        # ~40 deg — rotational diversity is what hand-eye/merge need).
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        ang = rng.uniform(0.2, np.deg2rad(40.0))
        G = np.eye(4)
        G[:3, :3] = so3_exp(axis * ang)
        G[:3, 3] = [rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(1.9, 2.5)]

        # world poses of the two rigidly-linked objects.
        Tg_o = {0: G, 1: G @ T_b0_b1}
        for c in (0, 1):
            obj = seen[c]
            T_c_o = gt_ext[c] @ Tg_o[obj.object_id]      # object -> camera
            Xc = (T_c_o[:3, :3] @ obj.pts_3d.T).T + T_c_o[:3, 3]
            uv, valid = gt_models[c].project(Xc)
            inb = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            rows = np.where(inb)[0]
            if len(rows) < 6:
                continue
            pts = uv[rows] + (rng.normal(scale=noise_px, size=(len(rows), 2))
                              if noise_px else 0.0)
            object_obs.append(ObjectObs(cam_id=c, frame_id=fr, object_id=obj.object_id,
                                        point_rows=rows, pts_2d=pts, T_c_o=T_c_o))
    img_size = {0: (w, h), 1: (w, h)}
    return objects, object_obs, img_size, gt_ext, gt_models
