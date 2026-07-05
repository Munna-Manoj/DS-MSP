"""Reader for MC-Calib's Blender benchmark data (OpenCV YAML via ``cv2.FileStorage``).

Loads the detected keypoints, the fused 3D object, the synthetic ground truth, and
MC-Calib's own calibration result — everything needed to drive ``rig.calibrate_rig`` on
identical 2D observations and compare extrinsics against both references.

File formats (per scenario directory ``<scn>/Results/``):

* ``detected_keypoints_data.yml`` — per ``camera_<i>``: ``frame_idxs`` / ``board_idxs``
  (flat, one entry per board-observation) and ``pts_2d`` / ``charuco_idxs`` (parallel
  sequences of flattened ``[u,v,...]`` and corner-id arrays).
* ``calibrated_objects_data.yml`` — ``object_<j>.points``: a ``(5, N)`` matrix whose rows
  are ``[x, y, z, board_id, corner_id]`` in the object frame.
* ``calibrated_cameras_data.yml`` — per camera: ``camera_matrix``, ``distortion_vector``,
  ``camera_pose_matrix`` (group-ref -> camera), ``img_width/height``.
* ``<scn>/GroundTruth.yml`` — ``K_<i>`` and ``P_<i>`` (4x4 pose) per camera.
"""

from __future__ import annotations

import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..models.radtan import RadTanModel
from ..data.observations import Object3D, ObjectObs


def _seq(node) -> list:
    return [node.at(i) for i in range(node.size())]


def _flat(node) -> np.ndarray:
    """Read an inline YAML sequence ``[ ... ]`` as a flat float array."""
    return np.array([node.at(i).real() for i in range(node.size())], float)


@dataclass
class CameraGT:
    """One camera's intrinsics + pose as read from an MC-Calib YAML file.

    Used for both the synthetic ground truth (``GroundTruth.yml``) and MC-Calib's
    own calibration result (``calibrated_cameras_data.yml``); the two files share
    this shape, differing only in which fields are populated.

    Parameters
    ----------
    K : (3, 3) array
        Pinhole intrinsic matrix.
    dist : array or None
        Raw distortion coefficient vector in the file's native order, or ``None``
        when the source file has none (e.g. ``GroundTruth.yml``).
    pose : (4, 4) array
        Camera pose as stored by the source file (group-reference -> camera
        convention for MC-Calib's own result; see the module docstring).
    model_name : str or None, default None
        Canonical DS-MSP model name (e.g. ``"ds"``, ``"kb"``) when the file states
        it explicitly (``camera_model`` string or MC-Calib's ``distortion_type``
        int); ``None`` for a plain MC-Calib file where the caller must infer the
        model from ``len(dist)``.
    """

    K: np.ndarray
    dist: Optional[np.ndarray]
    pose: np.ndarray            # 4x4 (group-ref -> camera convention, as stored)
    model_name: Optional[str] = None   # canonical model the intrinsics are stored in, when the
                                       # file states it (camera_model string / distortion_type int);
                                       # None for a plain MC-Calib file (caller infers from dist len)


@dataclass
class Scenario:
    """A fully loaded MC-Calib Blender benchmark scenario.

    Bundles everything :func:`load_scenario` reads from one
    ``Blender_Images/Scenario_*`` directory: the fused calibration object, the
    2D detections that drive ``ds_msp.rig.calibrate_rig``, and both reference
    calibrations (synthetic ground truth and MC-Calib's own result) to compare
    against.

    Parameters
    ----------
    name : str
        Scenario directory name (e.g. ``"Scenario_1"``).
    object : Object3D
        Fused multi-board calibration object shared by every camera.
    object_obs : list of ObjectObs
        One entry per ``(camera, frame)`` observation of the object.
    cam_ids : list of int
        0-based camera ids present in the scenario, sorted.
    img_size : dict of int to (int, int)
        Per-camera ``(width, height)`` in pixels.
    gt : dict of int to CameraGT
        Synthetic ground-truth intrinsics/pose per camera id, from
        ``GroundTruth.yml``; empty if the file is absent.
    mccalib : dict of int to CameraGT
        MC-Calib's own calibrated intrinsics/pose per camera id.
    mccalib_rms : dict of int to float
        Per-camera RMS reprojection error reported by MC-Calib, pixels; empty
        unless populated by the caller (:func:`load_scenario` leaves it empty).
    """

    name: str
    object: Object3D
    object_obs: List[ObjectObs]                 # one per (camera, frame)
    cam_ids: List[int]
    img_size: Dict[int, Tuple[int, int]]
    gt: Dict[int, CameraGT]
    mccalib: Dict[int, CameraGT]
    mccalib_rms: Dict[int, float]


def _load_object(path: str) -> Object3D:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    M = fs.getNode("object_0").getNode("points").mat()          # (5, N)
    fs.release()
    xyz = M[:3].T.astype(float)
    board_ids = M[3].astype(int)
    corner_ids = M[4].astype(int)
    b2o = {(int(b), int(c)): i for i, (b, c) in enumerate(zip(board_ids, corner_ids))}
    boards = sorted(set(int(b) for b in board_ids))
    return Object3D(
        object_id=0, board_ids=boards, ref_board_id=min(boards),
        T_co_b={b: np.eye(4) for b in boards},                  # baked into pts_3d
        pts_3d=xyz, pts_obj_2_board=np.c_[board_ids, corner_ids],
        pts_board_2_obj=b2o,
    )


def _load_detections(path: str, obj: Object3D):
    """Return ``{cam_id: {frame_id: (point_rows, pts_2d)}}`` and per-cam image size."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    nb = int(fs.getNode("nb_camera").real())
    per_cam: Dict[int, Dict[int, Tuple[list, list]]] = {}
    img_size: Dict[int, Tuple[int, int]] = {}
    for ci in range(nb):
        cn = fs.getNode(f"camera_{ci}")
        img_size[ci] = (int(cn.getNode("img_width").real()),
                        int(cn.getNode("img_height").real()))
        frame_idxs = _flat(cn.getNode("frame_idxs")).astype(int)
        board_idxs = _flat(cn.getNode("board_idxs")).astype(int)
        pts_seq = _seq(cn.getNode("pts_2d"))
        cid_seq = _seq(cn.getNode("charuco_idxs"))
        frames: Dict[int, Tuple[list, list]] = {}
        for k in range(len(frame_idxs)):
            f = int(frame_idxs[k])
            b = int(board_idxs[k])
            pts = _flat(pts_seq[k]).reshape(-1, 2)
            cids = _flat(cid_seq[k]).astype(int).ravel()
            rows, uvs = frames.setdefault(f, ([], []))
            for cid, uv in zip(cids, pts):
                key = (b, int(cid))
                if key in obj.pts_board_2_obj:
                    rows.append(obj.pts_board_2_obj[key])
                    uvs.append(uv)
        per_cam[ci] = frames
    fs.release()
    return per_cam, img_size


def _camera_model_field(cn) -> Optional[str]:
    """The canonical model an MC-Calib ``camera_<i>`` node states, or ``None`` for a plain file.

    Prefers DS-MSP's ``camera_model`` string (it names *every* model — the only field that can
    distinguish ucm/eucm/ds/dsplus, all of which MC-Calib writes with ``distortion_type`` 1).
    Falls back to MC-Calib's own ``distortion_type``/``disto_type`` int (0 Brown→radtan,
    1 Kannala→kb, 2 double-sphere→ds). Unknown strings are ignored (treated as "not stated")."""
    from ..models.registry import canonical_name
    cm = cn.getNode("camera_model")
    if cm is not None and not cm.empty() and cm.isString():
        try:
            return canonical_name(cm.string())
        except KeyError:
            return None
    for key in ("distortion_type", "disto_type"):
        n = cn.getNode(key)
        if n is not None and not n.empty() and not n.isString() and not n.isSeq() and not n.isMap():
            v = int(round(n.real()))
            if v in (0, 1, 2):
                return {0: "radtan", 1: "kb", 2: "ds"}[v]
    return None


def _load_cameras(path: str) -> Tuple[Dict[int, CameraGT], Dict[int, float]]:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    nb = int(fs.getNode("nb_camera").real())
    cams: Dict[int, CameraGT] = {}
    for ci in range(nb):
        cn = fs.getNode(f"camera_{ci}")
        K = cn.getNode("camera_matrix").mat()
        dist = cn.getNode("distortion_vector").mat()
        pose = cn.getNode("camera_pose_matrix").mat()
        cams[ci] = CameraGT(K=K, dist=dist.ravel() if dist is not None else None, pose=pose,
                            model_name=_camera_model_field(cn))
    fs.release()
    return cams, {}


def _load_groundtruth(path: str) -> Dict[int, CameraGT]:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    nb = int(fs.getNode("nb_camera").real())
    gt: Dict[int, CameraGT] = {}
    for ci in range(1, nb + 1):
        K = fs.getNode(f"K_{ci}").mat()
        P = fs.getNode(f"P_{ci}").mat()
        gt[ci - 1] = CameraGT(K=K, dist=None, pose=P)
    fs.release()
    return gt


def load_scenario(scn_dir: str) -> Scenario:
    """Load a ``Blender_Images/Scenario_*`` directory into a :class:`Scenario`."""
    name = os.path.basename(scn_dir.rstrip("/"))
    results = os.path.join(scn_dir, "Results")
    obj = _load_object(os.path.join(results, "calibrated_objects_data.yml"))
    per_cam, img_size = _load_detections(
        os.path.join(results, "detected_keypoints_data.yml"), obj)
    mccalib, _ = _load_cameras(os.path.join(results, "calibrated_cameras_data.yml"))
    gt_path = os.path.join(scn_dir, "GroundTruth.yml")
    gt = _load_groundtruth(gt_path) if os.path.exists(gt_path) else {}

    object_obs: List[ObjectObs] = []
    for cam_id, frames in per_cam.items():
        for frame_id, (rows, uvs) in frames.items():
            if not rows:
                continue
            object_obs.append(ObjectObs(
                cam_id=cam_id, frame_id=frame_id, object_id=0,
                point_rows=np.array(rows, int), pts_2d=np.array(uvs, float),
            ))
    return Scenario(
        name=name, object=obj, object_obs=object_obs,
        cam_ids=sorted(per_cam), img_size=img_size,
        gt=gt, mccalib=mccalib, mccalib_rms={},
    )


def _T_to_rodrigues(T: np.ndarray):
    """4x4 transform -> (rvec(3,), tvec(3,)) like MC-Calib's getPoseVec."""
    rvec = cv2.Rodrigues(np.asarray(T[:3, :3], float))[0].ravel()
    return rvec, np.asarray(T[:3, 3], float).ravel()


def save_mccalib_cameras(rig, path: str, *, cam_groups: Optional[Dict[int, int]] = None,
                         cam_order=None) -> None:
    """Write ``calibrated_cameras_data.yml`` in MC-Calib's exact OpenCV-YAML schema.

    Per ``Calibration::saveCamerasParams`` (McCalib.cpp:386): a top-level ``nb_camera`` and,
    for each camera, a ``camera_<i>`` map with ``camera_matrix`` (3x3), ``distortion_vector``
    (1xN), ``camera_model`` (string), ``camera_group``, ``img_width``, ``img_height`` and
    ``camera_pose_matrix`` — the **camera->world** pose, i.e. ``inv(T_c_g)`` (MC-Calib writes
    ``getCameraPoseMat().inv()``; ``RigState.T_c_g`` is the world->camera projection extrinsic).
    """
    from ..models.registry import mccalib_name
    order = list(cam_order) if cam_order is not None else sorted(rig.cameras)
    groups = cam_groups or {c: 0 for c in order}
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("nb_camera", int(len(order)))
    for c in order:
        model = rig.cameras[c]
        w, h = rig.img_size.get(c, (0, 0))
        cam2world = np.linalg.inv(np.asarray(rig.T_c_g[c], float))
        fs.startWriteStruct(f"camera_{c}", cv2.FileNode_MAP)
        fs.write("camera_matrix", np.asarray(model.K, float))
        fs.write("distortion_vector",
                 np.asarray(model.distortion, float).reshape(1, -1))
        # distortion_type keeps MC-Calib's field present for every camera (0=Brown/perspective,
        # 1=Kannala/fisheye-family); camera_model carries the exact model for the DS-MSP
        # extension models (ucm/eucm/ds/ocam) MC-Calib's two-model enum cannot name.
        fs.write("distortion_type", 0 if model.name == "radtan" else 1)
        fs.write("camera_model", mccalib_name(model.name))
        fs.write("camera_group", int(groups.get(c, 0)))
        fs.write("img_width", int(w))
        fs.write("img_height", int(h))
        fs.write("camera_pose_matrix", cam2world)
        fs.endWriteStruct()
    fs.release()


#: Per-model ``distortion_vector`` field order -- the exact inverse of each model's own
#: ``.distortion`` property (see e.g. ``ds_msp/models/dsplus.py``), needed to reconstruct a
#: real ``CameraModel`` instance from ``calibrated_cameras_data.yml``'s raw arrays.
_DISTORTION_LAYOUT: Dict[str, Tuple[str, ...]] = {
    "ds": ("xi", "alpha"),
    "dsplus": ("alpha", "lambda1", "lambda2", "tau_x", "tau_y"),
    "kb": ("k1", "k2", "k3", "k4"),
    "radtan": ("k1", "k2", "p1", "p2", "k3"),
    "ucm": ("alpha",),
    "eucm": ("alpha", "beta"),
    "ocam": ("c", "d", "e", "a0", "a1", "a2", "a3", "a4"),
}


def load_camera(path: str, cam_id: int):
    """Read one camera's calibrated intrinsics back into a ready ``CameraModel`` instance --
    the ``ds_msp.io.mccalib`` analogue of :func:`ds_msp.io.kalibr.load_kalibr`.

    Needs ``camera_model`` (or the legacy ``distortion_type``/``disto_type`` int) to be
    present in the file to know which model to reconstruct -- see :func:`_camera_model_field`.
    Unlike Kalibr's single-camera camchain, one MC-Calib file holds every camera in the rig,
    so this takes the 0-based ``cam_id`` (matching ``camera_<cam_id>`` in the file, the same
    indexing :func:`save_mccalib_cameras` writes)."""
    from ..models.registry import model_class
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    cn = fs.getNode(f"camera_{cam_id}")
    if cn.empty():
        fs.release()
        raise KeyError(f"no camera_{cam_id} in {path}")
    name = _camera_model_field(cn)
    K = cn.getNode("camera_matrix").mat()
    dist_node = cn.getNode("distortion_vector")
    dist = dist_node.mat() if not dist_node.empty() else None
    fs.release()
    if name is None:
        raise ValueError(f"camera_{cam_id} in {path} states no camera_model/distortion_type "
                         f"-- cannot tell which model to reconstruct")
    layout = _DISTORTION_LAYOUT[name]
    dist = np.asarray(dist, dtype=float).ravel() if dist is not None else np.zeros(0)
    if len(dist) != len(layout):
        raise ValueError(f"camera_{cam_id}: model {name!r} expects {len(layout)} distortion "
                         f"values {layout}, file has {len(dist)}")
    kwargs = dict(zip(layout, dist.tolist()))
    if name == "ocam":
        kwargs["cx"], kwargs["cy"] = float(K[0, 2]), float(K[1, 2])
    else:
        kwargs.update(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]))
    return model_class(name)(**kwargs)


def save_mccalib_objects(obj: Object3D, path: str) -> None:
    """Write ``calibrated_objects_data.yml`` (McCalib.cpp:427): per ``object_<j>`` a
    ``points`` matrix of shape ``(5, N)`` whose rows are ``[x, y, z, board_id, corner_id]``."""
    rows = obj.pts_obj_2_board                                   # (N,2) = [board_id, corner_id]
    pts = np.vstack([obj.pts_3d.T, rows[:, 0], rows[:, 1]]).astype(np.float32)  # (5,N)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.startWriteStruct(f"object_{obj.object_id}", cv2.FileNode_MAP)
    fs.write("points", pts)
    fs.endWriteStruct()
    fs.release()


def save_mccalib_object_poses(rig, path: str, *, object_id: int = 0) -> None:
    """Write ``calibrated_objects_pose_data.yml`` (McCalib.cpp:469): per ``object_<j>`` a
    ``poses`` matrix ``(6, M)`` of ``[rx, ry, rz, tx, ty, tz]`` over the frames the object is
    seen, ``T_g_o`` (object->group)."""
    keys = sorted(k for k in rig.object_poses if k[0] == object_id)
    pose_mat = np.zeros((6, len(keys)), float)
    for a, key in enumerate(keys):
        rvec, tvec = _T_to_rodrigues(rig.object_poses[key])
        pose_mat[:3, a] = rvec
        pose_mat[3:, a] = tvec
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.startWriteStruct(f"object_{object_id}", cv2.FileNode_MAP)
    fs.write("poses", pose_mat)
    fs.endWriteStruct()
    fs.release()


def _obs_reprojection(rig, o):
    """Detected vs reprojected pixels for one ObjectObs: ``(uv_det, uv_rep, valid)``.

    Reproject the object's 3D points through ``T_c_o = T_c_g[cam] @ T_g_o`` and the camera's
    model — the same composition MC-Calib uses (``getCameraPoseMat * getPoseInGroupMat``)."""
    cam = o.cam_id
    key = (o.object_id, o.frame_id)
    if cam not in rig.cameras or key not in rig.object_poses or cam not in rig.T_c_g:
        return None
    obj = next(iter(rig.objects.values()))
    X = obj.pts_3d[o.point_rows]
    T_c_o = np.asarray(rig.T_c_g[cam], float) @ np.asarray(rig.object_poses[key], float)
    Xc = (T_c_o[:3, :3] @ X.T).T + T_c_o[:3, 3]
    uv_rep, valid = rig.cameras[cam].project(Xc)
    return np.asarray(o.pts_2d, float), uv_rep, valid


def save_mccalib_reprojection_error(rig, object_obs, path: str, *, cam_group: int = 0) -> None:
    """Write ``reprojection_error_data.yml`` in MC-Calib's schema (McCalib.cpp:2278):
    ``nb_camera_group`` then per ``camera_group_<g>`` a ``frame_<idx>`` map holding, per
    ``camera_<id>``, ``nb_pts`` and an ``error_list`` (1xN per-point pixel distances), plus a
    ``camera_list`` per frame and a ``frame_list`` for the group."""
    by_frame: Dict[int, List] = {}
    for o in object_obs:
        by_frame.setdefault(o.frame_id, []).append(o)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("nb_camera_group", 1)
    fs.startWriteStruct(f"camera_group_{cam_group}", cv2.FileNode_MAP)
    frame_list = []
    for fr in sorted(by_frame):
        cam_list = []
        fs.startWriteStruct(f"frame_{fr}", cv2.FileNode_MAP)
        for o in by_frame[fr]:
            rep = _obs_reprojection(rig, o)
            if rep is None:
                continue
            uv_det, uv_rep, valid = rep
            err = np.linalg.norm(uv_rep[valid] - uv_det[valid], axis=1)
            cam_list.append(o.cam_id)
            fs.startWriteStruct(f"camera_{o.cam_id}", cv2.FileNode_MAP)
            fs.write("nb_pts", int(valid.sum()))
            fs.write("error_list", err.reshape(1, -1).astype(np.float64))
            fs.endWriteStruct()
        fs.write("camera_list", np.asarray(cam_list, np.int32).reshape(-1, 1))
        fs.endWriteStruct()
        frame_list.append(fr)
    fs.write("frame_list", np.asarray(frame_list, np.int32).reshape(-1, 1))
    fs.endWriteStruct()
    fs.release()


def _obs_image_path(obs_list, image_root, cam, fr, cam_prefix="Cam_", ext="png"):
    """Source image for a ``(cam, frame)`` observation group, for overlay drawing.

    Prefer the path recorded at detection time (``ObjectObs.image_path``) — robust to
    ``detect_rig`` rebasing ``frame_id`` so it can drift from the raw filename. Fall back to
    filename candidates, **0-indexed first** then MC-Calib's 1-indexed Blender naming.
    """
    for o in obs_list:
        p = getattr(o, "image_path", None)
        if p and os.path.exists(p):
            return p
    for cand in (f"{fr:05d}.{ext}", f"{fr + 1:05d}.{ext}", f"{fr:06d}.{ext}", f"{fr + 1:06d}.{ext}"):
        p = os.path.join(image_root, f"{cam_prefix}{cam + 1:03d}", cand)
        if os.path.exists(p):
            return p
    return None


def save_reprojection_images(rig, object_obs, image_root: str, save_dir: str, *,
                             cam_prefix: str = "Cam_", ext: str = "png",
                             workers: Optional[int] = None, progress_cb=None) -> int:
    """Draw detected (green) vs reprojected (red) corners per frame and save under
    ``<save_dir>/Reprojection/<cam:03d>/<frame:06d>.jpg`` — the MC-Calib layout
    (McCalib.cpp:1923). The source image is the one recorded at detection (``image_path``),
    falling back to filename lookup. Returns the number of images written.

    Parallelised across (camera, frame) tasks the same way as ``detect.charuco.detect_rig`` —
    each task's ``cv2.imread``/``cv2.circle``/``cv2.imwrite`` releases the GIL and touches only
    its own image array, so this scales the same way corner detection does (measured ~12.7s
    serial for 255 images on this repo's real MC-Calib dataset; this used to run silently and
    serially *after* the whole bundle adjustment converged, which read as the live view "just
    sitting there doing nothing" once the optimizer was actually done).
    ``progress_cb(cam_id, i, n, frame_id)``, if given, fires once per completed image."""
    root = os.path.join(save_dir, "Reprojection")
    by_cf: Dict[Tuple[int, int], List] = {}
    for o in object_obs:
        by_cf.setdefault((o.cam_id, o.frame_id), []).append(o)
    items = list(by_cf.items())
    counts = Counter(cam for (cam, _fr), _ in items)
    seen: Dict[int, int] = Counter()
    lock = threading.Lock()

    def _do(entry) -> bool:
        (cam, fr), obs_list = entry
        if progress_cb is not None:
            with lock:
                seen[cam] += 1
                progress_cb(cam, seen[cam], counts[cam], fr)
        img_path = _obs_image_path(obs_list, image_root, cam, fr, cam_prefix, ext)
        if img_path is None:
            return False
        image = cv2.imread(img_path)
        if image is None:
            return False
        for o in obs_list:
            rep = _obs_reprojection(rig, o)
            if rep is None:
                continue
            uv_det, uv_rep, valid = rep
            for i in np.where(valid)[0]:
                cv2.circle(image, (int(round(uv_rep[i, 0])), int(round(uv_rep[i, 1]))), 4,
                           (0, 0, 255), cv2.FILLED, 8)
                cv2.circle(image, (int(round(uv_det[i, 0])), int(round(uv_det[i, 1]))), 4,
                           (0, 255, 0), cv2.FILLED, 8)
        out_dir = os.path.join(root, f"{cam_prefix}{cam + 1:03d}")
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"{fr:06d}.jpg"), image)
        return True

    n_workers = (os.cpu_count() or 4) if workers is None else workers
    if n_workers and n_workers > 1 and len(items) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_do, items))
    else:
        results = [_do(e) for e in items]
    return sum(1 for ok in results if ok)


def _write_int_seq(fs, name, values):
    fs.startWriteStruct(name, cv2.FileNode_SEQ + cv2.FileNode_FLOW)
    for v in values:
        fs.write("", int(v))
    fs.endWriteStruct()


def _write_seq_of_seqs(fs, name, arrays, cast=float):
    fs.startWriteStruct(name, cv2.FileNode_SEQ)
    for arr in arrays:
        fs.startWriteStruct("", cv2.FileNode_SEQ + cv2.FileNode_FLOW)
        for v in arr:
            fs.write("", cast(v))
        fs.endWriteStruct()
    fs.endWriteStruct()


def _write_str_seq(fs, name, values):
    fs.startWriteStruct(name, cv2.FileNode_SEQ)
    for v in values:
        fs.write("", str(v))
    fs.endWriteStruct()


def _resolve_frame_path(image_root, cam_prefix, cam, fr, ext="png"):
    """Resolve the source image path for ``(cam, frame)`` (MC-Calib's frame_paths), matching the
    detection/reprojection image resolver. Returns the path string or ``""`` if not found."""
    for cand in (f"{fr:05d}.{ext}", f"{fr + 1:05d}.{ext}", f"{fr:06d}.{ext}", f"{fr + 1:06d}.{ext}"):
        p = os.path.join(image_root, f"{cam_prefix}{cam + 1:03d}", cand)
        if os.path.exists(p):
            return p
    return ""


def save_mccalib_detected_keypoints(object_obs, obj: Object3D, img_size, path: str, *,
                                    image_root: Optional[str] = None,
                                    cam_prefix: str = "Cam_") -> None:
    """Write ``detected_keypoints_data.yml`` in MC-Calib's schema (McCalib.cpp:507): per
    ``camera_<i>`` the parallel, per-board-observation sequences ``frame_idxs`` / ``board_idxs``
    (inline) and ``pts_2d`` / ``charuco_idxs`` (sequence of inline ``[u,v,...]`` / corner-id
    arrays), plus ``img_width`` / ``img_height``. DS-MSP fuses boards into one object, so each
    point is mapped back to its ``(board_id, corner_id)`` via ``obj.pts_obj_2_board`` and the
    per-frame observation is split by board — round-trip-identical to the reader."""
    by_cam: Dict[int, List] = {}
    for o in object_obs:
        by_cam.setdefault(o.cam_id, []).append(o)
    b2 = obj.pts_obj_2_board                                  # (N,2) = [board_id, corner_id]
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("nb_camera", int(len(by_cam)))
    for cam in sorted(by_cam):
        w, h = img_size.get(cam, (0, 0))
        frame_idxs, board_idxs, pts_2d, charuco, frame_paths = [], [], [], [], []
        for o in sorted(by_cam[cam], key=lambda x: x.frame_id):
            bc = b2[o.point_rows]                            # (n,2)
            fp = getattr(o, "image_path", None) or (
                _resolve_frame_path(image_root, cam_prefix, cam, o.frame_id) if image_root else "")
            for bid in np.unique(bc[:, 0]):
                m = bc[:, 0] == bid
                frame_idxs.append(o.frame_id)
                board_idxs.append(int(bid))
                pts_2d.append(np.asarray(o.pts_2d, float)[m].ravel())
                charuco.append(bc[m, 1])
                frame_paths.append(fp)
        fs.startWriteStruct(f"camera_{cam}", cv2.FileNode_MAP)
        fs.write("img_width", int(w))
        fs.write("img_height", int(h))
        _write_int_seq(fs, "frame_idxs", frame_idxs)
        if image_root:                                       # MC-Calib's frame_paths field
            _write_str_seq(fs, "frame_paths", frame_paths)
        _write_int_seq(fs, "board_idxs", board_idxs)
        _write_seq_of_seqs(fs, "pts_2d", pts_2d, cast=float)
        _write_seq_of_seqs(fs, "charuco_idxs", charuco, cast=int)
        fs.endWriteStruct()
    fs.release()


def save_detection_images(object_obs, image_root: str, save_dir: str, *,
                          cam_prefix: str = "Cam_", ext: str = "png",
                          workers: Optional[int] = None, progress_cb=None) -> int:
    """Draw detected corners (green) per frame and save under
    ``<save_dir>/Detection/<cam:03d>/<frame:06d>.jpg`` — MC-Calib's ``saveDetectionImages``
    layout. Returns the number of images written (0 if no source images found).

    Parallelised the same way as :func:`save_reprojection_images` (see its docstring for why
    this used to be a silent, unparallelised tail after the bundle adjustment already
    converged). ``progress_cb(cam_id, i, n, frame_id)``, if given, fires once per image."""
    root = os.path.join(save_dir, "Detection")
    by_cf: Dict[Tuple[int, int], List] = {}
    for o in object_obs:
        by_cf.setdefault((o.cam_id, o.frame_id), []).append(o)
    items = list(by_cf.items())
    counts = Counter(cam for (cam, _fr), _ in items)
    seen: Dict[int, int] = Counter()
    lock = threading.Lock()

    def _do(entry) -> bool:
        (cam, fr), obs_list = entry
        if progress_cb is not None:
            with lock:
                seen[cam] += 1
                progress_cb(cam, seen[cam], counts[cam], fr)
        img_path = _obs_image_path(obs_list, image_root, cam, fr, cam_prefix, ext)
        if img_path is None:
            return False
        image = cv2.imread(img_path)
        if image is None:
            return False
        for o in obs_list:
            for u, v in np.asarray(o.pts_2d, float):
                cv2.circle(image, (int(round(u)), int(round(v))), 4, (0, 255, 0), cv2.FILLED, 8)
        out_dir = os.path.join(root, f"{cam_prefix}{cam + 1:03d}")
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"{fr:06d}.jpg"), image)
        return True

    n_workers = (os.cpu_count() or 4) if workers is None else workers
    if n_workers and n_workers > 1 and len(items) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_do, items))
    else:
        results = [_do(e) for e in items]
    return sum(1 for ok in results if ok)


def save_mccalib_results(rig, save_dir: str, *, object3d: Optional[Object3D] = None,
                         object_obs=None, cam_groups: Optional[Dict[int, int]] = None,
                         camera_params_file_name: str = "", image_root: Optional[str] = None,
                         cam_prefix: str = "Cam_") -> Dict[str, str]:
    """Write the full MC-Calib result set into ``save_dir`` and return the paths written.

    Always writes ``calibrated_cameras_data.yml`` (or ``camera_params_file_name`` if given)
    and ``calibrated_objects_pose_data.yml``; writes ``calibrated_objects_data.yml`` when an
    ``Object3D`` is provided (or taken from ``rig.objects``).
    """
    os.makedirs(save_dir, exist_ok=True)
    cam_name = camera_params_file_name or "calibrated_cameras_data.yml"
    paths = {"cameras": os.path.join(save_dir, cam_name),
             "object_poses": os.path.join(save_dir, "calibrated_objects_pose_data.yml")}
    save_mccalib_cameras(rig, paths["cameras"], cam_groups=cam_groups)
    save_mccalib_object_poses(rig, paths["object_poses"])
    obj = object3d if object3d is not None else next(iter(getattr(rig, "objects", {}).values()), None)
    if obj is not None:
        paths["objects"] = os.path.join(save_dir, "calibrated_objects_data.yml")
        save_mccalib_objects(obj, paths["objects"])
    if object_obs is not None:
        paths["reprojection_error"] = os.path.join(save_dir, "reprojection_error_data.yml")
        save_mccalib_reprojection_error(rig, object_obs, paths["reprojection_error"])
        if obj is not None:
            paths["keypoints"] = os.path.join(save_dir, "detected_keypoints_data.yml")
            save_mccalib_detected_keypoints(object_obs, obj, rig.img_size, paths["keypoints"],
                                            image_root=image_root, cam_prefix=cam_prefix)
    return paths


def radtan_from_cameragt(cam: CameraGT) -> RadTanModel:
    """Build a DS-MSP ``RadTanModel`` from an MC-Calib camera (distortion_type 0)."""
    K = cam.K
    d = cam.dist if cam.dist is not None else np.zeros(5)
    d = np.asarray(d, float).ravel()
    k1, k2, p1, p2, k3 = (list(d) + [0, 0, 0, 0, 0])[:5]
    return RadTanModel(K[0, 0], K[1, 1], K[0, 2], K[1, 2], k1, k2, p1, p2, k3)
