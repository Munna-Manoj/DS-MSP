"""Multi-board fused-object reconstruction from raw detections — MC-Calib's
``calibrate3DObjects`` stage (McCalib.cpp:765-956) in NumPy.

MC-Calib does not need the fused 3-D object up front: it *reconstructs* it. After the
per-camera intrinsics are bootstrapped, every board is resected by PnP (``T_c_b``), the
inter-board relative poses are averaged over all images that see a board pair
(``computeBoardsPairPose`` -> ``initInterTransform``), and the boards are fused into one
rigid point cloud per covisibility component (``init3DObjects`` -> :func:`object3d.build_objects`).

This module reproduces that for ``number_board > 1`` so a multi-board rig calibrates
straight from a raw image folder (or pre-detected keypoints) — no pre-built
``calibrated_objects_data.yml`` required. The bootstrap intrinsics are only used to recover
the object geometry; the per-camera native models + extrinsics are re-optimized by the
joint BA downstream, exactly as MC-Calib refines after this stage.

Convention: a single rigid calibration object is assumed (the largest covisibility
component); a board that is never co-observed with the others forms its own component and is
dropped with a warning, matching the "one object per connected component, pick the rig's"
intent of the rig pipeline (which carries ``object_id = 0``).
"""

from __future__ import annotations

import glob
import os
import threading
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..detect.charuco import (BoardSpec, _frame_id_from_name, board_object_points,
                             detect_image, make_detectors)
from .object3d import build_objects
from .pose_init import robust_pose_irls
from .types import BoardObs, Object3D, ObjectObs

_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _T_from_rt(rvec, tvec) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float))[0]
    T[:3, 3] = np.asarray(tvec, float).ravel()
    return T


_TLS = threading.local()


def _thread_detectors(specs, legacy, tuned):
    """One detector set per worker thread, built once and reused across that thread's images
    (CharucoDetector construction is non-trivial; per-image rebuilds would dominate)."""
    d = getattr(_TLS, "dets", None)
    if d is None or _TLS.key != (id(specs), legacy, tuned):
        d = make_detectors(specs, legacy=legacy, tuned=tuned)
        _TLS.dets, _TLS.key = d, (id(specs), legacy, tuned)
    return d


def _detect_one_image(root_path, c, path, specs, cam_prefix, legacy, min_corners, subpix, tuned):
    """Detect every board in one image — the unit of work for image-level load balancing (the
    big 2592px cameras have far more work than the 640px ones, so balancing per image, not per
    camera, keeps every core busy)."""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return c, None, []
    dets = _thread_detectors(specs, legacy, tuned)
    frame_id = _frame_id_from_name(path)
    out = []
    for board_id, corner_ids, pts in detect_image(dets, gray, min_corners=min_corners,
                                                   subpix=subpix):
        out.append(BoardObs(cam_id=c, frame_id=frame_id, board_id=board_id,
                            corner_ids=corner_ids, pts_2d=pts, image_path=path))
    return c, (gray.shape[1], gray.shape[0]), out


def detect_board_obs_images(root_path: str, cam_ids: List[int], specs: List[BoardSpec], *,
                            cam_prefix: str = "Cam_", legacy: bool = True,
                            min_corners: int = 6, subpix: bool = False, tuned: bool = False,
                            workers: Optional[int] = None
                            ) -> Tuple[List[BoardObs], Dict[int, Tuple[int, int]]]:
    """Detect per-board ChArUco corners over ``<root>/<cam_prefix><cam+1:03d>/`` — the raw,
    *object-free* detections needed to reconstruct the fused object. Returns
    ``(board_obs, img_size)`` with ``board_obs`` carrying ``corner_ids`` / ``pts_2d`` and a
    not-yet-filled ``T_c_b`` (frame ids rebased to MC-Calib's 0-indexed convention).

    Detection is **parallelised across cameras** (one thread per camera; OpenCV's detector
    releases the GIL, so threads scale near-linearly with cores) — the dominant end-to-end
    cost, cut ~3.5x on an 8-camera rig. ``subpix`` adds ``cv2.cornerSubPix`` refinement."""
    if workers is None:
        workers = (os.cpu_count() or 4)
    # flat (camera, image) task list — image-level balancing keeps all cores busy despite the
    # 2592px vs 640px size imbalance across cameras.
    tasks = []
    for c in cam_ids:
        cam_dir = os.path.join(root_path, f"{cam_prefix}{c + 1:03d}")
        if not os.path.isdir(cam_dir):
            continue
        for path in sorted(f for f in glob.glob(os.path.join(cam_dir, "*"))
                           if f.lower().endswith(_IMG_EXT)):
            tasks.append((c, path))

    def _do(t):
        return _detect_one_image(root_path, t[0], t[1], specs, cam_prefix, legacy,
                                 min_corners, subpix, tuned)

    if workers and workers > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_do, tasks))
    else:
        results = [_do(t) for t in tasks]

    all_obs: List[BoardObs] = []
    img_size: Dict[int, Tuple[int, int]] = {}
    for c, wh, obs in results:
        if wh is not None and c not in img_size:
            img_size[c] = wh
        all_obs.extend(obs)
    if all_obs:
        base = min(o.frame_id for o in all_obs)
        for o in all_obs:
            o.frame_id -= base
    return all_obs, img_size


def detect_board_obs_keypoints(keypoints_path: str
                               ) -> Tuple[List[BoardObs], Dict[int, Tuple[int, int]]]:
    """Read board-level detections from MC-Calib ``detected_keypoints_data.yml`` (one
    :class:`BoardObs` per stored (camera, frame, board) entry). Unlike
    :func:`io.mccalib._load_detections` this keeps the boards *separate* (no object needed),
    which is what reconstruction consumes."""
    fs = cv2.FileStorage(keypoints_path, cv2.FILE_STORAGE_READ)
    nb = int(fs.getNode("nb_camera").real())
    all_obs: List[BoardObs] = []
    img_size: Dict[int, Tuple[int, int]] = {}
    for ci in range(nb):
        cn = fs.getNode(f"camera_{ci}")
        img_size[ci] = (int(cn.getNode("img_width").real()),
                        int(cn.getNode("img_height").real()))
        fidx = cn.getNode("frame_idxs")
        bidx = cn.getNode("board_idxs")
        pts_node = cn.getNode("pts_2d")
        cid_node = cn.getNode("charuco_idxs")
        n = fidx.size()
        for k in range(n):
            f = int(fidx.at(k).real())
            b = int(bidx.at(k).real())
            p = pts_node.at(k)
            cset = cid_node.at(k)
            pts = np.array([p.at(i).real() for i in range(p.size())], float).reshape(-1, 2)
            cids = np.array([cset.at(i).real() for i in range(cset.size())], int).ravel()
            all_obs.append(BoardObs(cam_id=ci, frame_id=f, board_id=b,
                                    corner_ids=cids, pts_2d=pts))
    fs.release()
    return all_obs, img_size


def _bootstrap_K(board_obs: List[BoardObs], board_points: Dict[int, np.ndarray],
                 img_size: Dict[int, Tuple[int, int]]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Per-camera pinhole+RadTan bootstrap (``cv2.calibrateCamera`` over all of that
    camera's board views as planar targets), with a focal-plausibility guard + consensus
    fallback for a camera that only ever sees a near-planar board. Only used to resect the
    boards; final intrinsics come from the joint BA."""
    by_cam: Dict[int, List[BoardObs]] = defaultdict(list)
    for o in board_obs:
        by_cam[o.cam_id].append(o)

    raw: Dict[int, dict] = {}
    for cam, lst in by_cam.items():
        objp = [board_points[o.board_id][o.corner_ids].astype(np.float32) for o in lst]
        imgp = [o.pts_2d.astype(np.float32) for o in lst]
        w, h = img_size[cam]
        K0 = np.array([[float(w), 0, w / 2.0], [0, float(w), h / 2.0], [0, 0, 1.0]])
        try:
            _ret, K, dist, _rv, _tv = cv2.calibrateCamera(
                objp, imgp, (w, h), K0, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        except cv2.error:
            K, dist = K0, np.zeros(5)
        diag = float(np.hypot(w, h))
        ok = 0.2 * diag < K[0, 0] < 4.0 * diag and 0.2 * diag < K[1, 1] < 4.0 * diag
        raw[cam] = dict(K=K, dist=np.asarray(dist, float).ravel(), ok=ok)

    good = [r["K"] for r in raw.values() if r["ok"]]
    consensus = np.median(np.stack(good), axis=0) if good else None
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for cam, r in raw.items():
        if r["ok"] or consensus is None:
            out[cam] = (r["K"], r["dist"])
        else:                                      # degenerate camera -> consensus focal/pp
            w, h = img_size[cam]
            Kc = consensus.copy()
            Kc[0, 2], Kc[1, 2] = w / 2.0, h / 2.0
            out[cam] = (Kc, np.zeros(5))
    return out


def reconstruct_object(board_obs: List[BoardObs], specs: List[BoardSpec],
                       img_size: Dict[int, Tuple[int, int]], *,
                       object_id: int = 0, init_models: Optional[Dict[int, object]] = None
                       ) -> Object3D:
    """Resect every board (-> ``T_c_b``) and fuse the boards into one rigid :class:`Object3D`
    (the largest covisibility component). Raw, object-free ``board_obs`` in, fused object out
    — MC-Calib's ``calibrate3DObjects`` result.

    ``init_models`` (``{cam_id: CameraModel}``, e.g. from ``cam_params_path``) resects each
    board with that camera's **native model** via robust model-aware PnP. This is essential
    for a wide-FOV fisheye: the default ``cv2.calibrateCamera`` bootstrap is a Brown/pinhole
    fit that cannot represent a ~190° lens, so its per-board PnP — and therefore the fused
    inter-board geometry — is corrupted. With the correct model the resection (and the whole
    reconstructed object) is right. Cameras without an init model fall back to the bootstrap."""
    board_points = {b: board_object_points(specs[b]) for b in range(len(specs))}
    boot_cams = [o.cam_id for o in board_obs
                 if not (init_models and o.cam_id in init_models)]
    Kd = (_bootstrap_K([o for o in board_obs if o.cam_id in set(boot_cams)],
                       board_points, img_size) if boot_cams else {})
    for o in board_obs:                            # resect each board by PnP
        objp = board_points[o.board_id][o.corner_ids].astype(np.float64)
        if len(objp) < 4:
            o.valid = False
            continue
        if init_models and o.cam_id in init_models:        # model-aware (correct for fisheye)
            T = robust_pose_irls(init_models[o.cam_id], objp, o.pts_2d.astype(np.float64),
                                 kernel="cauchy", gnc_iters=5, gnc_start=4.0)
            o.T_c_b = T
            o.valid = T is not None
        else:                                              # Brown bootstrap (pinhole rigs)
            K, dist = Kd[o.cam_id]
            ok, rv, tv = cv2.solvePnP(objp, o.pts_2d.astype(np.float64), K, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
            o.T_c_b = _T_from_rt(rv, tv) if ok else None
            o.valid = bool(ok)

    valid = [o for o in board_obs if o.valid and o.T_c_b is not None]
    objects = build_objects(valid, board_points)
    if not objects:
        raise ValueError("multi-board reconstruction found no valid board observations")
    obj = max(objects, key=lambda o: len(o.board_ids))
    if len(objects) > 1:
        dropped = [b for o in objects if o is not obj for b in o.board_ids]
        warnings.warn(
            f"reconstruct_object: boards {sorted(dropped)} are never co-observed with the "
            f"main object (boards {sorted(obj.board_ids)}); dropping them. Provide views "
            f"where the boards are seen together, or pass a pre-built object.")
    obj.object_id = object_id
    return obj


def object_obs_from_board_obs(board_obs: List[BoardObs], obj: Object3D, *,
                              min_corners: int = 4) -> List[ObjectObs]:
    """Map raw per-board detections onto the fused object's rows, fusing all boards seen in
    one image into a single :class:`ObjectObs` (one object pose per (camera, frame), all
    boards' corners pooled — the most-constrained PnP). Mirrors what the keypoint reader
    produces, so the rest of the pipeline is untouched."""
    by_image: Dict[Tuple[int, int], Tuple[list, list]] = defaultdict(lambda: ([], []))
    paths: Dict[Tuple[int, int], str] = {}
    for o in board_obs:
        rows, uvs = by_image[(o.cam_id, o.frame_id)]
        if getattr(o, "image_path", None):
            paths.setdefault((o.cam_id, o.frame_id), o.image_path)
        for cid, uv in zip(o.corner_ids, o.pts_2d):
            key = (int(o.board_id), int(cid))
            row = obj.pts_board_2_obj.get(key)
            if row is not None:
                rows.append(row)
                uvs.append(uv)
    obs: List[ObjectObs] = []
    for (cam, fr), (rows, uvs) in by_image.items():
        if len(rows) >= min_corners:
            obs.append(ObjectObs(cam_id=cam, frame_id=fr, object_id=obj.object_id,
                                 point_rows=np.array(rows, int),
                                 pts_2d=np.array(uvs, float),
                                 image_path=paths.get((cam, fr))))
    return obs


def reconstruct_from_images(root_path: str, cam_ids: List[int], specs: List[BoardSpec], *,
                            cam_prefix: str = "Cam_", legacy: bool = True,
                            min_corners: int = 6, init_models: Optional[Dict[int, object]] = None
                            ) -> Tuple[Object3D, List[ObjectObs], Dict[int, Tuple[int, int]]]:
    """Raw image folder -> ``(fused object, object_obs, img_size)`` for a multi-board rig.
    ``init_models`` resects boards with the native per-camera model (see
    :func:`reconstruct_object`) — needed for a wide-FOV fisheye rig."""
    board_obs, img_size = detect_board_obs_images(
        root_path, cam_ids, specs, cam_prefix=cam_prefix, legacy=legacy, min_corners=min_corners)
    obj = reconstruct_object(board_obs, specs, img_size, init_models=init_models)
    return obj, object_obs_from_board_obs(board_obs, obj), img_size


def reconstruct_from_keypoints(keypoints_path: str, specs: List[BoardSpec], *,
                               init_models: Optional[Dict[int, object]] = None
                               ) -> Tuple[Object3D, List[ObjectObs], Dict[int, Tuple[int, int]]]:
    """Pre-detected keypoints -> ``(fused object, object_obs, img_size)`` for a multi-board rig.
    ``init_models`` resects boards with the native per-camera model (see
    :func:`reconstruct_object`)."""
    board_obs, img_size = detect_board_obs_keypoints(keypoints_path)
    obj = reconstruct_object(board_obs, specs, img_size, init_models=init_models)
    return obj, object_obs_from_board_obs(board_obs, obj), img_size
