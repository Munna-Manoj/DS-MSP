"""What a result page needs to show a calibration on the capture itself: the photos, the
corners found in them and where the calibration puts those corners, the board's place in every
frame, and what the calibration's uncertainty means for a point measured with two cameras.

Everything is read from the result folder (and the photos it names); nothing is recomputed
except :func:`task_uncertainty`, which propagates the recorded pair covariance to a point.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPORT_FILE = "calibration_report.json"
ACCEPTANCE_FILE = "calibration_acceptance.json"
MANIFEST_FILE = "calibration_manifest.json"
KEYPOINTS_FILE = "detected_keypoints_data.yml"
MAX_IMAGE_WIDTH = 2000


def _json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _residuals(folder: Path) -> Optional[Dict[str, np.ndarray]]:
    acc = _json(folder / ACCEPTANCE_FILE)
    rel = ((acc.get("diagnostics") or {}).get("files") or {}).get("residuals")
    path = folder / rel if rel else folder / "diagnostics" / "residuals.npz"
    if not path.is_file():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# -- frames and corners ----------------------------------------------------------------------
def frame_list(folder: os.PathLike | str, cam: int) -> Dict[str, Any]:
    """Every frame in which camera ``cam`` saw the board, in capture order: corner count, RMS and
    largest error in pixels, corners outside the inlier band."""
    z = _residuals(Path(folder))
    if z is None:
        return {"cam": cam, "frames": []}
    m = (z["cam"] == cam) & z["valid"]
    r = np.hypot(z["dx"], z["dy"])
    frames = []
    for f in np.unique(z["frame"][m]):
        fm = m & (z["frame"] == f)
        rf = r[fm]
        frames.append({"frame": int(f), "n": int(fm.sum()), "rms_px": float(np.sqrt(np.mean(rf ** 2))),
                       "max_px": float(rf.max()), "n_outside": int((fm & ~z["within_band"]).sum())})
    return {"cam": cam, "frames": frames}


def frame_corners(folder: os.PathLike | str, cam: int, frame: int) -> Dict[str, Any]:
    """The corners of one photo: where each was detected (``u``, ``v``, pixel centres) and the
    step to where the calibration predicts it (``dx``, ``dy`` = predicted minus detected)."""
    z = _residuals(Path(folder))
    if z is None:
        return {"cam": cam, "frame": frame, "n": 0, "u": [], "v": [], "dx": [], "dy": [], "within": []}
    m = (z["cam"] == cam) & (z["frame"] == frame) & z["valid"]
    return {"cam": cam, "frame": frame, "n": int(m.sum()),
            "u": np.round(z["u"][m], 2).tolist(), "v": np.round(z["v"][m], 2).tolist(),
            "dx": np.round(z["dx"][m], 4).tolist(), "dy": np.round(z["dy"][m], 4).tolist(),
            "within": z["within_band"][m].tolist()}


#: Distances, in pixels, at which :func:`error_spread` counts the corners further out: 0.01 px
#: to 100 px in 40 steps per decade.
SPREAD_GRID_PX = np.round(0.01 * 10.0 ** (np.arange(161) / 40.0), 5)


def error_spread(folder: os.PathLike | str, max_points: int = 400) -> Dict[str, Any]:
    """How far the corners of a result lie from where its calibration expects them, for a
    setup page that shows what an outlier band would keep: the corner count ``n``, the median
    and largest distance, ``above[k]`` = corners further than ``grid_px[k]``, and a sample of
    at most ``max_points`` error vectors (``dx``, ``dy``) that always holds the largest 60.
    ``n`` is 0 when the result recorded no residuals."""
    z = _residuals(Path(folder))
    if z is None or not int(z["valid"].sum()):
        return {"n": 0, "grid_px": [], "above": [], "dx": [], "dy": []}
    m = z["valid"]
    dx, dy = z["dx"][m].astype(float), z["dy"][m].astype(float)
    r = np.hypot(dx, dy)
    order = np.argsort(-r, kind="stable")
    top, rest = order[:60], order[60:]
    room = max(0, max_points - len(top))
    if len(rest) > room:
        rest = rest[np.linspace(0, len(rest) - 1, room).astype(int)] if room else rest[:0]
    pick = np.concatenate([top, rest])
    srt = np.sort(r)
    above = len(r) - np.searchsorted(srt, SPREAD_GRID_PX, side="right")
    return {"n": int(len(r)), "median_px": float(np.median(r)), "max_px": float(r.max()),
            "grid_px": SPREAD_GRID_PX.tolist(), "above": above.astype(int).tolist(),
            "dx": np.round(dx[pick], 4).tolist(), "dy": np.round(dy[pick], 4).tolist()}


# -- photos ----------------------------------------------------------------------------------
def _flat_ints(node) -> List[int]:
    if node.empty():
        return []
    if node.isSeq():
        return [int(node.at(i).real()) for i in range(node.size())]
    m = node.mat()
    return [] if m is None else [int(v) for v in np.asarray(m).ravel()]


def photo_index(folder: os.PathLike | str) -> Dict[str, Any]:
    """Which photo is which ``(camera, frame)``.

    Exact when the result's keypoints file records ``frame_paths``. Otherwise the photos listed
    in the run manifest are matched by camera folder and file number, taking the lowest file
    number as frame 0; that is right unless the board was missing from the first photos, so
    ``how`` says ``"assumed"`` and a page should say so."""
    import cv2

    folder = Path(folder)
    index: Dict[Tuple[int, int], str] = {}
    # the run's own detections, else the detections file it was given (it kept their frames)
    given = ((_json(folder / MANIFEST_FILE).get("configuration") or {}).get("resolved") or {}).get("keypoints_path")
    for kp in (folder / KEYPOINTS_FILE, Path(str(given)) if given else None):
        if index or kp is None or not kp.is_file():
            continue
        fs = cv2.FileStorage(str(kp), cv2.FILE_STORAGE_READ)
        nb_node = fs.getNode("nb_camera")
        nb = int(nb_node.real()) if not nb_node.empty() else 0
        for ci in range(nb):
            cn = fs.getNode(f"camera_{ci}")
            paths = cn.getNode("frame_paths")
            if cn.empty() or paths.empty() or not paths.isSeq():
                continue
            frames = _flat_ints(cn.getNode("frame_idxs"))
            for k in range(min(len(frames), paths.size())):
                p = paths.at(k).string()
                if p:
                    index[(ci, frames[k])] = p
        fs.release()
    if index:
        return {"how": "recorded", "index": index}

    from ..detect.charuco import _frame_id_from_name

    manifest = _json(folder / MANIFEST_FILE)
    resolved = ((manifest.get("configuration") or {}).get("resolved") or {})
    prefix = resolved.get("cam_prefix") or "Cam_"
    pattern = resolved.get("frame_pattern")
    raw: Dict[Tuple[int, int], str] = {}
    for item in ((manifest.get("inputs") or {}).get("files") or []):
        path = str(item.get("path", ""))
        parent = os.path.basename(os.path.dirname(path))
        if not parent.startswith(prefix) or not parent[len(prefix):].isdigit():
            continue
        try:
            raw[(int(parent[len(prefix):]) - 1, _frame_id_from_name(path, pattern))] = path
        except ValueError:
            continue
    if not raw:
        return {"how": "none", "index": {}}
    base = min(f for _c, f in raw)
    return {"how": "assumed", "index": {(c, f - base): p for (c, f), p in raw.items()}}


def photo_path(folder: os.PathLike | str, cam: int, frame: int,
               inside: Sequence[Path]) -> Optional[Path]:
    """The photo of ``(cam, frame)`` when it exists and lies under one of ``inside``."""
    raw = photo_index(folder)["index"].get((cam, frame))
    if not raw:
        return None
    path = Path(raw).resolve()
    if not path.is_file() or not any(root == path or root in path.parents for root in inside):
        return None
    return path


def photo_jpeg(path: Path, width: int = 1280) -> bytes:
    """``path`` as a JPEG no wider than ``width`` (never enlarged), with the scale it was shrunk
    by left to the caller: a page positions overlays in original pixels."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"not a readable image: {path.name}")
    width = int(min(max(width, 64), MAX_IMAGE_WIDTH))
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError(f"could not encode {path.name}")
    return buf.tobytes()


# -- the rig and the board through the capture --------------------------------------------------
def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]


def _board_outline(points: np.ndarray) -> List[List[float]]:
    """Four corners of the smallest rectangle, in the board's own plane, around its points."""
    centre = points.mean(axis=0)
    _u, _s, vt = np.linalg.svd(points - centre, full_matrices=False)
    a, b = vt[0], vt[1]
    pa, pb = (points - centre) @ a, (points - centre) @ b
    pad = 0.5 * float(np.median(np.diff(np.unique(np.round(pa, 6))))) if len(np.unique(np.round(pa, 6))) > 1 else 0.0
    lo_a, hi_a, lo_b, hi_b = pa.min() - 2 * pad, pa.max() + 2 * pad, pb.min() - 2 * pad, pb.max() + 2 * pad
    return [(centre + x * a + y * b).tolist()
            for x, y in ((lo_a, lo_b), (hi_a, lo_b), (hi_a, hi_b), (lo_a, hi_b))]


def run_scene(folder: os.PathLike | str) -> Dict[str, Any]:
    """The target and its pose in every frame, in the reference camera's frame.

    ``boards``: per board its corner points and an outline rectangle, in target coordinates.
    ``frames``: ``{"frame", "T"}`` with ``T`` the 4x4 target-to-reference transform. ``photos``
    says how photos are matched to frames (``recorded``, ``assumed`` or ``none``)."""
    import cv2

    folder = Path(folder)
    out: Dict[str, Any] = {"boards": [], "frames": [], "photos": photo_index(folder)["how"]}
    obj_path, pose_path = folder / "calibrated_objects_data.yml", folder / "calibrated_objects_pose_data.yml"
    if not obj_path.is_file():
        return out
    fs = cv2.FileStorage(str(obj_path), cv2.FILE_STORAGE_READ)
    node = fs.getNode("object_0").getNode("points")
    pts = None if node.empty() else node.mat()
    fs.release()
    if pts is None:
        return out
    xyz, board_ids = np.asarray(pts[:3].T, float), np.asarray(pts[3], int)
    for b in sorted(set(board_ids.tolist())):
        p = xyz[board_ids == b]
        if len(p) >= 3:
            out["boards"].append({"id": int(b), "points": np.round(p, 6).tolist(),
                                  "outline": _board_outline(p)})
    z = _residuals(folder)
    if z is None or not pose_path.is_file():
        return out
    fs = cv2.FileStorage(str(pose_path), cv2.FILE_STORAGE_READ)
    node = fs.getNode("object_0").getNode("poses")
    poses = None if node.empty() else node.mat()
    fs.release()
    # The pose file lists the frames that have a pose, in order, without their numbers: they
    # are the frames whose corners have a residual.
    posed = np.unique(z["frame"][(z["object"] == 0) & np.isfinite(z["dx"])])
    if poses is None or poses.shape[1] != len(posed):
        return out
    for k, f in enumerate(posed):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = _rodrigues(poses[:3, k]), poses[3:, k]
        out["frames"].append({"frame": int(f), "T": np.round(T, 8).tolist()})
    return out


# -- what the uncertainty means for a measurement ---------------------------------------------
def _relative(T_a: np.ndarray, T_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rotation a->b and the centre of b in a's frame (the pair covariance's variables)."""
    T = T_b @ np.linalg.inv(T_a)
    return T[:3, :3], -T[:3, :3].T @ T[:3, 3]


def _so3_exp(w: np.ndarray) -> np.ndarray:
    return _rodrigues(w)


def task_uncertainty(folder: os.PathLike | str, a: int, b: int, distance: float,
                     pixel_noise: float = 0.5) -> Dict[str, Any]:
    r"""1-sigma position uncertainty of a point triangulated by cameras ``a`` and ``b``.

    The point lies ``distance`` (scene units) from the midpoint between the two cameras, along
    the direction both look in. Its position is found from one pixel in each camera, so it
    inherits three errors: the uncertainty of camera ``b``'s pose relative to ``a`` (the
    recorded 6x6 pair covariance ``P``, widened by the Student-t factor), each lens's
    uncertainty at that pixel (the camera's median lens sigma), and the noise of whatever finds
    the point in the image (``pixel_noise``, px, the caller's own number). With ``J_x`` the
    Jacobian of the four pixel coordinates in the point and ``J_p`` in the relative pose,

    ``Cov(X) = A (Σ_px + J_p P J_pᵀ) Aᵀ``,  ``A = (J_xᵀ J_x)⁻¹ J_xᵀ``.

    Returns sigmas along the viewing direction (``depth``) and across it (``sideways``, the
    larger of the two), each also split into its ``from_cameras`` and ``from_pixels`` parts,
    or ``{"ok": False, "reason"}`` when the two cameras cannot both see such a point. Random
    error only: it is a lower bound on the real error."""
    from ..io.mccalib import load_camera

    folder = Path(folder)
    report, acc = _json(folder / REPORT_FILE), _json(folder / ACCEPTANCE_FILE)
    ext = {e["cam_id"]: np.asarray(e["T_c_ref"], float) for e in report.get("extrinsics") or []}
    if a == b or a not in ext or b not in ext:
        return {"ok": False, "reason": "choose two different cameras of this rig"}
    if not (np.isfinite(distance) and distance > 0 and np.isfinite(pixel_noise) and pixel_noise >= 0):
        return {"ok": False, "reason": "the distance must be positive"}
    unc = acc.get("extrinsic_uncertainty") or {}
    pair = (unc.get("pairs") or {}).get(f"{min(a, b)}-{max(a, b)}")
    if not pair or not pair.get("covariance"):
        return {"ok": False, "reason": "this result has no uncertainty for that camera pair"}
    lo, hi = min(a, b), max(a, b)                      # the covariance is of hi relative to lo
    cam = {c: load_camera(str(folder / "calibrated_cameras_data.yml"), c) for c in (lo, hi)}
    R0, C0 = _relative(ext[lo], ext[hi])
    t_factor = float(unc.get("t_factor") or 1.0)
    P = np.asarray(pair["covariance"], float) * t_factor ** 2

    axis_lo, axis_hi = np.array([0.0, 0.0, 1.0]), R0.T @ np.array([0.0, 0.0, 1.0])
    look = axis_lo + axis_hi
    if np.linalg.norm(look) < 1e-6:
        return {"ok": False, "reason": "these two cameras face opposite ways"}
    look /= np.linalg.norm(look)
    X0 = 0.5 * C0 + distance * look                    # in camera lo's frame

    def pixels(X: np.ndarray, d: np.ndarray) -> Optional[np.ndarray]:
        R, C = R0 @ _so3_exp(d[:3]), C0 + d[3:]
        out = []
        for model, Y in ((cam[lo], X), (cam[hi], R @ (X - C))):
            uv, ok = model.project(Y[None, :])
            if not bool(np.asarray(ok).ravel()[0]) or not np.isfinite(uv).all():
                return None
            out.append(np.asarray(uv, float).ravel())
        return np.concatenate(out)

    zero = np.zeros(6)
    u0 = pixels(X0, zero)
    intr = {i["cam_id"]: i for i in report.get("intrinsics") or []}
    inside = u0 is not None and all(
        0 <= u0[2 * k] <= intr[c]["width"] - 1 and 0 <= u0[2 * k + 1] <= intr[c]["height"] - 1
        for k, c in enumerate((lo, hi)) if c in intr)
    if not inside:
        return {"ok": False, "reason": "at this distance the point is outside one camera's image"}

    def jac(f, x0: np.ndarray, h: np.ndarray) -> Optional[np.ndarray]:
        cols = []
        for j in range(len(x0)):
            e = np.zeros(len(x0))
            e[j] = h[j]
            p, m = f(x0 + e), f(x0 - e)
            if p is None or m is None:
                return None
            cols.append((p - m) / (2 * h[j]))
        return np.stack(cols, axis=1)

    hx = np.full(3, 1e-5 * max(distance, np.linalg.norm(C0), 1e-9))
    hp = np.r_[np.full(3, 1e-6), np.full(3, 1e-6 * max(np.linalg.norm(C0), 1e-9))]
    Jx = jac(lambda x: pixels(x, zero), X0, hx)
    Jp = jac(lambda d: pixels(X0, d), zero, hp)
    if Jx is None or Jp is None or np.linalg.matrix_rank(Jx) < 3:
        return {"ok": False, "reason": "the two views do not fix a point there"}
    lens = (acc.get("lens") or {}).get("cameras") or {}
    lens_px = [float((lens.get(str(c)) or {}).get("sigma_median_px") or 0.0) for c in (lo, hi)]
    var_px = np.repeat([pixel_noise ** 2 + lens_px[0] ** 2, pixel_noise ** 2 + lens_px[1] ** 2], 2)
    A = np.linalg.solve(Jx.T @ Jx, Jx.T)
    cov_cam = A @ (Jp @ P @ Jp.T) @ A.T
    cov_pix = A @ np.diag(var_px) @ A.T

    def split(cov: np.ndarray) -> Tuple[float, float]:
        depth = float(np.sqrt(max(look @ cov @ look, 0.0)))
        Q = np.eye(3) - np.outer(look, look)
        side = float(np.sqrt(max(np.linalg.eigvalsh(Q @ cov @ Q)[-1], 0.0)))
        return depth, side

    depth, side = split(cov_cam + cov_pix)
    d_cam, s_cam = split(cov_cam)
    d_pix, s_pix = split(cov_pix)
    return {"ok": True, "a": a, "b": b, "distance": float(distance),
            "baseline": float(np.linalg.norm(C0)), "pixel_noise": float(pixel_noise),
            "lens_sigma_px": lens_px, "depth": depth, "sideways": side,
            "from_cameras": {"depth": d_cam, "sideways": s_cam},
            "from_pixels": {"depth": d_pix, "sideways": s_pix},
            "t_factor": t_factor,
            "note": "random error only; the real error is usually larger"}


def task_curve(folder: os.PathLike | str, a: int, b: int, pixel_noise: float,
               distances: Sequence[float]) -> List[Dict[str, Any]]:
    """:func:`task_uncertainty` over several distances (points the cameras cannot both see
    are left out)."""
    out = []
    for d in distances:
        r = task_uncertainty(folder, a, b, float(d), pixel_noise)
        if r.get("ok"):
            out.append({"distance": r["distance"], "depth": r["depth"], "sideways": r["sideways"]})
    return out


def pair_geometry(folder: os.PathLike | str, a: int, b: int) -> Dict[str, Any]:
    """How camera ``b`` sits relative to camera ``a``: distance between them, ``b``'s position
    in ``a``'s frame, and ``b``'s rotation as yaw (about a's down axis), pitch (about its right
    axis) and roll (about its viewing axis), degrees, applied in that order; with the recorded
    1-sigma of the rotation and the position when the result has them."""
    report, acc = _json(Path(folder) / REPORT_FILE), _json(Path(folder) / ACCEPTANCE_FILE)
    ext = {e["cam_id"]: np.asarray(e["T_c_ref"], float) for e in report.get("extrinsics") or []}
    if a == b or a not in ext or b not in ext:
        return {"ok": False, "reason": "choose two different cameras of this rig"}
    R, C = _relative(ext[a], ext[b])
    Rb = R.T                                            # b's axes in a's frame
    # Rb = Ry(yaw) Rx(pitch) Rz(roll), camera axes x right, y down, z forward
    pitch = float(np.degrees(np.arcsin(np.clip(-Rb[1, 2], -1, 1))))
    yaw = float(np.degrees(np.arctan2(Rb[0, 2], Rb[2, 2])))
    roll = float(np.degrees(np.arctan2(Rb[1, 0], Rb[1, 1])))
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    pair = ((acc.get("extrinsic_uncertainty") or {}).get("pairs") or {}).get(f"{min(a, b)}-{max(a, b)}") or {}
    return {"ok": True, "a": a, "b": b, "distance": float(np.linalg.norm(C)),
            "position": C.tolist(), "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll,
            "angle_deg": angle, "rotation_sigma_deg": pair.get("rotation_sigma_deg_t"),
            "position_sigma": pair.get("centre_sigma_t")}
