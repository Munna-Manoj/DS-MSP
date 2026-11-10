"""Checking the lens type of each camera against its photos, or against corner detections
saved by an earlier run, before a calibration is started.

A lens type that cannot bend enough for the lens still calibrates with a normal-looking error
and gives a wrong rig. This module catches that early, on a few photos: it fits a flexible
fisheye lens (Kannala-Brandt) to the board views with the lens-independent seed the
calibration itself uses (:func:`ds_msp.calib.lens_seed.calibrate_from_views`), reads the field
of view from it, fits the lens type the user chose to the same views, and compares the two.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Below this horizontal field of view an ordinary (radial-tangential) lens type is suggested.
STANDARD_LENS_MAX_FOV_DEG = 100.0
#: A lens type "does not fit" when its typical error exceeds the flexible lens's by this factor
#: and by this many pixels.
MISFIT_RATIO = 2.5
MISFIT_MARGIN_PX = 0.3
MIN_VIEWS = 3


def _field_angle(model, uv: np.ndarray) -> np.ndarray:
    """Angle from the optical axis (deg) of each pixel's ray; NaN where the lens has none."""
    rays, ok = model.unproject(np.asarray(uv, float))
    rays = np.asarray(rays, float).reshape(-1, 3)
    ok = np.asarray(ok, bool).ravel() & np.isfinite(rays).all(axis=1)
    ang = np.degrees(np.arctan2(np.linalg.norm(rays[:, :2], axis=1), rays[:, 2]))
    ang[~ok] = np.nan
    return ang


def advise_from_views(object_points: Sequence[np.ndarray], pixels: Sequence[np.ndarray],
                      width: int, height: int, chosen: str, *, where: str = "sample photos"
                      ) -> Dict[str, Any]:
    """Judge lens type ``chosen`` for one camera from board views (3-D board points and their
    pixels, one entry per board per photo). ``where`` names the views in the messages.

    Returns ``verdict`` (``"ok"``, ``"misfit"``, ``"caution"`` or ``"unknown"``), a ``message``,
    the measured field of view (``fov_h_deg`` across the image width, ``measured_to_deg`` the
    largest angle a detected corner reached), the typical error of the flexible lens
    (``reference_px``) and of the chosen one (``chosen_px``), and ``suggested``."""
    from ..calib.lens_seed import calibrate_from_views
    from ..models.registry import canonical_name, model_class

    chosen = canonical_name(chosen)
    out: Dict[str, Any] = {"views": len(object_points), "chosen": chosen, "verdict": "unknown",
                           "suggested": None, "fov_h_deg": None, "measured_to_deg": None,
                           "reference_px": None, "chosen_px": None}
    if len(object_points) < MIN_VIEWS:
        out["message"] = (f"The board was found in {len(object_points)} of the {where}; "
                          f"{MIN_VIEWS} are needed to judge the lens.")
        return out
    ref = calibrate_from_views(model_class("kb"), object_points, pixels, width, height)
    if ref is None or not np.isfinite(ref["median_px"]):
        out["message"] = (f"The {where} do not pin the lens down; it needs views with the "
                          f"board tilted.")
        return out
    lens = ref["model"]
    # Field of view across the image width: the largest angle the fitted lens reaches on each
    # side of the middle row. A fisheye's image circle can end before the image edge, so the
    # edge pixel itself may have no ray.
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    u = np.linspace(0.0, width - 1.0, 201)
    row = _field_angle(lens, np.c_[u, np.full_like(u, cy)])
    left, right = row[u <= cx], row[u >= cx]
    seen = _field_angle(lens, np.vstack([np.asarray(p, float) for p in pixels]))
    out["reference_px"] = float(ref["median_px"])
    out["measured_to_deg"] = float(np.nanmax(seen)) if np.isfinite(seen).any() else None
    fov = (float(np.nanmax(left) + np.nanmax(right))
           if np.isfinite(left).any() and np.isfinite(right).any() else None)
    out["fov_h_deg"] = fov
    wide = fov is None or fov >= STANDARD_LENS_MAX_FOV_DEG
    out["suggested"] = "kb" if wide else "radtan"

    fit = ref if chosen == "kb" else calibrate_from_views(model_class(chosen), object_points, pixels,
                                                          width, height)
    chosen_px = None if fit is None or not np.isfinite(fit["median_px"]) else float(fit["median_px"])
    out["chosen_px"] = chosen_px
    limit = max(MISFIT_RATIO * out["reference_px"], out["reference_px"] + MISFIT_MARGIN_PX)
    reach = "" if out["measured_to_deg"] is None else f", measured out to {out['measured_to_deg']:.0f}° from its centre"
    fov_text = "of unknown width" if fov is None else f"about {fov:.0f}° wide{reach}"
    if chosen_px is None or chosen_px > limit:
        out["verdict"] = "misfit"
        shown = "could not be fitted at all" if chosen_px is None else f"leaves {chosen_px:.2f} px"
        out["message"] = (f"This lens is {fov_text}. The chosen lens type {shown} on the {where}, "
                          f"where a fisheye lens type leaves {out['reference_px']:.2f} px.")
        out["suggested"] = "kb"
    else:
        out["verdict"] = "ok"
        out["message"] = f"This lens is {fov_text}. The chosen lens type fits ({chosen_px:.2f} px)."
        if not wide and chosen == "ucm":
            # Fits the pixels and still gives a wrong rig: on real rigs with ordinary lenses UCM
            # moved camera angles by 0.96 and 1.42 degrees at a normal reprojection error
            # (docs/RIG_REAL_DATA_EVALUATION.md, result 4). The pixel fit cannot show that.
            out["verdict"] = "caution"
            out["suggested"] = "radtan"
            out["message"] = (f"This lens is {fov_text}. UCM fits the {where} ({chosen_px:.2f} px), "
                              f"but it cannot represent an ordinary lens's distortion and has "
                              f"moved camera angles by about 1° on real rigs.")
        elif not wide and chosen != "radtan":
            out["suggested"] = chosen                 # a fisheye type on a narrow lens that fits
        elif wide and chosen != "kb":
            out["suggested"] = chosen
    return out


def sample_views(root_path: str, cam: int, values: Dict[str, Any], *, per_camera: int = 12
                 ) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[Tuple[int, int]], int]:
    """Board views of camera ``cam`` from up to ``per_camera`` photos spread over its capture:
    ``(object_points, pixels, (width, height), photos_tried)``."""
    import cv2

    from ..detect.charuco import board_object_points, detect_image, make_detectors
    from .photos import _images, board_specs, guess_prefix

    root = Path(root_path)
    prefix = values.get("cam_prefix") or guess_prefix(root) or "Cam_"
    files = _images(root / f"{prefix}{cam + 1:03d}")
    specs = board_specs(values)
    detectors = make_detectors(specs)
    boards = [np.asarray(board_object_points(s), float) for s in specs]
    step = max(1, len(files) // per_camera)
    picks = files[step // 2::step][:per_camera]
    points: List[np.ndarray] = []
    pixels: List[np.ndarray] = []
    size: Optional[Tuple[int, int]] = None
    for f in picks:
        gray = cv2.imread(os.fspath(f), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        size = (int(gray.shape[1]), int(gray.shape[0]))
        for board, ids, uv in detect_image(detectors, gray, min_corners=8):
            points.append(boards[board][np.asarray(ids, int)])
            pixels.append(np.asarray(uv, float))
    return points, pixels, size, len(picks)


@functools.lru_cache(maxsize=2)
def _keypoints(path: str, _mtime_ns: int, _size: int, convention: Optional[str]):
    from ..rig.reconstruct import detect_board_obs_keypoints

    return detect_board_obs_keypoints(path, convention)


def keypoint_views(keypoints_path: str, cam: int, values: Dict[str, Any], *, per_camera: int = 12
                   ) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[Tuple[int, int]], int]:
    """Board views of camera ``cam`` from up to ``per_camera`` of its saved detections, spread
    over the file: ``(object_points, pixels, (width, height), views_tried)``. A detection that
    names a board or a corner the board description in ``values`` does not have is left out,
    so a wrong description gives fewer views than were tried."""
    from ..detect.charuco import board_object_points
    from .photos import board_specs

    st = os.stat(keypoints_path)
    obs, sizes = _keypoints(os.fspath(keypoints_path), st.st_mtime_ns, st.st_size,
                            values.get("keypoints_pixel_convention") or None)
    boards = [np.asarray(board_object_points(s), float) for s in board_specs(values)]
    mine = [o for o in obs if o.cam_id == cam and len(o.corner_ids) >= 8]
    step = max(1, len(mine) // per_camera)
    picks = mine[step // 2::step][:per_camera]
    points: List[np.ndarray] = []
    pixels: List[np.ndarray] = []
    for o in picks:
        ids = np.asarray(o.corner_ids, int)
        if not 0 <= o.board_id < len(boards) or ids.min() < 0 or ids.max() >= len(boards[o.board_id]):
            continue
        points.append(boards[o.board_id][ids])
        pixels.append(np.asarray(o.pts_2d, float))
    size = sizes.get(cam)
    return points, pixels, (size if size and size[0] > 0 and size[1] > 0 else None), len(picks)


def _views(values: Dict[str, Any], cam: int):
    """The views to judge camera ``cam`` on, and what to call them: its photos when the form
    names a photo folder, else the saved detections it names."""
    if values.get("root_path"):
        return (*sample_views(str(values["root_path"]), cam, values), "sample photos")
    if values.get("keypoints_path"):
        return (*keypoint_views(str(values["keypoints_path"]), cam, values), "saved detections")
    raise ValueError("neither a photo folder nor a detections file is set")


def lens_advice(values: Dict[str, Any], cam: int) -> Dict[str, Any]:
    """:func:`advise_from_views` for camera ``cam`` of the photo folder (``root_path``) or the
    detections file (``keypoints_path``) in ``values``, with the lens type the form holds."""
    models = list(values.get("camera_models") or [])
    chosen = str(models[cam] if cam < len(models) else (models[-1] if models else "radtan"))
    points, pixels, size, tried, where = _views(values, cam)
    if size is None:
        return {"cam": cam, "views": 0, "chosen": chosen, "verdict": "unknown", "suggested": None,
                "fov_h_deg": None, "measured_to_deg": None, "reference_px": None, "chosen_px": None,
                "photos_tried": tried, "source": where,
                "message": ("This camera has no readable photos." if where == "sample photos"
                            else "The detections file has no image size for this camera.")}
    out = advise_from_views(points, pixels, size[0], size[1], chosen, where=where)
    if tried and not points and where == "saved detections":
        out["message"] = ("The saved detections name corners this board description does not "
                          "have. Set the board exactly as it was when they were found.")
    out.update(cam=cam, photos_tried=tried, source=where)
    return out


def lens_file_health(values: Dict[str, Any], cam: int, lens_path: str) -> Dict[str, Any]:
    """Whether the lens values a file gives for camera ``cam`` explain that camera's photos, or
    its saved detections when ``values`` names no photo folder.

    With the lens held as given, each sample board view is posed on its own (robust PnP) and
    the remaining pixel error is measured. It is compared with the error of a lens fitted to
    the same views. A file that belongs to another camera, another image size or another lens
    setting shows up as a large error before any calibration runs.

    Returns ``verdict`` (``"ok"``, ``"misfit"`` or ``"unknown"``), ``message``, ``given_px``
    (typical error with the given lens), ``reference_px`` (with a fitted lens) and ``views``."""
    import cv2

    from ..calib.lens_seed import calibrate_from_views
    from ..io.mccalib import load_camera
    from ..models.registry import model_class
    from ..ops.pose import solve_pnp_robust
    from .sources import scan_intrinsics

    out: Dict[str, Any] = {"cam": cam, "verdict": "unknown", "given_px": None,
                           "reference_px": None, "views": 0}
    listed = {c["index"]: c for c in scan_intrinsics(Path(lens_path))["cameras"]}
    if cam not in listed:
        out["message"] = "The lens file has no values for this camera."
        return out
    try:
        model = load_camera(str(lens_path), cam)
    except (KeyError, ValueError) as exc:
        out["message"] = f"The lens file's entry for this camera cannot be used: {exc}."
        return out
    points, pixels, size, _tried, where = _views(values, cam)
    out["views"] = len(points)
    out["source"] = where
    if size is None or not points:
        out["message"] = (f"The board was not found in this camera's {where}, so the lens values "
                          f"cannot be checked.")
        return out
    fw, fh = listed[cam].get("width"), listed[cam].get("height")
    if fw and fh and (fw, fh) != size:
        out["verdict"] = "misfit"
        out["message"] = (f"The lens file is for {fw} x {fh} images; the {where} are from "
                          f"{size[0]} x {size[1]} images.")
        return out
    errors: List[np.ndarray] = []
    for X, uv in zip(points, pixels):
        ok, rvec, tvec, _inl = solve_pnp_robust(model, X, uv)
        if not ok or rvec is None or tvec is None:
            errors.append(np.full(len(X), np.inf))
            continue
        R = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]
        pred, valid = model.project(X @ R.T + np.asarray(tvec, float).ravel())
        e = np.linalg.norm(np.asarray(pred, float) - uv, axis=1)
        e[~np.asarray(valid, bool).ravel()] = np.inf
        errors.append(e)
    given = float(np.median(np.concatenate(errors)))
    out["given_px"] = given if np.isfinite(given) else None
    ref = (calibrate_from_views(model_class("kb"), points, pixels, size[0], size[1])
           if len(points) >= MIN_VIEWS else None)
    ref_px = None if ref is None or not np.isfinite(ref["median_px"]) else float(ref["median_px"])
    out["reference_px"] = ref_px
    limit = 1.0 if ref_px is None else max(MISFIT_RATIO * ref_px, ref_px + MISFIT_MARGIN_PX)
    if not np.isfinite(given) or given > limit:
        out["verdict"] = "misfit"
        shown = "cannot place the board at all" if not np.isfinite(given) else f"leave {given:.2f} px"
        base = "" if ref_px is None else f", where a lens fitted to them leaves {ref_px:.2f} px"
        out["message"] = (f"These lens values {shown} on the {where}{base}. Check that the "
                          f"file belongs to this camera (same camera order) and this image size.")
    else:
        out["verdict"] = "ok"
        out["message"] = f"These lens values fit the {where} ({given:.2f} px)."
    return out
