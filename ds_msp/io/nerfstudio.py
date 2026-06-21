"""
nerfstudio ``transforms.json`` I/O (C9 ecosystem interop).

Exports a DS-MSP calibration + poses to the nerfstudio / instant-ngp
``transforms.json`` format consumed by many Gaussian-Splatting and NeRF trainers.

Two conventions matter and are handled here so callers don't have to:

- **Intrinsics** are global (one shared camera): ``fl_x, fl_y, cx, cy, w, h`` plus a
  ``camera_model`` (``OPENCV`` or ``OPENCV_FISHEYE``) and its distortion keys.
- **Poses** are per-frame ``transform_matrix`` = **camera-to-world in the OpenGL/Blender
  convention** (camera looks down −Z, +Y up). DS-MSP / COLMAP use OpenCV (camera looks
  down +Z, +Y down), so we convert: ``c2w_gl = inv(T_cam_world) · diag(1,−1,−1,1)``.
  The public API takes/returns 4×4 ``T_cam_world`` (world-to-camera, OpenCV) matrices.
"""

from __future__ import annotations

import json
import os
from typing import List, Sequence

import numpy as np

from .colmap import model_to_colmap, colmap_to_model

__all__ = ["export_nerfstudio", "read_nerfstudio"]

# OpenCV camera axes -> OpenGL camera axes (flip Y and Z). Self-inverse.
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])

_DISTORTION_KEYS = {
    "OPENCV": ("k1", "k2", "p1", "p2"),
    "OPENCV_FISHEYE": ("k1", "k2", "k3", "k4"),
}


def _c2w_gl_from_Tcw(T_cam_world: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(T_cam_world, dtype=np.float64)) @ _CV_TO_GL


def _Tcw_from_c2w_gl(c2w_gl: np.ndarray) -> np.ndarray:
    c2w_cv = np.asarray(c2w_gl, dtype=np.float64) @ _CV_TO_GL
    return np.linalg.inv(c2w_cv)


def export_nerfstudio(
    path: str,
    model,
    width: int,
    height: int,
    poses: np.ndarray,
    image_names: Sequence[str],
) -> str:
    """Write a nerfstudio ``transforms.json``.

    Parameters
    ----------
    path : str
        Output ``.json`` path.
    model : CameraModel
        DS-MSP camera model (KB / RadTan / pinhole; convert others to KB first).
    width, height : int
        Image resolution.
    poses : (N, 4, 4) array
        ``T_cam_world`` (world-to-camera, OpenCV) per image.
    image_names : sequence of str
        File names (used as ``file_path``), one per pose.

    Returns
    -------
    path : str
    """
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4)
    if len(poses) != len(image_names):
        raise ValueError(f"poses ({len(poses)}) and image_names ({len(image_names)}) differ")

    colmap_model, params = model_to_colmap(model)
    fx, fy, cx, cy = params[:4]
    out: dict = {
        "camera_model": colmap_model,
        "w": int(width), "h": int(height),
        "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
    }
    for key, val in zip(_DISTORTION_KEYS[colmap_model], params[4:]):
        out[key] = val

    out["frames"] = [
        {"file_path": name, "transform_matrix": _c2w_gl_from_Tcw(T).tolist()}
        for T, name in zip(poses, image_names)
    ]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


def read_nerfstudio(path: str) -> dict:
    """Read a nerfstudio ``transforms.json``.

    Returns a dict with ``model`` (DS-MSP model), ``width``, ``height``,
    ``poses`` ((N,4,4) ``T_cam_world``) and ``image_names``.
    """
    with open(path, "r") as f:
        data = json.load(f)

    colmap_model = data["camera_model"]
    params: List[float] = [data["fl_x"], data["fl_y"], data["cx"], data["cy"]]
    params += [data[k] for k in _DISTORTION_KEYS[colmap_model] if k in data]
    model = colmap_to_model(colmap_model, params)

    frames = data["frames"]
    poses = np.stack([_Tcw_from_c2w_gl(np.array(fr["transform_matrix"])) for fr in frames]) \
        if frames else np.empty((0, 4, 4))
    return {
        "model": model, "width": int(data["w"]), "height": int(data["h"]),
        "poses": poses, "image_names": [fr["file_path"] for fr in frames],
    }
