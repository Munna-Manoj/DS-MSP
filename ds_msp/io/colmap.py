"""
COLMAP sparse-model I/O (C9 ecosystem interop).

Reads/writes the COLMAP text sparse model (``cameras.txt`` / ``images.txt`` /
``points3D.txt``) so a DS-MSP calibration + poses + sparse points can feed external
Structure-from-Motion, MVS, and Gaussian-Splatting tools that consume COLMAP models
(and the nerfstudio / openMVG / OpenMVS formats).

Camera-model mapping (DS-MSP ↔ COLMAP), verified against COLMAP's
``src/colmap/sensor/models.h`` parameter orderings:

  ===========  =================  =====================================
  DS-MSP       COLMAP model       params
  ===========  =================  =====================================
  KB           OPENCV_FISHEYE     [fx, fy, cx, cy, k1, k2, k3, k4]
  RadTan       OPENCV             [fx, fy, cx, cy, k1, k2, p1, p2]
  (pinhole)    PINHOLE            [fx, fy, cx, cy]
  ===========  =================  =====================================

COLMAP has no native Double-Sphere / EUCM / UCM model, so those must be converted
to KB first (``ds_msp.adapt.convert(model, KannalaBrandtModel)``) — we refuse to
silently approximate them.

Pose convention: COLMAP stores **world-to-camera** ``(qvec, tvec)`` with
``X_cam = R(qvec) · X_world + tvec`` and ``qvec`` in **(w, x, y, z)** order. The
public API takes/returns 4×4 ``T_cam_world`` matrices so callers never touch
quaternion ordering.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from ..models.kb import KannalaBrandtModel
from ..models.radtan import RadTanModel

__all__ = [
    "ColmapCamera",
    "ColmapImage",
    "ColmapPoint3D",
    "model_to_colmap",
    "colmap_to_model",
    "export_colmap",
    "read_colmap",
]


# --------------------------------------------------------------------------- #
# Camera-model mapping
# --------------------------------------------------------------------------- #
def model_to_colmap(model) -> tuple[str, List[float]]:
    """Map a DS-MSP camera model to a ``(colmap_model_name, params)`` pair."""
    name = model.name
    if name == "kb":
        return "OPENCV_FISHEYE", [model.fx, model.fy, model.cx, model.cy,
                                  model.k1, model.k2, model.k3, model.k4]
    if name == "radtan":
        if abs(model.k3) > 1e-12:
            warnings.warn(
                "COLMAP OPENCV has no k3; dropping k3=%.3g on export." % model.k3
            )
        return "OPENCV", [model.fx, model.fy, model.cx, model.cy,
                          model.k1, model.k2, model.p1, model.p2]
    raise NotImplementedError(
        f"COLMAP has no native model for '{name}'. Convert to Kannala-Brandt first: "
        f"ds_msp.adapt.convert(model, ds_msp.models.kb.KannalaBrandtModel) "
        f"→ exports as OPENCV_FISHEYE."
    )


def colmap_to_model(colmap_model: str, params: Sequence[float]):
    """Reconstruct a DS-MSP camera model from a COLMAP ``(model, params)`` pair."""
    p = [float(v) for v in params]
    if colmap_model == "OPENCV_FISHEYE":
        fx, fy, cx, cy, k1, k2, k3, k4 = p
        return KannalaBrandtModel(fx, fy, cx, cy, k1, k2, k3, k4)
    if colmap_model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p
        return RadTanModel(fx, fy, cx, cy, k1, k2, p1, p2, 0.0)
    if colmap_model == "PINHOLE":
        fx, fy, cx, cy = p
        return RadTanModel(fx, fy, cx, cy)
    if colmap_model == "SIMPLE_PINHOLE":
        f, cx, cy = p
        return RadTanModel(f, f, cx, cy)
    raise NotImplementedError(f"Unsupported COLMAP camera model '{colmap_model}'.")


# --------------------------------------------------------------------------- #
# Quaternion <-> rotation (COLMAP w,x,y,z order)
# --------------------------------------------------------------------------- #
def _qvec_from_R(R: np.ndarray) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def _R_from_qvec(qvec: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in qvec)
    return Rotation.from_quat([x, y, z, w]).as_matrix()


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: List[float]


@dataclass
class ColmapImage:
    id: int
    qvec: np.ndarray  # (4,) w,x,y,z  (world->cam)
    tvec: np.ndarray  # (3,)          (world->cam)
    camera_id: int
    name: str
    xys: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    point3D_ids: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))

    @property
    def T_cam_world(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = _R_from_qvec(self.qvec)
        T[:3, 3] = self.tvec
        return T


@dataclass
class ColmapPoint3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray  # (3,) uint8
    error: float = 0.0


def _fmt(v: float) -> str:
    # 12 significant figures → round-trips O(1) values to < 1e-11 absolute.
    return f"{float(v):.12g}"


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def _write_cameras(path: str, cameras: Sequence[ColmapCamera]) -> None:
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for c in cameras:
            params = " ".join(_fmt(v) for v in c.params)
            f.write(f"{c.id} {c.model} {int(c.width)} {int(c.height)} {params}\n")


def _write_images(path: str, images: Sequence[ColmapImage]) -> None:
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}\n")
        for im in images:
            q = " ".join(_fmt(v) for v in im.qvec)
            t = " ".join(_fmt(v) for v in im.tvec)
            f.write(f"{im.id} {q} {t} {im.camera_id} {im.name}\n")
            # second line: 2D observations (empty when no correspondences)
            pts = []
            for (x, y), pid in zip(np.asarray(im.xys).reshape(-1, 2), im.point3D_ids):
                pts.append(f"{_fmt(x)} {_fmt(y)} {int(pid)}")
            f.write(" ".join(pts) + "\n")


def _write_points3D(path: str, points: Sequence[ColmapPoint3D]) -> None:
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}\n")
        for p in points:
            x, y, z = (float(v) for v in p.xyz)
            r, g, b = (int(v) for v in p.rgb)
            f.write(f"{p.id} {_fmt(x)} {_fmt(y)} {_fmt(z)} {r} {g} {b} {_fmt(p.error)}\n")


# --------------------------------------------------------------------------- #
# High-level export
# --------------------------------------------------------------------------- #
def export_colmap(
    out_dir: str,
    model,
    width: int,
    height: int,
    poses: np.ndarray,
    image_names: Sequence[str],
    points3d: Optional[np.ndarray] = None,
    point_colors: Optional[np.ndarray] = None,
    camera_id: int = 1,
) -> str:
    """Write a COLMAP text sparse model from a DS-MSP calibration + poses + points.

    Parameters
    ----------
    out_dir : str
        Directory to write ``cameras.txt`` / ``images.txt`` / ``points3D.txt`` into.
    model : CameraModel
        DS-MSP camera model (KB / RadTan / pinhole; convert others to KB first).
    width, height : int
        Image resolution.
    poses : (N, 4, 4) array
        ``T_cam_world`` (world-to-camera) per image, matching ``image_names``.
    image_names : sequence of str
        File names, one per pose.
    points3d : (M, 3) array, optional
        Sparse point cloud (world frame). Many Gaussian-Splatting trainers require a
        sparse point cloud (they do not support random initialization).
    point_colors : (M, 3) uint8 array, optional
        Per-point RGB; defaults to mid-grey.
    camera_id : int
        Single shared camera id (all images share one intrinsic model).

    Returns
    -------
    out_dir : str
    """
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4)
    if len(poses) != len(image_names):
        raise ValueError(f"poses ({len(poses)}) and image_names ({len(image_names)}) differ")

    colmap_model, params = model_to_colmap(model)
    cameras = [ColmapCamera(camera_id, colmap_model, int(width), int(height), params)]

    images = []
    for i, (T, name) in enumerate(zip(poses, image_names), start=1):
        images.append(ColmapImage(
            id=i, qvec=_qvec_from_R(T[:3, :3]), tvec=T[:3, 3].copy(),
            camera_id=camera_id, name=name,
        ))

    points: List[ColmapPoint3D] = []
    if points3d is not None:
        xyz = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
        if point_colors is None:
            rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)
        else:
            rgb = np.asarray(point_colors).reshape(-1, 3).astype(np.uint8)
        for j, (p, c) in enumerate(zip(xyz, rgb), start=1):
            points.append(ColmapPoint3D(id=j, xyz=p, rgb=c, error=0.0))

    os.makedirs(out_dir, exist_ok=True)
    _write_cameras(os.path.join(out_dir, "cameras.txt"), cameras)
    _write_images(os.path.join(out_dir, "images.txt"), images)
    _write_points3D(os.path.join(out_dir, "points3D.txt"), points)
    return out_dir


# --------------------------------------------------------------------------- #
# Reader
# --------------------------------------------------------------------------- #
def _data_lines(path: str):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def read_colmap(in_dir: str) -> dict:
    """Read a COLMAP text sparse model.

    Returns a dict with keys: ``cameras`` (list[ColmapCamera]), ``model`` (the
    DS-MSP model for the first camera), ``images`` (list[ColmapImage], sorted by id),
    ``poses`` ((N,4,4) ``T_cam_world``), ``image_names`` (list[str]), ``points3d``
    ((M,3) or None) and ``point_colors`` ((M,3) uint8 or None).
    """
    cameras: List[ColmapCamera] = []
    for line in _data_lines(os.path.join(in_dir, "cameras.txt")):
        parts = line.split()
        cameras.append(ColmapCamera(
            id=int(parts[0]), model=parts[1], width=int(parts[2]), height=int(parts[3]),
            params=[float(v) for v in parts[4:]],
        ))

    images: List[ColmapImage] = []
    img_lines = list(_data_lines(os.path.join(in_dir, "images.txt")))
    i = 0
    while i < len(img_lines):
        h = img_lines[i].split()
        qvec = np.array([float(v) for v in h[1:5]])
        tvec = np.array([float(v) for v in h[5:8]])
        img = ColmapImage(id=int(h[0]), qvec=qvec, tvec=tvec,
                          camera_id=int(h[8]), name=h[9])
        # The points2D line is only present when correspondences were written; a
        # pose-only export omits the (otherwise empty) second line entirely.
        if i + 1 < len(img_lines) and not _looks_like_image_header(img_lines[i + 1]):
            i += 1  # skip the observations line
        images.append(img)
        i += 1
    images.sort(key=lambda im: im.id)

    points_path = os.path.join(in_dir, "points3D.txt")
    xyz_list, rgb_list = [], []
    if os.path.exists(points_path):
        for line in _data_lines(points_path):
            parts = line.split()
            xyz_list.append([float(v) for v in parts[1:4]])
            rgb_list.append([int(v) for v in parts[4:7]])
    points3d = np.array(xyz_list) if xyz_list else None
    point_colors = np.array(rgb_list, dtype=np.uint8) if rgb_list else None

    poses = np.stack([im.T_cam_world for im in images]) if images else np.empty((0, 4, 4))
    model = colmap_to_model(cameras[0].model, cameras[0].params) if cameras else None
    return {
        "cameras": cameras, "model": model, "images": images, "poses": poses,
        "image_names": [im.name for im in images],
        "points3d": points3d, "point_colors": point_colors,
    }


def _looks_like_image_header(line: str) -> bool:
    """A COLMAP image header is ``ID QW QX QY QZ TX TY TZ CAM_ID NAME`` (≥10 fields,
    last is a non-numeric file name). The observations line is triples of numbers."""
    parts = line.split()
    if len(parts) < 10:
        return False
    try:
        float(parts[-1])  # name is non-numeric → ValueError on a real header
        return False
    except ValueError:
        return True
