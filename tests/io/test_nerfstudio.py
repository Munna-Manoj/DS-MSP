"""Tests for nerfstudio transforms.json I/O (C9)."""

import json

import numpy as np
from scipy.spatial.transform import Rotation

from ds_msp.io.nerfstudio import (
    _CV_TO_GL,
    _c2w_gl_from_Tcw,
    _Tcw_from_c2w_gl,
    export_nerfstudio,
    read_nerfstudio,
)
from ds_msp.models.kb import KannalaBrandtModel


def _make_poses(n, seed=0):
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(n):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(rng.uniform(-1.2, 1.2, 3)).as_matrix()
        T[:3, 3] = rng.uniform(-3, 3, 3)
        poses.append(T)
    return np.stack(poses)


def test_opencv_to_opengl_conversion_is_invertible():
    T = _make_poses(1, seed=5)[0]
    np.testing.assert_allclose(_Tcw_from_c2w_gl(_c2w_gl_from_Tcw(T)), T, atol=1e-12)
    # the axis flip is its own inverse
    np.testing.assert_allclose(_CV_TO_GL @ _CV_TO_GL, np.eye(4), atol=0)


def test_export_read_roundtrip(tmp_path):
    cam = KannalaBrandtModel(190.97, 190.73, 254.9, 256.9, 0.01, 0.02, -0.003, 0.001)
    poses = _make_poses(5, seed=2)
    names = [f"images/frame_{i:04d}.png" for i in range(len(poses))]
    path = str(tmp_path / "transforms.json")

    export_nerfstudio(path, cam, 512, 512, poses, names)
    got = read_nerfstudio(path)

    np.testing.assert_allclose(got["model"].params, cam.params, atol=1e-9)
    assert (got["width"], got["height"]) == (512, 512)
    assert got["image_names"] == names
    np.testing.assert_allclose(got["poses"], poses, atol=1e-9)


def test_transforms_json_has_expected_fields(tmp_path):
    cam = KannalaBrandtModel(190.0, 190.0, 255.0, 255.0, 0.0, 0.0, 0.0, 0.0)
    path = str(tmp_path / "transforms.json")
    export_nerfstudio(path, cam, 640, 480, _make_poses(2, seed=4), ["a.png", "b.png"])

    data = json.loads(open(path).read())
    assert data["camera_model"] == "OPENCV_FISHEYE"
    assert data["w"] == 640 and data["h"] == 480
    assert {"fl_x", "fl_y", "cx", "cy", "k1", "k2", "k3", "k4"} <= set(data)
    assert len(data["frames"]) == 2
    assert np.array(data["frames"][0]["transform_matrix"]).shape == (4, 4)
