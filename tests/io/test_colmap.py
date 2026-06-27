"""Tests for COLMAP sparse-model I/O."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ds_msp.io import colmap
from ds_msp.io.colmap import colmap_to_model, export_colmap, model_to_colmap, read_colmap
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.models.kb import KannalaBrandtModel
from ds_msp.models.radtan import RadTanModel


def _make_poses(n, seed=0):
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(n):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(rng.uniform(-1.2, 1.2, 3)).as_matrix()
        T[:3, 3] = rng.uniform(-3, 3, 3)
        poses.append(T)
    return np.stack(poses)


def test_kb_maps_to_opencv_fisheye_and_back():
    cam = KannalaBrandtModel(190.97, 190.73, 254.9, 256.9, 0.01, 0.02, -0.003, 0.001)
    name, params = model_to_colmap(cam)
    assert name == "OPENCV_FISHEYE"
    assert params == [cam.fx, cam.fy, cam.cx, cam.cy, cam.k1, cam.k2, cam.k3, cam.k4]
    back = colmap_to_model(name, params)
    assert isinstance(back, KannalaBrandtModel)
    np.testing.assert_allclose(back.params, cam.params, rtol=0, atol=0)


def test_radtan_maps_to_opencv():
    cam = RadTanModel(458.6, 457.3, 367.2, 248.4, -0.28, 0.07, 1e-4, 2e-5)
    name, params = model_to_colmap(cam)
    assert name == "OPENCV"
    assert params == [cam.fx, cam.fy, cam.cx, cam.cy, cam.k1, cam.k2, cam.p1, cam.p2]


def test_double_sphere_export_refused():
    cam = DoubleSphereModel(180.0, 180.0, 255.0, 255.0, -0.18, 0.59)
    with pytest.raises(NotImplementedError, match="Convert to Kannala-Brandt"):
        model_to_colmap(cam)


def test_export_read_roundtrip(tmp_path):
    cam = KannalaBrandtModel(190.97, 190.73, 254.9, 256.9, 0.01, 0.02, -0.003, 0.001)
    poses = _make_poses(6)
    names = [f"frame_{i:04d}.png" for i in range(len(poses))]
    rng = np.random.default_rng(1)
    pts = rng.uniform(-5, 5, (40, 3))
    cols = rng.integers(0, 256, (40, 3), dtype=np.uint8)

    export_colmap(str(tmp_path), cam, 512, 512, poses, names, points3d=pts, point_colors=cols)
    got = read_colmap(str(tmp_path))

    # intrinsics
    np.testing.assert_allclose(got["model"].params, cam.params, atol=1e-9)
    assert got["cameras"][0].model == "OPENCV_FISHEYE"
    assert (got["cameras"][0].width, got["cameras"][0].height) == (512, 512)
    # poses (world->cam) recovered as rotation + translation
    np.testing.assert_allclose(got["poses"][:, :3, :3], poses[:, :3, :3], atol=1e-9)
    np.testing.assert_allclose(got["poses"][:, :3, 3], poses[:, :3, 3], atol=1e-9)
    # names, points, colors
    assert got["image_names"] == names
    np.testing.assert_allclose(got["points3d"], pts, atol=1e-9)
    np.testing.assert_array_equal(got["point_colors"], cols)


def test_pose_only_export_has_no_observation_lines(tmp_path):
    cam = KannalaBrandtModel(190.0, 190.0, 255.0, 255.0, 0.0, 0.0, 0.0, 0.0)
    poses = _make_poses(4, seed=7)
    names = [f"img{i}.jpg" for i in range(len(poses))]
    export_colmap(str(tmp_path), cam, 640, 480, poses, names)

    got = read_colmap(str(tmp_path))
    assert got["points3d"] is None
    np.testing.assert_allclose(got["poses"][:, :3, :3], poses[:, :3, :3], atol=1e-9)
    np.testing.assert_allclose(got["poses"][:, :3, 3], poses[:, :3, 3], atol=1e-9)


def test_quaternion_convention_world_to_camera():
    # COLMAP stores world->cam; T_cam_world property must reconstruct R exactly.
    T = _make_poses(1, seed=3)[0]
    q = colmap._qvec_from_R(T[:3, :3])
    np.testing.assert_allclose(colmap._R_from_qvec(q), T[:3, :3], atol=1e-12)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-12


def test_pose_count_mismatch_raises(tmp_path):
    cam = KannalaBrandtModel(190.0, 190.0, 255.0, 255.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="differ"):
        export_colmap(str(tmp_path), cam, 512, 512, _make_poses(3), ["a.png", "b.png"])
