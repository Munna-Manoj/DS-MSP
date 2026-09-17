"""TI LDC mesh export is model-agnostic: every CameraModel exports a mesh automatically.

Parametrized over the model registry, so a model added in the future is covered the moment
it is registered, with no change to this file or to ``ds_msp/ldc.py``. A deliberately
unregistered, duck-typed model proves the exporter depends on the contract alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from ds_msp.core.contracts import CameraModel
from ds_msp.ldc import TI_LDC_MeshGenerator, TI_LDC_PointUndistorter
from ds_msp.models.registry import _BY_NAME
from ds_msp.ops.undistort import Undistorter

REGISTRY = sorted(_BY_NAME)
STEP_LOG2 = 4
STEP = 2 ** STEP_LOG2


def _nominal_size(model):
    return max(64, int(round(model.cx * 2))), max(48, int(round(model.cy * 2)))


def _expected_mesh_shape(width, height):
    padded_w = ((width + STEP - 1) // STEP) * STEP
    padded_h = ((height + STEP - 1) // STEP) * STEP
    return padded_h // STEP + 1, padded_w // STEP + 1, 2


@pytest.mark.parametrize("name", REGISTRY)
def test_every_registered_model_exports_a_mesh_consistent_with_the_service_layer(name):
    model = _BY_NAME[name].sample()
    width, height = _nominal_size(model)
    res = TI_LDC_MeshGenerator(model).generate_mesh_and_intrinsics(
        width, height, downsample_factor=STEP_LOG2, balance=0.5
    )
    mesh, mesh_float, K_new, valid = (
        res["mesh_lut"], res["mesh_lut_float"], res["K_new"], res["valid_mask"]
    )
    assert mesh.shape == mesh_float.shape == _expected_mesh_shape(width, height)
    assert mesh.dtype == np.int16 and valid.shape == mesh.shape[:2]
    assert valid.all(), "sample models must be fully valid at balance 0.5"
    assert np.array_equal(mesh, np.round(mesh_float * 8.0).astype(np.int16))

    # Same rectified frame as the model-agnostic software undistorter for this model.
    service = Undistorter(model, width, height)
    assert np.allclose(K_new, service.new_K(0.5))

    # Node displacements equal the service layer's independent distort_points path.
    mesh_h, mesh_w = mesh.shape[:2]
    hu, vu = np.meshgrid(np.arange(mesh_w) * STEP, np.arange(mesh_h) * STEP)
    nodes = np.stack([hu.ravel(), vu.ravel()], axis=-1).astype(np.float64)
    distorted, ok = service.distort_points(nodes, K_new)
    assert ok.all()
    assert np.allclose(mesh_float.reshape(-1, 2), distorted - nodes, atol=1e-6)

    config = res["config"]
    assert config["camera_model"]["name"] == name
    assert tuple(config["camera_model"]["params"]) == tuple(model.param_names)
    assert config["n_invalid_nodes"] == 0 and config["q3_overflow"] is False


@pytest.mark.parametrize("name", REGISTRY)
def test_hardware_style_mesh_inverse_matches_closed_form_near_center(name):
    model = _BY_NAME[name].sample()
    width, height = _nominal_size(model)
    res = TI_LDC_MeshGenerator(model).generate_mesh_and_intrinsics(
        width, height, downsample_factor=STEP_LOG2, balance=1.0
    )
    hardware = TI_LDC_PointUndistorter(
        res["mesh_lut_float"], res["K_new"], STEP_LOG2, width, height
    )
    pts = np.array([
        [width * 0.5, height * 0.5], [width * 0.6, height * 0.45],
        [width * 0.4, height * 0.6], [width * 0.7, height * 0.7],
    ])
    from_mesh, ok_mesh = hardware.undistort_points(pts)
    closed_form, ok_cf = Undistorter(model, width, height).undistort_points(pts, res["K_new"])
    assert ok_mesh.all() and ok_cf.all()
    assert np.allclose(from_mesh, closed_form, atol=0.15)


class _FutureModel:
    """A contract-shaped model that is NOT in the registry: pinhole + one radial term.

    Rays beyond ``fov_limit`` (as ``|x/z|`` or ``|y/z|``) are declared invalid so the
    exporter's handling of non-projectable nodes can be exercised.
    """

    name = "future"
    param_names = ("fx", "fy", "cx", "cy", "k")

    def __init__(self, fx=300.0, fy=300.0, cx=320.0, cy=240.0, k=0.05,
                 *, fov_limit=10.0, gain=1.0):
        self.fx, self.fy, self.cx, self.cy, self.k = fx, fy, cx, cy, k
        self.fov_limit, self.gain = fov_limit, gain

    @property
    def params(self):
        return np.array([self.fx, self.fy, self.cx, self.cy, self.k])

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def distortion(self):
        return np.array([self.k])

    def project(self, P):
        P = np.asarray(P, dtype=np.float64)
        z = P[..., 2]
        valid = z > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(valid, P[..., 0] / z, 0.0)
            y = np.where(valid, P[..., 1] / z, 0.0)
        valid &= (np.abs(x) < self.fov_limit) & (np.abs(y) < self.fov_limit)
        r2 = x * x + y * y
        s = self.gain * (1.0 + self.k * r2)
        uv = np.stack([self.fx * x * s + self.cx, self.fy * y * s + self.cy], axis=-1)
        uv[~valid] = 0.0
        return uv, valid

    def unproject(self, uv):
        uv = np.asarray(uv, dtype=np.float64)
        x = (uv[..., 0] - self.cx) / self.fx
        y = (uv[..., 1] - self.cy) / self.fy
        for _ in range(20):  # fixed-point inverse of the radial term
            r2 = x * x + y * y
            s = self.gain * (1.0 + self.k * r2)
            x = (uv[..., 0] - self.cx) / self.fx / s
            y = (uv[..., 1] - self.cy) / self.fy / s
        rays = np.stack([x, y, np.ones_like(x)], axis=-1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        return rays, np.ones(x.shape, dtype=bool)

    def project_jacobian(self, P):
        raise NotImplementedError

    @classmethod
    def from_params(cls, p):
        return cls(*p)

    @classmethod
    def param_bounds(cls):
        return np.full(5, -np.inf), np.full(5, np.inf)

    def initialize_from_correspondences(self, K_seed, rays, pixels):
        return None

    def to_dict(self):
        return {"model": self.name, **dict(zip(self.param_names, self.params.tolist()))}

    @classmethod
    def from_dict(cls, d):
        return cls(*(d[k] for k in cls.param_names))


def test_unregistered_contract_model_exports_without_any_exporter_change():
    model = _FutureModel()
    assert isinstance(model, CameraModel)
    assert model.name not in _BY_NAME
    res = TI_LDC_MeshGenerator(model).generate_mesh_and_intrinsics(640, 480)
    assert res["mesh_lut"].shape == _expected_mesh_shape(640, 480)
    assert res["valid_mask"].all()
    assert res["config"]["camera_model"] == {
        "name": "future",
        "params": {"fx": 300.0, "fy": 300.0, "cx": 320.0, "cy": 240.0, "k": 0.05},
    }
    assert "double_sphere_params" not in res["config"]


def test_invalid_nodes_hold_zero_displacement_and_are_flagged():
    model = _FutureModel(fov_limit=0.45)  # narrower than the balance=0.0 pinhole grid
    with pytest.warns(UserWarning, match="invalid"):
        res = TI_LDC_MeshGenerator(model).generate_mesh_and_intrinsics(640, 480, balance=0.0)
    valid = res["valid_mask"]
    assert not valid.all() and valid.any()
    assert np.all(res["mesh_lut_float"][~valid] == 0.0)
    assert np.all(res["mesh_lut"][~valid] == 0)
    assert res["config"]["n_invalid_nodes"] == int((~valid).sum())


def test_q3_overflow_is_clipped_to_int16_and_flagged():
    model = _FutureModel(gain=60.0)  # displacements far beyond 4096 px
    with pytest.warns(UserWarning, match="int16"):
        res = TI_LDC_MeshGenerator(model).generate_mesh_and_intrinsics(640, 480)
    assert res["config"]["q3_overflow"] is True
    assert res["mesh_lut"].max() == 32767 and res["mesh_lut"].min() == -32768
    assert np.abs(res["mesh_lut_float"]).max() * 8 > 32767


def test_legacy_double_sphere_camera_keeps_its_record_and_numbers():
    from ds_msp.model import DoubleSphereCamera
    from ds_msp.models import DoubleSphereModel

    fx, fy, cx, cy, xi, alpha = 711.57, 711.24, 949.18, 518.81, 0.183, 0.809
    legacy = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha)
    contract = DoubleSphereModel(fx, fy, cx, cy, xi, alpha)
    res_legacy = TI_LDC_MeshGenerator(legacy).generate_mesh_and_intrinsics(1920, 1080)
    res_contract = TI_LDC_MeshGenerator(contract).generate_mesh_and_intrinsics(1920, 1080)
    assert np.array_equal(res_legacy["mesh_lut"], res_contract["mesh_lut"])
    assert np.allclose(res_legacy["K_new"], res_contract["K_new"])
    assert res_contract["config"]["double_sphere_params"] == {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy, "xi": xi, "alpha": alpha,
    }
    assert res_legacy["config"]["camera_model"]["name"] == "DoubleSphereCamera"


def test_objects_without_the_contract_are_rejected_early():
    with pytest.raises(TypeError, match="project"):
        TI_LDC_MeshGenerator(object())


pytestmark = pytest.mark.req("FR-INTEROP-002")
