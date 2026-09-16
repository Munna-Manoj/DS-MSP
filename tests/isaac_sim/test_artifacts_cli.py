import json

import numpy as np
import pytest

from ds_msp.cli import run as run_root_cli
from ds_msp.data.observations import RigState
from ds_msp.io.kalibr import save_kalibr
from ds_msp.io.mccalib import save_mccalib_cameras
from ds_msp.isaac_sim.artifacts import load_calibration, model_from_params
from ds_msp.isaac_sim.cli import run as run_lut_cli
from ds_msp.models.registry import _BY_NAME


def test_named_and_ordered_parameter_construction():
    cls = _BY_NAME["ds"]
    ordered = model_from_params("ds", [700, 701, 640, 360, 0.1, 0.6])
    named = model_from_params(
        "double_sphere",
        ["alpha=0.6", "cy=360", "fx=700", "xi=0.1", "cx=640", "fy=701"],
    )
    assert isinstance(named, cls)
    assert np.array_equal(named.params, ordered.params)


@pytest.mark.parametrize("name", sorted(_BY_NAME))
def test_kalibr_artifacts_load_for_every_registered_model(name, tmp_path):
    model = _BY_NAME[name].sample()
    if name == "radtan":
        model.k3 = 0.0  # Kalibr's native RadTan dialect has no k3 field.
    path = tmp_path / f"{name}.yaml"
    save_kalibr(model, str(path), 640, 480)
    loaded = load_calibration(path)
    assert type(loaded.model) is type(model)
    assert np.allclose(loaded.model.params, model.params)
    assert loaded.resolution == (640, 480)


@pytest.mark.parametrize("name", sorted(_BY_NAME))
def test_mccalib_artifacts_load_for_every_registered_model(name, tmp_path):
    model = _BY_NAME[name].sample()
    rig = RigState(
        cameras={0: model}, T_c_g={0: np.eye(4)}, ref_cam_id=0,
        object_poses={}, objects={}, img_size={0: (800, 600)},
    )
    path = tmp_path / f"{name}.yml"
    save_mccalib_cameras(rig, str(path))
    loaded = load_calibration(path, camera="camera_0")
    assert type(loaded.model) is type(model)
    assert np.allclose(loaded.model.params, model.params, atol=1e-6)
    assert loaded.resolution == (800, 600)


def test_generic_model_json_loads_and_resolution_can_be_overridden(tmp_path):
    model = _BY_NAME["eucm"].sample()
    path = tmp_path / "camera.json"
    path.write_text(json.dumps({**model.to_dict(), "resolution": [100, 80]}))
    loaded = load_calibration(path, resolution=(640, 480))
    assert loaded.model.name == "eucm"
    assert loaded.resolution == (640, 480)


def test_dedicated_and_unified_cli_forms_generate_a_bundle(tmp_path):
    args = [
        "--model", "ds", "--resolution", "640", "480", "--params",
        "fx=300", "fy=301", "cx=320", "cy=240", "xi=0.1", "alpha=0.6",
        "--texture-size", "32", "24", "--output-dir", str(tmp_path / "dedicated"),
    ]
    assert run_lut_cli(args) == 0
    assert len(list((tmp_path / "dedicated").glob("*_isaac_lut.json"))) == 1

    root_args = [
        "--lut", *("--param" if item == "--params" else item for item in args[:-2]),
        "--output-dir", str(tmp_path / "root"),
    ]
    assert run_root_cli(root_args) == 0
    assert len(list((tmp_path / "root").glob("*_camera.usda"))) == 1


def test_cli_path_alias_loads_calibration_artifact(tmp_path):
    path = tmp_path / "camera.json"
    path.write_text(json.dumps({
        "model": "ucm", "resolution": [640, 480],
        "params": [300, 301, 320, 240, 0.6],
    }))
    output = tmp_path / "lut"
    assert run_lut_cli([
        "--path", str(path), "--texture-size", "24", "16", "--output-dir", str(output)
    ]) == 0
    assert next(output.glob("*_isaac_lut.json")).is_file()


pytestmark = pytest.mark.req("FR-INTEROP-003")
