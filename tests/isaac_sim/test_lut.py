import json

import numpy as np
import pytest

from ds_msp.isaac_sim.exr import read_rgb32_exr
from ds_msp.isaac_sim.lut import (
    FORMAT_VERSION,
    IsaacProjectionAdapter,
    export_lut,
    octahedral_directions,
    texel_centers,
)
from ds_msp.models.registry import _BY_NAME


MODEL_FACTORIES = [cls.sample for cls in _BY_NAME.values()]


def _nominal_size(model):
    return max(64, int(round(model.cx * 2))), max(48, int(round(model.cy * 2)))


def _read_rgb(path):
    rgb = read_rgb32_exr(path)
    assert rgb.dtype == np.float32 and rgb.ndim == 3 and rgb.shape[2] == 3
    return rgb


@pytest.mark.parametrize("factory", MODEL_FACTORIES, ids=lambda factory: factory().name)
def test_every_registered_model_generates_exact_rtx_lut_samples(factory, tmp_path):
    model = factory()
    width, height = _nominal_size(model)
    texture_width, texture_height = 47, 35
    bundle = export_lut(
        model, width, height, tmp_path,
        texture_width=texture_width, texture_height=texture_height,
    )
    assert bundle.round_trip_max_error_px < 1e-5

    enter = _read_rgb(bundle.paths.ray_enter)
    u, v = np.meshgrid(texel_centers(texture_width), texel_centers(texture_height))
    expected_rays, valid = IsaacProjectionAdapter(model, width, height).unproject(u, v)
    assert np.array_equal(enter, expected_rays.astype(np.float32))
    assert np.all(enter[~valid] == np.array([0.0, 0.0, 1.0], np.float32))

    exit_ = _read_rgb(bundle.paths.ray_exit)
    directions = octahedral_directions(texture_width, texture_height)
    expected_ndc, _ = IsaacProjectionAdapter(model, width, height).project(directions)
    assert np.array_equal(exit_[..., :2], expected_ndc.astype(np.float32))
    assert np.all(exit_[..., 2] == 0.0)


def test_manifest_has_loadable_schema_values_and_baked_principal_point_rule(tmp_path):
    model = _BY_NAME["kb"].sample()
    bundle = export_lut(model, 640, 480, tmp_path, texture_width=32, texture_height=24)
    manifest = json.loads(bundle.manifest.read_text())
    assert manifest["format"] == FORMAT_VERSION
    assert manifest["model"]["name"] == "kb"
    attrs = manifest["isaac_sim"]["attributes"]
    assert manifest["isaac_sim"]["api_schema"] == "OmniLensDistortionLutAPI"
    assert attrs["omni:lensdistortion:model"] == "lut"
    # cx/cy are baked into the maps; setting them again on the schema would double-shift.
    assert attrs["omni:lensdistortion:lut:opticalCenter"] == [320.0, 240.0]
    for key in (
        "omni:lensdistortion:lut:rayEnterDirectionTexture",
        "omni:lensdistortion:lut:rayExitPositionTexture",
    ):
        assert (bundle.manifest.parent / attrs[key]).is_file()
    usda = bundle.camera_usda.read_text()
    assert 'prepend apiSchemas = ["OmniLensDistortionLutAPI"]' in usda
    assert "omni:lensdistortion:lut:rayEnterDirectionTexture" in usda


def test_existing_bundle_is_checksum_validated_before_reuse(tmp_path):
    model = _BY_NAME["ucm"].sample()
    first = export_lut(model, 640, 480, tmp_path, texture_width=24, texture_height=16)
    second = export_lut(model, 640, 480, tmp_path, texture_width=24, texture_height=16)
    assert second.paths == first.paths
    first.paths.ray_enter.write_bytes(first.paths.ray_enter.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        export_lut(model, 640, 480, tmp_path, texture_width=24, texture_height=16)


def test_invalid_projection_directions_are_written_off_screen():
    model = _BY_NAME["radtan"].sample()
    adapter = IsaacProjectionAdapter(model, 640, 480)
    # RTX +Z is behind a conventional forward-facing RadTan camera.
    ndc, valid = adapter.project(np.array([[0.0, 0.0, 1.0]]))
    assert not valid[0]
    assert np.array_equal(ndc[0], [-1.0, -1.0])


pytestmark = pytest.mark.req("FR-INTEROP-003")
