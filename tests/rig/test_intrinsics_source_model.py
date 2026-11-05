"""Explicit source model labels must never silently change their meaning."""

from types import SimpleNamespace

import numpy as np
import pytest

from ds_msp.io.mccalib import CameraGT
from ds_msp.rig.calib_param import _resolve_init_intrinsics, _source_model


def _camera(dist, model_name=None):
    return CameraGT(
        K=np.array([[420., 0., 320.], [0., 421., 240.], [0., 0., 1.]]),
        dist=None if dist is None else np.asarray(dist, float),
        pose=np.eye(4), model_name=model_name,
    )


@pytest.mark.parametrize("name,dist", [
    ("ds", [0., 0., 0., 0.]),
    ("double_sphere", [0., 0., 0., 0.]),
    ("ocam", [0., 0., 0., 0.]),
    ("kb", [0., 0., 0., 0., 0.]),
    ("radtan", [0., 0., 0., 0.]),
    ("ucm", [0., 0.]),
    ("eucm", [0.]),
    ("dsplus", [0., 0.]),
    ("radtan", None),
])
def test_explicit_malformed_source_is_rejected(name, dist):
    with pytest.raises(ValueError, match="intrinsics need"):
        _source_model(_camera(dist, name))


@pytest.mark.parametrize("dist,name,coefficients", [
    ([0.1, -0.02, 0.003, -0.0004], "kb", [0.1, -0.02, 0.003, -0.0004]),
    ([0.1, -0.02], "radtan", [0.1, -0.02, 0., 0., 0.]),
    (None, "radtan", [0., 0., 0., 0., 0.]),
])
def test_nameless_legacy_source_keeps_length_inference_and_padding(dist, name, coefficients):
    model = _source_model(_camera(dist))
    assert model.name == name
    np.testing.assert_array_equal(model.params, [420., 421., 320., 240., *coefficients])


@pytest.mark.parametrize("fix_intrinsic", [False, True])
def test_malformed_explicit_source_cannot_be_converted_into_matching_config(
        tmp_path, monkeypatch, fix_intrinsic):
    # Previously this declared DS source was interpreted as KB and then converted
    # back into DS, even with fix_intrinsic=True and a matching config model.
    path = tmp_path / "intrinsics.yml"
    path.touch()
    monkeypatch.setattr("ds_msp.rig.calib_param._load_cameras",
                        lambda _: ({0: _camera([0., 0., 0., 0.], "ds")}, None))
    config = SimpleNamespace(cam_params_path=str(path), fix_intrinsic=fix_intrinsic)
    with pytest.raises(ValueError, match="ds intrinsics need 2"):
        _resolve_init_intrinsics(config, [0], {0: "ds"}, {0: (640, 480)})
