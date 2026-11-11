"""A rig run writes the LDC meshes and Isaac LUTs its config asks for (FR-INTEROP-005)."""
import json
import os
import warnings

import pytest

from ds_msp.io.camera_file import list_cameras
from ds_msp.io.mccalib import save_mccalib_detected_keypoints
from ds_msp.ldc import read_ldc_mesh
from ds_msp.models.double_sphere import DoubleSphereModel
from ds_msp.rig.calib_param import calibrate_from_config, load_config

from ._synth import make_rig

pytestmark = pytest.mark.req("FR-INTEROP-005")


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    root = tmp_path_factory.mktemp("exports")
    obj, obs, img_size, _ext, _models = make_rig(
        n_cam=2, n_frame=12, noise_px=0.05, seed=5, multi_board=False,
        model_factory=lambda _c, _rng: DoubleSphereModel(500, 500, 640, 480, -0.1, 0.55))
    kp = root / "kp" / "detected_keypoints_data.yml"
    kp.parent.mkdir()
    save_mccalib_detected_keypoints(obs, obj, img_size, str(kp))
    cfg = root / "calib_param.yml"
    cfg.write_text(f'%YAML:1.0\nnumber_camera: 2\nnumber_board: 1\nnumber_x_square: 5\n'
                   f'number_y_square: 5\nsquare_size: 0.1\nkeypoints_path: "{kp}"\n'
                   f'camera_models: [ "ds", "ds" ]\n'
                   f'verbose: "false"\nwebviewer: "false"\n')
    return root, cfg


def _run(cfg, out, **overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return calibrate_from_config(str(cfg), {"save_path": str(out) + os.sep, **overrides})


def test_exports_are_off_by_default_and_checked_when_set(capture):
    root, cfg = capture
    plain = load_config(str(cfg))
    assert (plain.export_ldc, plain.export_isaac_lut, plain.ldc_output_size) == (False, False, [])
    set_ = load_config(str(cfg), {"export_ldc": "true", "ldc_output_size": "640,480",
                                  "save_path": str(root)})
    assert set_.export_ldc and set_.ldc_output_size == [640, 480]
    for bad in ({"ldc_balance": "2"}, {"ldc_downsample_factor": "9"},
                {"ldc_output_size": "640"}, {"ldc_output_size": "0,480"}):
        with pytest.raises(ValueError):
            load_config(str(cfg), {**bad, "save_path": str(root)})
    with pytest.raises(ValueError, match="save_path"):
        load_config(str(cfg), {"export_ldc": "true"})


def test_a_run_writes_one_mesh_and_lut_per_camera_identically_twice(capture):
    root, cfg = capture
    sets = {"export_ldc": "true", "export_isaac_lut": "true",
            "isaac_lut_texture_size": "32,24", "ldc_balance": "1"}
    a = _run(cfg, root / "a", **sets)
    _run(cfg, root / "b", **sets)
    assert {"ldc_meshes", "isaac_luts", "exports_summary"} <= set(a["paths"])
    summary = json.loads((root / "a" / "exports.json").read_text())
    cams = list_cameras(root / "a" / "calibrated_cameras_data.yml")
    assert [r["camera"] for r in summary["ldc"]["cameras"]] == [c.camera for c in cams]
    for entry, ldc, lut in zip(cams, summary["ldc"]["cameras"], summary["isaac_lut"]["cameras"]):
        assert "record" in ldc and "manifest" in lut, (ldc, lut)
        if "record" in ldc:
            mesh, record = read_ldc_mesh(root / "a" / "ldc" / ldc["record"])
            assert (record["output_width"], record["output_height"]) == entry.loaded.resolution
            assert record["camera_model"]["name"] == entry.loaded.model.name
    for sub in ("ldc", "isaac_lut"):
        assert (root / "a" / sub).exists() == (root / "b" / sub).exists()
        if not (root / "a" / sub).exists():
            continue
        names = sorted(p.name for p in (root / "a" / sub).iterdir())
        assert names == sorted(p.name for p in (root / "b" / sub).iterdir())
        for name in names:
            assert (root / "a" / sub / name).read_bytes() == (root / "b" / sub / name).read_bytes()
    assert (root / "a" / "exports.json").read_bytes() == (root / "b" / "exports.json").read_bytes()


def test_a_camera_that_cannot_be_exported_is_reported_and_the_run_stands(capture, monkeypatch):
    import ds_msp.isaac_sim as isaac

    def broken(*_a, **_k):
        raise ValueError("no valid pixels")

    monkeypatch.setattr(isaac, "export_lut", broken)
    root, cfg = capture
    res = _run(cfg, root / "broken", export_isaac_lut="true")
    assert res["rig"].cameras
    rows = json.loads((root / "broken" / "exports.json").read_text())["isaac_lut"]["cameras"]
    assert all(r["problem"] == "no valid pixels" for r in rows)
    assert (root / "broken" / "calibrated_cameras_data.yml").exists()
