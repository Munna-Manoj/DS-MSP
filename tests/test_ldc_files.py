"""LDC meshes are written as files with a record, measured, and exported from the command line."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ds_msp.cli import run as run_root_cli
from ds_msp.data.observations import RigState
from ds_msp.io.mccalib import save_mccalib_cameras
from ds_msp.ldc import TI_LDC_MeshGenerator, mesh_accuracy, read_ldc_mesh, write_ldc_mesh
from ds_msp.ldc_cli import run as run_ldc_cli
from ds_msp.models.registry import _BY_NAME

pytestmark = pytest.mark.req("FR-INTEROP-004")

DS = dict(fx=380.0, fy=380.0, cx=639.5, cy=399.5, xi=-0.2, alpha=0.6)


def _ds():
    return _BY_NAME["ds"](**DS)


def _mesh(model=None, factor=4, size=(1280, 800), balance=0.5):
    return TI_LDC_MeshGenerator(model or _ds()).generate_mesh_and_intrinsics(
        size[0], size[1], factor, balance)


def test_written_mesh_reads_back_exactly_and_twice_is_byte_identical(tmp_path):
    result = _mesh()
    first = write_ldc_mesh(result, tmp_path / "a", "cam0", accuracy=mesh_accuracy(_ds(), result))
    second = write_ldc_mesh(_mesh(), tmp_path / "b", "cam0", accuracy=mesh_accuracy(_ds(), _mesh()))
    mesh, record = read_ldc_mesh(first.record)
    assert np.array_equal(mesh, result["mesh_lut"])
    assert (record["mesh_height"], record["mesh_width"]) == result["mesh_lut"].shape[:2]
    assert record["checked_on_board"] is False
    assert first.mesh.read_bytes() == second.mesh.read_bytes()
    assert first.record.read_bytes() == second.record.read_bytes()


def test_mesh_bytes_are_little_endian_h_then_v_row_by_row(tmp_path):
    result = _mesh()
    files = write_ldc_mesh(result, tmp_path, "cam0")
    raw = np.frombuffer(files.mesh.read_bytes(), dtype="<i2")
    m = result["mesh_lut"]
    assert raw[0] == m[0, 0, 0] and raw[1] == m[0, 0, 1] and raw[2] == m[0, 1, 0]
    assert raw[2 * m.shape[1]] == m[1, 0, 0]


def test_existing_files_are_not_replaced_without_permission(tmp_path):
    write_ldc_mesh(_mesh(), tmp_path, "cam0")
    with pytest.raises(FileExistsError):
        write_ldc_mesh(_mesh(), tmp_path, "cam0")
    write_ldc_mesh(_mesh(), tmp_path, "cam0", overwrite=True)


def test_a_changed_mesh_file_is_refused(tmp_path):
    files = write_ldc_mesh(_mesh(), tmp_path, "cam0")
    data = bytearray(files.mesh.read_bytes())
    data[10] ^= 1
    files.mesh.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="checksum"):
        read_ldc_mesh(files.record)


@pytest.mark.parametrize("stem", ["", "a/b", "..", "a\\b"])
def test_stem_must_be_a_plain_name(tmp_path, stem):
    with pytest.raises(ValueError):
        write_ldc_mesh(_mesh(), tmp_path, stem)


def test_a_node_on_every_pixel_leaves_only_the_q3_rounding():
    result = _mesh(factor=0, size=(320, 200))
    acc = mesh_accuracy(_ds(), result)
    assert acc["measured_fraction"] == 1.0
    assert acc["max_px"] <= np.hypot(1 / 16, 1 / 16) + 1e-9


@pytest.mark.parametrize("name", sorted(_BY_NAME))
def test_a_denser_mesh_is_never_less_accurate(name):
    model = _BY_NAME[name].sample()
    size = (max(64, int(round(model.cx * 2))), max(48, int(round(model.cy * 2))))
    coarse = mesh_accuracy(model, _mesh(model, 5, size, 1.0))
    fine = mesh_accuracy(model, _mesh(model, 2, size, 1.0))
    assert fine["median_px"] <= coarse["median_px"] + 1e-9


def test_in_view_fraction_counts_output_pixels_whose_source_is_on_the_sensor():
    result = _mesh(balance=0.0)
    full = mesh_accuracy(_ds(), result, input_size=(1280, 800))["in_view_fraction"]
    tiny = mesh_accuracy(_ds(), result, input_size=(640, 400))["in_view_fraction"]
    assert 0.0 < tiny < full <= 1.0


def test_cli_exports_every_camera_of_a_rig_result(tmp_path, capsys):
    cams = {0: _ds(), 1: _BY_NAME["kb"].sample()}
    rig = RigState(cameras=cams, T_c_g={0: np.eye(4), 1: np.eye(4)}, ref_cam_id=0,
                   object_poses={}, objects={}, img_size={0: (1280, 800), 1: (640, 480)})
    path = tmp_path / "calibrated_cameras_data.yml"
    save_mccalib_cameras(rig, str(path))
    assert run_ldc_cli(["--input", str(path)]) == 0
    out = capsys.readouterr().out
    assert "camera_0: ds 1280x800" in out and "camera_1: kb 640x480" in out
    for name, size in (("camera_0", (1280, 800)), ("camera_1", (640, 480))):
        mesh, record = read_ldc_mesh(tmp_path / "ldc" / f"{name}_ldc.json")
        assert (record["output_width"], record["output_height"]) == size
        assert record["source"]["file"] == "calibrated_cameras_data.yml"
        assert record["accuracy"]["median_px"] is not None


def test_cli_one_named_camera_and_list(tmp_path, capsys):
    rig = RigState(cameras={0: _ds(), 1: _ds()}, T_c_g={0: np.eye(4), 1: np.eye(4)},
                   ref_cam_id=0, object_poses={}, objects={},
                   img_size={0: (1280, 800), 1: (1280, 800)})
    path = tmp_path / "cams.yml"
    save_mccalib_cameras(rig, str(path))
    assert run_ldc_cli(["--input", str(path), "--list-cameras"]) == 0
    assert capsys.readouterr().out.splitlines() == ["camera_0: ds 1280x800", "camera_1: ds 1280x800"]
    assert run_ldc_cli(["--input", str(path), "--camera", "camera_1", "--output-dir",
                        str(tmp_path / "one"), "--output-size", "640", "400"]) == 0
    assert [p.name for p in sorted((tmp_path / "one").iterdir())] == [
        "camera_1_ldc.json", "camera_1_ldc_mesh.bin"]


def test_typed_values_and_unified_command_agree(tmp_path):
    params = [f"{k}={v}" for k, v in DS.items()]
    common = ["--model", "ds", "--resolution", "1280", "800", "--params", *params]
    assert run_ldc_cli([*common, "--output-dir", str(tmp_path / "a")]) == 0
    assert run_root_cli(["ldc", *common, "--output-dir", str(tmp_path / "b")]) == 0
    a, b = tmp_path / "a" / "cam0_ldc_mesh.bin", tmp_path / "b" / "cam0_ldc_mesh.bin"
    assert a.read_bytes() == b.read_bytes()
    record = json.loads((tmp_path / "a" / "cam0_ldc.json").read_text())
    assert record["source"]["format"] == "typed" and record["source"]["file"] is None


@pytest.mark.parametrize("bad", [
    ["--model", "ds", "--params", "1", "2"],                     # no resolution
    ["--balance", "2", "--model", "ds", "--resolution", "9", "9", "--params", "1"],
    ["--input", "missing.yml"],
])
def test_bad_input_is_a_message_not_a_traceback(tmp_path, bad, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        run_ldc_cli(bad)
    assert exc.value.code == 2


def test_existing_output_is_refused_by_the_cli(tmp_path):
    common = ["--model", "ds", "--resolution", "1280", "800", "--output-dir", str(tmp_path),
              "--params", *[f"{k}={v}" for k, v in DS.items()]]
    assert run_ldc_cli(common) == 0
    with pytest.raises(SystemExit):
        run_ldc_cli(common)
    assert run_ldc_cli([*common, "--overwrite"]) == 0


@pytest.mark.req("FR-INTEROP-005")
def test_export_command_writes_the_rig_run_layout_and_refuses_to_repeat(tmp_path, capsys):
    cams = {0: _ds(), 1: _BY_NAME["kb"].sample()}
    sizes = {0: (1280, 800), 1: (int(round(2 * cams[1].cx)), int(round(2 * cams[1].cy)))}
    rig = RigState(cameras=cams, T_c_g={0: np.eye(4), 1: np.eye(4)}, ref_cam_id=0,
                   object_poses={}, objects={}, img_size=sizes)
    path = tmp_path / "calibrated_cameras_data.yml"
    save_mccalib_cameras(rig, str(path))
    out = tmp_path / "out"
    args = ["export", "--input", str(path), "--ldc", "--isaac-lut", "--lut-texture-size", "32",
            "24", "--output-dir", str(out)]
    assert run_root_cli(args) == 0
    summary = json.loads((out / "exports.json").read_text())
    assert [r["camera"] for r in summary["ldc"]["cameras"]] == ["camera_0", "camera_1"]
    assert all("record" in r for r in summary["ldc"]["cameras"])
    assert all("manifest" in r for r in summary["isaac_lut"]["cameras"])
    assert (out / "isaac_lut" / "camera_1_isaac_lut.json").exists()
    with pytest.raises(SystemExit):
        run_root_cli(args)
    before = (out / "ldc" / "camera_0_ldc_mesh.bin").read_bytes()
    assert run_root_cli([*args, "--overwrite"]) == 0
    assert (out / "ldc" / "camera_0_ldc_mesh.bin").read_bytes() == before
    with pytest.raises(SystemExit):
        run_root_cli(["export", "--input", str(path), "--output-dir", str(tmp_path / "x")])


@pytest.mark.req("FR-INTEROP-005")
def test_export_command_reports_a_camera_it_cannot_export(tmp_path, monkeypatch):
    import ds_msp.isaac_sim as isaac

    monkeypatch.setattr(isaac, "export_lut", lambda *a, **k: (_ for _ in ()).throw(
        ValueError("no valid pixels")))
    params = [f"{k}={v}" for k, v in DS.items()]
    code = run_root_cli(["export", "--model", "ds", "--resolution", "1280", "800", "--params",
                         *params, "--ldc", "--isaac-lut", "--output-dir", str(tmp_path)])
    assert code == 1
    summary = json.loads((tmp_path / "exports.json").read_text())
    assert summary["isaac_lut"]["cameras"][0]["problem"] == "no valid pixels"
    assert "record" in summary["ldc"]["cameras"][0]
