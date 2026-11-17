"""LDC meshes are written as files with a record, measured, and exported from the command line."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ds_msp.cli import run as run_root_cli
from ds_msp.data.observations import RigState
from ds_msp.io.mccalib import save_mccalib_cameras
from ds_msp.ldc import (
    LDC_BLOCK_INPUT_BUDGET_PX,
    TI_LDC_MeshGenerator,
    ldc_board_check,
    mesh_accuracy,
    read_ldc_mesh,
    write_ldc_mesh,
)
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
    assert record["format_version"] == 2 and "tiovxldc" in record["layout_checked_on"]
    assert record["tiovxldc"] == {"lut-file": "cam0_ldc_mesh.bin", "ldc-ds-factor": 4,
                                  "ldc-table-width": 1280, "ldc-table-height": 800,
                                  "out-block-width": 64, "out-block-height": 32, "pixel-pad": 1}
    assert record["board_check"]["max_block_input_px"] <= LDC_BLOCK_INPUT_BUDGET_PX
    assert first.mesh.read_bytes() == second.mesh.read_bytes()
    assert first.record.read_bytes() == second.record.read_bytes()


def test_mesh_bytes_follow_the_ti_driver_layout(tmp_path):
    """(v, h) per 32-bit word, rows padded to 16 nodes: the layout checked on a TDA4VM board."""
    result = _mesh()
    files = write_ldc_mesh(result, tmp_path, "cam0")
    m = result["mesh_lut"]
    raw = np.frombuffer(files.mesh.read_bytes(), dtype="<i2").reshape(51, 96, 2)
    assert np.array_equal(raw[:, :81, 0], m[..., 1]) and np.array_equal(raw[:, :81, 1], m[..., 0])
    assert not raw[:, 81:].any()
    words = np.frombuffer(files.mesh.read_bytes(), dtype="<u4").reshape(51, 96)
    assert np.array_equal((words[:, :81] >> 16).astype(np.uint16).view(np.int16), m[..., 0])


def test_table_size_matches_ti_sample_and_odd_sizes_cover_the_last_part_cell(tmp_path):
    # TI's imx390_ldc_lut_1920x1080.bin (ds factor 2) is 537664 bytes: 496 x 271 words.
    files = write_ldc_mesh(_mesh(factor=2, size=(1920, 1080)), tmp_path / "ti", "cam0")
    assert files.mesh.stat().st_size == 537664
    # 1224 x 1000 at 16 px nodes: the table covers 1232 x 1008, so the last 8 columns and
    # rows have nodes beyond them (the board uses zero displacement past the table).
    result = _mesh(size=(1224, 1000))
    files = write_ldc_mesh(result, tmp_path / "odd", "cam0")
    mesh, record = read_ldc_mesh(files.record)
    assert (record["mesh_height"], record["mesh_width"], record["row_stride_nodes"]) == (64, 78, 80)
    assert (record["tiovxldc"]["ldc-table-width"], record["tiovxldc"]["ldc-table-height"]) == (1232, 1008)
    assert (record["output_width"], record["output_height"]) == (1224, 1000)
    assert np.array_equal(mesh, result["mesh_lut"])
    assert mesh[-1].any()
    assert files.mesh.stat().st_size == 64 * 80 * 4


def _zoom(scale, size=(960, 768), factor=4):
    """A mesh that magnifies the input uniformly: each output block reads scale x its size."""
    w, h = size
    step = 2 ** factor
    nu, nv = np.meshgrid(np.arange(0, w + step, step, dtype=float),
                         np.arange(0, h + step, step, dtype=float))
    cx, cy = (w - 1) / 2, (h - 1) / 2
    src = np.stack([cx + scale * (nu - cx), cy + scale * (nv - cy)], -1)
    mesh = np.round((src - np.stack([nu, nv], -1)) * 8).astype(np.int16)
    return {"mesh_lut": mesh, "valid_mask": np.ones(mesh.shape[:2], bool),
            "config": {"output_width": w, "output_height": h, "downsample_factor": factor}}


def test_block_size_keeps_each_blocks_input_within_the_board_memory():
    # Board: a 130 x 130 input box per 64 x 64 block was right, 146 x 146 was wrong.
    small = ldc_board_check(_zoom(2.0), (960, 768))
    assert (small["out_block_width"], small["out_block_height"]) == (64, 64)
    big = ldc_board_check(_zoom(2.25), (960, 768))
    assert (big["out_block_width"], big["out_block_height"]) == (64, 32)
    assert big["max_block_input_px"] <= LDC_BLOCK_INPUT_BUDGET_PX < 146 * 146
    assert not big["notes"]


def test_no_block_size_fits_is_reported():
    check = ldc_board_check(_zoom(40.0), None)
    assert check["out_block_width"] is None
    assert "block memory" in check["notes"][0]


@pytest.mark.parametrize("width, block_width", [(960, 64), (952, 64), (944, 16), (928, 32),
                                                (936, 16)])
def test_block_width_divides_the_output_row_pitch(width, block_width):
    # The LDC writes whole blocks; a pitch that is not a multiple of the block width made the
    # last block of each row spill into the next row on the board (912, 928, 936, 944 wide).
    check = ldc_board_check(_zoom(1.0, size=(width, 768)), None)
    assert check["out_block_width"] == block_width


def test_block_size_respects_the_driver_mesh_block_limit():
    # TI's VHWA driver refuses (width >> ds) * (height >> ds) * 4 > 5 KB: no 64 x 64 at ds 0.
    check = ldc_board_check(_zoom(1.0, size=(256, 192), factor=0), None)
    bw, bh = check["out_block_width"], check["out_block_height"]
    assert bw * bh * 4 <= 5 * 1024


def test_a_folded_mesh_is_counted_and_warned(tmp_path):
    result = _zoom(1.0)
    mesh = result["mesh_lut"].astype(np.int32)
    nu = np.arange(mesh.shape[1]) * 16.0
    mesh[10:20, 30:40, 0] = np.round((960 - 2 * nu[30:40]) * 8)  # mirror a patch in x
    result["mesh_lut"] = mesh.astype(np.int16)
    check = ldc_board_check(result, (960, 768))
    assert check["folded_cells"] > 0 and "fold" in check["notes"][-1]
    result["config"].update(balance=0.5, camera_model={"name": "test"}, n_invalid_nodes=0,
                            q3_overflow=False)
    result["K_new"] = np.eye(3)
    with pytest.warns(UserWarning, match="fold"):
        files = write_ldc_mesh(result, tmp_path, "cam0", input_size=(960, 768))
    record = json.loads(files.record.read_text())
    assert record["board_check"]["folded_cells"] == check["folded_cells"]
    assert record["board_notes"] == check["notes"]


def test_a_format_1_record_is_refused(tmp_path):
    files = write_ldc_mesh(_mesh(), tmp_path, "cam0")
    record = json.loads(files.record.read_text())
    files.record.write_text(json.dumps({**record, "format_version": 1}))
    with pytest.raises(ValueError, match=r"0\.28\.0.*misreads.*Export the mesh again"):
        read_ldc_mesh(files.record)


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
    for row in summary["ldc"]["cameras"]:
        record = json.loads((out / "ldc" / row["record"]).read_text())
        assert row["board_fits"] is True and row["folded_cells"] == 0
        assert row["out_block"] == [record["tiovxldc"]["out-block-width"],
                                    record["tiovxldc"]["out-block-height"]]
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
