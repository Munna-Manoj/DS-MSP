"""One reader for every calibration file format DS-MSP supports, used by every export."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ds_msp.data.observations import RigState
from ds_msp.io.camera_file import list_cameras, load_calibration
from ds_msp.io.colmap import export_colmap
from ds_msp.io.kalibr import to_kalibr_cam
from ds_msp.io.mccalib import save_mccalib_cameras
from ds_msp.io.nerfstudio import export_nerfstudio
from ds_msp.models.registry import _BY_NAME

pytestmark = pytest.mark.req("FR-IO-005")


def _rig_file(tmp_path, names):
    cams = {i: _BY_NAME[n].sample() for i, n in enumerate(names)}
    rig = RigState(
        cameras=cams, T_c_g={i: np.eye(4) for i in cams}, ref_cam_id=0,
        object_poses={}, objects={}, img_size={i: (800 + i, 600) for i in cams},
    )
    path = tmp_path / "calibrated_cameras_data.yml"
    save_mccalib_cameras(rig, str(path))
    return path, cams


def test_every_camera_of_a_rig_result_is_listed_in_order(tmp_path):
    names = sorted(_BY_NAME)
    path, cams = _rig_file(tmp_path, names)
    found = list_cameras(path)
    assert [c.camera for c in found] == [f"camera_{i}" for i in range(len(names))]
    for i, entry in enumerate(found):
        assert entry.problem is None
        assert entry.loaded.model.name == names[i]
        assert np.allclose(entry.loaded.model.params, cams[i].params, atol=1e-6)
        assert entry.loaded.resolution == (800 + i, 600)


def test_a_rig_result_with_more_than_ten_cameras_keeps_numeric_order(tmp_path):
    path, _ = _rig_file(tmp_path, ["ds"] * 12)
    assert [c.camera for c in list_cameras(path)][9:] == ["camera_9", "camera_10", "camera_11"]


def test_kalibr_camchain_with_two_cameras_lists_both(tmp_path):
    a, b = _BY_NAME["kb"].sample(), _BY_NAME["ds"].sample()
    path = tmp_path / "camchain.yaml"
    import yaml
    path.write_text(yaml.safe_dump({"cam0": to_kalibr_cam(a, 640, 480),
                                    "cam1": to_kalibr_cam(b, 1280, 720)}))
    found = list_cameras(path)
    assert [(c.camera, c.loaded.model.name, c.loaded.resolution) for c in found] == [
        ("cam0", "kb", (640, 480)), ("cam1", "ds", (1280, 720)),
    ]


def test_colmap_cameras_file_or_its_folder_both_load(tmp_path):
    model = _BY_NAME["kb"].sample()
    export_colmap(str(tmp_path), model, 640, 480, np.eye(4)[None], ["a.png"], camera_id=3)
    for given in (tmp_path, tmp_path / "cameras.txt"):
        found = list_cameras(given)
        assert [c.camera for c in found] == ["3"]
        assert found[0].loaded.source_format == "colmap"
        assert np.allclose(found[0].loaded.model.params, model.params)
        assert load_calibration(given).resolution == (640, 480)  # default: first camera


def test_nerfstudio_transforms_loads(tmp_path):
    model = _BY_NAME["kb"].sample()
    path = export_nerfstudio(str(tmp_path / "transforms.json"), model, 640, 480,
                             np.eye(4)[None], ["a.png"])
    loaded = load_calibration(path)
    assert loaded.source_format == "nerfstudio"
    assert loaded.resolution == (640, 480)
    assert np.allclose(loaded.model.params, model.params)


def test_a_camera_that_cannot_be_read_is_listed_with_its_reason(tmp_path):
    import cv2
    path = tmp_path / "old.yml"
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("nb_camera", 2)
    for i in range(2):
        fs.startWriteStruct(f"camera_{i}", cv2.FileNode_MAP)
        fs.write("camera_matrix", np.array([[300.0, 0, 320], [0, 300, 240], [0, 0, 1]]))
        fs.write("distortion_vector", np.zeros((1, 4)))
        if i == 1:
            fs.write("camera_model", "kb")
        fs.write("img_width", 640)
        fs.write("img_height", 480)
        fs.endWriteStruct()
    fs.release()
    found = list_cameras(path)
    assert found[0].loaded is None and "camera_model" in found[0].problem
    assert found[1].loaded is not None


def test_a_single_model_file_is_one_camera(tmp_path):
    model = _BY_NAME["eucm"].sample()
    path = tmp_path / "camera.json"
    path.write_text(json.dumps({**model.to_dict(), "resolution": [100, 80]}))
    found = list_cameras(path)
    assert [c.camera for c in found] == ["cam0"]
    assert found[0].loaded.resolution == (100, 80)


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_cameras(tmp_path / "nope.yml")
