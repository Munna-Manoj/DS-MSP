"""Starting a calibration from files other than photos: a config on disk, saved detections,
and lens values already known."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from ds_msp.studio import sources  # noqa: E402
from ds_msp.studio.server import create_app  # noqa: E402

pytestmark = pytest.mark.req("FR-UI-010")
RUN = "lab/run_a"


def test_config_on_disk_resolves_paths_like_the_cli(tmp_path):
    (tmp_path / "cfg").mkdir()
    (tmp_path / "data" / "Images").mkdir(parents=True)
    path = tmp_path / "cfg" / "calib_param.yml"
    path.write_text('%YAML:1.0\nnumber_camera: 2\nroot_path: "../data/Images"\n'
                    f'cam_params_path: "{tmp_path / "abs.yml"}"\nkeypoints_path: "None"\n')
    doc = sources.config_file(path)
    assert doc["values"] == {"number_camera": 2, "root_path": str(tmp_path / "data" / "Images"),
                             "cam_params_path": str(tmp_path / "abs.yml")}
    assert doc["path"] == str(path)


def test_saved_detections_are_summarised(results_root):
    doc = sources.scan_keypoints(results_root / RUN / "detected_keypoints_data.yml")
    assert doc["number_camera"] == 3 and doc["pixel_convention"] == "pixel_center"
    assert doc["cameras"][0]["frames"] == 30 and doc["cameras"][0]["width"] > 0
    assert doc["number_board"] >= 1 and doc["problems"] == []
    # the target file beside it would be reused by a multi-board run: it is named
    assert doc["reused_target"].endswith("calibrated_objects_data.yml")
    with pytest.raises(ValueError, match="not a detections file"):
        sources.scan_keypoints(results_root / RUN / "calibrated_objects_data.yml")


def test_known_lenses_are_summarised(results_root):
    doc = sources.scan_intrinsics(results_root / RUN / "calibrated_cameras_data.yml")
    assert [c["index"] for c in doc["cameras"]] == [0, 1, 2]
    assert doc["cameras"][0]["model"] == "radtan" and 500 < doc["cameras"][0]["fx"] < 1100
    with pytest.raises(ValueError):
        sources.scan_intrinsics(results_root / RUN / "detected_keypoints_data.yml")


def test_source_routes_stay_inside_the_roots(results_root, tmp_path):
    c = TestClient(create_app([str(results_root)]))
    kp = results_root / RUN / "detected_keypoints_data.yml"
    assert c.get("/api/source/keypoints", params={"path": str(kp)}).json()["number_camera"] == 3
    cams = results_root / RUN / "calibrated_cameras_data.yml"
    assert len(c.get("/api/source/intrinsics", params={"path": str(cams)}).json()["cameras"]) == 3
    assert c.get("/api/source/keypoints", params={"path": str(cams)}).status_code == 422
    outside = tmp_path / "x.yml"
    outside.write_text("%YAML:1.0\nnumber_camera: 1\n")
    assert c.get("/api/source/config", params={"path": str(outside)}).status_code == 400
    assert c.get("/api/source/config", params={"path": str(results_root / "nope.yml")}).status_code == 404
