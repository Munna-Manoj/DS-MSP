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
    with pytest.raises(ValueError, match="is a target file, not a file of found corners"):
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


def _terminal_result(root, sets=()):
    """A result folder as a terminal run leaves it: no calib_param.yml inside, a manifest that
    records the config it was made from."""
    import hashlib
    import json

    (root / "cfg").mkdir()
    (root / "data" / "Images").mkdir(parents=True)
    cfg = root / "cfg" / "calib_param.yml"
    cfg.write_text('%YAML:1.0\nnumber_camera: 2\nroot_path: "../data/Images"\nnumber_x_square: 6\n'
                   'save_path: "../out"\nexclude_views: [ "0:4" ]\n')
    out = root / "out"
    out.mkdir()
    (out / "calibration_manifest.json").write_text(json.dumps({
        "configuration": {"entrypoint": "config", "config_path": str(cfg), "overrides": list(sets)},
        "inputs": {"files": [{"path": str(cfg), "exists": True,
                              "sha256": hashlib.sha256(cfg.read_bytes()).hexdigest()}]}}))
    return cfg, out


@pytest.mark.req("FR-UI-012")
def test_settings_of_a_terminal_result_come_from_its_manifest(tmp_path):
    cfg, out = _terminal_result(tmp_path, sets=["number_x_square=9", "webviewer=false", "nonsense=1"])
    doc = sources.settings_of(out, [tmp_path])
    # the --set value wins, relative paths are the config's own, the output folder is not carried
    assert doc["values"] == {"number_camera": 2, "root_path": str(tmp_path / "data" / "Images"),
                             "number_x_square": 9, "exclude_views": ["0:4"]}
    assert doc["changed"] is False
    cfg.write_text(cfg.read_text() + "number_y_square: 5\n")
    assert sources.settings_of(out, [tmp_path])["changed"] is True
    with pytest.raises(FileNotFoundError, match="outside"):
        sources.settings_of(out, [tmp_path / "out"])
    cfg.unlink()
    with pytest.raises(FileNotFoundError, match="no longer there"):
        sources.settings_of(out, [tmp_path])


@pytest.mark.req("FR-UI-012")
def test_config_route_serves_studio_and_terminal_results(tmp_path):
    _cfg, out = _terminal_result(tmp_path)
    c = TestClient(create_app([str(tmp_path)]))
    doc = c.get("/api/config", params={"id": "0:out"}).json()
    assert doc["name"] == "out" and doc["values"]["exclude_views"] == ["0:4"]
    (out / "calib_param.yml").write_text('%YAML:1.0\nnumber_camera: 3\nexclude_views: [ "1:2", "0:7:1" ]\n')
    doc = c.get("/api/config", params={"id": "0:out"}).json()
    assert doc["values"] == {"number_camera": 3, "exclude_views": ["1:2", "0:7:1"]}
    # what the form sends back is written as the loader reads it
    from ds_msp.studio import configio
    values, problems = configio.check_values(doc["values"])
    assert problems == [] and 'exclude_views: [ "1:2", "0:7:1" ]' in configio.render_config(values)


def test_a_wrong_file_is_answered_with_what_it_is(tmp_path, results_root):
    """Choosing the wrong kind of file names what was chosen and what belongs there."""
    run = results_root / RUN
    with pytest.raises(ValueError, match=r"detected_keypoints_data.yml is a file of found corners, "
                                         r"not a lens file\. A lens file lists every camera"):
        sources.scan_intrinsics(run / "detected_keypoints_data.yml")
    with pytest.raises(ValueError, match="is a lens file, not a file of found corners"):
        sources.scan_keypoints(run / "calibrated_cameras_data.yml")
    cfg = tmp_path / "calib_param.yml"
    cfg.write_text("%YAML:1.0\nnumber_camera: 2\nnumber_x_square: 6\n")
    with pytest.raises(ValueError, match="calib_param.yml is a settings file, not a lens file"):
        sources.scan_intrinsics(cfg)
    kalibr = tmp_path / "camchain.yaml"
    kalibr.write_text("cam0:\n  camera_model: pinhole\n  intrinsics: [600, 600, 320, 240]\n")
    with pytest.raises(ValueError, match="camchain.yaml is a Kalibr camera file, not a lens file.*"
                                         "other formats are not read here"):
        sources.scan_intrinsics(kalibr)
    odd = tmp_path / "notes.yml"
    odd.write_text("hello: world\n")
    with pytest.raises(ValueError, match="notes.yml is not a lens file"):
        sources.scan_intrinsics(odd)
    assert sources.what_is(run / "calibrated_cameras_data.yml").startswith("a lens file")
    assert sources.what_is(tmp_path / "missing.yml") == ""


@pytest.mark.req("FR-UI-016")
def test_a_config_from_another_computer_points_at_what_sits_next_to_it(tmp_path):
    (tmp_path / "Cam_001").mkdir()
    (tmp_path / "lens.yml").write_text("%YAML:1.0\nnb_camera: 1\ncamera_0:\n  fx: 1\n")
    path = tmp_path / "calib_param.yml"
    path.write_text('%YAML:1.0\nnumber_camera: 1\nroot_path: "/nowhere/capture/"\n'
                    'cam_params_path: "/nowhere/capture/lens.yml"\n'
                    'accuracy_profile_path: "/nowhere/profile.json"\n')
    doc = sources.config_here(path)
    assert doc["values"]["root_path"] == str(tmp_path)
    assert doc["values"]["cam_params_path"] == str(tmp_path / "lens.yml")
    assert {m["key"] for m in doc["repointed"]} == {"root_path", "cam_params_path"}
    assert "accuracy_profile_path" not in doc["values"]
    assert doc["missing"] == [{"key": "accuracy_profile_path", "was": "/nowhere/profile.json"}]


@pytest.mark.req("FR-UI-016")
def test_a_photo_folder_says_which_files_in_it_can_be_used(tmp_path):
    (tmp_path / "calib_param.yml").write_text("%YAML:1.0\nnumber_camera: 1\n")
    (tmp_path / "lens.yml").write_text("%YAML:1.0\nnb_camera: 1\ncamera_0:\n  fx: 1\n")
    (tmp_path / "kp.yml").write_text("%YAML:1.0\ncamera_0:\n  frame_idxs: [0]\n")
    (tmp_path / "other.yml").write_text("%YAML:1.0\nsomething: 1\n")
    found = {f["name"]: f["kind"] for f in sources.companions(tmp_path)}
    assert found == {"calib_param.yml": "settings", "lens.yml": "lens", "kp.yml": "corners"}
