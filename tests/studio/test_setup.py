"""Setting up a calibration from Studio: config values to the file the CLI reads and back,
the photo-folder scan, the board check, and the folder browser's confinement."""
import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from ds_msp.detect.charuco import make_detectors  # noqa: E402
from ds_msp.rig.calib_param import load_config  # noqa: E402
from ds_msp.studio import configio, photos  # noqa: E402
from ds_msp.studio.server import WRITE_HEADER, create_app  # noqa: E402

pytestmark = pytest.mark.req("FR-UI-007")
SEND = {WRITE_HEADER: "1"}


def test_values_round_trip_through_the_cli_loader(tmp_path):
    values = {"number_camera": 3, "number_board": 2, "number_x_square": 7, "number_y_square": 5,
              "square_size": 0.192, "camera_models": ["kb", "kb", "radtan"],
              "root_path": str(tmp_path), "cam_prefix": "Cam_", "fix_intrinsic": False,
              "lens_gate": "refuse", "noise_bound": 0.5, "save_path": str(tmp_path / "out")}
    clean, problems = configio.check_values(values)
    assert problems == [] and clean == values
    path = tmp_path / "calib_param.yml"
    configio.write_config(path, clean)
    cfg = load_config(str(path))
    assert cfg.number_camera == 3 and cfg.camera_models == ["kb", "kb", "radtan"]
    assert [b.square_size for b in cfg.boards] == [0.192, 0.192] and cfg.boards[0].n_x == 7
    assert cfg.lens_gate == "refuse" and cfg.noise_bound == 0.5 and cfg.fix_intrinsic is False
    back = configio.read_values(path.read_text())
    assert back["values"] == clean and back["unknown"] == [] and back["problems"] == []


def test_a_key_that_was_not_set_is_left_to_the_loader(tmp_path):
    """Writing every key would mark each as set and stop a profile from raising it."""
    text = configio.render_config({"number_camera": 2, "profile": "production"})
    assert "lens_gate" not in text and "convergence_gate" not in text
    assert text.startswith("%YAML:1.0")


def test_check_values_names_each_problem():
    clean, problems = configio.check_values({
        "number_camera": 0, "square_size": "abc", "lens_gate": "maybe", "not_a_key": 1,
        "camera_models": ["kb", "kb", "kb"], "number_x_square": "6", "keypoints_path": ""})
    by_key = {p["key"]: p["message"] for p in problems}
    assert by_key == {"number_camera": "must be at least 1", "square_size": "must be a number",
                      "lens_gate": "must be one of off, warn, refuse",
                      "not_a_key": "is not a calib_param.yml key"}
    assert clean == {"camera_models": ["kb", "kb", "kb"], "number_x_square": 6}
    _clean, problems = configio.check_values({"number_camera": 2, "camera_models": "kb,ds,ds"})
    assert problems == [{"key": "camera_models", "message": "lists 3 lens models for 2 cameras"}]
    assert configio.check_values({})[1] == [{"key": "number_camera", "message": "needs a value"}]


def test_read_values_reads_an_mccalib_style_file():
    doc = configio.read_values('''%YAML:1.0
number_x_square: 6      # squares
number_camera: 2
refine_corner: 1
distortion_model: 1
fix_intrinsic: "true"   # hold: them
cam_params_path: "None"
root_path: "/data/rig/Images"
camera_models: [ kb, radtan ]
''')
    assert doc["values"] == {"number_x_square": 6, "number_camera": 2, "distortion_model": 1,
                             "fix_intrinsic": True, "root_path": "/data/rig/Images",
                             "camera_models": ["kb", "radtan"]}
    assert doc["unknown"] == ["refine_corner"]
    with pytest.raises(ValueError, match="calib_param.yml"):
        configio.read_values("%YAML:1.0\nthis is : not : yaml : [")


def _photo_root(tmp_path, n_cam=2, n_img=3, board=True):
    spec_values = {"number_x_square": 5, "number_y_square": 4}
    det = make_detectors(photos.board_specs(spec_values))[0]
    img = det.getBoard().generateImage((500, 400), marginSize=30) if board \
        else np.full((400, 500), 128, np.uint8)
    canvas = np.full((480, 640), 200, np.uint8)
    canvas[40:440, 70:570] = img
    root = tmp_path / "Images"
    for c in range(n_cam):
        d = root / f"Cam_{c + 1:03d}"
        d.mkdir(parents=True)
        for k in range(n_img):
            cv2.imwrite(str(d / f"{k:04d}.png"), canvas)
    return root, spec_values


def test_scan_reads_the_layout_the_calibration_reads(tmp_path):
    root, _ = _photo_root(tmp_path)
    doc = photos.scan_photos(str(root))
    assert doc["cam_prefix"] == "Cam_" and doc["number_camera"] == 2 and doc["problems"] == []
    assert doc["cameras"][1] == {"index": 1, "folder": "Cam_002", "images": 3, "width": 640,
                                 "height": 480, "frames": 3}
    assert doc["shared_frames"] == 3


def test_scan_says_what_is_wrong(tmp_path):
    flat = tmp_path / "flat"
    flat.mkdir()
    cv2.imwrite(str(flat / "0001.png"), np.zeros((8, 8), np.uint8))
    assert "its own subfolder" in photos.scan_photos(str(flat))["problems"][0]
    assert photos.scan_photos(str(tmp_path / "missing"))["problems"] == ["This folder does not exist."]

    root, _ = _photo_root(tmp_path)
    (root / "Cam_004").mkdir()                                   # gap: 003 is missing
    (root / "Cam_002" / "0000.png").rename(root / "Cam_002" / "1700000000.123456.png")
    for k in (1, 2):
        (root / "Cam_002" / f"{k:04d}.png").rename(root / "Cam_002" / f"{k + 50:04d}.png")
    text = " ".join(photos.scan_photos(str(root))["problems"])
    assert "Cam_004 is not used" in text and "timestamp" in text
    assert "no frame index is shared" in text.lower()


def test_board_check_finds_the_right_board_and_not_a_wrong_one(tmp_path):
    root, spec = _photo_root(tmp_path)
    good = photos.check_board(str(root), {**spec, "number_camera": 2})
    assert good["found"] == good["tried"] == 6 and "all 6" in good["message"]
    assert good["cameras"][0]["corners_max"] == good["cameras"][0]["corners_of"] == 12
    blank_root, _ = _photo_root(tmp_path / "blank", board=False)
    none = photos.check_board(str(blank_root), {**spec, "number_camera": 2})
    assert none["found"] == 0 and "not found in any" in none["message"]


@pytest.fixture()
def client(tmp_path):
    root, _ = _photo_root(tmp_path / "work")
    return TestClient(create_app([str(tmp_path / "work")])), root


def test_browse_lists_folders_and_stays_inside_the_roots(client):
    c, root = client
    top = c.get("/api/browse").json()
    assert [d["ref"] for d in top["dirs"]] == ["0:"]
    work = c.get("/api/browse", params={"ref": "0:"}).json()
    assert [d["name"] for d in work["dirs"]] == ["Images"] and work["parent"] is None
    images = c.get("/api/browse", params={"ref": "0:Images"}).json()
    assert [(d["name"], d["images"]) for d in images["dirs"]] == [("Cam_001", 3), ("Cam_002", 3)]
    assert images["path"] == str(root) and images["parent"] == "0:"
    for ref in ("0:..", "0:../..", "1:", "0:Images/../../.."):
        assert c.get("/api/browse", params={"ref": ref}).status_code == 404


def test_scan_and_board_check_routes(client, tmp_path):
    c, root = client
    doc = c.post("/api/photos/scan", json={"root_path": str(root)}, headers=SEND).json()
    assert doc["number_camera"] == 2
    values = {"root_path": str(root), "number_camera": 2, "number_x_square": 5, "number_y_square": 4}
    assert c.post("/api/photos/check-board", json={"values": values}, headers=SEND).json()["found"] == 6
    outside = c.post("/api/photos/scan", json={"root_path": str(tmp_path)}, headers=SEND)
    assert outside.status_code == 400 and "outside" in outside.json()["detail"]


def test_config_check_uses_the_cli_loader(client):
    c, root = client
    ok = c.post("/api/config/check", headers=SEND, json={"values": {
        "root_path": str(root), "number_camera": 2, "camera_models": ["kb", "kb"]}}).json()
    assert ok["ok"] and "number_camera: 2" in ok["yaml"]
    # production refuses to run without uncertainty limits: the loader's own message comes back
    prod = c.post("/api/config/check", headers=SEND, json={"values": {
        "root_path": str(root), "number_camera": 2, "profile": "production"}}).json()
    assert not prod["ok"] and "max_rotation_sigma_deg" in prod["problems"][0]["message"]
    nowhere = c.post("/api/config/check", headers=SEND, json={"values": {"number_camera": 2}}).json()
    assert nowhere["problems"] == [{"key": "root_path", "message": "choose the folder with the photos"}]


def test_changes_need_the_studio_header_and_the_same_origin(client):
    c, root = client
    body = {"root_path": str(root)}
    assert c.post("/api/photos/scan", json=body).status_code == 403
    cross = c.post("/api/photos/scan", json=body, headers={**SEND, "origin": "http://evil.example"})
    assert cross.status_code == 403
    same = c.post("/api/photos/scan", json=body, headers={**SEND, "origin": "http://testserver"})
    assert same.status_code == 200


def test_requests_for_another_host_name_are_refused(tmp_path):
    c = TestClient(create_app([str(tmp_path)], allowed_hosts=["127.0.0.1", "localhost"]),
                   base_url="http://localhost:8765")
    assert c.get("/api/health").status_code == 200
    rebound = TestClient(create_app([str(tmp_path)], allowed_hosts=["127.0.0.1", "localhost"]),
                         base_url="http://evil.example:8765")
    assert rebound.get("/api/health").status_code == 403


def test_read_only_studio_has_no_setup_or_runs(tmp_path):
    c = TestClient(create_app([str(tmp_path)], read_only=True))
    assert c.get("/api/health").json()["can_run"] is False
    assert c.get("/api/jobs").json() == []
    r = c.post("/api/jobs", json={"name": "x", "values": {}}, headers=SEND)
    assert r.status_code == 403 and "read-only" in r.json()["detail"]
