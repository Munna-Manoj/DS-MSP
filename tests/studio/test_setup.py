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


def _board_photo(n_x, n_y, *, offset=0, size=(960, 720), bend=0.0, seed=0, fill=0.75):
    """A photo of an ``n_x`` by ``n_y`` ChArUco board: tilted, and with ``bend`` > 0 pulled
    through a strong barrel distortion like a fisheye lens."""
    import cv2
    import numpy as np

    from ds_msp.detect.charuco import _DICT, BoardSpec, _make_board

    board = _make_board(BoardSpec(n_x, n_y, 0.04, 0.04 * fill, 1.0), cv2.aruco.getPredefinedDictionary(_DICT),
                        offset, True)
    cell = 60
    flat = board.generateImage((n_x * cell, n_y * cell), marginSize=0)
    w, h = size
    rng = np.random.default_rng(seed)
    scale = 0.62 * min(w / flat.shape[1], h / flat.shape[0])
    bw, bh = flat.shape[1] * scale, flat.shape[0] * scale
    x0, y0 = (w - bw) / 2, (h - bh) / 2
    dst = (np.array([[x0, y0], [x0 + bw, y0], [x0 + bw, y0 + bh], [x0, y0 + bh]])
           + rng.uniform(-0.08, 0.08, (4, 2)) * [bw, bh]).astype(np.float32)
    src = np.float32([[0, 0], [flat.shape[1], 0], [flat.shape[1], flat.shape[0]], [0, flat.shape[0]]])
    img = cv2.warpPerspective(flat, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=255)
    if bend:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        u, v = (xx - w / 2) / (w / 2), (yy - h / 2) / (w / 2)
        k = 1 + bend * (u * u + v * v)
        img = cv2.remap(img, u * k * (w / 2) + w / 2, v * k * (w / 2) + h / 2, cv2.INTER_LINEAR, borderValue=255)
    return img


@pytest.mark.req("FR-UI-014")
@pytest.mark.parametrize("n_x,n_y,bend", [(6, 5, 0.0), (5, 5, 0.0), (5, 7, 0.0), (9, 6, 0.0),
                                          (4, 3, 0.0), (12, 8, 0.0), (7, 5, 0.45), (6, 9, 0.45)])
def test_board_size_is_read_from_the_markers(tmp_path, n_x, n_y, bend):
    import cv2

    from ds_msp.studio.photos import find_board

    cam = tmp_path / "Cam_001"
    cam.mkdir()
    for k in range(3):
        cv2.imwrite(str(cam / f"{k:04d}.png"), _board_photo(n_x, n_y, bend=bend, seed=k))
    out = find_board(str(tmp_path), {"number_camera": 1})
    assert out["ok"] and (out["number_x_square"], out["number_y_square"]) == (n_x, n_y), out


@pytest.mark.req("FR-UI-015")
@pytest.mark.parametrize("fill", [0.5, 0.6, 0.75, 0.9])
@pytest.mark.parametrize("bend", [0.0, 0.45])
def test_marker_width_is_measured_from_the_photos(tmp_path, fill, bend):
    """The ratio of marker to square is read from markers that touch at a corner, to within
    0.02 here, through a tilt and a strong barrel distortion."""
    import cv2

    from ds_msp.studio.photos import find_board

    cam = tmp_path / "Cam_001"
    cam.mkdir()
    for k in range(4):
        cv2.imwrite(str(cam / f"{k:04d}.png"), _board_photo(7, 5, bend=bend, seed=k, fill=fill))
    out = find_board(str(tmp_path), {"number_camera": 1})
    assert out["marker_fill_pairs"] >= 20
    assert out["marker_fill"] == pytest.approx(fill, abs=0.02), out
    # the board size is read as before, whatever the marker width
    assert out["ok"] and (out["number_x_square"], out["number_y_square"]) == (7, 5), out


@pytest.mark.req("FR-UI-015")
def test_marker_width_is_not_stated_from_a_few_markers(tmp_path):
    import cv2

    from ds_msp.studio.photos import find_board

    cam = tmp_path / "Cam_001"
    cam.mkdir()
    img = _board_photo(6, 5)
    img[:, 420:] = 255            # a strip of the board: a handful of touching markers
    cv2.imwrite(str(cam / "0000.png"), img)
    out = find_board(str(tmp_path), {"number_camera": 1})
    assert 0 < out["marker_fill_pairs"] < 20 and out["marker_fill"] is None, out


@pytest.mark.req("FR-UI-014")
def test_board_size_is_not_guessed_without_enough_of_the_board(tmp_path):
    import cv2
    import numpy as np

    from ds_msp.studio.photos import find_board

    cam = tmp_path / "Cam_001"
    cam.mkdir()
    cv2.imwrite(str(cam / "0000.png"), np.full((480, 640), 200, np.uint8))
    blank = find_board(str(tmp_path), {"number_camera": 1})
    assert not blank["ok"] and "No board markers" in blank["message"]
    # only the top rows in view: the last markers are never seen, so the size is not settled
    img = _board_photo(6, 8)
    img[330:] = 255
    cv2.imwrite(str(cam / "0000.png"), img)
    part = find_board(str(tmp_path), {"number_camera": 1})
    assert not part["ok"] or (part["number_x_square"], part["number_y_square"]) == (6, 8), part


@pytest.mark.req("FR-UI-014")
def test_board_check_returns_the_corners_and_thumbnails_stay_inside(tmp_path):
    import cv2

    from ds_msp.studio.photos import check_board, thumbnail

    cam = tmp_path / "Cam_001"
    cam.mkdir()
    for k in range(2):
        cv2.imwrite(str(cam / f"{k:04d}.png"), _board_photo(6, 5, size=(1600, 1200), seed=k))
    doc = check_board(str(tmp_path), {"number_camera": 1, "number_x_square": 6, "number_y_square": 5})
    sample = doc["cameras"][0]["sample"]
    assert sample["width"] == 1600 and len(sample["corners"]) == 20 and sample["photo"] in (0, 1)
    assert sample["boards"] == [0] * 20
    assert all(0 <= u < 1600 and 0 <= v < 1200 for u, v in sample["corners"])
    jpg = thumbnail(str(tmp_path), 0, sample["photo"])
    small = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert small.shape == (480, 640)
    assert thumbnail(str(tmp_path), 0, 9) is None and thumbnail(str(tmp_path), 4, 0) is None
    c = TestClient(create_app([str(tmp_path)]))
    assert c.get("/api/photos/thumb", params={"path": str(tmp_path), "cam": 0}).headers["content-type"] == "image/jpeg"
    assert c.get("/api/photos/thumb", params={"path": "/etc", "cam": 0}).status_code == 400
    found = c.post("/api/photos/find-board", json={"values": {"root_path": str(tmp_path), "number_camera": 1}},
                   headers={"x-dsmsp-studio": "1"}).json()
    assert found["ok"] and found["number_x_square"] == 6
