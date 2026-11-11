"""Pictures of an LDC mesh and an Isaac LUT at work (FR-UI-026): they are drawn the way the board
and Isaac Sim use the files, so a correct mesh straightens the test room exactly as an ideal
pinhole camera would see it."""
import base64
import time

import cv2
import numpy as np
import pytest

from ds_msp.ldc import TI_LDC_MeshGenerator
from ds_msp.models.registry import _BY_NAME
from ds_msp.studio import lensview

pytestmark = pytest.mark.req("FR-UI-026")
DS = dict(fx=380.0, fy=380.0, cx=639.5, cy=399.5, xi=-0.2, alpha=0.6)


def _pinhole_room(K, w, h):
    u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
    return lensview.room_colours(rays, np.ones((h, w), bool))


def _mesh(model, size, factor=3, balance=1.0):
    r = TI_LDC_MeshGenerator(model).generate_mesh_and_intrinsics(size[0], size[1], factor, balance)
    return r, {"mesh_lut": r["mesh_lut"], "K_new": r["K_new"], "downsample_factor": factor, "output_size": size}


def test_the_room_has_a_colour_per_wall_and_black_where_nothing_is_seen():
    o = 0.13                                   # off the grid lines and the centre cross
    rays = np.array([[o, o, 1], [o, o, -1], [1, o, o], [-1, o, o], [o, 1, o], [o, -1, o], [0.3, 0.2, 1]], float)
    c = lensview.room_colours(rays, np.array([1, 1, 1, 1, 1, 1, 0], bool)).astype(int)
    assert len({tuple(x) for x in c[:6]}) == 6
    assert (c[6] == 0).all()


@pytest.mark.parametrize("name", ["ds", "kb", "eucm"])
def test_a_corrected_room_matches_an_ideal_pinhole_view(name):
    model = _BY_NAME["ds"](**DS) if name == "ds" else _BY_NAME[name].sample()
    size = (int(round(2 * model.cx + 1)), int(round(2 * model.cy + 1)))
    r, mesh = _mesh(model, size)
    got = lensview.ldc_view(model, size, mesh["mesh_lut"], mesh["K_new"], 3, size, size[0], None, False)
    want = _pinhole_room(r["K_new"], *size)
    seen = got.any(axis=-1)
    differs = np.abs(got.astype(int) - want.astype(int)).max(axis=-1) > 40
    assert seen.mean() > 0.5
    assert differs[seen].mean() < 0.02          # only pixels on a grid line's edge


def test_a_photo_is_corrected_like_the_room():
    model = _BY_NAME["ds"](**DS)
    size = (1280, 800)
    r, mesh = _mesh(model, size)
    photo = lensview.camera_view(model, size, size[0], None)       # the room as the camera sees it
    by_photo = lensview.ldc_view(model, size, mesh["mesh_lut"], mesh["K_new"], 3, size, 640, photo, False)
    by_rays = lensview.ldc_view(model, size, mesh["mesh_lut"], mesh["K_new"], 3, size, 640, None, False)
    assert by_photo.shape == by_rays.shape == (400, 640, 3)
    assert (np.abs(by_photo.astype(int) - by_rays.astype(int)).max(axis=-1) > 60).mean() < 0.05


def test_the_overlay_marks_a_coarse_mesh_and_not_a_fine_one():
    model = _BY_NAME["ds"](**DS)
    size = (1280, 800)
    tint = []
    for factor in (2, 6):
        _r, mesh = _mesh(model, size, factor, 0.0)
        a = lensview.ldc_view(model, size, mesh["mesh_lut"], mesh["K_new"], factor, size, 640, None, False)
        b = lensview.ldc_view(model, size, mesh["mesh_lut"], mesh["K_new"], factor, size, 640, None, True)
        tint.append((np.abs(a.astype(int) - b.astype(int)).max(axis=-1) > 0).mean())
    assert tint[0] < 0.01 < tint[1]


def test_the_lut_drawn_from_its_file_matches_the_preview(tmp_path):
    from ds_msp.isaac_sim import export_lut
    from ds_msp.isaac_sim.lut import _read_rgb32_exr

    model = _BY_NAME["ds"](**DS)
    bundle = export_lut(model, 1280, 800, tmp_path, texture_width=320, texture_height=200, stem="c")
    from_file = lensview.lut_view_from_texture(_read_rgb32_exr(bundle.paths.ray_enter), 320)
    preview = lensview.lut_view_from_model(model, (1280, 800), 320)
    assert from_file.shape == preview.shape
    assert (np.abs(from_file.astype(int) - preview.astype(int)).max(axis=-1) > 40).mean() < 0.03
    # the lens view and Isaac's view of the room agree: the LUT keeps the image the right way up
    lens = lensview.camera_view(model, (1280, 800), 320, None)
    assert (np.abs(lens.astype(int) - preview.astype(int)).max(axis=-1) > 40).mean() < 0.03


def test_pictures_are_quick_enough_to_follow_a_slider():
    model = _BY_NAME["kb"].sample()
    size = (int(round(2 * model.cx + 1)), int(round(2 * model.cy + 1)))
    _r, mesh = _mesh(model, size, 4, 0.5)
    t = time.perf_counter()
    lensview.encode({"a": lensview.ldc_view(model, size, mesh["mesh_lut"], mesh["K_new"], 4, size, 720, None, True)})
    assert time.perf_counter() - t < 3.0


fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def test_the_routes_draw_a_preview_and_the_written_files(tmp_path):
    from fastapi.testclient import TestClient

    from ds_msp.studio.server import WRITE_HEADER, create_app

    c = TestClient(create_app([str(tmp_path)]))
    send = {WRITE_HEADER: "1"}
    cam = {"model": "ds", **{k: v for k, v in DS.items()}, "width": 1280, "height": 800}
    spec = {"source": {"kind": "typed", "cameras": [cam]}, "cameras": ["camera_0"],
            "ldc": {"downsample_factor": 4, "balance": 0.5}, "lut": {"texture_size": [160, 100]}}
    for kind, keys in (("ldc", {"camera", "corrected"}), ("lut", {"render"})):
        d = c.post("/api/export/view", json={"spec": spec, "camera": "camera_0", "kind": kind,
                                             "overlay": True, "width": 400}, headers=send).json()
        assert set(d["images"]) == keys and d["from_files"] is False
        img = cv2.imdecode(np.frombuffer(base64.b64decode(d["images"][sorted(keys)[0]]["src"].split(",")[1]), np.uint8), 1)
        assert img.shape[1] == 400
    r = c.post("/api/export/view", json={"spec": spec, "camera": "camera_9", "kind": "ldc"}, headers=send)
    assert r.status_code == 400

    job = c.post("/api/jobs/export", json={"root": 0, "name": "e", "spec": spec}, headers=send).json()
    end = time.monotonic() + 120
    while c.get("/api/job", params={"id": job["id"]}).json()["state"] in ("queued", "running"):
        assert time.monotonic() < end
        time.sleep(0.2)
    for kind in ("ldc", "lut"):
        d = c.get("/api/run/export-view", params={"id": job["id"], "camera": "camera_0", "kind": kind,
                                                  "width": 320}).json()
        assert d["from_files"] is True and d["images"]
    assert c.get("/api/run/export-view", params={"id": job["id"], "camera": "camera_5"}).status_code == 400
