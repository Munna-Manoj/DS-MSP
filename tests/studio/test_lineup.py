"""Saying which folder is which camera, what stops a run before it starts, and typed lens values."""
import json
import os

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from ds_msp.io.mccalib import load_camera  # noqa: E402
from ds_msp.studio import lineup  # noqa: E402
from ds_msp.studio.photos import scan_photos  # noqa: E402
from ds_msp.studio.server import WRITE_HEADER, create_app  # noqa: E402
from ds_msp.studio.sources import scan_intrinsics  # noqa: E402

SEND = {WRITE_HEADER: "1"}
FRAMES = [f"{i:06d}.png" for i in range(3)]


def _layout(root, *folders, files=FRAMES):
    for folder in folders:
        (root / folder).mkdir(parents=True)
        for name in files:
            cv2.imwrite(str(root / folder / name), np.zeros((60, 80), np.uint8))
    return root


@pytest.mark.req("FR-UI-019")
@pytest.mark.parametrize("folders", [
    ("front", "left"), ("cam1", "cam2"), ("Cam_1", "Cam_2"), ("cam00", "cam01"),
    ("Cam_000", "Cam_001"), ("Cam_001", "Cam_003"), ("Cam_001", "Camera_001"),
    ("Cam_001", "Cam_002", "front")])
def test_a_folder_with_photos_is_never_left_out_silently(tmp_path, folders):
    doc = scan_photos(str(_layout(tmp_path, *folders)))
    assert doc["needs_lineup"] and doc["problems"]
    assert {f["name"] for f in doc["folders"]} == set(folders)


@pytest.mark.req("FR-UI-019")
def test_correct_names_need_no_lineup(tmp_path):
    (tmp_path / "notes").mkdir()
    doc = scan_photos(str(_layout(tmp_path, "Cam_001", "Cam_002")))
    assert not doc["needs_lineup"] and doc["problems"] == [] and doc["number_camera"] == 2


@pytest.mark.req("FR-UI-019")
def test_folders_are_proposed_in_natural_order_and_odd_ones_out_are_not(tmp_path):
    names = [f["name"] for f in scan_photos(str(_layout(tmp_path / "a", "cam10", "cam2")))["folders"]]
    assert names == ["cam2", "cam10"]
    by = {f["name"]: f for f in scan_photos(str(_layout(tmp_path / "b", "Cam_001", "Cam_002", "front")))["folders"]}
    assert by["Cam_002"]["proposed"] and not by["front"]["proposed"] and "differently" in by["front"]["reason"]
    assert by["front"]["width"] == 80


@pytest.mark.req("FR-UI-019")
def test_an_explicit_prefix_is_a_choice_and_is_only_noted(tmp_path):
    doc = scan_photos(str(_layout(tmp_path, "Cam_001", "Cam_002", "front")), "Cam_")
    assert not doc["needs_lineup"] and doc["problems"] == [] and "front" in doc["notes"][0]


@pytest.mark.req("FR-UI-019")
def test_a_lineup_links_folders_in_the_chosen_order_and_changes_nothing(tmp_path):
    root = _layout(tmp_path / "photos", "front", "rear", "junk")
    before = sorted(os.listdir(root))
    dest = lineup.stage(root, ["rear", "front"], tmp_path / "work")
    doc = scan_photos(str(dest))
    assert doc["number_camera"] == 2 and doc["problems"] == [] and not doc["needs_lineup"]
    assert (dest / "Cam_001").resolve() == (root / "rear").resolve()
    assert doc["lineup"]["cameras"][1] == {"camera": "Cam_002", "folder": "front"}
    assert sorted(os.listdir(root)) == before
    assert lineup.stage(root, ["rear", "front"], tmp_path / "work") == dest


@pytest.mark.req("FR-UI-019")
@pytest.mark.parametrize("folders", [[], ["a", "a"], ["../a"], ["empty"]])
def test_a_lineup_refuses_what_cannot_be_a_camera(tmp_path, folders):
    root = _layout(tmp_path / "p", "a")
    (root / "empty").mkdir()
    with pytest.raises(ValueError):
        lineup.stage(root, folders, tmp_path / "work")


@pytest.mark.req("FR-UI-019")
def test_a_bad_frame_pattern_is_a_problem_not_a_crash(tmp_path):
    root = _layout(tmp_path, "Cam_001")
    assert "cannot be used" in scan_photos(str(root), None, "run(")["problems"][0]
    assert "cannot be used" in scan_photos(str(root), None, r"\d+")["problems"][0]


def _lens(**over):
    return {"model": "radtan", "width": 80, "height": 60, "fx": 50.0, "fy": 51.0, "cx": 40.0, "cy": 30.0,
            "k1": 0.1, "k2": -0.02, "p1": 0.001, "p2": 0.002, "k3": 0.003, **over}


@pytest.mark.req("FR-UI-021")
@pytest.mark.parametrize("model,extra", [
    ("radtan", {"k1": 0.1, "k2": -0.02, "p1": 0.001, "p2": 0.002, "k3": 0.003}),
    ("kb", {"k1": 0.01, "k2": 0.02, "k3": 0.03, "k4": 0.04}),
    ("ds", {"xi": -0.2, "alpha": 0.6}), ("eucm", {"alpha": 0.6, "beta": 1.1}), ("ucm", {"alpha": 0.6})])
def test_typed_values_read_back_as_the_same_lens(tmp_path, model, extra):
    cam = {"model": model, "width": 80, "height": 60, "fx": 50.0, "fy": 51.0, "cx": 40.0, "cy": 30.0, **extra}
    path = lineup.write_lens_values([cam, cam], tmp_path)
    got = load_camera(str(path), 1)
    assert got.name == model
    np.testing.assert_allclose(got.params, [cam[k] for k in lineup.lens_fields(model)])
    assert [c["model"] for c in scan_intrinsics(path)["cameras"]] == [model, model]


@pytest.mark.req("FR-UI-021")
@pytest.mark.parametrize("over,word", [
    ({"fx": ""}, "fx"), ({"k3": None}, "k3"), ({"fx": 0}, "above 0"), ({"fy": "nan"}, "finite"),
    ({"width": 1.5}, "whole"), ({"model": "nope"}, "unknown")])
def test_typed_values_must_be_complete_and_possible(tmp_path, over, word):
    with pytest.raises(ValueError, match=word):
        lineup.write_lens_values([_lens(**over)], tmp_path)


def _keys(values):
    return {p["key"] for p in lineup.run_problems(values)}


@pytest.mark.req("FR-UI-020")
def test_a_run_needs_the_square_size(tmp_path):
    assert "square_size" in _keys({"number_camera": 1})


@pytest.mark.req("FR-UI-020")
def test_keeping_lens_values_needs_them(tmp_path):
    assert "cam_params_path" in _keys({"square_size": 1, "fix_intrinsic": True, "number_camera": 2})


@pytest.mark.req("FR-UI-020")
def test_strict_checking_never_works_out_lenses_from_the_photos(tmp_path):
    lens = str(lineup.write_lens_values([_lens()], tmp_path))
    base = {"square_size": 1, "number_camera": 1, "profile": "production"}
    assert _keys(base) == {"fix_intrinsic", "cam_params_path"}
    assert _keys({**base, "cam_params_path": lens}) == {"fix_intrinsic"}
    assert _keys({**base, "cam_params_path": lens, "fix_intrinsic": True}) == set()


@pytest.mark.req("FR-UI-020")
def test_the_standard_mode_may_work_out_lenses_from_scratch():
    assert _keys({"square_size": 1, "number_camera": 2}) == set()


@pytest.mark.req("FR-UI-020")
def test_kept_lens_values_must_cover_and_match_every_camera(tmp_path):
    lens = str(lineup.write_lens_values([_lens()], tmp_path))
    base = {"square_size": 1, "fix_intrinsic": True, "cam_params_path": lens}
    assert _keys({**base, "number_camera": 2}) == {"cam_params_path"}
    assert _keys({**base, "number_camera": 1, "camera_models": ["kb"]}) == {"camera_models"}
    assert _keys({**base, "number_camera": 1, "camera_models": ["radtan"]}) == set()


@pytest.mark.req("FR-UI-020")
def test_scan_problems_and_a_wrong_camera_count_stop_a_run(tmp_path):
    root = _layout(tmp_path / "bad", "front", "rear")
    assert "root_path" in _keys({"square_size": 1, "root_path": str(root), "number_camera": 2})
    good = _layout(tmp_path / "good", "Cam_001", "Cam_002")
    assert _keys({"square_size": 1, "root_path": str(good), "number_camera": 3}) == {"number_camera"}


@pytest.fixture
def client(tmp_path):
    root = tmp_path / "served"
    _layout(root / "photos", "front", "rear")
    return TestClient(create_app([str(root)])), root


@pytest.mark.req("FR-UI-019")
def test_lineup_over_the_api(client):
    c, root = client
    body = {"root_path": str(root / "photos"), "folders": ["rear", "front"]}
    assert c.post("/api/photos/lineup", json=body).status_code == 403
    staged = c.post("/api/photos/lineup", json=body, headers=SEND).json()["root_path"]
    doc = c.post("/api/photos/scan", json={"root_path": staged}, headers=SEND).json()
    assert doc["number_camera"] == 2 and doc["lineup"]["cameras"][0]["folder"] == "rear"
    thumb = c.get("/api/photos/folder-thumb", params={"path": str(root / "photos"), "folder": "front"})
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    assert c.get("/api/photos/folder-thumb", params={"path": str(root), "folder": "../x"}).status_code == 400


@pytest.mark.req("FR-UI-021")
def test_typed_values_over_the_api(client):
    c, _root = client
    doc = c.post("/api/lens/values", json={"cameras": [_lens()]}, headers=SEND).json()
    assert doc["cameras"][0]["fx"] == 50.0 and lineup.in_work_dir(doc["path"])
    bad = c.post("/api/lens/values", json={"cameras": [_lens(fx="")]}, headers=SEND)
    assert bad.status_code == 400 and "fx" in bad.json()["detail"]


@pytest.mark.req("FR-UI-021")
def test_a_result_keeps_its_own_lineup_and_lens_values(client):
    c, root = client
    staged = c.post("/api/photos/lineup", json={"root_path": str(root / "photos"), "folders": ["front", "rear"]},
                    headers=SEND).json()["root_path"]
    lens = c.post("/api/lens/values", json={"cameras": [_lens(), _lens()]}, headers=SEND).json()["path"]
    job = c.post("/api/jobs", headers=SEND, json={"root": 0, "name": "run", "values": {
        "root_path": staged, "number_camera": 2, "square_size": 1, "fix_intrinsic": True,
        "cam_params_path": lens, "camera_models": ["radtan", "radtan"]}})
    assert job.status_code == 200, job.text
    folder = root / "run"
    assert json.loads((folder / lineup.LINEUP_FILE).read_text())["cameras"][0]["folder"] == "front"
    assert (folder / lineup.LENS_FILE).is_file()
    assert str(folder / lineup.LENS_FILE) in (folder / "calib_param.yml").read_text()
    c.post("/api/job/cancel", json={"id": job.json()["id"]}, headers=SEND)
