"""Studio exports LDC meshes and Isaac LUTs for cameras the user already has: from a file, a
Studio result or typed values, through ``ds-msp export`` (FR-UI-025)."""
import json
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from ds_msp.data.observations import RigState  # noqa: E402
from ds_msp.io.mccalib import save_mccalib_cameras  # noqa: E402
from ds_msp.models.registry import _BY_NAME  # noqa: E402
from ds_msp.studio.server import WRITE_HEADER, create_app  # noqa: E402

pytestmark = pytest.mark.req("FR-UI-025")
SEND = {WRITE_HEADER: "1"}
DS = {"model": "ds", "fx": 380, "fy": 380, "cx": 639.5, "cy": 399.5, "xi": -0.2, "alpha": 0.6,
      "width": 1280, "height": 800}


@pytest.fixture()
def setup(tmp_path):
    cams = {0: _BY_NAME["ds"](380, 380, 639.5, 399.5, -0.2, 0.6), 1: _BY_NAME["kb"].sample()}
    sizes = {0: (1280, 800), 1: (int(round(2 * cams[1].cx)), int(round(2 * cams[1].cy)))}
    rig = RigState(cameras=cams, T_c_g={0: np.eye(4), 1: np.eye(4)}, ref_cam_id=0,
                   object_poses={}, objects={}, img_size=sizes)
    path = tmp_path / "lenses" / "calibrated_cameras_data.yml"
    path.parent.mkdir()
    save_mccalib_cameras(rig, str(path))
    return tmp_path, path, TestClient(create_app([str(tmp_path)]))


def _wait(c, job_id, timeout=120.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        doc = c.get("/api/job", params={"id": job_id}).json()
        if doc["state"] not in ("queued", "running"):
            return doc
        time.sleep(0.2)
    raise AssertionError("export did not finish")


def test_a_file_lists_its_cameras(setup):
    root, path, c = setup
    doc = c.post("/api/export/cameras", json={"source": {"kind": "file", "path": str(path)}},
                 headers=SEND).json()
    assert [(r["camera"], r["model"]) for r in doc["cameras"]] == [("camera_0", "ds"),
                                                                    ("camera_1", "kb")]
    bad = c.post("/api/export/cameras", json={"source": {"kind": "file", "path": "/etc/hosts"}},
                 headers=SEND)
    assert bad.status_code == 400 and "outside" in bad.json()["detail"]


def test_the_check_previews_each_mesh_without_writing(setup):
    root, path, c = setup
    spec = {"source": {"kind": "file", "path": str(path)}, "cameras": ["camera_0"],
            "ldc": {"downsample_factor": 4, "balance": 0.5}}
    before = sorted(p.name for p in root.rglob("*"))
    doc = c.post("/api/export/check", json={"spec": spec}, headers=SEND).json()
    assert doc["cameras"][0]["ldc"]["max_px"] < 1.0
    assert sorted(p.name for p in root.rglob("*")) == before
    for bad in ({"ldc": {"balance": 3}}, {"ldc": None, "lut": None}, {"cameras": ["camera_7"]},
                {"cameras": []}):
        r = c.post("/api/export/check", json={"spec": {**spec, **bad}}, headers=SEND)
        assert r.status_code == 400 and r.json()["problems"]


def test_an_export_job_runs_the_command_and_matches_the_terminal(setup):
    root, path, c = setup
    spec = {"source": {"kind": "file", "path": str(path)}, "cameras": ["camera_0", "camera_1"],
            "ldc": {"downsample_factor": 4, "balance": 0.5}, "lut": {"texture_size": [32, 24]}}
    job = c.post("/api/jobs/export", json={"root": 0, "name": "meshes", "spec": spec},
                 headers=SEND).json()
    doc = _wait(c, job["id"])
    assert doc["state"] == "done", doc
    assert doc["command"].startswith("ds-msp export --input ")
    rows = doc["exports"]["ldc"]["cameras"]
    assert [r["camera"] for r in rows] == ["camera_0", "camera_1"] and all("record" in r for r in rows)
    got = c.get("/api/run/exports", params={"id": job["id"]}).json()["exports"]
    assert got == doc["exports"]
    f = c.get("/api/run/export-file", params={"id": job["id"], "path": "ldc/camera_0_ldc.json"})
    assert f.status_code == 200 and json.loads(f.content)["format_version"] == 1
    assert c.get("/api/run/export-file", params={"id": job["id"],
                                                 "path": "../lenses/calibrated_cameras_data.yml"}
                 ).status_code == 404

    from ds_msp.cli import run as run_root_cli
    import shlex
    args = shlex.split(doc["command"])[2:]
    k = args.index("--output-dir")
    args[k + 1] = str(root / "terminal")
    assert run_root_cli(["export", *args]) == 0
    for sub in ("ldc", "isaac_lut"):
        for f in sorted((root / "meshes" / sub).iterdir()):
            assert f.read_bytes() == (root / "terminal" / sub / f.name).read_bytes(), f.name
    assert (root / "meshes" / "exports.json").read_bytes() == \
        (root / "terminal" / "exports.json").read_bytes()


def test_typed_values_are_written_beside_the_export(setup):
    root, path, c = setup
    spec = {"source": {"kind": "typed", "cameras": [DS]}, "cameras": ["camera_0"],
            "ldc": {"downsample_factor": 4, "balance": 0.5}}
    listed = c.post("/api/export/cameras", json={"source": spec["source"]}, headers=SEND).json()
    assert listed["cameras"][0]["model"] == "ds"
    job = c.post("/api/jobs/export", json={"root": 0, "name": "typed", "spec": spec},
                 headers=SEND).json()
    doc = _wait(c, job["id"])
    assert doc["state"] == "done", doc
    assert (root / "typed" / "typed_cameras.yml").is_file()
    assert "typed_cameras.yml" in doc["command"]
    bad = {**spec, "source": {"kind": "typed", "cameras": [{**DS, "fx": ""}]}}
    r = c.post("/api/jobs/export", json={"root": 0, "name": "typed2", "spec": bad}, headers=SEND)
    assert r.status_code == 400
    assert not (root / "typed2").exists()


def test_a_name_in_use_or_a_bad_spec_creates_nothing(setup):
    root, path, c = setup
    (root / "taken").mkdir()
    spec = {"source": {"kind": "file", "path": str(path)}, "cameras": ["camera_0"],
            "ldc": {"downsample_factor": 4, "balance": 0.5}}
    r = c.post("/api/jobs/export", json={"root": 0, "name": "taken", "spec": spec}, headers=SEND)
    assert r.status_code == 400 and "already exists" in r.json()["detail"]
    r = c.post("/api/jobs/export", json={"root": 0, "name": "x", "spec": {**spec, "ldc": None}},
               headers=SEND)
    assert r.status_code == 400 and not (root / "x").exists()


def test_a_read_only_studio_refuses_exports(setup):
    root, path, _c = setup
    c = TestClient(create_app([str(root)], read_only=True))
    spec = {"source": {"kind": "file", "path": str(path)}, "cameras": ["camera_0"],
            "ldc": {"downsample_factor": 4, "balance": 0.5}}
    r = c.post("/api/jobs/export", json={"root": 0, "name": "x", "spec": spec}, headers=SEND)
    assert r.status_code in (403, 404, 405)
