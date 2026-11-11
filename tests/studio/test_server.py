"""The Studio JSON API: listing, reading, file access and its confinement to the roots."""
import os
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from ds_msp.studio.server import WRITE_HEADER, create_app  # noqa: E402

pytestmark = pytest.mark.req("FR-UI-001", "FR-UI-003")


@pytest.fixture(scope="module")
def client(results_root):
    return TestClient(create_app([str(results_root)]))


def test_lists_and_opens_a_run(client):
    assert client.get("/api/health").json()["ok"] is True
    runs = client.get("/api/runs").json()
    assert [r["id"] for r in runs] == ["0:lab/run_a"]
    assert runs[0]["n_cameras"] == 3
    doc = client.get("/api/run", params={"id": "0:lab/run_a"}).json()
    assert doc["id"] == "0:lab/run_a" and doc["summary"]["n_cameras"] == 3


def test_serves_known_files_only(client):
    ok = client.get("/api/run/file", params={"id": "0:lab/run_a", "name": "calibration_report.txt"})
    assert ok.status_code == 200 and "DS-MSP" in ok.text
    for name in ("../calibration_report.txt", "/etc/passwd", "rig.pkl"):
        r = client.get("/api/run/file", params={"id": "0:lab/run_a", "name": name})
        assert r.status_code == 404


@pytest.mark.parametrize("run_id", ["0:..", "0:../..", "0:lab/../../x", "1:lab/run_a", "x", "0:empty_dir"])
def test_refuses_paths_outside_the_roots(client, run_id):
    assert client.get("/api/run", params={"id": run_id}).status_code == 404


def test_schema_endpoint(client):
    doc = client.get("/api/schema/rig-config").json()
    assert "lens_gate" in doc["properties"] and doc["required"] == ["number_camera"]


def test_unknown_api_path_is_404_not_the_page(client):
    assert client.get("/api/nope").status_code == 404
    page = client.get("/")
    assert page.status_code == 200 and "DS-MSP Studio" in page.text


def test_token_gate(results_root):
    c = TestClient(create_app([str(results_root)], token="s3cret"))
    assert c.get("/api/runs").status_code == 401
    assert c.get("/api/runs", headers={"x-dsmsp-token": "wrong"}).status_code == 401
    assert c.get("/api/runs", headers={"x-dsmsp-token": "s3cret"}).status_code == 200
    first = c.get("/?token=s3cret")
    assert first.status_code == 200 and "dsmsp_token" in first.headers.get("set-cookie", "")
    assert c.get("/api/runs").status_code == 200                # cookie kept by the client


def test_missing_root_is_an_error(tmp_path):
    with pytest.raises(NotADirectoryError):
        create_app([str(tmp_path / "nope")])


def test_cli_refuses_a_missing_root(tmp_path, capsys):
    from ds_msp.studio.cli import run
    assert run(["--root", str(tmp_path / "nope"), "--no-browser"]) == 2
    assert "not a folder" in capsys.readouterr().err


def test_top_level_command_dispatches_to_studio(tmp_path):
    from ds_msp.cli import run
    assert run(["studio", "--root", str(tmp_path / "nope"), "--no-browser"]) == 2


@pytest.mark.req("FR-UI-005")
def test_diagnostics_endpoint(client):
    d = client.get("/api/run/diagnostics", params={"id": "0:lab/run_a", "cam": 0}).json()
    assert d["cam"] == 0 and d["residuals"]["n_shown"] > 0 and d["lens"]["nx"] == 24
    assert client.get("/api/run/diagnostics", params={"id": "0:..", "cam": 0}).status_code == 404
    assert client.get("/api/run/diagnostics", params={"id": "0:lab/run_a", "cam": -1}).status_code == 422


@pytest.mark.req("FR-UI-016")
def test_folders_outside_the_roots_need_anywhere(results_root, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "lens.yml").write_text("x: 1\n")
    head = {"x-dsmsp-studio": "1"}
    limited = TestClient(create_app([str(results_root)]))
    assert limited.get("/api/browse", params={"ref": f"abs:{outside}"}).status_code == 404
    assert limited.post("/api/roots/add", json={"path": str(outside)}, headers=head).status_code == 403
    assert limited.post("/api/photos/scan", json={"root_path": str(outside)}, headers=head).status_code == 400
    assert limited.get("/api/browse").json()["anywhere"] is False

    free = TestClient(create_app([str(results_root)], anywhere=True))
    top = free.get("/api/browse").json()
    assert top["anywhere"] is True and {p["name"] for p in top["places"]} == {"Home folder", "Whole computer"}
    doc = free.get("/api/browse", params={"ref": f"abs:{outside}"}).json()
    assert doc["path"] == str(outside) and [f["name"] for f in doc["files"]] == ["lens.yml"]
    assert doc["parent"] == f"abs:{outside.parent}"
    assert free.post("/api/photos/scan", json={"root_path": str(outside)}, headers=head).status_code == 200
    # the whole disk is too broad to search for results; a folder is fine, and only once
    assert free.post("/api/roots/add", json={"path": "/"}, headers=head).status_code == 400
    k = free.post("/api/roots/add", json={"path": str(outside)}, headers=head).json()["index"]
    assert free.post("/api/roots/add", json={"path": str(outside)}, headers=head).json()["index"] == k == 1
    assert [d["path"] for d in free.get("/api/browse").json()["dirs"]][k] == str(outside)

    # read-only always stays inside the roots
    viewer = TestClient(create_app([str(results_root)], anywhere=True, read_only=True))
    assert viewer.get("/api/browse", params={"ref": f"abs:{outside}"}).status_code == 404


@pytest.mark.req("FR-UI-016")
def test_reading_anywhere_is_behind_a_token(monkeypatch, tmp_path):
    from ds_msp.studio import cli

    seen = {}
    monkeypatch.setattr("ds_msp.studio.server.create_app",
                        lambda *a, **k: seen.update(k) or (_ for _ in ()).throw(NotADirectoryError("stop")))
    assert cli.serve([str(tmp_path)], open_browser=False) == 2
    assert seen["anywhere"] is True and seen["token"]
    seen.clear()
    assert cli.serve([str(tmp_path)], open_browser=False, only_roots=True) == 2
    assert seen["anywhere"] is False and seen["token"] is None
    seen.clear()
    assert cli.serve([str(tmp_path)], open_browser=False, read_only=True) == 2
    assert seen["anywhere"] is False and seen["token"] is None


@pytest.mark.req("FR-UI-016")
@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root can read every folder")
def test_a_locked_folder_is_listed_and_does_not_break_the_list(results_root, tmp_path):
    (tmp_path / "open").mkdir()
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        free = TestClient(create_app([str(results_root)], anywhere=True))
        doc = free.get("/api/browse", params={"ref": f"abs:{tmp_path}"})
        assert doc.status_code == 200
        assert {d["name"]: d["locked"] for d in doc.json()["dirs"]} == {"locked": True, "open": False}
        inside = free.get("/api/browse", params={"ref": f"abs:{locked}"})
        assert inside.status_code == 403 and "permission" in inside.json()["detail"]
        scan = free.post("/api/photos/scan", json={"root_path": str(locked / "x")},
                         headers={WRITE_HEADER: "1"})
        assert scan.status_code == 403 and "permission" in scan.json()["detail"]
    finally:
        locked.chmod(0o755)
