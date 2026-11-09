"""The Studio JSON API: listing, reading, file access and its confinement to the roots."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from ds_msp.studio.server import create_app  # noqa: E402

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
