"""Calibrations started from Studio: the job runs the rig CLI on the config Studio wrote, is
followed through the CLI's progress log, can be queued and cancelled, and survives a restart
of Studio as a record."""
import json
import os
import stat
import time
import warnings

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from ds_msp.io.mccalib import load_camera, save_mccalib_detected_keypoints  # noqa: E402
from ds_msp.rig.calib_param import calibrate_from_config  # noqa: E402
from ds_msp.studio import jobs as jobs_mod  # noqa: E402
from ds_msp.studio.jobs import JobError, JobManager, error_headline, folder_name, log_tail  # noqa: E402
from ds_msp.studio.server import WRITE_HEADER, create_app  # noqa: E402
from tests.rig._synth import make_rig  # noqa: E402

pytestmark = pytest.mark.req("FR-UI-008")
SEND = {WRITE_HEADER: "1"}


@pytest.fixture(scope="module")
def keypoints(tmp_path_factory):
    """A detected-keypoints file of a 3-camera rig seeing one 5 x 5 board: a calibration from
    it takes a few seconds and needs no images."""
    root = tmp_path_factory.mktemp("studio_jobs")
    obj, obs, img_size, _ext, _models = make_rig(n_cam=3, n_frame=20, noise_px=0.05, seed=2,
                                                 multi_board=False)
    path = root / "capture" / "detected_keypoints_data.yml"
    path.parent.mkdir()
    save_mccalib_detected_keypoints(obs, obj, img_size, str(path))
    values = {"number_camera": 3, "number_board": 1, "number_x_square": 5, "number_y_square": 5,
              "square_size": 0.1, "keypoints_path": str(path), "camera_models": ["radtan"]}
    return root, values


def _wait(read, done, timeout=120.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        doc = read()
        if done(doc):
            return doc
        time.sleep(0.2)
    raise AssertionError(f"timed out: {doc}")


def _script(tmp_path, body):
    """A stand-in for the Python interpreter, to test the runner without a calibration."""
    path = tmp_path / "fake_python.sh"
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_a_job_is_the_cli_run_on_the_written_config(keypoints, tmp_path):
    root, values = keypoints
    c = TestClient(create_app([str(root)]))
    started = c.post("/api/jobs", headers=SEND, json={"root": 0, "name": "first run", "values": values})
    assert started.status_code == 200, started.text
    job = started.json()
    assert job["id"] == "0:first_run" and job["state"] in ("queued", "running")
    assert job["command"].startswith("ds-msp-calibrate-rig --config ")
    done = _wait(lambda: c.get("/api/job", params={"id": job["id"]}).json(),
                 lambda d: d["state"] not in ("queued", "running"))
    assert done["state"] == "done" and done["exit_code"] == 0 and done["has_result"], done["log"][-5:]
    progress = done["progress"]
    assert progress["phase"] == "done" and progress["final_rms"] < 0.5
    names = [s["name"] for s in progress["stages"]]
    assert names[0].startswith("(0)") and any(n.startswith("(c) global joint BA") for n in names)
    assert any("calibrated" in line for line in done["log"])

    folder = root / "first_run"
    record = json.loads((folder / "studio_job.json").read_text())
    assert record["command"][1:4] == ["-m", "ds_msp.rig.cli", "--config"]
    assert [r["id"] for r in c.get("/api/runs").json()] == ["0:first_run"]
    assert c.get("/api/run", params={"id": "0:first_run"}).json()["summary"]["n_cameras"] == 3

    # the same config through the CLI's own entry gives the same lenses
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrate_from_config(str(folder / "calib_param.yml"),
                              {"save_path": str(tmp_path / "cli") + os.sep, "verbose": "false",
                               "webviewer": "false"})
    for cam in range(3):
        a = load_camera(str(folder / "calibrated_cameras_data.yml"), cam)
        b = load_camera(str(tmp_path / "cli" / "calibrated_cameras_data.yml"), cam)
        np.testing.assert_allclose(a.params, b.params, rtol=1e-6)

    # its settings can start the next run, without the old result folder
    again = c.get("/api/config", params={"id": "0:first_run"}).json()
    assert again["values"] == {**values, "camera_models": ["radtan"]} and again["name"] == "first_run"

    # a second Studio over the same folder still lists the job
    assert [j["state"] for j in JobManager([root]).list()] == ["done"]


def test_bad_settings_create_nothing(keypoints):
    root, values = keypoints
    c = TestClient(create_app([str(root)]))
    r = c.post("/api/jobs", headers=SEND, json={"root": 0, "name": "bad", "values": {
        **values, "square_size": "x"}})
    assert r.status_code == 400 and r.json()["problems"] == [
        {"key": "square_size", "message": "must be a number"}]
    r = c.post("/api/jobs", headers=SEND, json={"root": 0, "name": "prod", "values": {
        **values, "profile": "production"}})
    assert r.status_code == 400 and "max_rotation_sigma_deg" in r.json()["detail"]
    r = c.post("/api/jobs", headers=SEND, json={"root": 0, "name": "../escape", "values": values})
    assert r.status_code == 400 and r.json()["problems"][0]["key"] == "name"
    r = c.post("/api/jobs", headers=SEND, json={"root": 0, "name": "capture", "values": values})
    assert r.status_code == 400 and "already exists" in r.json()["detail"]
    assert not (root / "bad").exists() and not (root / "prod").exists()
    assert c.get("/api/jobs").json() == [] or all(j["name"] != "bad" for j in c.get("/api/jobs").json())


@pytest.mark.skipif(os.name != "posix", reason="uses a shell script as the interpreter")
def test_jobs_queue_and_cancel(keypoints, tmp_path):
    _root, values = keypoints
    work = tmp_path / "work"
    work.mkdir()
    values = dict(values)
    mgr = JobManager([work, _root], python=_script(tmp_path, "echo started; sleep 60"))
    first = mgr.submit(0, "one", values)
    second = mgr.submit(0, "two", values)
    _wait(lambda: mgr.snapshot(first.id)["log"], lambda log: log == ["started"], timeout=20)
    listed = {j["name"]: j for j in mgr.list()}
    assert listed["two"]["state"] == "queued" and listed["two"]["queue_position"] == 1
    assert mgr.cancel(second.id).state == "cancelled"
    mgr.cancel(first.id)
    ended = _wait(lambda: mgr.get(first.id), lambda j: j.state != "running", timeout=20)
    assert ended.state == "cancelled" and ended.error is None
    assert mgr.snapshot(first.id)["log"] == ["started"]
    assert json.loads((work / "one" / "studio_job.json").read_text())["state"] == "cancelled"


@pytest.mark.skipif(os.name != "posix", reason="uses a shell script as the interpreter")
def test_a_failed_run_reports_the_error_it_ended_on(keypoints, tmp_path):
    _root, values = keypoints
    work = tmp_path / "work"
    work.mkdir()
    mgr = JobManager([work, _root], python=_script(
        tmp_path, 'echo "Traceback (most recent call last):"; '
                  'echo "ValueError: camera 2 has no images"; exit 1'))
    job = mgr.submit(0, "broken", dict(values))
    ended = _wait(lambda: mgr.get(job.id), lambda j: j.state not in ("queued", "running"), timeout=20)
    assert ended.state == "failed" and ended.exit_code == 1
    assert ended.error == "Camera 2 has no images"
    # a job left "running" by a Studio that stopped is shown as interrupted, not as running
    record = work / "broken" / "studio_job.json"
    doc = json.loads(record.read_text())
    record.write_text(json.dumps({**doc, "state": "running", "pid": 1}))
    reopened = JobManager([work, _root]).list()[0]
    assert reopened["state"] == "interrupted" and "stopped" in reopened["error"]


def test_names_and_log_reading(tmp_path):
    assert folder_name("  cabin rig 2 ") == "cabin_rig_2"
    for bad in ("", ".hidden", "a/b", "..", "x" * 81, "né"):
        with pytest.raises(JobError):
            folder_name(bad)
    log = tmp_path / "log"
    log.write_bytes(b"[detect] cam 0: 1/9\r[detect] cam 0: 9/9\n\x1b[32mPASS\x1b[0m done\n\n")
    assert log_tail(log) == ["[detect] cam 0: 9/9", "PASS done"]
    # without a terminal each count is its own line: a run of them keeps the latest per camera
    log.write_text("start\n[detect] cam 0: 1/3  a.png\n[detect] cam 1: 1/3  a.png\n"
                   "[detect] cam 0: 2/3  b.png\n[detect] cam 0: 3/3  c.png\nsolving\n"
                   "[detect] cam 0: 1/3  a.png\n")
    assert log_tail(log) == ["start", "[detect] cam 0: 3/3  c.png", "[detect] cam 1: 1/3  a.png",
                             "solving", "[detect] cam 0: 1/3  a.png"]
    assert error_headline(["x", "RuntimeError: calibration failed: non-finite cost", "tail"]) == \
        "Calibration failed: non-finite cost"
    assert error_headline(["nothing here"]) is None
    assert jobs_mod.ACTIVE == ("queued", "running")


@pytest.mark.req("FR-UI-010")
def test_a_result_is_repeated_from_its_fingerprint(keypoints, tmp_path):
    """Check the inputs against the manifest, repeat the run into a new folder, and get the same
    calibration files; a changed input is named, not silently used."""
    from ds_msp.studio import replay

    root, values = keypoints
    c = TestClient(create_app([str(root)]))
    first = c.post("/api/jobs", headers=SEND, json={"root": 0, "name": "original", "values": values}).json()
    _wait(lambda: c.get("/api/job", params={"id": first["id"]}).json(), lambda d: d["state"] == "done")

    fp = c.get("/api/run/fingerprint", params={"id": "0:original"}).json()
    assert fp["ok"] and fp["can_repeat"] and fp["n_inputs"] >= 2
    assert fp["config_path"].endswith("original/calib_param.yml") and len(fp["run_id"]) >= 8
    check = c.get("/api/run/fingerprint/check", params={"id": "0:original"}).json()
    assert check["same"] and check["checked"] == fp["n_inputs"]

    again = c.post("/api/jobs/repeat", headers=SEND, json={"id": "0:original", "root": 0, "name": "repeat"})
    assert again.status_code == 200, again.text
    assert again.json()["repeat_of"] == "0:original"
    done = _wait(lambda: c.get("/api/job", params={"id": "0:repeat"}).json(), lambda d: d["state"] not in ("queued", "running"))
    assert done["state"] == "done", done["log"][-5:]
    assert done["repeat"]["ok"] and done["repeat"]["same"], done["repeat"]
    names = {r["what"] for r in done["repeat"]["rows"]}
    assert {"corner detections used", "target used", "calibrated_cameras_data.yml"} <= names
    assert "--set save_path=" in done["command"] and "original/calib_param.yml" in done["command"]

    # an input that changed since is reported by name
    kp = root / "capture" / "detected_keypoints_data.yml"
    text = kp.read_text()
    try:
        kp.write_text(text + "\n# edited\n")
        changed = replay.check_inputs(root / "original")
        assert not changed["same"] and changed["changed"] == [str(kp)]
    finally:
        kp.write_text(text)

    # what a manifest may ask the CLI to do is checked, not passed through
    with pytest.raises(ValueError, match="outside the folders"):
        replay.replay_command(root / "original", tmp_path / "x", [tmp_path])
    bad = c.post("/api/jobs/repeat", headers=SEND, json={"id": "0:original", "root": 0, "name": "repeat"})
    assert bad.status_code == 400 and "already exists" in bad.json()["detail"]
    assert replay.fingerprint(root / "capture")["ok"] is False
