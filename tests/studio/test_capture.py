"""A photo folder as a capture with a history: earlier calibrations of the same photos are
listed, and the detections of one are reused when the photos and the board have not changed."""
import json
import os
import stat
import time
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from ds_msp.studio import capture
from ds_msp.studio.jobs import JobError, JobManager

pytestmark = pytest.mark.req("FR-UI-011")
VALUES = {"number_camera": 2, "number_x_square": 6, "number_y_square": 5, "square_size": 0.08}


def _capture(tmp_path):
    photos = tmp_path / "rig" / "Images"
    files = []
    for cam in ("Cam_001", "Cam_002"):
        (photos / cam).mkdir(parents=True)
        for k in range(3):
            f = photos / cam / f"{k:04d}.png"
            cv2.imwrite(str(f), np.full((40, 60), 100 + k, np.uint8))
            files.append(f)
    return photos, files


def _earlier_run(tmp_path, photos, files, name="run_1", *, n_x=6, opencv=None, after_s=60, keypoints=None):
    """A result folder as a calibration made ``after_s`` seconds after the photos left it."""
    folder = tmp_path / "rig" / name
    folder.mkdir(parents=True)
    made = datetime.now(timezone.utc) + timedelta(seconds=after_s)
    manifest = {
        "created_utc": made.isoformat(),
        "configuration": {"resolved": {
            "root_path": str(photos), "cam_prefix": "Cam_", "frame_pattern": None, "number_camera": 2,
            "keypoints_path": keypoints,
            "boards": [{"n_x": n_x, "n_y": 5, "length_square": 0.04, "length_marker": 0.03,
                        "square_size": 0.2, "id_offset": None}]}},
        "environment": {"packages": {"opencv-python": opencv or cv2.__version__}},
        "inputs": {"files": [{"path": str(f), "size_bytes": f.stat().st_size} for f in files]},
    }
    (folder / "calibration_manifest.json").write_text(json.dumps(manifest))
    (folder / "calibration_report.json").write_text(json.dumps({
        "capture": {"n_cameras": 2}, "models": {"0": "kb", "1": "kb"}, "overall_errors_px": {"median": 0.08}}))
    (folder / "detected_keypoints_data.yml").write_text("%YAML:1.0\nnb_camera: 2\n")
    return folder


def test_history_lists_earlier_calibrations_of_the_same_photos(tmp_path):
    photos, files = _capture(tmp_path)
    _earlier_run(tmp_path, photos, files, "run_1", after_s=60)
    _earlier_run(tmp_path, photos, files, "run_2", after_s=120)
    other_photos, other_files = _capture(tmp_path / "elsewhere")
    _earlier_run(tmp_path / "elsewhere", other_photos, other_files, "run_x")
    doc = capture.history(str(photos), [tmp_path], {**VALUES, "root_path": str(photos)})
    assert [r["name"] for r in doc["runs"]] == ["run_2", "run_1"]          # newest first
    assert doc["runs"][0]["models"] == ["kb"] and doc["runs"][0]["median_px"] == 0.08
    assert all(r["reuse"]["ok"] for r in doc["runs"]) and doc["reusable"] == "0:rig/run_2"
    assert doc["runs"][0]["reuse"]["why"] == "6 photos unchanged"


def test_detections_are_reused_only_when_they_still_fit(tmp_path):
    photos, files = _capture(tmp_path)
    _earlier_run(tmp_path, photos, files, "same")
    _earlier_run(tmp_path, photos, files, "other_board", n_x=9)
    _earlier_run(tmp_path, photos, files, "other_opencv", opencv="3.4.1")
    _earlier_run(tmp_path, photos, files, "from_detections", keypoints="/somewhere/kp.yml")
    values = {**VALUES, "root_path": str(photos)}
    why = {r["name"]: r["reuse"] for r in capture.history(str(photos), [tmp_path], values)["runs"]}
    assert why["same"]["ok"]
    assert not why["other_board"]["ok"] and "board layout" in why["other_board"]["why"]
    assert not why["other_opencv"]["ok"] and "OpenCV 3.4.1" in why["other_opencv"]["why"]
    assert not why["from_detections"]["ok"]
    # the square size scales the target, not the corners: detections stay usable
    assert capture.history(str(photos), [tmp_path], {**values, "square_size": 0.5})["reusable"]
    # a lens type or solver setting does not matter either
    assert capture.history(str(photos), [tmp_path], {**values, "camera_models": ["ds", "ds"], "noise_bound": 0.3})["reusable"]

    # a photo added, removed or edited since then stops the reuse, with the reason
    cv2.imwrite(str(photos / "Cam_001" / "0009.png"), np.zeros((40, 60), np.uint8))
    added = {r["name"]: r["reuse"] for r in capture.history(str(photos), [tmp_path], values)["runs"]}
    assert not added["same"]["ok"] and "1 added" in added["same"]["why"]
    (photos / "Cam_001" / "0009.png").unlink()
    future = time.time() + 3600
    os.utime(files[0], (future, future))
    again = {r["name"]: r["reuse"] for r in capture.history(str(photos), [tmp_path], values)["runs"]}
    assert not again["same"]["ok"] and "modified after" in again["same"]["why"]


@pytest.mark.skipif(os.name != "posix", reason="uses a shell script as the interpreter")
def test_a_reusing_job_copies_the_detections_and_leaves_the_earlier_result_alone(tmp_path):
    photos, files = _capture(tmp_path)
    earlier = _earlier_run(tmp_path, photos, files, "first")
    before = {p.name: p.read_bytes() for p in earlier.iterdir()}
    fake = tmp_path / "fake_python.sh"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    mgr = JobManager([tmp_path], python=str(fake))
    values = {**VALUES, "root_path": str(photos), "camera_models": ["kb", "kb"]}
    job = mgr.submit(0, "second", values, parent="rig", reuse="0:rig/first")
    assert job.reused_from == "0:rig/first"
    new = tmp_path / "rig" / "second"
    copied = new / "reused_detections" / "detected_keypoints_data.yml"
    assert copied.read_bytes() == before["detected_keypoints_data.yml"]
    assert sorted(p.name for p in (new / "reused_detections").iterdir()) == ["detected_keypoints_data.yml"]
    config = (new / "calib_param.yml").read_text()
    assert f'keypoints_path: "{copied}"' in config and f'root_path: "{photos}"' in config
    assert {p.name: p.read_bytes() for p in earlier.iterdir()} == before       # untouched

    with pytest.raises(JobError, match="cannot be reused"):
        mgr.submit(0, "third", {**values, "number_x_square": 9}, parent="rig", reuse="0:rig/first")
    assert not (tmp_path / "rig" / "third").exists()
