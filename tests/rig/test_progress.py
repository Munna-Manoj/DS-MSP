"""The machine-readable progress log (FR-RIG-033): it records the run and changes nothing."""
import json

import pytest

from ds_msp.rig.progress import ProgressLog

pytestmark = pytest.mark.req("FR-RIG-033")


class _View:
    def __init__(self):
        self.calls = []
        self.url = "http://view"

    def detect_progress(self, *a):
        self.calls.append(("detect", a))

    def bind_scene(self, obj, obs):
        self.calls.append(("scene", len(obs)))

    def set_stage(self, name):
        self.calls.append(("stage", name))

    def save_progress(self, *a):
        self.calls.append(("save", a))

    def __call__(self, *a):
        self.calls.append(("iter", a[:4]))

    def finish(self, rig=None, **kw):
        self.calls.append(("finish", kw))


class _Obs:
    def __init__(self, cam_id):
        self.cam_id = cam_id


def test_progress_log_records_events_and_forwards_every_call(tmp_path):
    view = _View()
    log = ProgressLog(str(tmp_path / "p.jsonl"), view, min_interval=0.0)
    log.detect_progress(0, 1, 2, "a.png")
    log.detect_progress(0, 2, 2, "b.png")
    log.bind_scene(object(), [_Obs(0), _Obs(1), _Obs(1)])
    log.set_stage("(c) global joint BA")
    log(3, 60, 0.42, 1.5, None)
    log(4, 60, float("nan"), float("inf"), None)
    log.save_progress(1, 1, 1, 7)
    log.finish(None, rms=0.31)
    assert log.url == "http://view"                       # anything else reaches the view
    assert [c[0] for c in view.calls] == ["detect", "detect", "scene", "stage", "iter", "iter",
                                          "save", "finish"]
    events = [json.loads(line) for line in (tmp_path / "p.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["start", "detect", "detect", "scene", "stage", "iter",
                                            "iter", "save", "done"]
    assert events[2]["cameras"] == {"0": [2, 2]}
    assert events[3] == {"t": events[3]["t"], "event": "scene", "cameras": 2, "observations": 3}
    assert events[4]["name"] == "(c) global joint BA" and events[4]["index"] == 1
    assert events[5]["rms"] == 0.42 and events[6]["rms"] is None and events[6]["cost"] is None
    assert events[-1]["rms"] == 0.31
    assert all(a["t"] <= b["t"] for a, b in zip(events, events[1:]))


def test_progress_log_throttles_and_works_without_a_view(tmp_path):
    log = ProgressLog(str(tmp_path / "p.jsonl"), None, min_interval=3600.0)
    for k in range(1, 6):
        log.detect_progress(0, k, 5, "x.png")             # first and last are written
        log(k, 60, 0.5, 1.0, None)                        # only the first
    log.finish(None, rms=0.2)
    log(9, 60, 0.5, 1.0, None)                            # after the end: ignored, no error
    kinds = [json.loads(line)["event"] for line in (tmp_path / "p.jsonl").read_text().splitlines()]
    assert kinds == ["start", "detect", "iter", "detect", "done"]
    with pytest.raises(AttributeError):
        _ = log.url
