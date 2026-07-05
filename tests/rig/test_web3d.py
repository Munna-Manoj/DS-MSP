"""Tests for ``ds_msp.rig.web3d.WebLive3DAnimator`` -- the Three.js cinematic replacement for
the braille-terminal ``Live3DAnimator``. Same duck-typed ``on_iter`` contract, so these mirror
``test_report.py``'s live-animator tests: real geometry in, valid state out, throttled render
cost, graceful degradation when a server can't start. Verifies FR-RIG-008 (live, real-data-
driven animation during bundle adjustment) for the browser-based renderer.
"""
import json
import urllib.request

import numpy as np
import pytest

from ds_msp.rig import web3d

from .test_report import _FakeNonTTY, _FakeTTY, _minimal_rig_and_obs

pytestmark = pytest.mark.req("FR-RIG-008", "FR-RIG-009", "FR-RIG-010", "FR-RIG-011", "FR-RIG-012",
                              "FR-RIG-013")


def _shutdown(anim):
    if anim._server is not None:
        anim._server.shutdown()
        anim._server.server_close()


def _fetch(anim, name):
    url = anim._url.rsplit("/", 1)[0] + "/" + name
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.read().decode()


def test_web_animator_serves_valid_state_and_index_over_http():
    obj, obs, rig = _minimal_rig_and_obs()
    anim = web3d.WebLive3DAnimator(obj, obs, verbose=True, auto_open=False,
                                   stream=_FakeNonTTY(), min_interval=0.0)
    try:
        assert anim._server is not None
        assert anim._url is not None

        anim.set_stage("(b) rig extrinsics")
        anim(1, 10, 0.83, 12.3, rig)

        index_html = _fetch(anim, "index.html")
        assert "<title>" in index_html
        assert "importmap" in index_html
        assert "THREE" in index_html or "three" in index_html

        state = json.loads(_fetch(anim, "state.json"))
        assert state["status"] == "running"
        assert state["stage"] == "(b) rig extrinsics"
        assert state["rms"] == pytest.approx(0.83)
        assert state["cost"] == pytest.approx(12.3)
        cam_ids = {c["id"] for c in state["cameras"]}
        assert cam_ids == {0, 1}
        for c in state["cameras"]:
            assert len(c["pos"]) == 3
            assert len(c["ax_x"]) == len(c["ax_y"]) == len(c["ax_z"]) == 3
        assert len(state["beams"]) > 0                  # real per-point reprojection rays
        assert state["history"] == [pytest.approx(0.83)]
    finally:
        _shutdown(anim)


def _rig_no_single_frame_covers_every_camera():
    """3 cameras, 2 frames: frame 0 is co-observed by cams 0+1 (the "richest" frame by total
    point count), cam 2 only ever appears in frame 1. Reproduces the real bug -- picking one
    shared richest (object_id, frame_id) pair for the whole rig silently drops cam 2 forever,
    even though it has real observations and a real, live-updating reprojection error."""
    from ds_msp.models.radtan import RadTanModel
    from ds_msp.rig.types import Object3D, ObjectObs, RigState

    pts_3d = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.1, 0.0]])
    obj = Object3D(object_id=0, board_ids=[0], ref_board_id=0, T_co_b={0: np.eye(4)},
                   pts_3d=pts_3d,
                   pts_obj_2_board=np.array([[0, i] for i in range(len(pts_3d))]),
                   pts_board_2_obj={(0, i): i for i in range(len(pts_3d))})
    cams = {c: RadTanModel(500.0, 500.0, 320, 240, 0, 0, 0, 0, 0) for c in (0, 1, 2)}
    T_c_g = {0: np.eye(4), 1: np.eye(4), 2: np.eye(4)}
    T_c_g[1][:3, 3] = [0.2, 0.0, 0.0]
    T_c_g[2][:3, 3] = [-0.2, 0.0, 0.0]
    T_g_o0 = np.eye(4)
    T_g_o0[:3, 3] = [0.0, 0.0, 2.0]
    T_g_o1 = np.eye(4)
    T_g_o1[:3, 3] = [0.05, 0.0, 2.0]
    object_poses = {(0, 0): T_g_o0, (0, 1): T_g_o1}

    obs = []
    for cam_id, frame_id, T_g_o in ((0, 0, T_g_o0), (1, 0, T_g_o0), (2, 1, T_g_o1)):
        cam = cams[cam_id]
        Xg = (T_g_o[:3, :3] @ pts_3d.T).T + T_g_o[:3, 3]
        Xc = (T_c_g[cam_id][:3, :3] @ Xg.T).T + T_c_g[cam_id][:3, 3]
        uv, valid = cam.project(Xc)
        obs.append(ObjectObs(cam_id=cam_id, frame_id=frame_id, object_id=0,
                             point_rows=np.arange(len(pts_3d)), pts_2d=uv))
    rig = RigState(cameras=cams, T_c_g=T_c_g, ref_cam_id=0, object_poses=object_poses,
                   objects={0: obj}, img_size={c: (640, 480) for c in cams})
    return obj, obs, rig


def test_web_animator_gives_every_camera_a_value_even_with_no_shared_frame():
    obj, obs, rig = _rig_no_single_frame_covers_every_camera()
    anim = web3d.WebLive3DAnimator(obj, obs, verbose=True, auto_open=False,
                                   stream=_FakeNonTTY(), min_interval=0.0)
    try:
        anim(0, 1, 1.0, 1.0, rig)
        state = json.loads(_fetch(anim, "state.json"))
        by_id = {c["id"]: c for c in state["cameras"]}
        assert set(by_id) == {0, 1, 2}
        for cid, c in by_id.items():
            assert c["err"] is not None, f"cam {cid} has no reprojection error value"
    finally:
        _shutdown(anim)


def test_web_animator_camera_positions_match_real_extrinsics():
    """The state payload must carry the *actual* solved geometry, not a placeholder -- same
    bar ``test_live3d_animator_tty_renders_real_geometry_not_decoration`` holds the terminal
    animator to."""
    obj, obs, rig = _minimal_rig_and_obs(cam1_dx=0.7)
    anim = web3d.WebLive3DAnimator(obj, obs, verbose=True, auto_open=False,
                                   stream=_FakeNonTTY(), min_interval=0.0)
    try:
        anim(0, 5, 1.0, 1.0, rig)
        state = json.loads(_fetch(anim, "state.json"))
        by_id = {c["id"]: c for c in state["cameras"]}
        assert by_id[0]["pos"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
        # T_c_g[1] translates by +0.7 in x (global->camera), so camera 1's position in the
        # global frame is -t (R=I here) -- same convention as Live3DAnimator's `pos = -R.T @ t`.
        assert by_id[1]["pos"] == pytest.approx([-0.7, 0.0, 0.0], abs=1e-6)
    finally:
        _shutdown(anim)


def test_web_animator_throttles_like_the_terminal_version():
    obj, obs, rig = _minimal_rig_and_obs()
    anim = web3d.WebLive3DAnimator(obj, obs, verbose=True, auto_open=False,
                                   stream=_FakeNonTTY(), min_interval=10.0)
    try:
        writes = []
        real_write = anim._write_state
        anim._write_state = lambda s: (writes.append(s), real_write(s))[1]
        for it in range(5):
            anim(it, 5, 1.0, 1.0, rig)
        assert len(writes) == 1                # only the first, un-throttled call renders
    finally:
        _shutdown(anim)


def test_web_animator_ignores_nan_rms_and_respects_verbose_false():
    obj, obs, rig = _minimal_rig_and_obs()
    anim = web3d.WebLive3DAnimator(obj, obs, verbose=True, auto_open=False,
                                   stream=_FakeNonTTY(), min_interval=0.0)
    try:
        anim(0, 5, float("nan"), 1.0, rig)      # must not write or crash
        state = json.loads(_fetch(anim, "state.json"))
        assert state["status"] == "starting"    # untouched since construction
    finally:
        _shutdown(anim)

    quiet = web3d.WebLive3DAnimator(obj, obs, verbose=False, auto_open=False)
    assert quiet._server is None
    quiet(0, 5, 1.0, 1.0, rig)                  # must not raise with no server
    quiet.finish(rig, rms=0.5)                  # must not raise


def test_web_animator_launches_before_any_scene_is_bound():
    """The live view now starts (server + browser) at construction time, before detection has
    even run -- see the class docstring for why. An empty/never-bound scene must not prevent
    the server from starting; it just means nothing renders in the 3D view yet."""
    anim = web3d.WebLive3DAnimator(verbose=True, auto_open=False)
    try:
        assert anim._server is not None
        anim.finish(None)                        # must not raise with no rig ever bound
    finally:
        _shutdown(anim)


def test_web_animator_never_starts_a_server_when_not_verbose():
    from ds_msp.rig.types import Object3D

    empty_obj = Object3D(object_id=0, board_ids=[0], ref_board_id=0, T_co_b={0: np.eye(4)},
                         pts_3d=np.zeros((0, 3)),
                         pts_obj_2_board=np.zeros((0, 2), dtype=int), pts_board_2_obj={})
    anim = web3d.WebLive3DAnimator(empty_obj, [], verbose=False, auto_open=False)
    assert anim._server is None
    anim.finish(None)                            # must not raise


def test_web_animator_finish_writes_final_stats_with_no_nan_and_a_valid_mvp():
    obj, obs, rig = _minimal_rig_and_obs()
    anim = web3d.WebLive3DAnimator(obj, obs, verbose=True, auto_open=False,
                                   stream=_FakeNonTTY(), min_interval=0.0, finish_grace_s=0)
    try:
        anim.set_stage("(d) structure refinement 1/1")
        anim(0, 1, 1.0, 1.0, rig)
        anim.finish(rig, rms=0.42)

        raw = _fetch(anim, "state.json")
        assert "NaN" not in raw and "Infinity" not in raw       # would break JSON.parse in-browser
        state = json.loads(raw)
        assert state["status"] == "finished"
        final = state["final"]
        assert {c["id"] for c in final["cameras"]} == {0, 1}
        for c in final["cameras"]:
            assert "median" in c and "model" in c
            # real solved T_c_g pose, for the post-finale digital-twin reveal (not the pond depth)
            assert len(c["pos"]) == 3
            assert len(c["ax_x"]) == len(c["ax_y"]) == len(c["ax_z"]) == 3
        assert final["mvp"] in (0, 1)
        assert final["overall"]["n"] > 0
        # temporal board replay -- the real solved per-frame corner positions, reusing
        # report._frame_payload verbatim (see _final_payload's docstring)
        assert len(final["frames"]) >= 1
        frame0 = final["frames"][0]
        assert "object_id" in frame0 and "frame_id" in frame0
        assert len(frame0["points"]) > 0
        for p in frame0["points"]:
            assert len(p) == 4                   # [x, y, z, err_or_null]
    finally:
        _shutdown(anim)


def test_web_animator_final_camera_pose_matches_real_extrinsics():
    """The post-finale reveal is only honest if the pose it ships is the SAME real T_c_g the
    live view already draws during the run, not a second, possibly-divergent computation."""
    obj, obs, rig = _minimal_rig_and_obs(cam1_dx=0.7)
    anim = web3d.WebLive3DAnimator(obj, obs, verbose=True, auto_open=False,
                                   stream=_FakeNonTTY(), min_interval=0.0, finish_grace_s=0)
    try:
        anim(0, 1, 1.0, 1.0, rig)
        live_state = json.loads(_fetch(anim, "state.json"))
        live_pos = {c["id"]: c["pos"] for c in live_state["cameras"]}

        anim.finish(rig, rms=0.5)
        final_state = json.loads(_fetch(anim, "state.json"))
        final_pos = {c["id"]: c["pos"] for c in final_state["final"]["cameras"]}

        for cid in (0, 1):
            assert final_pos[cid] == pytest.approx(live_pos[cid], abs=1e-9)
    finally:
        _shutdown(anim)


def test_web_animator_finish_blocks_for_the_configured_grace_period():
    """finish() must not return until the grace period has elapsed -- nothing else guarantees
    the browser (which polls every 150ms, and can be throttled far slower than that by a
    backgrounded tab) has actually fetched the final "finished" state before this process is
    free to exit and kill the ephemeral HTTP server. Without this, the finale (splash, victory
    leap, fireworks) can be lost entirely even though the browser-side logic itself is correct."""
    import time as _time

    obj, obs, rig = _minimal_rig_and_obs()
    anim = web3d.WebLive3DAnimator(obj, obs, verbose=True, auto_open=False,
                                   stream=_FakeNonTTY(), min_interval=0.0, finish_grace_s=0.3)
    try:
        t0 = _time.time()
        anim.finish(rig, rms=0.5)
        elapsed = _time.time() - t0
        assert elapsed >= 0.3, f"finish() returned after only {elapsed:.3f}s, expected >= 0.3s"
    finally:
        _shutdown(anim)

    anim2 = web3d.WebLive3DAnimator(obj, obs, verbose=True, auto_open=False,
                                    stream=_FakeNonTTY(), min_interval=0.0, finish_grace_s=0)
    try:
        t0 = _time.time()
        anim2.finish(rig, rms=0.5)
        assert _time.time() - t0 < 0.1, "finish_grace_s=0 must not block"
    finally:
        _shutdown(anim2)


def test_web_animator_survives_a_headless_no_socket_environment(monkeypatch):
    """A sandboxed/CI environment without loopback sockets must degrade to plain progress
    lines, not crash the calibration run over a visualization nicety."""
    obj, obs, rig = _minimal_rig_and_obs()

    def _boom(*a, **k):
        raise OSError("socket creation blocked")

    monkeypatch.setattr(web3d.http.server, "ThreadingHTTPServer", _boom)
    anim = web3d.WebLive3DAnimator(obj, obs, verbose=True, auto_open=False, stream=_FakeTTY())
    assert anim._server is None
    anim(0, 5, 1.0, 1.0, rig)                    # falls back to plain progress lines
    anim.finish(rig, rms=0.5)                    # must not raise


def test_web_animator_bind_scene_populates_frame_obs_after_construction():
    obj, obs, rig = _minimal_rig_and_obs()
    anim = web3d.WebLive3DAnimator(verbose=True, auto_open=False, stream=_FakeNonTTY())
    try:
        assert anim._frame_obs == []                # nothing bound yet
        anim.bind_scene(obj, obs)
        assert {o.cam_id for o in anim._frame_obs} == {0, 1}
        anim(0, 5, 1.0, 1.0, rig)
        state = json.loads(_fetch(anim, "state.json"))
        assert {c["id"] for c in state["cameras"]} == {0, 1}
    finally:
        _shutdown(anim)


def test_web_animator_detect_progress_streams_before_scene_is_bound():
    anim = web3d.WebLive3DAnimator(verbose=True, auto_open=False, stream=_FakeNonTTY())
    try:
        anim.detect_progress(0, 1, 10, "/x/00000.png")
        anim.detect_progress(1, 1, 5, "/x/00000.png")
        state = json.loads(_fetch(anim, "state.json"))
        assert state["status"] == "detecting"
        assert state["detect"]["0"] == {"i": 1, "n": 10}
        assert state["detect"]["1"] == {"i": 1, "n": 5}
        assert state["cameras"] == []                # no rig geometry exists yet -- honestly empty
    finally:
        _shutdown(anim)


def test_web_animator_set_stage_pushes_immediately_without_a_call(): # noqa: E501 -- descriptive name
    """Some stages (front-end intrinsics, saving debug images) have no per-iteration callback
    of their own -- set_stage must push a real update on its own, not wait for the next
    __call__ that may never come during that stage."""
    obj, obs, rig = _minimal_rig_and_obs()
    anim = web3d.WebLive3DAnimator(obj, obs, verbose=True, auto_open=False, stream=_FakeNonTTY())
    try:
        anim.set_stage("(0) per-camera front-end intrinsics")
        state = json.loads(_fetch(anim, "state.json"))
        assert state["stage"] == "(0) per-camera front-end intrinsics"
        assert state["status"] == "running"
    finally:
        _shutdown(anim)


def test_web_animator_save_progress_keeps_last_known_cameras_visible():
    obj, obs, rig = _minimal_rig_and_obs()
    anim = web3d.WebLive3DAnimator(obj, obs, verbose=True, auto_open=False, stream=_FakeNonTTY(),
                                   min_interval=0.0)
    try:
        anim(0, 1, 1.0, 1.0, rig)                    # establishes real camera geometry in state
        anim.save_progress(0, 3, 10, 7)
        state = json.loads(_fetch(anim, "state.json"))
        assert state["save"]["0"] == {"i": 3, "n": 10}
        assert {c["id"] for c in state["cameras"]} == {0, 1}   # not blanked by the save update
    finally:
        _shutdown(anim)
