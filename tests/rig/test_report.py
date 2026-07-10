"""Tests for ``ds_msp.rig.report``: distribution stats, verdict thresholds, the
self-contained HTML digital-twin report, and the live optimization animator.

Verifies FR-RIG-006 (live progress + full error-distribution stats + verdict, not just a
single max-RMS number), FR-RIG-007 (self-contained interactive HTML report), and FR-RIG-008
(live, real-data-driven terminal animation during bundle adjustment).
"""
import io
import json
import re
import time

import numpy as np
import pytest

from ds_msp.io.mccalib import CameraGT, Scenario
from ds_msp.models.radtan import RadTanModel
from ds_msp.rig import report as rpt
from ds_msp.rig.pipeline import calibrate_scenario

from ._synth import make_rig

pytestmark = pytest.mark.req("FR-RIG-006", "FR-RIG-007", "FR-RIG-008")

W, H = 640, 480


def _small_scenario(n_cam=2, n_frame=6, seed=0):
    def fac(cam_id, rng):
        fx = 700.0 * rng.uniform(0.98, 1.02)
        return RadTanModel(fx, fx, W / 2, H / 2, -0.05, 0.01, 0.0, 0.0, 0.0)
    obj, obs, img, gt_ext, gtm = make_rig(n_cam=n_cam, n_frame=n_frame, noise_px=0.2,
                                          seed=seed, w=W, h=H, model_factory=fac)
    gt = {c: CameraGT(K=gtm[c].K, dist=None, pose=np.linalg.inv(gt_ext[c]))
          for c in range(n_cam)}
    return Scenario(name="report_smoke", object=obj, object_obs=obs, cam_ids=sorted(img),
                    img_size=img, gt=gt, mccalib=gt, mccalib_rms={})


# --------------------------------------------------------------------------- stats math

def test_stats_matches_numpy_on_a_known_array():
    e = np.array([0.1, 0.2, 0.3, 0.4, 5.0])
    s = rpt._stats(e, inlier_px=1.0)
    assert s.n == 5
    assert s.mean == pytest.approx(float(np.mean(e)))
    assert s.median == pytest.approx(float(np.median(e)))
    assert s.p95 == pytest.approx(float(np.percentile(e, 95)))
    assert s.max == pytest.approx(5.0)
    assert s.rms == pytest.approx(float(np.sqrt(np.mean(e ** 2))))
    assert s.inlier_frac == pytest.approx(4 / 5)         # 4 of 5 entries < 1.0 px


def test_stats_empty_array_is_nan_not_a_crash():
    s = rpt._stats(np.zeros(0), inlier_px=1.0)
    assert s.n == 0
    assert np.isnan(s.mean) and np.isnan(s.median) and np.isnan(s.rms)
    assert s.inlier_frac == 0.0
    assert np.isnan(s.inlier_rms)
    assert s.n_gross == 0


# --------------------------------------------------------------- robust reporting (FR-RIG-018)

@pytest.mark.req("FR-RIG-018")
def test_stats_inlier_rms_and_n_gross_exclude_blunders():
    e = np.array([0.1, 0.2, 0.3, 0.4, 40.0, 60.0])   # 2 of 6 are gross mis-detections
    s = rpt._stats(e, inlier_px=1.0, gross_px=5.0)
    assert s.n_gross == 2
    non_gross = e[e < 5.0]
    assert s.inlier_rms == pytest.approx(float(np.sqrt(np.mean(non_gross ** 2))))
    # the raw (non-robust) rms is still reported too, and it's much worse -- nothing hidden
    assert s.rms > 10 * s.inlier_rms


@pytest.mark.req("FR-RIG-018")
def test_stats_inlier_rms_is_nan_when_every_corner_is_gross():
    e = np.array([40.0, 60.0])
    s = rpt._stats(e, inlier_px=1.0, gross_px=5.0)
    assert s.n_gross == 2
    assert np.isnan(s.inlier_rms)


@pytest.mark.req("FR-RIG-018")
def test_render_report_notes_blunders_only_when_gross_present():
    clean = rpt.ErrorStats(n=10, mean=0.3, median=0.3, p95=0.6, max=1.0, rms=0.35,
                           inlier_frac=1.0, inlier_rms=0.35, n_gross=0)
    text_clean = rpt.render_report({0: "radtan"}, {0: clean}, clean, level="PASS",
                                   message="ok", color=False)
    assert "note:" not in text_clean
    assert "inl_rms" in text_clean                     # column always shown

    dirty_cam = rpt.ErrorStats(n=10, mean=4.0, median=0.4, p95=30.0, max=42.7, rms=13.6,
                               inlier_frac=0.8, inlier_rms=0.636, n_gross=2)
    overall = rpt.ErrorStats(n=10, mean=4.0, median=0.4, p95=30.0, max=42.7, rms=13.6,
                             inlier_frac=0.8, inlier_rms=0.636, n_gross=2)
    text_dirty = rpt.render_report({0: "radtan"}, {0: dirty_cam}, overall, level="PASS",
                                   message="ok", color=False)
    assert "note:" in text_dirty
    assert "DOWN-WEIGHTED" in text_dirty
    assert "0.636" in text_dirty                        # points the reader at inl_rms, not rms


# --------------------------------------------------------------------------- verdict

def test_verdict_pass_warn_fail_thresholds():
    good = rpt.ErrorStats(n=100, mean=0.3, median=0.3, p95=0.6, max=1.0, rms=0.35,
                          inlier_frac=1.0)
    level, _ = rpt.verdict(good, pass_px=1.0, warn_px=3.0)
    assert level == "PASS"

    borderline = rpt.ErrorStats(n=100, mean=1.5, median=1.5, p95=4.0, max=8.0, rms=2.0,
                                inlier_frac=0.8)
    level, _ = rpt.verdict(borderline, pass_px=1.0, warn_px=3.0)
    assert level == "WARN"

    bad = rpt.ErrorStats(n=100, mean=5.0, median=5.0, p95=12.0, max=30.0, rms=6.0,
                         inlier_frac=0.2)
    level, _ = rpt.verdict(bad, pass_px=1.0, warn_px=3.0)
    assert level == "FAIL"

    empty = rpt.ErrorStats(n=0, mean=float("nan"), median=float("nan"), p95=float("nan"),
                           max=float("nan"), rms=float("nan"), inlier_frac=0.0)
    level, msg = rpt.verdict(empty)
    assert level == "FAIL"
    assert "no reprojection" in msg


# --------------------------------------------------------------------------- end-to-end

def test_camera_and_overall_stats_smoke(tmp_path):
    scn = _small_scenario()
    res = calibrate_scenario(scn, {0: "radtan", 1: "ucm"}, save_dir=str(tmp_path))
    per_cam, overall = rpt.camera_and_overall_stats(res["rig"], scn.object_obs)

    assert set(per_cam) == {0, 1}
    for c, s in per_cam.items():
        assert s.n > 0
        assert 0.0 <= s.median <= s.p95 <= s.max
    # overall combines every camera's observations
    assert overall.n == sum(s.n for s in per_cam.values())

    level, message = rpt.verdict(overall)
    assert level in ("PASS", "WARN", "FAIL")
    assert message


def test_write_html_report_is_self_contained_and_valid(tmp_path):
    scn = _small_scenario()
    res = calibrate_scenario(scn, {0: "radtan", 1: "ucm"}, save_dir=str(tmp_path))
    per_cam, overall = rpt.camera_and_overall_stats(res["rig"], scn.object_obs)
    level, message = rpt.verdict(overall)

    out = tmp_path / "report.html"
    rpt.write_html_report(str(out), res["rig"], scn, res["models"], per_cam, overall,
                          level, message)
    html = out.read_text()

    # single self-contained file: no network-fetched script/style/font/image
    assert not re.search(r'(?:src|href)\s*=\s*"(?:https?:)?//', html)
    assert "<script>" in html and "</script>" in html

    i = html.index("const DATA = ") + len("const DATA = ")
    data, _ = json.JSONDecoder().raw_decode(html[i:])
    assert data["verdict"]["level"] == level
    assert {c["id"] for c in data["cameras"]} == {0, 1}
    for c in data["cameras"]:
        assert c["stats"]["n"] == per_cam[c["id"]].n
    assert len(data["frames"]) > 0
    assert all(len(p) == 4 for f in data["frames"] for p in f["points"])


def test_write_html_report_caps_frame_count(tmp_path):
    scn = _small_scenario(n_frame=20)
    res = calibrate_scenario(scn, {0: "radtan", 1: "radtan"}, save_dir=str(tmp_path))
    per_cam, overall = rpt.camera_and_overall_stats(res["rig"], scn.object_obs)
    level, message = rpt.verdict(overall)

    out = tmp_path / "report.html"
    rpt.write_html_report(str(out), res["rig"], scn, res["models"], per_cam, overall,
                          level, message, max_frames=5)
    html = out.read_text()
    i = html.index("const DATA = ") + len("const DATA = ")
    data, _ = json.JSONDecoder().raw_decode(html[i:])
    assert len(data["frames"]) <= 5


# --------------------------------------------------------------------------- live-line UX

def test_live_line_and_stage_do_not_raise_on_a_non_tty_stream():
    stream = _FakeNonTTY()
    rpt.live_line("hello", stream=stream)
    rpt.end_live(stream=stream)
    with rpt.Stage("unit-test stage", verbose=True, stream=stream):
        pass
    assert "unit-test stage" in stream.getvalue()


# --------------------------------------------------------------------------- Live3DAnimator

class _FakeTTY(io.StringIO):
    def isatty(self):
        return True


class _FakeNonTTY(io.StringIO):
    def isatty(self):
        return False


def _minimal_rig_and_obs(fx=500.0, cam1_dx=0.3):
    """A tiny, self-contained (object, object_obs, rig) triple: 6 board points, 2 cameras,
    1 frame — real projection math, no synthetic-rig-generator dependency."""
    from ds_msp.models.radtan import RadTanModel
    from ds_msp.rig.types import Object3D, ObjectObs, RigState

    pts_3d = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0],
                       [0.1, 0.1, 0.0], [0.05, 0.05, 0.0], [0.02, 0.08, 0.0]])
    obj = Object3D(object_id=0, board_ids=[0], ref_board_id=0, T_co_b={0: np.eye(4)},
                   pts_3d=pts_3d,
                   pts_obj_2_board=np.array([[0, i] for i in range(len(pts_3d))]),
                   pts_board_2_obj={(0, i): i for i in range(len(pts_3d))})
    cams = {0: RadTanModel(fx, fx, 320, 240, 0, 0, 0, 0, 0),
           1: RadTanModel(fx, fx, 320, 240, 0, 0, 0, 0, 0)}
    T_c_g = {0: np.eye(4), 1: np.eye(4)}
    T_c_g[1][:3, 3] = [cam1_dx, 0.0, 0.0]
    T_g_o = np.eye(4)
    T_g_o[:3, 3] = [0.0, 0.0, 2.0]              # board 2m in front of the rig
    object_poses = {(0, 0): T_g_o}

    obs = []
    for cam_id, cam in cams.items():
        Xg = (T_g_o[:3, :3] @ pts_3d.T).T + T_g_o[:3, 3]
        Xc = (T_c_g[cam_id][:3, :3] @ Xg.T).T + T_c_g[cam_id][:3, 3]
        uv, valid = cam.project(Xc)
        obs.append(ObjectObs(cam_id=cam_id, frame_id=0, object_id=0,
                             point_rows=np.arange(len(pts_3d)), pts_2d=uv))
    rig = RigState(cameras=cams, T_c_g=T_c_g, ref_cam_id=0, object_poses=object_poses,
                   objects={0: obj}, img_size={0: (640, 480), 1: (640, 480)})
    return obj, obs, rig


def test_braille_canvas_sets_the_expected_dot_and_survives_out_of_range():
    canvas = rpt.BrailleCanvas(cols=4, rows=4)
    canvas.set(0, 0, color=(10, 20, 30))
    lines = canvas.render_lines()
    assert chr(0x2801) in lines[0]              # dot (0,0) -> braille bit 0x01
    assert "\x1b[38;2;10;20;30m" in lines[0]
    canvas.set(-5, -5)                          # out of range must not raise or wrap
    canvas.set(999, 999)


def test_err_color_is_green_low_red_high_like_the_html_report():
    lo = rpt._err_color(0.05)
    hi = rpt._err_color(5.0)
    nan = rpt._err_color(float("nan"))
    assert lo[1] > lo[0] and lo[1] > lo[2]        # low error -> green-dominant
    assert hi[0] > hi[1]                          # high error -> red-dominant
    assert nan != lo and nan != hi                # NaN (unobserved) gets its own neutral color


def test_live3d_animator_tty_renders_real_geometry_not_decoration():
    obj, obs, rig = _minimal_rig_and_obs()
    stream = _FakeTTY()
    anim = rpt.Live3DAnimator(obj, obs, verbose=True, stream=stream, min_interval=0.0)
    anim(1, 10, 0.9, 12.0, rig)
    anim(2, 10, 0.4, 5.0, rig)
    anim.finish()
    out = stream.getvalue()
    # braille glyphs (real dot-plotted geometry) and per-cell truecolor are present
    assert any(0x2800 < ord(ch) <= 0x28FF for ch in out)
    assert "\x1b[38;2;" in out
    # a live TTY redraw uses ANSI cursor-up between frames
    assert "\x1b[" in out
    # the real fed numbers appear in the header, not placeholders
    assert "rms=0.4000px" in out
    assert "iter 2/10" in out


def test_live3d_animator_non_tty_throttles_plain_lines():
    obj, obs, rig = _minimal_rig_and_obs()
    stream = _FakeNonTTY()
    anim = rpt.Live3DAnimator(obj, obs, verbose=True, stream=stream)
    anim._throttle_every = 3
    for i, r in enumerate([1.0, 0.8, 0.6, 0.4, 0.2, 0.1], 1):
        anim(i, 6, r, r * 5, rig)
    out = stream.getvalue()
    assert "\x1b[" not in out                 # no ANSI redraw when not a TTY
    assert out.count("[optimizing]") == 3     # steps 1, 3, 6 (first + every throttle_every)


def test_live3d_animator_ignores_nan_and_respects_verbose_false():
    obj, obs, rig = _minimal_rig_and_obs()
    stream = _FakeTTY()
    anim = rpt.Live3DAnimator(obj, obs, verbose=True, stream=stream, min_interval=0.0)
    anim(1, 10, float("nan"), 0.0, rig)
    assert stream.getvalue() == ""            # NaN rms (e.g. zero-residual edge case) is skipped

    stream2 = _FakeTTY()
    quiet = rpt.Live3DAnimator(obj, obs, verbose=False, stream=stream2)
    quiet(1, 10, 1.0, 5.0, rig)
    assert stream2.getvalue() == ""


def test_live3d_animator_render_cost_is_cheap_enough_not_to_matter():
    """The stated design goal: rendering must not be the thing that makes calibration slow.
    100 renders of a 2-camera/6-point scene should take a small fraction of a second — the
    render itself, not the throttle, is what's being measured (min_interval=0 disables it)."""
    obj, obs, rig = _minimal_rig_and_obs()
    stream = _FakeTTY()
    anim = rpt.Live3DAnimator(obj, obs, verbose=True, stream=stream, min_interval=0.0)
    t0 = time.time()
    for i in range(1, 101):
        anim(i, 100, 1.0 / i, 5.0, rig)
    dt = time.time() - t0
    assert dt < 2.0, f"100 renders took {dt:.2f}s — too slow to run live during a solve"


def test_live3d_animator_keeps_every_camera_in_frame_through_a_full_orbit():
    """Regression: cameras must stay inside the canvas at every orbit angle, not just the
    angle a screenshot happened to be taken at. A board that sits meters in front of the rig
    (real geometry — see ``_minimal_rig_and_obs``'s ``cam1_dx``) with a small board-vs-rig
    baseline ratio used to leave cameras outside the projected frustum whenever the ~0.045
    rad/frame auto-orbit swept them perpendicular to the view direction, because framing
    radius only counted 35% of the true camera-to-board distance."""
    obj, obs, rig = _minimal_rig_and_obs(cam1_dx=0.5)
    stream = _FakeTTY()
    anim = rpt.Live3DAnimator(obj, obs, verbose=True, stream=stream, min_interval=0.0)
    cam_positions = []
    for T_c_g in rig.T_c_g.values():
        R, t = T_c_g[:3, :3], T_c_g[:3, 3]
        cam_positions.append(-R.T @ t)

    def proj(p, eye, right, up, fwd):
        rel = p - eye
        vx, vy, vz = rel @ right, rel @ up, rel @ fwd
        if vz <= 1e-3:
            return None
        f = anim.canvas.h * 1.15 / vz
        return anim.canvas.w / 2 + vx * f, anim.canvas.h / 2 - vy * f

    n_steps = int(2 * np.pi / 0.045) + 5        # one full orbit plus a few extra frames
    for i in range(1, n_steps + 1):
        anim(i, n_steps, 0.5, 5.0, rig)
        if i < 20:                              # let the EMA-smoothed framing settle first
            continue
        eye, right, up, fwd = rpt._orbit_basis(anim._center, anim._radius, anim._az, anim._el)
        for cp in cam_positions:
            pp = proj(cp, eye, right, up, fwd)
            assert pp is not None, f"camera behind view plane at az={anim._az:.3f}"
            x, y = pp
            assert 0 <= x < anim.canvas.w and 0 <= y < anim.canvas.h, (
                f"camera projected off-canvas at az={anim._az:.3f}: ({x:.1f}, {y:.1f}) "
                f"vs canvas {anim.canvas.w}x{anim.canvas.h}")


def test_refine_groups_on_iter_never_drops_a_camera_across_groups():
    """FR-RIG-008 multi-group regression: a live viewer driven off ``refine_groups``'s
    ``on_iter`` must see every rig camera on every callback, not just the group currently
    being solved — otherwise cameras visually "disappear" mid-animation while another group
    refines (ds_msp/rig/bundle.py:471-476)."""
    from ds_msp.rig import bundle

    def fac(cam_id, rng):
        return RadTanModel(700.0, 700.0, W / 2, H / 2, -0.05, 0.01, 0.0, 0.0, 0.0)
    obj, obs, img, gt_ext, gtm = make_rig(n_cam=4, n_frame=10, noise_px=0.2, seed=0,
                                          w=W, h=H, model_factory=fac)
    from ds_msp.rig.types import RigState
    rig = RigState(cameras=gtm, T_c_g=gt_ext, ref_cam_id=0,
                   object_poses={(0, f): np.eye(4) for f in {o.frame_id for o in obs}},
                   objects={0: obj}, img_size=img)
    # per-object warm-up: cameras/extrinsics fixed at GT, only object poses move off np.eye(4)
    rig = bundle.refine(rig, obs, fix_intrinsics=True, fix_extrinsics=True, max_iter=5)

    seen_camera_sets = []

    def on_iter(it, max_iter, rms, cost, partial_rig):
        seen_camera_sets.append((set(partial_rig.cameras), set(partial_rig.T_c_g)))

    groups = [[0, 1], [2, 3]]
    bundle.refine_groups(rig, obs, groups, max_iter=5, on_iter=on_iter)
    assert seen_camera_sets, "on_iter never fired"
    all_cams = {0, 1, 2, 3}
    for cams, tcg in seen_camera_sets:
        assert cams == all_cams, f"cameras dropped mid-group-refine: {cams}"
        assert tcg == all_cams, f"extrinsics dropped mid-group-refine: {tcg}"


def test_calibrate_scenario_drives_on_iter_with_real_solver_rms(tmp_path):
    """End-to-end: the callback threaded through calibrate_scenario -> calibrate_rig ->
    bundle.refine -> core.optimize.{lm_solve,schur_lm} actually fires with the real,
    monotonically-sane RMS values the solver converges through — not a mock. Also carries a
    real RigState snapshot of the mid-solve geometry (cameras/extrinsics/object poses), not
    just the scalar trace — that snapshot is what the live terminal animator renders from."""
    from ds_msp.rig.types import RigState

    scn = _small_scenario()
    calls = []
    calibrate_scenario(
        scn, {0: "radtan", 1: "ucm"}, save_dir=str(tmp_path),
        on_iter=lambda it, max_iter, rms, cost, rig: calls.append((it, rms, cost, rig)))
    assert len(calls) > 0
    assert all(np.isfinite(rms) and rms >= 0 for _, rms, _, _ in calls)
    assert all(cost >= 0 for _, _, cost, _ in calls)
    assert all(isinstance(rig, RigState) for _, _, _, rig in calls)
    assert all(rig.T_c_g for _, _, _, rig in calls)
