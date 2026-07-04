"""Tests for ``ds_msp.calib.report``: distribution stats, verdict thresholds, live progress,
and the single-camera summary rendering.

Verifies FR-CALIB-008 (live per-image detection progress + a full reprojection-error
distribution + a PASS/WARN/FAIL verdict, not just one bare RMS number).
"""
import io
import math

import pytest

from ds_msp.calib import report as rpt

pytestmark = pytest.mark.req("FR-CALIB-008")


class _FakeNonTTY(io.StringIO):
    def isatty(self):
        return False


# --------------------------------------------------------------------------- ErrorStats

def test_error_stats_from_result_reads_bundle_calibrate_keys():
    result = {"n_obs": 1954, "mean_px": 0.372, "median_px": 0.287, "p95_px": 0.959,
             "max_px": 3.761, "rms_px": 0.509}
    s = rpt.ErrorStats.from_result(result)
    assert s.n == 1954
    assert s.mean == pytest.approx(0.372)
    assert s.median == pytest.approx(0.287)
    assert s.p95 == pytest.approx(0.959)
    assert s.max == pytest.approx(3.761)
    assert s.rms == pytest.approx(0.509)
    assert s.to_dict()["n"] == 1954


# --------------------------------------------------------------------------- verdict

def test_verdict_pass_warn_fail_thresholds():
    good = rpt.ErrorStats(n=100, mean=0.3, median=0.3, p95=0.6, max=1.0, rms=0.35)
    level, _ = rpt.verdict(good, pass_px=1.0, warn_px=3.0)
    assert level == "PASS"

    borderline = rpt.ErrorStats(n=100, mean=1.5, median=1.5, p95=4.0, max=8.0, rms=2.0)
    level, _ = rpt.verdict(borderline, pass_px=1.0, warn_px=3.0)
    assert level == "WARN"

    bad = rpt.ErrorStats(n=100, mean=5.0, median=5.0, p95=12.0, max=30.0, rms=6.0)
    level, _ = rpt.verdict(bad, pass_px=1.0, warn_px=3.0)
    assert level == "FAIL"

    empty = rpt.ErrorStats(n=0, mean=math.nan, median=math.nan, p95=math.nan,
                           max=math.nan, rms=math.nan)
    level, msg = rpt.verdict(empty)
    assert level == "FAIL"
    assert "no reprojection" in msg


def test_verdict_boundary_is_inclusive():
    exactly_at_bar = rpt.ErrorStats(n=10, mean=1.0, median=1.0, p95=3.0, max=3.0, rms=1.0)
    level, _ = rpt.verdict(exactly_at_bar, pass_px=1.0, warn_px=3.0)
    assert level == "PASS"


# --------------------------------------------------------------------------- rendering

def test_render_report_shape_and_verdict_badge():
    s = rpt.ErrorStats(n=1954, mean=0.372, median=0.287, p95=0.959, max=3.761, rms=0.509)
    level, message = rpt.verdict(s)
    text = rpt.render_report("charuco", "kb", 58, 58, s, level, message, color=False)
    assert "charuco" in text and "kb" in text
    assert "58/58" in text
    assert "1954" in text
    assert "PASS" in text
    assert "\x1b[" not in text            # color=False -> no ANSI codes


def test_render_report_color_adds_ansi_codes():
    s = rpt.ErrorStats(n=10, mean=0.1, median=0.1, p95=0.2, max=0.3, rms=0.15)
    level, message = rpt.verdict(s)
    text = rpt.render_report("checkerboard", "ds", 10, 10, s, level, message, color=True)
    assert "\x1b[32m" in text             # PASS -> green


# --------------------------------------------------------------------------- live progress

def test_live_line_and_stage_do_not_raise_on_a_non_tty_stream():
    stream = _FakeNonTTY()
    rpt.live_line("hello", stream=stream)
    rpt.end_live(stream=stream)
    with rpt.Stage("unit-test stage", verbose=True, stream=stream):
        pass
    assert "unit-test stage" in stream.getvalue()


def test_make_detect_progress_fires_and_can_be_silenced():
    stream = _FakeNonTTY()
    cb = rpt.make_detect_progress(verbose=True, stream=stream)
    cb(1, 3, "/some/path/img_000.png")
    assert "img_000.png" in stream.getvalue()
    assert "1/3" in stream.getvalue()

    quiet_stream = _FakeNonTTY()
    quiet_cb = rpt.make_detect_progress(verbose=False, stream=quiet_stream)
    quiet_cb(1, 3, "/some/path/img_000.png")
    assert quiet_stream.getvalue() == ""
