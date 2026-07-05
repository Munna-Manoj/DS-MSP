"""Human-readable single-camera calibration reporting: live terminal progress during
detection, a distribution-stats summary (not just one RMS number), and a pass/warn/fail
verdict — the ``ds_msp.calib`` analogue of :mod:`ds_msp.rig.report`.

A fresh, independent module rather than an import of ``ds_msp.rig.report``: the import-linter
contract forbids a capability (``ds_msp.calib``) from importing a pipeline (``ds_msp.rig``),
and rig's own module is release-gated, tested against real 8-camera data (ADR-0006) -- not
worth touching for a bit of code sharing. The terminal-facing pieces below (``Stage``,
``live_line``, the stats/verdict shape) are ported in spirit, not copy-pasted line for line:
single-camera calibration has no per-camera table dimension (there is exactly one camera), so
``render_report`` is a single summary line, not a per-camera table.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------------------
# Live single-line progress (used for the long silent stretch: per-image detection)
# ---------------------------------------------------------------------------------------

_last_len = 0


def live_line(msg: str, *, stream=None, throttle_every: int = 25) -> None:
    """Update one live status line. On a TTY this rewrites the current line in place; when
    output is redirected (CI logs, ``| tee``) it throttles to avoid spamming a log with one
    line per image — full detail every ``throttle_every`` calls plus the first/last."""
    global _last_len
    stream = stream or sys.stdout
    if stream.isatty():
        pad = max(0, _last_len - len(msg))
        stream.write("\r" + msg + (" " * pad))
        stream.flush()
        _last_len = len(msg)
    else:
        stream.write(msg + "\n")


def end_live(*, stream=None) -> None:
    """Terminate a live-line run — newline on a TTY, no-op otherwise."""
    global _last_len
    stream = stream or sys.stdout
    if stream.isatty() and _last_len:
        stream.write("\n")
        stream.flush()
    _last_len = 0


class Stage:
    """Context manager that brackets a calibration stage with a start/elapsed banner, so a
    stage that takes seconds to minutes is visibly *running*, not silent."""

    def __init__(self, title: str, verbose: bool = True, stream=None):
        self.title = title
        self.verbose = verbose
        self.stream = stream or sys.stdout
        self.t0 = 0.0

    def __enter__(self) -> "Stage":
        self.t0 = time.time()
        if self.verbose:
            self.stream.write(f"-> {self.title}...\n")
            self.stream.flush()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        end_live(stream=self.stream)
        if self.verbose and exc_type is None:
            dt = time.time() - self.t0
            self.stream.write(f"   done ({dt:.1f}s)\n")
            self.stream.flush()


def make_detect_progress(*, verbose: bool = True, stream=None):
    """A ``Board.detect`` ``progress_cb(i, n, path)`` that live-updates one terminal line per
    image, mirroring ``ds_msp.rig.calib_param``'s ``_compose_progress`` (minus the per-camera
    dimension single-camera calibration doesn't have)."""
    import os

    def _cb(i: int, n: int, path: str) -> None:
        if not verbose:
            return
        live_line(f"[detect] {i}/{n}  {os.path.basename(path)}", stream=stream)

    return _cb


# ---------------------------------------------------------------------------------------
# Distribution stats — mean / median / p95 / max / rms, not just one RMS number
# ---------------------------------------------------------------------------------------

@dataclass
class ErrorStats:
    """Reprojection-error distribution summary, in pixels.

    ``n`` is the number of residual observations the other five statistics are computed
    over (not the number of images) — see :meth:`from_result`.
    """

    n: int
    mean: float
    median: float
    p95: float
    max: float
    rms: float

    def to_dict(self) -> Dict[str, float]:
        """Return ``{"n", "mean", "median", "p95", "max", "rms"}`` as a plain dict."""
        return {"n": self.n, "mean": self.mean, "median": self.median, "p95": self.p95,
                "max": self.max, "rms": self.rms}

    @classmethod
    def from_result(cls, result: Dict) -> "ErrorStats":
        """Build from :func:`ds_msp.calib.bundle.calibrate`'s (and thus
        :func:`ds_msp.calib.single_camera.calibrate_camera`'s) result dict — it already carries
        these fields, computed over every reprojection residual."""
        return cls(n=int(result["n_obs"]), mean=float(result["mean_px"]),
                   median=float(result["median_px"]), p95=float(result["p95_px"]),
                   max=float(result["max_px"]), rms=float(result["rms_px"]))


# ---------------------------------------------------------------------------------------
# Verdict — was this calibration any good?
# ---------------------------------------------------------------------------------------

# Defaults, not universal truths: ChArUco/AprilGrid corner detection is itself good to only
# ~0.1-0.3 px (docs/learn/robust_losses_and_evaluation.md), so a converged median well under
# 1 px is the generic sub-pixel bar; p95 catches a fit that is good on average but has a
# systematic tail (bad frames, a wrong model). Override with pass_px/warn_px for a dataset
# with a known, different bar (e.g. this repo's MC-Calib ChArUco set: camera_0 reference
# median 0.481 px, mean 0.593 px — see CLAUDE.md).
DEFAULT_PASS_PX = 1.0
DEFAULT_WARN_PX = 3.0


def verdict(overall: ErrorStats, pass_px: float = DEFAULT_PASS_PX,
           warn_px: float = DEFAULT_WARN_PX) -> Tuple[str, str]:
    """Return ``(level, message)`` with ``level`` in ``{"PASS", "WARN", "FAIL"}``."""
    if overall.n == 0 or not (overall.median == overall.median):  # NaN check, no numpy import
        return "FAIL", "no reprojection observations were produced"
    if overall.median <= pass_px and overall.p95 <= 3 * pass_px:
        return "PASS", (f"median {overall.median:.3f}px, p95 {overall.p95:.3f}px "
                        f"<= {pass_px:.2f}/{3 * pass_px:.2f}px")
    if overall.median <= warn_px:
        return "WARN", (f"median {overall.median:.3f}px is above the {pass_px:.2f}px bar "
                        f"but under {warn_px:.2f}px — check detection yield and board geometry")
    return "FAIL", (f"median {overall.median:.3f}px exceeds the {warn_px:.2f}px bar — "
                    "likely a wrong model, bad init, or outlier-contaminated detections")


# ---------------------------------------------------------------------------------------
# Terminal rendering
# ---------------------------------------------------------------------------------------

_COLOR = {"PASS": "32", "WARN": "33", "FAIL": "31"}


def _c(code: str, s: str, *, enabled: bool) -> str:
    return f"\x1b[{code}m{s}\x1b[0m" if enabled else s


def render_report(board_type: str, model_name: str, n_detected: int, n_total: int,
                  overall: ErrorStats, level: str, message: str, *,
                  color: Optional[bool] = None) -> str:
    """Render the calibration summary + verdict as plain text (ANSI-colored on a TTY unless
    ``color`` overrides). One board+model, not a per-camera table — single-camera calibration
    has exactly one of each."""
    if color is None:
        color = sys.stdout.isatty()
    lines = [f"board: {board_type}   model: {model_name}   "
            f"images: {n_detected}/{n_total} detected"]
    header = f"{'n':>6} {'mean':>7} {'median':>7} {'p95':>7} {'max':>7} {'rms':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    lines.append(f"{overall.n:>6d} {overall.mean:>7.3f} {overall.median:>7.3f} "
                f"{overall.p95:>7.3f} {overall.max:>7.3f} {overall.rms:>7.3f}")
    lines.append("")
    badge = _c(_COLOR.get(level, "0"), f" {level} ", enabled=color)
    lines.append(f"verdict:{badge} {message}")
    return "\n".join(lines)


def print_report(board_type: str, model_name: str, n_detected: int, n_total: int,
                 overall: ErrorStats, level: str, message: str) -> None:
    """Print :func:`render_report`'s output to stdout. Arguments are forwarded verbatim."""
    print(render_report(board_type, model_name, n_detected, n_total, overall, level, message))
