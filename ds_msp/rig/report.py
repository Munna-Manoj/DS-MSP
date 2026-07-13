"""Human-readable calibration reporting: live terminal progress, a distribution-stats
summary (not just one RMS number), a pass/warn/fail verdict, and a self-contained
interactive HTML "digital twin" report.

Exists because the previous CLI experience was: silence for minutes, then a single
``max reprojection RMS: X px`` line with no distribution, no live progress, and no way to
*see* whether the recovered extrinsics/intrinsics are actually plausible. See
``docs/learn/robust_losses_and_evaluation.md`` for why a single RMS number is misleading on
a robust fit — report the full distribution instead.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import bundle
from .types import RigState

# ---------------------------------------------------------------------------------------
# Live single-line progress (used for the long silent stretches: per-image detection, BA)
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
    stage that takes tens of seconds to minutes is visibly *running*, not silent."""

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


_BRAILLE_BITS = {
    (0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (0, 3): 0x40,
    (1, 0): 0x08, (1, 1): 0x10, (1, 2): 0x20, (1, 3): 0x80,
}


class BrailleCanvas:
    """A terminal pseudo-3D canvas. Unicode braille (U+2800-U+28FF) packs an 8-dot (2x4) grid
    per character cell — the standard trick (``drawille`` and kin) for real sub-character
    resolution in a plain terminal, giving ~2x4x the addressable points of one dot per
    character. Color is applied per *cell* (ANSI 24-bit truecolor), not per dot — the last dot
    drawn into a cell sets that cell's color. At calibration scale (tens of points, a handful
    of camera frustums) two points landing in the same cell is rare, and when it happens the
    color is still a defensible read (both are "roughly this error level here")."""

    def __init__(self, cols: int, rows: int):
        self.cols, self.rows = cols, rows
        self.w, self.h = cols * 2, rows * 4
        self._bits = [[0] * cols for _ in range(rows)]
        self._color = [[None] * cols for _ in range(rows)]

    def clear(self) -> None:
        """Clear every dot in the canvas (colors are left in place, unused until re-set)."""
        for row in self._bits:
            for i in range(len(row)):
                row[i] = 0

    def set(self, x: float, y: float, color: Optional[Tuple[int, int, int]] = None) -> None:
        """Light one dot at sub-character resolution.

        Parameters
        ----------
        x, y : float
            Dot coordinates in the canvas's ``(w, h) = (cols*2, rows*4)`` dot grid;
            rounded to the nearest integer dot. Out-of-bounds coordinates are
            silently ignored (no-op).
        color : tuple of int, optional
            ``(r, g, b)`` in ``0-255``, applied to the dot's whole character cell.
            If omitted, the cell's existing color (if any) is left unchanged.
        """
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < self.w and 0 <= yi < self.h):
            return
        cx, cy = xi // 2, yi // 4
        self._bits[cy][cx] |= _BRAILLE_BITS[(xi % 2, yi % 4)]
        if color is not None:
            self._color[cy][cx] = color

    def line(self, x0: float, y0: float, x1: float, y1: float,
            color: Optional[Tuple[int, int, int]] = None) -> None:
        """Draw a straight line from ``(x0, y0)`` to ``(x1, y1)`` in dot coordinates.

        Parameters
        ----------
        x0, y0, x1, y1 : float
            Endpoints in the canvas's dot grid (see :meth:`set`).
        color : tuple of int, optional
            ``(r, g, b)`` applied to every dot's cell along the line.
        """
        n = max(2, int(max(abs(x1 - x0), abs(y1 - y0))))
        for i in range(n + 1):
            t = i / n
            self.set(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, color)

    def render_lines(self) -> List[str]:
        """Render the canvas to printable text.

        Returns
        -------
        list of str
            One string per row (``self.rows`` entries), each row's braille
            characters ANSI-colored per non-empty cell.
        """
        lines = []
        for r in range(self.rows):
            parts = []
            for c in range(self.cols):
                bits = self._bits[r][c]
                ch = chr(0x2800 + bits)
                col = self._color[r][c]
                if col is not None and bits:
                    parts.append(f"\x1b[38;2;{col[0]};{col[1]};{col[2]}m{ch}\x1b[0m")
                else:
                    parts.append(ch)
            lines.append("".join(parts))
        return lines


def _orbit_basis(center: np.ndarray, radius: float, az: float, el: float):
    """Eye position + right/up/forward basis for an auto-orbiting camera looking at
    ``center`` from ``radius`` away — the same convention as the HTML report's JS viewer, so
    the terminal and browser views agree on what "orbiting" looks like."""
    dist = radius * 2.4 + 1e-6
    eye = center + dist * np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
    fwd = center - eye
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, world_up)
    rn = np.linalg.norm(right)
    right = right / rn if rn > 1e-9 else np.array([1.0, 0.0, 0.0])
    up = np.cross(right, fwd)
    return eye, right, up, fwd


def _err_color(e: float) -> Tuple[int, int, int]:
    """Green -> yellow -> red by reprojection error (px), same scale as the HTML report."""
    if not np.isfinite(e):
        return (75, 81, 100)
    t = max(0.0, min(1.0, e / 3.0))
    stops = [(0.0, (74, 222, 128)), (0.5, (250, 204, 21)), (1.0, (248, 113, 113))]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t0 <= t <= t1:
            k = (t - t0) / (t1 - t0 or 1)
            return tuple(int(c0[j] + (c1[j] - c0[j]) * k) for j in range(3))
    return stops[-1][1]


class Live3DAnimator:
    """Live terminal 'digital twin' of the rig during bundle adjustment: the *real* camera
    frustums at their current ``T_c_g`` plus one actually-captured frame's board points,
    reprojected through the *current* mid-solve intrinsics/extrinsics/object-pose and colored
    by their real, current reprojection error — auto-orbiting so a static terminal grid still
    reads as 3D. This replaces the earlier loss-curve sparkline, which was too abstract to be
    meaningful: this shows the thing actually being solved for, using the real evolving
    geometry the solver's ``on_iter`` callback now carries (``core/optimize.py`` passes its
    opaque ``state``; ``rig/bundle.py::refine`` reconstructs it into a ``RigState`` before
    calling us — see that module for why this doesn't slow the solve).

    Render cost is capped by ``min_interval`` (default ~12 fps) regardless of how fast the
    solver iterates — cheap iterations just skip rendering rather than queuing frames, so a
    fast solve never pays more than a bounded, small terminal-I/O tax per calibration.
    """

    def __init__(self, obj, object_obs, *, verbose: bool = True, stream=None,
                cols: int = 64, rows: int = 20, min_interval: float = 0.08):
        self.verbose = verbose
        self.stream = stream or sys.stdout
        self.canvas = BrailleCanvas(cols, rows)
        self.min_interval = min_interval
        self._last_render = 0.0
        self._az, self._el = 0.6, 0.4
        self._n_lines = 0
        self._started = False
        self._tty = self.verbose and self.stream.isatty()
        self._throttle_every = 5
        self._step = 0
        self._stage = ""
        # EMA-smoothed view center/radius: recomputing these fresh from the *current*,
        # possibly still-wildly-converging board pose every frame made the viewport itself
        # chase the board — the camera view visibly "flew" independent of whether the board
        # was actually moving that much. Smoothing decouples "the view is stable" from "the
        # board is still converging," which is the honest way to show convergence (a fixed
        # vantage point watching points settle), not the vantage point lurching after them.
        self._center = None
        self._radius = None

        # Pick one real, richly-observed (object_id, frame_id) to animate live — an actual
        # captured pose, not a synthetic one.
        by_key: Dict[Tuple[int, int], List] = {}
        for o in object_obs:
            by_key.setdefault((o.object_id, o.frame_id), []).append(o)
        if by_key:
            key = max(by_key, key=lambda k: sum(len(o.point_rows) for o in by_key[k]))
            self._object_id, self._frame_id = key
            self._frame_obs = by_key[key]
        else:
            self._object_id = self._frame_id = None
            self._frame_obs = []
        self._pts_3d = obj.pts_3d

    def set_stage(self, name: str) -> None:
        """Optional hook a caller iterating multiple BA stages (per-object warm-up, per-group
        extrinsics, global joint, structure polish — see ``rig/calibrate.py::calibrate_rig``)
        can call between stages so the live view labels which one is running, instead of
        silently jumping from "only the board moves" to "cameras move too" with no context."""
        self._stage = name

    def __call__(self, it: int, max_iter: int, rms: float, cost: float, rig) -> None:
        """The ``on_iter(it, max_iter, rms, cost, rig)`` signature every solver calls."""
        if not self.verbose or not self._frame_obs or not np.isfinite(rms):
            return
        self._step += 1
        now = time.time()
        if self._tty:
            if self._started and now - self._last_render < self.min_interval:
                return
            self._last_render = now
            self._az += 0.045
            self._render(it, max_iter, rms, cost, rig)
        elif self._step == 1 or self._step % self._throttle_every == 0:
            stage = f"{self._stage}: " if self._stage else ""
            self.stream.write(f"  [optimizing] {stage}step {self._step}  rms={rms:.4f}px  "
                              f"cost={cost:.3g}\n")
            self.stream.flush()

    def _update_framing(self, Xg_all: np.ndarray, cam_positions: List[np.ndarray],
                        *, snap: bool = False) -> None:
        center_now = Xg_all.mean(axis=0) if len(Xg_all) else np.zeros(3)
        board_radius = (float(np.max(np.linalg.norm(Xg_all - center_now, axis=1)))
                        if len(Xg_all) else 0.0)
        # ``cam_positions`` distance is measured from the *board* centroid, not from a rig
        # baseline — on a real rig the board is typically meters away from the cameras during
        # a captured frame, so this is really "how far the camera sphere must reach to include
        # every camera", not "how spread apart the cameras are from each other". A previous
        # version down-weighted this to 0.35x to keep small boards legible when cameras are
        # far away, but that under-covers the true extent: at orbit angles where a camera sits
        # roughly perpendicular to the view direction, its transverse offset is the *full*
        # (unweighted) cam_spread, so a 0.35x radius left it projected off-canvas — exactly the
        # "camera missing from the frustum view" bug. Full inclusion (matching the HTML
        # report's ``bounds()``, which never down-weights) is the correctness floor; a small
        # legibility margin keeps points from touching the very edge.
        cam_spread = (float(np.max(np.linalg.norm(np.array(cam_positions) - center_now, axis=1)))
                     if cam_positions else 0.0)
        radius_now = max(board_radius, cam_spread) * 1.08 + 1e-6
        if self._center is None or snap:
            self._center, self._radius = center_now, radius_now
        else:
            alpha = 0.15                    # EMA weight: smooth, not frozen — still adapts
            self._center = (1 - alpha) * self._center + alpha * center_now
            self._radius = (1 - alpha) * self._radius + alpha * radius_now

    def _render(self, it: int, max_iter: int, rms: float, cost: float, rig,
               *, label: Optional[str] = None) -> None:
        key = (self._object_id, self._frame_id)
        T_g_o = rig.object_poses.get(key)
        if T_g_o is None or not rig.T_c_g:
            return
        canvas = self.canvas
        canvas.clear()

        cam_positions = []
        for T_c_g in rig.T_c_g.values():
            R, t = T_c_g[:3, :3], T_c_g[:3, 3]
            cam_positions.append(-R.T @ t)
        Xg_all = (T_g_o[:3, :3] @ self._pts_3d.T).T + T_g_o[:3, 3]
        # ``_update_framing`` fully includes camera positions (see its docstring): a partial
        # weighting used to leave cameras outside the projected frustum at some orbit angles.
        # The view itself is EMA-smoothed so it doesn't chase a still-converging board
        # frame-to-frame.
        self._update_framing(Xg_all, cam_positions, snap=(label is not None))
        center, radius = self._center, self._radius
        eye, right, up, fwd = _orbit_basis(center, radius, self._az, self._el)

        def proj(p):
            """Perspective-project a world-frame point ``p`` (3,) into canvas dot
            coordinates, or ``None`` if it is behind the orbit camera (``vz <= 1e-3``)."""
            rel = p - eye
            vx, vy, vz = rel @ right, rel @ up, rel @ fwd
            if vz <= 1e-3:
                return None
            f = canvas.h * 1.15 / vz
            return canvas.w / 2 + vx * f, canvas.h / 2 - vy * f

        size = radius * 0.16
        for T_c_g in rig.T_c_g.values():
            R, t = T_c_g[:3, :3], T_c_g[:3, 3]
            pos = -R.T @ t
            ax_x, ax_y, ax_z = R.T[:, 0], R.T[:, 1], R.T[:, 2]
            tip = pos + ax_z * size
            corners = [tip + ax_x * u * size * 0.6 + ax_y * v * size * 0.6
                      for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            p_pos, p_tip = proj(pos), proj(tip)
            frustum_color = (125, 211, 252)
            if p_pos and p_tip:
                canvas.line(*p_pos, *p_tip, color=frustum_color)
            proj_corners = [proj(pc) for pc in corners]
            for i in range(4):
                pa, pb = proj_corners[i], proj_corners[(i + 1) % 4]
                if pa and pb:
                    canvas.line(*pa, *pb, color=frustum_color)
                if p_tip and pa:
                    canvas.line(*p_tip, *pa, color=frustum_color)

        for o in self._frame_obs:
            cam = rig.cameras.get(o.cam_id)
            T_c_g = rig.T_c_g.get(o.cam_id)
            if cam is None or T_c_g is None:
                continue
            Xo = self._pts_3d[o.point_rows]
            Xg = (T_g_o[:3, :3] @ Xo.T).T + T_g_o[:3, 3]
            Xc = (T_c_g[:3, :3] @ Xg.T).T + T_c_g[:3, 3]
            uv, valid = cam.project(Xc)
            err = np.full(len(Xo), np.nan)
            err[valid] = np.linalg.norm(uv[valid] - o.pts_2d[valid], axis=1)
            for p3, e in zip(Xg, err):
                pp = proj(p3)
                if pp is not None:
                    canvas.set(*pp, color=_err_color(e))

        lines = canvas.render_lines()
        title = label or f"optimizing ({self._stage or 'live rig'}, real reprojection error)"
        header = f"{title}  iter {it}/{max_iter}  rms={rms:.4f}px"
        if not (isinstance(cost, float) and np.isnan(cost)):
            header += f"  cost={cost:.3g}"
        block = header + "\n" + "\n".join(lines)
        if self._started:
            self.stream.write(f"\x1b[{self._n_lines}A")
        self.stream.write(block + "\n")
        self.stream.flush()
        self._n_lines = 1 + len(lines)
        self._started = True

    def finish(self, final_rig=None, *, it: Optional[int] = None,
              max_iter: Optional[int] = None, rms: Optional[float] = None,
              cost: Optional[float] = None) -> None:
        """Stop the live animation. If ``final_rig`` is given, render one last, **un-
        throttled** frame from it first — the throttle (``min_interval``) exists to bound
        render cost during the solve and can otherwise skip the very last iteration, leaving
        a slightly-stale frame on screen right when the user most wants to see the converged
        result. The view snaps to this final geometry instead of easing into it (``snap``),
        since this is the answer, not one more step of a trajectory."""
        if final_rig is not None and self.verbose and self._tty and self._frame_obs:
            self._render(it if it is not None else self._step,
                        max_iter if max_iter is not None else self._step,
                        rms if rms is not None else float("nan"),
                        cost if cost is not None else float("nan"),
                        final_rig, label="converged (final result)")
        self._started = False
        self._n_lines = 0


# ---------------------------------------------------------------------------------------
# Distribution stats — mean / median / p95 / max / rms / inlier fraction, not just max RMS
# ---------------------------------------------------------------------------------------

@dataclass
class ErrorStats:
    """Distribution of reprojection errors (pixels), not just a single RMS number.

    See ``docs/learn/robust_losses_and_evaluation.md`` for why a single RMS value
    is misleading on a robust fit — this reports the shape of the error
    distribution instead. Built by :func:`camera_and_overall_stats` /
    :func:`_stats` from per-observation errors.

    Parameters
    ----------
    n : int
        Number of error samples (0 if none were available; other fields are NaN).
    mean, median, p95, max, rms : float
        Mean, median, 95th percentile, maximum, and RMS reprojection error, pixels.
    inlier_frac : float
        Fraction of errors below the ``inlier_px`` threshold used to compute this
        instance (see :func:`camera_and_overall_stats`).
    """

    n: int
    mean: float
    median: float
    p95: float
    max: float
    rms: float
    inlier_frac: float
    inlier_rms: float = float("nan")   # RMS over the non-gross corners (robust; ignores blunders)
    n_gross: int = 0                   # #corners above the gross-outlier line (mis-detections)

    def to_dict(self) -> Dict[str, float]:
        """Return the stats as a plain ``{field: value}`` dict (for JSON/HTML export)."""
        return {"n": self.n, "mean": self.mean, "median": self.median, "p95": self.p95,
                "max": self.max, "rms": self.rms, "inlier_frac": self.inlier_frac,
                "inlier_rms": self.inlier_rms, "n_gross": self.n_gross}


# The gross-outlier line: a corner reprojecting past this on a sub-pixel calibration is a
# blunder (mis-decoded ChArUco id, wrong association), not noise. The robust bundle already
# down-weights it to ~zero in the *estimate* (docs/learn/robust_losses_and_evaluation.md), so
# the calibration is correct — but a naive max/rms still *counts* it and makes a correct fit
# read as a failure. ``inlier_rms``/``n_gross`` report the robust picture without dropping
# anything, honouring the "down-weight, don't drop" philosophy.
GROSS_PX = 5.0


def _stats(e: np.ndarray, inlier_px: float, gross_px: float = GROSS_PX) -> ErrorStats:
    if e.size == 0:
        nan = float("nan")
        return ErrorStats(0, nan, nan, nan, nan, nan, 0.0, nan, 0)
    non_gross = e[e < gross_px]
    return ErrorStats(
        n=int(e.size),
        mean=float(np.mean(e)),
        median=float(np.median(e)),
        p95=float(np.percentile(e, 95)),
        max=float(np.max(e)),
        rms=float(np.sqrt(np.mean(e ** 2))),
        inlier_frac=float(np.mean(e < inlier_px)),
        inlier_rms=float(np.sqrt(np.mean(non_gross ** 2))) if non_gross.size else float("nan"),
        n_gross=int(np.count_nonzero(e >= gross_px)),
    )


def camera_and_overall_stats(rig: RigState, object_obs, inlier_px: float = 1.0,
                             gross_px: float = GROSS_PX
                             ) -> Tuple[Dict[int, ErrorStats], ErrorStats]:
    """Per-camera and rig-wide :class:`ErrorStats` over all reprojection errors (px)."""
    per_obs = bundle.per_observation_errors(rig, object_obs)
    per_cam = {c: _stats(e, inlier_px, gross_px) for c, e in per_obs.items()}
    all_e = np.concatenate(list(per_obs.values())) if per_obs else np.zeros(0)
    return per_cam, _stats(all_e, inlier_px, gross_px)


# ---------------------------------------------------------------------------------------
# Verdict — was this calibration any good?
# ---------------------------------------------------------------------------------------

# Defaults, not universal truths: ChArUco/AprilGrid corner detection is itself good to only
# ~0.1-0.3 px (docs/learn/robust_losses_and_evaluation.md), so a converged median well under
# 1 px is the generic sub-pixel bar; p95 catches a fit that is good on average but has a
# systematic tail (bad frames, a wrong per-camera model). Override with --pass-px/--warn-px
# for a dataset with a known, different bar (e.g. this repo's MC-Calib ChArUco set: camera_0
# reference median 0.481 px, mean 0.593 px — see CLAUDE.md).
DEFAULT_PASS_PX = 1.0
DEFAULT_WARN_PX = 3.0


def verdict(overall: ErrorStats, pass_px: float = DEFAULT_PASS_PX,
           warn_px: float = DEFAULT_WARN_PX) -> Tuple[str, str]:
    """Return ``(level, message)`` with ``level`` in ``{"PASS", "WARN", "FAIL"}``."""
    if overall.n == 0 or not np.isfinite(overall.median):
        return "FAIL", "no reprojection observations were produced"
    if overall.median <= pass_px and overall.p95 <= 3 * pass_px:
        return "PASS", (f"median {overall.median:.3f}px, p95 {overall.p95:.3f}px "
                        f"<= {pass_px:.2f}/{3 * pass_px:.2f}px")
    if overall.median <= warn_px:
        return "WARN", (f"median {overall.median:.3f}px is above the {pass_px:.2f}px bar "
                        f"but under {warn_px:.2f}px — check per-camera table below")
    return "FAIL", (f"median {overall.median:.3f}px exceeds the {warn_px:.2f}px bar — "
                    "likely a wrong model, bad init, or outlier-contaminated detections")


# ---------------------------------------------------------------------------------------
# Terminal rendering
# ---------------------------------------------------------------------------------------

_COLOR = {"PASS": "32", "WARN": "33", "FAIL": "31"}


def _c(code: str, s: str, *, enabled: bool) -> str:
    return f"\x1b[{code}m{s}\x1b[0m" if enabled else s


def render_report(models: Dict[int, str], per_cam: Dict[int, ErrorStats], overall: ErrorStats,
                  level: str, message: str, *, color: Optional[bool] = None) -> str:
    """Render the final per-camera stats table + verdict as plain text (ANSI-colored on a
    TTY unless ``color`` overrides)."""
    if color is None:
        color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    lines: List[str] = []
    # ``rms``/``max`` are the raw all-corner numbers (a gross mis-detection inflates them even
    # on a correct, robustly-fit calibration); ``inl_rms`` is the robust RMS over non-blunder
    # corners — the honest accuracy of the fit. Both shown so nothing is hidden and nothing is
    # dropped (the blunders are down-weighted, not removed).
    header = (f"{'cam':>4}  {'model':13s} {'n':>6} {'mean':>7} {'median':>7} "
             f"{'p95':>7} {'max':>7} {'rms':>7} {'inl_rms':>8} {'inlier%':>8}")
    lines.append(header)
    lines.append("-" * len(header))
    for c in sorted(per_cam):
        s = per_cam[c]
        lines.append(
            f"{c:>4}  {models.get(c, '?'):13s} {s.n:>6d} {s.mean:>7.3f} {s.median:>7.3f} "
            f"{s.p95:>7.3f} {s.max:>7.3f} {s.rms:>7.3f} {s.inlier_rms:>8.3f} "
            f"{100 * s.inlier_frac:>7.2f}%")
    lines.append("-" * len(header))
    lines.append(
        f"{'all':>4}  {'':13s} {overall.n:>6d} {overall.mean:>7.3f} {overall.median:>7.3f} "
        f"{overall.p95:>7.3f} {overall.max:>7.3f} {overall.rms:>7.3f} "
        f"{overall.inlier_rms:>8.3f} {100 * overall.inlier_frac:>7.2f}%")
    lines.append("")
    # If gross outliers are present, say so plainly: they inflate max/rms but were down-weighted
    # to ~zero in the solve, so the calibration is governed by median/inl_rms, not the tail. This
    # turns an alarming-looking (but correct) result into an explained one.
    gross = {c: per_cam[c].n_gross for c in sorted(per_cam) if per_cam[c].n_gross}
    if gross:
        detail = ", ".join(f"cam {c}: {n}" for c, n in gross.items())
        lines.append(
            f"note: {overall.n_gross} corner(s) reproject > {GROSS_PX:.0f}px ({detail}) — "
            f"likely mis-detections (e.g. a ChArUco board a different OpenCV build mis-decodes).")
        lines.append(
            f"      they are DOWN-WEIGHTED in the solve, not fitted; judge accuracy by median / "
            f"inl_rms ({overall.inlier_rms:.3f}px), not max / rms.")
        lines.append("")
    badge = _c(_COLOR.get(level, "0"), f" {level} ", enabled=color)
    lines.append(f"verdict:{badge} {message}")
    return "\n".join(lines)


def print_report(models: Dict[int, str], per_cam: Dict[int, ErrorStats], overall: ErrorStats,
                 level: str, message: str) -> None:
    """Print the per-camera stats table + verdict to stdout.

    Parameters
    ----------
    models : dict of int to str
        Camera id -> model name, for the table's "model" column.
    per_cam : dict of int to ErrorStats
        Per-camera reprojection error distribution.
    overall : ErrorStats
        Rig-wide reprojection error distribution.
    level : str
        Verdict level, e.g. ``"PASS"``, ``"WARN"``, ``"FAIL"``.
    message : str
        Human-readable verdict explanation.
    """
    print(render_report(models, per_cam, overall, level, message))


# ---------------------------------------------------------------------------------------
# Self-contained interactive HTML "digital twin" report
# ---------------------------------------------------------------------------------------

def _camera_payload(rig: RigState, c: int, stats: ErrorStats) -> Dict:
    cam = rig.cameras[c]
    K = cam.K
    w, h = rig.img_size.get(c, (0, 0))
    return {
        "id": c,
        "model": getattr(cam, "name", "?"),
        "w": int(w), "h": int(h),
        "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "T_c_g": rig.T_c_g[c].tolist(),
        "stats": stats.to_dict(),
    }


def _frame_payload(rig: RigState, object_obs, max_frames: int) -> List[Dict]:
    """One entry per (object, frame) actually solved: the board's 3D points in the rig's
    world frame plus, per point, the worst reprojection error among the cameras that saw it
    this frame — a real capture-session animation, not a synthetic one."""
    by_key: Dict[Tuple[int, int], Dict[int, float]] = {}
    for o in object_obs:
        if o.cam_id not in rig.cameras:
            continue
        key = (o.object_id, o.frame_id)
        if key not in rig.object_poses:
            continue
        Xo = rig.objects[o.object_id].pts_3d[o.point_rows]
        Xg = (rig.object_poses[key][:3, :3] @ Xo.T).T + rig.object_poses[key][:3, 3]
        Xc = (rig.T_c_g[o.cam_id][:3, :3] @ Xg.T).T + rig.T_c_g[o.cam_id][:3, 3]
        uv, valid = rig.cameras[o.cam_id].project(Xc)
        err = np.full(len(o.point_rows), np.nan)
        err[valid] = np.linalg.norm(uv[valid] - o.pts_2d[valid], axis=1)
        errs_by_row = by_key.setdefault(key, {})
        for row, e in zip(o.point_rows.tolist(), err.tolist()):
            if np.isnan(e):
                continue
            errs_by_row[row] = max(errs_by_row.get(row, 0.0), e)

    frames = []
    keys = sorted(by_key)
    if len(keys) > max_frames:                     # keep the report file small
        step = len(keys) / max_frames
        keys = [keys[int(i * step)] for i in range(max_frames)]
    for key in keys:
        object_id, frame_id = key
        T_g_o = rig.object_poses[key]
        obj = rig.objects[object_id]
        Xg = (T_g_o[:3, :3] @ obj.pts_3d.T).T + T_g_o[:3, 3]
        errs_by_row = by_key[key]
        pts = [[float(x), float(y), float(z), errs_by_row.get(row, None)]
              for row, (x, y, z) in enumerate(Xg.tolist())]
        frames.append({"object_id": object_id, "frame_id": frame_id, "points": pts})
    return frames


def build_report_data(rig: RigState, scn, models: Dict[int, str],
                      per_cam: Dict[int, ErrorStats], overall: ErrorStats,
                      level: str, message: str, *, max_frames: int = 60) -> Dict:
    """Assemble the JSON payload embedded in the HTML report — everything the browser-side
    viewer needs, no server, no network fetch."""
    return {
        "scenario": getattr(scn, "name", ""),
        "generated_utc": None,           # stamped by the caller if desired (no wall-clock here)
        "ref_cam_id": rig.ref_cam_id,
        "cameras": [_camera_payload(rig, c, per_cam[c]) for c in sorted(rig.cameras)
                   if c in per_cam],
        "overall": overall.to_dict(),
        "verdict": {"level": level, "message": message},
        "frames": _frame_payload(rig, scn.object_obs, max_frames),
    }


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DS-MSP calibration report -- __SCENARIO__</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.4 -apple-system, "Segoe UI", Roboto, sans-serif;
         background: #0b0d12; color: #e6e8ee; }
  header { padding: 10px 16px; border-bottom: 1px solid #232733; display: flex;
           align-items: center; gap: 14px; flex-wrap: wrap; }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; color: #9fb4ff; }
  .badge { padding: 2px 10px; border-radius: 999px; font-weight: 700; font-size: 12px; }
  .PASS { background: #103d24; color: #4ade80; }
  .WARN { background: #3d3410; color: #facc15; }
  .FAIL { background: #3d1414; color: #f87171; }
  #wrap { display: flex; height: calc(100vh - 46px); }
  #view { flex: 1; position: relative; }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; }
  canvas:active { cursor: grabbing; }
  #side { width: 360px; overflow-y: auto; border-left: 1px solid #232733; padding: 12px; }
  table { border-collapse: collapse; width: 100%; font-size: 12px; margin-bottom: 14px; }
  th, td { padding: 3px 6px; text-align: right; border-bottom: 1px solid #1b1e27; }
  th:first-child, td:first-child { text-align: left; }
  th { color: #8890a4; font-weight: 600; }
  .hint { color: #8890a4; font-size: 12px; margin: 6px 0 14px; }
  #hud { position: absolute; left: 10px; bottom: 10px; background: rgba(10,12,18,.7);
         padding: 6px 10px; border-radius: 6px; font-size: 12px; color: #b7bccb; }
  #hud button { background: #1b2030; color: #dde2ee; border: 1px solid #2a3040;
                border-radius: 4px; padding: 2px 8px; margin-right: 6px; cursor: pointer; }
  #legend { position: absolute; right: 10px; top: 10px; background: rgba(10,12,18,.7);
            padding: 6px 10px; border-radius: 6px; font-size: 11px; color: #b7bccb; }
  #legend .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
                margin-right: 4px; vertical-align: middle; }
</style>
</head>
<body>
<header>
  <h1>DS-MSP calibration report</h1>
  <span>__SCENARIO__</span>
  <span class="badge __LEVEL__">__LEVEL__</span>
  <span>__MESSAGE__</span>
</header>
<div id="wrap">
  <div id="view">
    <canvas id="cv"></canvas>
    <div id="hud">
      <button id="play">pause</button>
      frame <span id="fidx">0</span>/<span id="fn">0</span>
      &nbsp;|&nbsp; drag to orbit, wheel to zoom
    </div>
    <div id="legend">
      reprojection error (px)<br>
      <span class="sw" style="background:#4ade80"></span>&lt;0.5
      <span class="sw" style="background:#facc15"></span>~1.5
      <span class="sw" style="background:#f87171"></span>&gt;3
      <span class="sw" style="background:#4b5164"></span>not seen
    </div>
  </div>
  <div id="side">
    <h3 style="margin-top:0">Per-camera reprojection (px)</h3>
    <table id="camtable"></table>
    <h3>Overall</h3>
    <table id="overalltable"></table>
    <p class="hint">Cameras are drawn as frustums at their calibrated extrinsics
      (<code>T_c_g</code>, reference camera = identity). Points are the fused board's 3D
      corners in the rig frame at each solved frame pose, colored by the worst reprojection
      error among the cameras that observed them in that frame. This is the actual solved
      geometry, not a synthetic replay.</p>
  </div>
</div>
<script>
const DATA = __DATA_JSON__;

// ---- tiny vanilla-JS 3D viewer: perspective projection + mouse-orbit, no dependencies ----
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
function resize() {
  const r = cv.getBoundingClientRect();
  cv.width = r.width * devicePixelRatio; cv.height = r.height * devicePixelRatio;
}
window.addEventListener('resize', resize);
resize();

let az = 0.6, el = 0.35, dist = null, target = [0, 0, 0];

function bounds() {
  let pts = [];
  for (const c of DATA.cameras) pts.push(camPos(c));
  for (const f of DATA.frames) for (const p of f.points) pts.push([p[0], p[1], p[2]]);
  if (!pts.length) return { c: [0, 0, 0], r: 1 };
  let c = [0, 0, 0];
  for (const p of pts) { c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; }
  c = c.map(v => v / pts.length);
  let r = 0.001;
  for (const p of pts) r = Math.max(r, Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]));
  return { c, r };
}

function invert4(T) {
  const R = [[T[0][0], T[0][1], T[0][2]], [T[1][0], T[1][1], T[1][2]], [T[2][0], T[2][1], T[2][2]]];
  const t = [T[0][3], T[1][3], T[2][3]];
  const Rt = [[R[0][0], R[1][0], R[2][0]], [R[0][1], R[1][1], R[2][1]], [R[0][2], R[1][2], R[2][2]]];
  const nt = [
    -(Rt[0][0] * t[0] + Rt[0][1] * t[1] + Rt[0][2] * t[2]),
    -(Rt[1][0] * t[0] + Rt[1][1] * t[1] + Rt[1][2] * t[2]),
    -(Rt[2][0] * t[0] + Rt[2][1] * t[1] + Rt[2][2] * t[2]),
  ];
  return { R: Rt, t: nt };
}

function camPos(c) { return invert4(c.T_c_g).t; }

function camAxes(c) {
  const { R } = invert4(c.T_c_g);
  // columns of R (world-frame camera axes): x-right, y-down, z-forward
  return {
    x: [R[0][0], R[1][0], R[2][0]],
    y: [R[0][1], R[1][1], R[2][1]],
    z: [R[0][2], R[1][2], R[2][2]],
  };
}

function addv(a, b, s) { s = s === undefined ? 1 : s; return [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s]; }

function project(p, view) {
  // world -> view (orbit camera) -> perspective
  const [x, y, z] = [p[0] - view.eye[0], p[1] - view.eye[1], p[2] - view.eye[2]];
  const vx = x * view.right[0] + y * view.right[1] + z * view.right[2];
  const vy = x * view.up[0] + y * view.up[1] + z * view.up[2];
  const vz = x * view.fwd[0] + y * view.fwd[1] + z * view.fwd[2];
  if (vz <= 0.01) return null;
  const f = view.f / vz;
  return [cv.width / 2 + vx * f, cv.height / 2 - vy * f, vz];
}

function makeView() {
  const b = bounds();
  if (dist === null) dist = b.r * 2.6 + 1e-6;
  target = b.c;
  const eye = [
    target[0] + dist * Math.cos(el) * Math.sin(az),
    target[1] + dist * Math.sin(el),
    target[2] + dist * Math.cos(el) * Math.cos(az),
  ];
  let fwd = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
  const n = Math.hypot(...fwd) || 1; fwd = fwd.map(v => v / n);
  let right = [fwd[1] * 0 - fwd[2] * 0, 0, 0];
  // world up = (0,-1,0) in typical camera-vision convention (y-down); use (0,1,0) screen-up
  const worldUp = [0, 1, 0];
  right = [
    fwd[1] * worldUp[2] - fwd[2] * worldUp[1],
    fwd[2] * worldUp[0] - fwd[0] * worldUp[2],
    fwd[0] * worldUp[1] - fwd[1] * worldUp[0],
  ];
  const rn = Math.hypot(...right) || 1; right = right.map(v => v / rn);
  const up = [
    right[1] * fwd[2] - right[2] * fwd[1],
    right[2] * fwd[0] - right[0] * fwd[2],
    right[0] * fwd[1] - right[1] * fwd[0],
  ];
  return { eye, fwd, right, up, f: cv.height * 0.9 };
}

function errColor(e) {
  if (e === null || e === undefined) return '#4b5164';
  const t = Math.max(0, Math.min(1, e / 3.0));
  // green -> yellow -> red
  const stops = [[0.0, [74, 222, 128]], [0.5, [250, 204, 21]], [1.0, [248, 113, 113]]];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i], [t1, c1] = stops[i + 1];
    if (t >= t0 && t <= t1) {
      const k = (t - t0) / (t1 - t0 || 1);
      const c = c0.map((v, j) => Math.round(v + (c1[j] - v) * k));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
  }
  return '#f87171';
}

function drawLine(view, a, b, style, width) {
  const pa = project(a, view), pb = project(b, view);
  if (!pa || !pb) return;
  ctx.strokeStyle = style; ctx.lineWidth = width || 1.5;
  ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
}

function drawFrustum(view, cam) {
  const pos = camPos(cam);
  const ax = camAxes(cam);
  const b = bounds();
  const size = Math.max(b.r * 0.12, 0.02);
  const aspect = (cam.h || 1) / (cam.w || 1);
  const corners = [[-1, -aspect], [1, -aspect], [1, aspect], [-1, aspect]].map(([u, v]) =>
    addv(addv(addv(pos, ax.z, size), ax.x, u * size), ax.y, v * size));
  for (let i = 0; i < 4; i++) drawLine(view, pos, corners[i], '#7dd3fc', 1);
  for (let i = 0; i < 4; i++) drawLine(view, corners[i], corners[(i + 1) % 4], '#7dd3fc', 1.5);
  const label = project(addv(pos, ax.z, size * 1.4), view);
  if (label) {
    ctx.fillStyle = '#bfe6ff'; ctx.font = `${12 * devicePixelRatio}px monospace`;
    ctx.fillText(`cam ${cam.id} (${cam.model})`, label[0] + 6, label[1]);
  }
}

let frameIdx = 0, playing = true, lastT = 0;
function draw(t) {
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = '#0b0d12'; ctx.fillRect(0, 0, cv.width, cv.height);
  const view = makeView();
  for (const c of DATA.cameras) drawFrustum(view, c);
  if (DATA.frames.length) {
    if (playing && t - lastT > 700) { frameIdx = (frameIdx + 1) % DATA.frames.length; lastT = t; }
    const fr = DATA.frames[frameIdx];
    document.getElementById('fidx').textContent = frameIdx + 1;
    document.getElementById('fn').textContent = DATA.frames.length;
    for (const p of fr.points) {
      const pr = project([p[0], p[1], p[2]], view);
      if (!pr) continue;
      ctx.fillStyle = errColor(p[3]);
      ctx.beginPath(); ctx.arc(pr[0], pr[1], 2.5 * devicePixelRatio, 0, 7); ctx.fill();
    }
  }
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);

// ---- orbit controls ----
let dragging = false, lastX = 0, lastY = 0;
cv.addEventListener('mousedown', e => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener('mouseup', () => dragging = false);
window.addEventListener('mousemove', e => {
  if (!dragging) return;
  az += (e.clientX - lastX) * 0.006;
  el = Math.max(-1.5, Math.min(1.5, el + (e.clientY - lastY) * 0.006));
  lastX = e.clientX; lastY = e.clientY;
});
cv.addEventListener('wheel', e => {
  dist = Math.max(1e-6, dist * (1 + Math.sign(e.deltaY) * 0.1));
  e.preventDefault();
}, { passive: false });
document.getElementById('play').addEventListener('click', e => {
  playing = !playing; e.target.textContent = playing ? 'pause' : 'play';
});

// ---- side tables ----
function fmt(v) { return (v === null || v === undefined || Number.isNaN(v)) ? '--' : v.toFixed(3); }
(function buildTables() {
  const ct = document.getElementById('camtable');
  ct.innerHTML = '<tr><th>cam</th><th>model</th><th>n</th><th>mean</th><th>median</th>' +
    '<th>p95</th><th>max</th><th>rms</th></tr>' + DATA.cameras.map(c =>
    `<tr><td>${c.id}</td><td>${c.model}</td><td>${c.stats.n}</td><td>${fmt(c.stats.mean)}</td>` +
    `<td>${fmt(c.stats.median)}</td><td>${fmt(c.stats.p95)}</td><td>${fmt(c.stats.max)}</td>` +
    `<td>${fmt(c.stats.rms)}</td></tr>`).join('');
  const ot = document.getElementById('overalltable');
  const o = DATA.overall;
  ot.innerHTML = '<tr><th>n</th><th>mean</th><th>median</th><th>p95</th><th>max</th><th>rms</th></tr>' +
    `<tr><td>${o.n}</td><td>${fmt(o.mean)}</td><td>${fmt(o.median)}</td><td>${fmt(o.p95)}</td>` +
    `<td>${fmt(o.max)}</td><td>${fmt(o.rms)}</td></tr>`;
})();
</script>
</body>
</html>
"""


def write_html_report(path: str, rig: RigState, scn, models: Dict[int, str],
                      per_cam: Dict[int, ErrorStats], overall: ErrorStats,
                      level: str, message: str, *, max_frames: int = 60) -> str:
    """Write a single self-contained ``.html`` file: embedded JSON data + a vanilla-JS
    canvas viewer (no external script/style/font references, works offline from ``file://``).
    Returns ``path``."""
    data = build_report_data(rig, scn, models, per_cam, overall, level, message,
                             max_frames=max_frames)
    html = (_HTML_TEMPLATE
           .replace("__SCENARIO__", str(data["scenario"]))
           .replace("__LEVEL__", level)
           .replace("__MESSAGE__", message)
           .replace("__DATA_JSON__", json.dumps(data)))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(html)
    return path
