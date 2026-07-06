"""Live cinematic 3D visualization of a rig calibration in progress, replacing the
braille-terminal :class:`~ds_msp.rig.report.Live3DAnimator`. Same duck-typed callback
contract (``set_stage``, ``__call__(it, max_iter, rms, cost, rig)``, ``finish``) so it drops
into ``rig/calibrate.py``'s ``on_iter`` hook and both CLI entry points unchanged, but instead
of redrawing a braille canvas in the terminal, it runs a tiny local HTTP server and streams
the *real, currently-converging* rig state to a Three.js scene in the browser: a pond, where
each camera's depth is driven live by its own actual reprojection error -- worse error sits
near the floor, better error rises toward the surface, and the best-converged camera breaks
through at the finale while the rest sink back down (see the class docstring below for the
full concept). Everything in the scene is procedurally generated (shaders + primitives); no
borrowed 3D character/creature assets.

Exists because a static-cell terminal grid is a poor medium for "here is 3D geometry
converging" -- there is no orbit, no depth cue beyond ASCII density, and nothing to look at
during the tens of seconds to minutes a real BA run takes. This module keeps the same low
per-iteration cost discipline as the terminal animator (throttled render rate, error computed
on one representative frame, not the whole dataset) so watching it never taxes the solve.
"""

from __future__ import annotations

import functools
import http.server
import json
import os
import sys
import tempfile
import threading
import time
import webbrowser
from typing import Dict, List, Optional, Tuple

import numpy as np


def _json_safe(o):
    """Recursively replace NaN/Inf with ``None`` before serializing. Python's ``json.dump``
    happily emits a literal, non-standard ``NaN``/``Infinity`` token for these (``allow_nan``
    defaults to ``True``) -- harmless to re-parse in Python, but a hard ``SyntaxError`` for the
    browser's ``JSON.parse``, e.g. a zero-observation camera's ``ErrorStats.median`` at
    ``finish()``. Applied once at the single write choke-point rather than at every call site
    that builds a state dict."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_json_safe(v) for v in o]
    return o


class WebLive3DAnimator:
    """Live cinematic "digital twin" of a rig calibration, rendered in a browser tab via
    Three.js instead of the terminal.

    Concept -- "the pond"
    ---------------------
    Every camera starts on the pond floor and rises through the water as its own live
    reprojection error falls -- worse error sits deep, better error climbs toward the
    surface. That single mapping (real per-camera error -> depth) is the entire
    visualization's data-driven core; everything else (water, fish, floor "pearls" at the
    real solved board-point positions, particle bursts) is procedurally generated atmosphere,
    not a borrowed 3D asset. At the finale the best-converged camera (the real MVP, by lowest
    final median error) breaks the surface with a splash; every other camera sinks back to the
    floor -- see ``onFinale`` in the embedded viewer script.

    The hierarchical-BA stages (``rig/calibrate.py::calibrate_rig``'s ``set_stage`` calls) are
    choreographed as acts with title cards and a shifting camera shot, so the otherwise-silent
    minutes of a real BA run read as a story with a beginning, middle, and end. A live
    leaderboard (ranked by per-camera reprojection error, with a sub-pixel badge) and an RMS
    "vitals" sparkline turn the wait into something to actually watch.

    Cost discipline mirrors ``Live3DAnimator``: renders are throttled to ``min_interval``
    seconds regardless of solver iteration rate, and the only per-frame reprojection-error
    computation is over one representative (object, frame) pair, not the whole dataset -- see
    ``_camera_frame_errors``. The browser polls a small JSON state file over HTTP; nothing here
    talks back to the solver, so a browser tab left open (or never opened) costs the calibration
    run nothing beyond that bounded per-iteration JSON write.

    Launches before there is anything to show
    -------------------------------------------
    The server (and browser tab) start the moment this object is constructed, *before*
    detection has even run -- measured on this repo's real MC-Calib dataset, corner detection
    (~5s) and the front-end's per-camera intrinsic fit (~19s, no progress callback of its own)
    together can dominate a minute of otherwise totally silent wait before a single BA
    iteration exists, during which a construct-with-the-scene-in-hand design would leave the
    browser blank. ``obj``/``object_obs`` are therefore optional at construction; call
    :meth:`bind_scene` once they exist (both CLI entry points do this right after detection),
    and feed :meth:`detect_progress` / :meth:`save_progress` as the ``progress_cb`` for
    detection and the (also-silent-until-now) MC-Calib debug-image writers respectively, so
    every phase of a run streams *something* real instead of the view going dark between them.
    """

    def __init__(self, obj=None, object_obs=None, *, verbose: bool = True, auto_open: bool = True,
                min_interval: float = 0.08, history_len: int = 400, stream=None,
                finish_grace_s: float = 2.0):
        self.verbose = verbose
        self.stream = stream or sys.stdout
        self.min_interval = min_interval
        self.finish_grace_s = finish_grace_s
        self._last_render = 0.0
        self._step = 0
        self._stage = ""
        self._started = False
        self._history: List[float] = []
        self._history_len = history_len
        self._pts_3d = None
        self._object_obs: List = []
        self._frame_obs: List = []
        self._tty = self.verbose and self.stream.isatty()
        self._throttle_every = 5

        self._detect_started = False
        self._detect_counts: Dict[int, Tuple[int, int]] = {}
        self._last_detect_write = 0.0
        self._save_started = False
        self._save_counts: Dict[int, Tuple[int, int]] = {}
        self._last_save_write = 0.0

        self._dir: Optional[str] = None
        self._state_path = self._tmp_path = None
        self._server: Optional[http.server.ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._url: Optional[str] = None
        self._last_state: Dict = {"status": "starting", "stage": "", "step": 0, "it": 0,
                                  "max_iter": 0, "rms": None, "cost": None, "cameras": [],
                                  "beams": [], "history": []}

        if self.verbose:
            try:
                self._start_server()
                if auto_open and self._tty:
                    webbrowser.open(self._url)
                self.stream.write(f"[ds-msp][rig] live 3D view: {self._url}\n")
                self.stream.flush()
            except OSError as e:
                # A headless/sandboxed environment (no loopback socket, no browser) should
                # degrade to plain progress lines, not crash the calibration run over a
                # visualization nicety.
                self.stream.write(f"[ds-msp][rig] live 3D view unavailable ({e}); "
                                  "falling back to progress lines\n")
                self._server = None

        if obj is not None and object_obs is not None:
            self.bind_scene(obj, object_obs)

    def bind_scene(self, obj, object_obs) -> None:
        """Bind the real fused object + observations once detection/reconstruction has
        produced them. Picks one representative frame PER CAMERA (its own richest-observed
        one), not one shared frame for the whole rig -- a single shared frame (what
        ``Live3DAnimator`` picks, fine for a small terminal grid) silently drops every camera
        that doesn't happen to co-observe that one frame, which on a real multi-board rig
        where cameras point in different directions is routinely most of them, leaving their
        leaderboard entry/glow permanently blank for the whole run. Bounded cost is unchanged:
        still exactly one ``ObjectObs``'s worth of work per camera, just chosen independently."""
        self._pts_3d = obj.pts_3d
        self._object_obs = object_obs
        best_by_cam: Dict[int, object] = {}
        for o in object_obs:
            cur = best_by_cam.get(o.cam_id)
            if cur is None or len(o.point_rows) > len(cur.point_rows):
                best_by_cam[o.cam_id] = o
        self._frame_obs = list(best_by_cam.values())

    # -- server plumbing ------------------------------------------------------------------

    def _start_server(self) -> None:
        self._dir = tempfile.mkdtemp(prefix="dsmsp_web3d_")
        self._state_path = os.path.join(self._dir, "state.json")
        self._tmp_path = self._state_path + ".tmp"
        with open(os.path.join(self._dir, "index.html"), "w") as f:
            f.write(_VIEWER_HTML)
        self._write_state({"status": "starting", "stage": "", "step": 0, "it": 0,
                           "max_iter": 0, "rms": None, "cost": None, "cameras": [],
                           "beams": [], "history": []})

        class _QuietHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, fmt, *args):      # noqa: A002 - stdlib signature
                """Suppress ``BaseHTTPRequestHandler``'s default per-request stderr logging."""
                pass

        handler = functools.partial(_QuietHandler, directory=self._dir)
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._server.daemon_threads = True
        port = self._server.server_address[1]
        self._url = f"http://127.0.0.1:{port}/index.html"
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()

    def _write_state(self, state: Dict) -> None:
        self._last_state = state
        if self._state_path is None:
            return
        with open(self._tmp_path, "w") as f:
            json.dump(_json_safe(state), f)
        os.replace(self._tmp_path, self._state_path)

    # -- public duck-typed hook API (matches Live3DAnimator) ------------------------------

    def set_stage(self, name: str) -> None:
        """See ``Live3DAnimator.set_stage`` -- called between the hierarchical-BA stages so
        the browser can trigger the matching act's title card and camera shot. Unlike the
        terminal animator, this also pushes an immediate state update (patched from the last
        known state) rather than only taking effect on the next ``__call__``: some stages
        (front-end intrinsics, saving debug images) have no per-iteration callback of their
        own at all, so without this the browser would sit frozen on the *previous* stage's
        label for that whole silent stretch instead of showing the new one."""
        self._stage = name
        if self._server is not None:
            state = dict(self._last_state)
            state["stage"] = name
            if state.get("status") not in ("finished",):
                state["status"] = "running"
            self._write_state(state)

    def detect_progress(self, cam_id: int, i: int, n: int, path: str) -> None:
        """A ``detect.charuco.detect_rig`` / ``rig.reconstruct.detect_board_obs_images``
        -compatible ``progress_cb`` -- streams live per-camera corner-detection counts to the
        browser instead of leaving it blank during detection (previously invisible; the live
        view didn't exist yet at this point in a run)."""
        if self._server is None:
            return
        now = time.time()
        is_new_cam = cam_id not in self._detect_counts
        if (self._detect_started and not is_new_cam
                and now - self._last_detect_write < 0.15 and i != n):
            return
        self._last_detect_write = now
        self._detect_started = True
        self._detect_counts[cam_id] = (i, n)
        self._write_state({
            "status": "detecting", "stage": "(pre) scanning for the board", "step": 0,
            "it": 0, "max_iter": 0, "rms": None, "cost": None, "cameras": [], "beams": [],
            "history": [],
            "detect": {str(c): {"i": ii, "n": nn} for c, (ii, nn) in self._detect_counts.items()},
        })

    def save_progress(self, cam_id: int, i: int, n: int, frame_id: int) -> None:
        """A ``io.mccalib.save_reprojection_images`` / ``save_detection_images`` -compatible
        ``progress_cb`` -- these used to run fully serially and silently *after* the whole
        bundle adjustment already converged (measured ~12.5s each on this repo's real MC-Calib
        dataset), the single biggest "the live view just sits there doing nothing" gap; now
        parallel (see io/mccalib.py) and reported live instead."""
        if self._server is None:
            return
        now = time.time()
        is_new_cam = cam_id not in self._save_counts
        if (self._save_started and not is_new_cam
                and now - self._last_save_write < 0.15 and i != n):
            return
        self._last_save_write = now
        self._save_started = True
        self._save_counts[cam_id] = (i, n)
        state = dict(self._last_state)
        if state.get("status") not in ("finished",):
            state["status"] = "running"
        state["save"] = {str(c): {"i": ii, "n": nn} for c, (ii, nn) in self._save_counts.items()}
        self._write_state(state)

    def __call__(self, it: int, max_iter: int, rms: float, cost: float, rig) -> None:
        """The ``on_iter(it, max_iter, rms, cost, rig)`` signature every solver calls."""
        if not self.verbose or not self._frame_obs or not np.isfinite(rms):
            return
        self._step += 1
        now = time.time()
        if self._server is not None:
            if self._started and now - self._last_render < self.min_interval:
                return
            self._last_render = now
            self._started = True
            self._write_state(self._build_state(it, max_iter, rms, cost, rig))
        elif self._step == 1 or self._step % self._throttle_every == 0:
            stage = f"{self._stage}: " if self._stage else ""
            self.stream.write(f"  [optimizing] {stage}step {self._step}  rms={rms:.4f}px  "
                              f"cost={cost:.3g}\n")
            self.stream.flush()

    def finish(self, final_rig=None, *, it: Optional[int] = None,
              max_iter: Optional[int] = None, rms: Optional[float] = None,
              cost: Optional[float] = None) -> None:
        """Stop the live view. If ``final_rig`` is given, push one last, un-throttled
        "finished" state carrying full per-camera final stats (median/mean/p95/rms, MVP
        camera) -- see ``Live3DAnimator.finish`` for why the final frame must bypass the
        render throttle. The browser-side scene transitions to its finale (splash, victory
        leap, fireworks, stats card) on receiving ``status: "finished"``.

        Blocks for a short grace period after writing that state -- nothing otherwise
        guarantees the browser has actually fetched it before this process exits and kills
        the ephemeral HTTP server (a *measured* bug, not a theoretical one: the browser polls
        every 150ms, but background/inactive browser tabs commonly throttle ``setTimeout`` to
        1000ms or more, and both CLI entry points do real work -- writing MC-Calib output,
        printing the report -- between this call and process exit, but nothing upstream of
        this method ever confirmed delivery. Without the wait, a backgrounded tab can lose the
        final payload entirely: the finale JS logic itself is correct (verified by replaying
        the exact real payload through it), it just never runs because it never receives
        ``status: "finished"`` before the server disappears)."""
        if final_rig is not None and self.verbose and self._server is not None:
            state = self._build_state(
                it if it is not None else self._step,
                max_iter if max_iter is not None else self._step,
                rms if rms is not None else float("nan"),
                cost if cost is not None else float("nan"),
                final_rig, status="finished")
            state["final"] = self._final_payload(final_rig)
            self._write_state(state)
            self.stream.write(f"[ds-msp][rig] live 3D view (final): {self._url}\n")
            self.stream.flush()
            if self.finish_grace_s > 0:
                time.sleep(self.finish_grace_s)
        self._started = False

    # -- state construction -----------------------------------------------------------------

    def _camera_frame_errors(self, rig) -> Tuple[Dict[int, float], List[Dict]]:
        """Per-camera mean reprojection error + per-point "beams" on each camera's own
        representative frame (see ``__init__``) -- the same bounded-cost approach as
        ``Live3DAnimator._render``, not a full-dataset recompute every rendered frame, but
        evaluated once per camera instead of once for the whole rig."""
        errs: Dict[int, float] = {}
        beams: List[Dict] = []
        for o in self._frame_obs:
            T_g_o = rig.object_poses.get((o.object_id, o.frame_id))
            cam = rig.cameras.get(o.cam_id)
            T_c_g = rig.T_c_g.get(o.cam_id)
            if T_g_o is None or cam is None or T_c_g is None:
                continue
            Xo = self._pts_3d[o.point_rows]
            Xg = (T_g_o[:3, :3] @ Xo.T).T + T_g_o[:3, 3]
            Xc = (T_c_g[:3, :3] @ Xg.T).T + T_c_g[:3, 3]
            uv, valid = cam.project(Xc)
            e = np.full(len(Xo), np.nan)
            e[valid] = np.linalg.norm(uv[valid] - o.pts_2d[valid], axis=1)
            finite = e[np.isfinite(e)]
            if finite.size:
                errs[o.cam_id] = float(np.mean(finite))
            for p3, ei in zip(Xg.tolist(), e.tolist()):
                beams.append({"cam": o.cam_id, "p": p3,
                             "e": (None if not np.isfinite(ei) else float(ei))})
        return errs, beams

    def _build_state(self, it: int, max_iter: int, rms: float, cost: float, rig, *,
                     status: str = "running") -> Dict:
        cam_errs, beams = self._camera_frame_errors(rig)
        cams = []
        for cid in sorted(rig.T_c_g):
            T_c_g = rig.T_c_g[cid]
            R, t = T_c_g[:3, :3], T_c_g[:3, 3]
            pos = (-R.T @ t)
            cams.append({
                "id": cid,
                "model": getattr(rig.cameras.get(cid), "name", "?"),
                "pos": pos.tolist(),
                "ax_x": R.T[:, 0].tolist(),
                "ax_y": R.T[:, 1].tolist(),
                "ax_z": R.T[:, 2].tolist(),
                "err": cam_errs.get(cid),
            })
        if np.isfinite(rms):
            self._history.append(float(rms))
            if len(self._history) > self._history_len:
                self._history = self._history[-self._history_len:]
        return {
            "status": status,
            "stage": self._stage,
            "step": self._step,
            "it": int(it), "max_iter": int(max_iter),
            "rms": float(rms) if np.isfinite(rms) else None,
            "cost": float(cost) if isinstance(cost, (int, float)) and np.isfinite(cost) else None,
            "cameras": cams,
            "beams": beams,
            "history": list(self._history),
        }

    def _final_payload(self, final_rig) -> Dict:
        """Final per-camera stats/MVP, plus the real solved geometry needed for the post-
        finale "digital twin" reveal: each camera's actual ``T_c_g`` pose (not the pond's
        error-driven depth) and, reusing ``report._frame_payload`` verbatim rather than
        reimplementing it, one entry per (object, frame) actually solved carrying the board's
        real 3D corner positions and per-point worst reprojection error -- the same data the
        static HTML report animates, so the live view's temporal board replay is provably the
        same real capture-session data, not a second, divergent implementation."""
        from .report import _frame_payload, camera_and_overall_stats
        per_cam, overall = camera_and_overall_stats(final_rig, self._object_obs)
        cams = []
        mvp_id, mvp_median = None, float("inf")
        for cid in sorted(per_cam):
            s = per_cam[cid]
            T_c_g = final_rig.T_c_g.get(cid)
            pose = {}
            if T_c_g is not None:
                R, t = T_c_g[:3, :3], T_c_g[:3, 3]
                pose = {"pos": (-R.T @ t).tolist(), "ax_x": R.T[:, 0].tolist(),
                       "ax_y": R.T[:, 1].tolist(), "ax_z": R.T[:, 2].tolist()}
            cams.append({"id": cid,
                        "model": getattr(final_rig.cameras.get(cid), "name", "?"),
                        **pose, **s.to_dict()})
            if np.isfinite(s.median) and s.median < mvp_median:
                mvp_id, mvp_median = cid, s.median
        frames = _frame_payload(final_rig, self._object_obs, max_frames=120)
        return {"cameras": cams, "overall": overall.to_dict(), "mvp": mvp_id, "frames": frames}


_VIEWER_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DS-MSP -- live rig calibration</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { width: 100%; height: 100%; overflow: hidden; background: #021014;
               color: #e6e8ee; font: 14px/1.4 -apple-system, "Segoe UI", Roboto, sans-serif; }
  #cv { position: fixed; inset: 0; display: block; }
  .panel { position: fixed; background: rgba(4,20,26,.72); backdrop-filter: blur(6px);
           border: 1px solid #163741; border-radius: 10px; padding: 10px 12px;
           transition: opacity .6s ease; }
  #hud { left: 12px; bottom: 12px; font-size: 12px; color: #b7bccb; display: flex;
         align-items: center; gap: 10px; }
  #hud button { background: #0d2b33; color: #dde2ee; border: 1px solid #1e4a56;
                border-radius: 6px; padding: 4px 10px; cursor: pointer; font-size: 12px; }
  #hud button:hover { background: #123642; }
  #vitals { left: 12px; top: 12px; width: 220px; }
  #vitals .num { font-size: 30px; font-weight: 700; letter-spacing: -1px; }
  #vitals .num small { font-size: 13px; font-weight: 500; color: #8890a4; }
  #vitals canvas { width: 100%; height: 46px; display: block; margin-top: 4px; }
  #vitals .stage { font-size: 11px; color: #7dd3fc; margin-top: 6px; min-height: 14px; }
  #board { right: 12px; top: 12px; width: 250px; }
  #board h3 { font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
              color: #8890a4; margin-bottom: 6px; font-weight: 600; }
  #board .row { display: flex; align-items: center; gap: 8px; padding: 3px 0;
                border-bottom: 1px solid #12333d; font-size: 12px; transition: opacity .3s; }
  #board .row .rank { width: 16px; color: #8890a4; }
  #board .row .cam { flex: 1; }
  #board .row .err { font-variant-numeric: tabular-nums; font-weight: 600; }
  #board .row .badge { font-size: 10px; }
  .lvl-good { color: #4ade80; } .lvl-mid { color: #facc15; } .lvl-bad { color: #f87171; }
  #card { position: fixed; inset: 0; display: flex; flex-direction: column; align-items: center;
          justify-content: center; text-align: center; pointer-events: none; opacity: 0;
          transition: opacity .5s ease; }
  #card .kicker { font-size: 13px; letter-spacing: .25em; color: #7dd3fc; font-weight: 700;
                  text-transform: uppercase; margin-bottom: 8px; text-shadow: 0 0 20px rgba(125,211,252,.6); }
  #card .title { font-size: 40px; font-weight: 800; letter-spacing: -.5px;
                 text-shadow: 0 0 30px rgba(0,0,0,.8); }
  #card .sub { font-size: 15px; color: #b7bccb; margin-top: 10px; max-width: 560px; }
  /* a SIDE panel, not a full-screen overlay -- the whole point of the post-finale reveal is
     the 3D scene (real camera poses + board replay), so the stats readout must never cover
     it. Anchored to the right edge, capped to a fraction of the viewport width, vertically
     scrollable if the credits list is long, never blocking the center/left where the reveal
     plays. */
  #finale { position: fixed; top: 50%; right: 16px; transform: translateY(-50%);
            width: min(300px, 28vw); max-height: 78vh; overflow-y: auto;
            opacity: 0; pointer-events: none; transition: opacity .8s ease; z-index: 3; }
  #finale .box { background: rgba(4,20,26,.88); border: 1px solid #1e4a56; border-radius: 14px;
                 padding: 18px 20px; text-align: center; box-shadow: 0 20px 80px rgba(0,0,0,.6); }
  #finale .title { font-size: 18px; font-weight: 800; color: #7dd3fc; }
  #finale .stats { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin: 14px 0; }
  #finale .stats div { background: #06181d; border-radius: 8px; padding: 7px 4px; }
  #finale .stats .v { font-size: 17px; font-weight: 700; }
  #finale .stats .k { font-size: 9px; color: #8890a4; text-transform: uppercase; }
  #finale .credits { max-height: 220px; overflow-y: auto; text-align: left; font-size: 11px;
                     margin-top: 10px; }
  #finale .credits div { display: flex; justify-content: space-between; padding: 2px 4px;
                         border-bottom: 1px solid #12333d; }
  #status { position: fixed; left: 50%; top: 12px; transform: translateX(-50%); font-size: 12px;
            color: #8890a4; }
  #fallback { position: fixed; inset: 0; display: none; align-items: center; justify-content: center;
              text-align: center; padding: 20px; }
  .letterbox { position: fixed; left: 0; right: 0; height: 9vh; background: #000; z-index: 5;
               transform: scaleY(0); transition: transform .5s cubic-bezier(.7,0,.3,1); }
  .letterbox.on { transform: scaleY(1); }
  #lbTop { top: 0; transform-origin: top; }
  #lbBottom { bottom: 0; transform-origin: bottom; }
  #flash { position: fixed; inset: 0; pointer-events: none; opacity: 0; z-index: 4;
           transition: opacity .12s ease-out; }
  #callout { position: fixed; left: 50%; top: 78%; transform: translate(-50%, -50%) scale(.85);
             font-size: 22px; font-weight: 800; letter-spacing: .08em; opacity: 0; z-index: 6;
             pointer-events: none; text-shadow: 0 0 24px currentColor; transition: opacity .25s, transform .25s; }
  #preflight { position: fixed; inset: 0; display: none; align-items: center; justify-content: center;
               z-index: 7; background: rgba(2,16,20,.55); }
  #preflight .box { background: rgba(4,20,26,.9); border: 1px solid #1e4a56; border-radius: 14px;
                    padding: 26px 32px; min-width: 360px; text-align: center; }
  #preflight .title { font-size: 15px; letter-spacing: .2em; color: #7dd3fc; font-weight: 700;
                      text-transform: uppercase; margin-bottom: 4px; }
  #preflight .sub { font-size: 12px; color: #8890a4; margin-bottom: 16px; }
  #preflight .cam-row { display: flex; align-items: center; gap: 8px; font-size: 12px;
                        margin: 5px 0; text-align: left; }
  #preflight .cam-row .lbl { width: 56px; color: #b7bccb; flex-shrink: 0; }
  #preflight .cam-row .bar { flex: 1; height: 6px; border-radius: 3px; background: #0d2b33;
                             overflow: hidden; }
  #preflight .cam-row .bar > div { height: 100%; background: linear-gradient(90deg,#0ea5e9,#7dd3fc);
                                   transition: width .15s ease; }
  #preflight .cam-row .n { width: 64px; text-align: right; color: #8890a4;
                           font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
<canvas id="cv"></canvas>
<div id="status">connecting...</div>

<div id="preflight">
  <div class="box">
    <div class="title" id="preflightTitle">SCANNING FOR THE BOARD</div>
    <div class="sub" id="preflightSub">Detecting ChArUco corners, camera by camera -- in parallel.</div>
    <div id="preflightRows"></div>
  </div>
</div>

<div class="panel" id="vitals">
  <div class="num" id="rmsnum">-- <small>px rms</small></div>
  <canvas id="spark" width="400" height="46"></canvas>
  <div class="stage" id="stagelabel"></div>
</div>

<div class="panel" id="board">
  <h3>Leaderboard -- reprojection error</h3>
  <div id="rows"></div>
</div>

<div class="panel" id="hud">
  <button id="modebtn">take manual control</button>
  <button id="orientBtn" style="display:none">orient rig</button>
  <span id="hint">director mode: watching the calibration</span>
</div>

<div id="card">
  <div class="kicker" id="cardKicker">MISSION</div>
  <div class="title" id="cardTitle">RIG CALIBRATION</div>
  <div class="sub" id="cardSub"></div>
</div>

<div id="finale">
  <div class="box">
    <div class="title">MISSION COMPLETE</div>
    <div class="stats" id="finaleStats"></div>
    <div class="credits" id="finaleCredits"></div>
  </div>
</div>

<div class="letterbox" id="lbTop"></div>
<div class="letterbox" id="lbBottom"></div>
<div id="flash"></div>
<div id="callout"></div>

<div id="fallback">
  <div>
    <h2>3D engine unavailable</h2>
    <p style="color:#8890a4;margin-top:8px;max-width:480px">
      Couldn't load Three.js from the CDN (no internet access from this browser?). The
      calibration itself is unaffected -- check the terminal for the text progress and final
      report.
    </p>
  </div>
</div>

<script type="importmap">
{ "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
} }
</script>
<script type="module">
let engineReady = false;
setTimeout(() => { if (!engineReady) document.getElementById('fallback').style.display = 'flex'; }, 6000);

let THREE, OrbitControls, EffectComposer, RenderPass, UnrealBloomPass, ShaderPass, TransformControls;
document.getElementById('status').textContent = 'filling the pond...';
try {
  THREE = await import('three');
  ({ OrbitControls } = await import('three/addons/controls/OrbitControls.js'));
  ({ EffectComposer } = await import('three/addons/postprocessing/EffectComposer.js'));
  ({ RenderPass } = await import('three/addons/postprocessing/RenderPass.js'));
  ({ UnrealBloomPass } = await import('three/addons/postprocessing/UnrealBloomPass.js'));
  ({ ShaderPass } = await import('three/addons/postprocessing/ShaderPass.js'));
  ({ TransformControls } = await import('three/addons/controls/TransformControls.js'));
} catch (err) {
  document.getElementById('fallback').style.display = 'flex';
  throw err;
}
engineReady = true;

// ---------------------------------------------------------------------------------------
// error color ramp -- same green/yellow/red convention as report.py's _err_color
// ---------------------------------------------------------------------------------------
function errColor(e) {
  if (e === null || e === undefined || Number.isNaN(e)) return new THREE.Color(0x4b5164);
  const t = Math.max(0, Math.min(1, e / 3.0));
  const stops = [[0.0, [74, 222, 128]], [0.5, [250, 204, 21]], [1.0, [248, 113, 113]]];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i], [t1, c1] = stops[i + 1];
    if (t >= t0 && t <= t1) {
      const k = (t - t0) / (t1 - t0 || 1);
      const c = c0.map((v, j) => Math.round(v + (c1[j] - v) * k));
      return new THREE.Color(`rgb(${c[0]},${c[1]},${c[2]})`);
    }
  }
  return new THREE.Color(0xf87171);
}
function errClass(e) {
  if (e === null || e === undefined) return '';
  if (e < 0.5) return 'lvl-good';
  if (e < 1.5) return 'lvl-mid';
  return 'lvl-bad';
}
const MODEL_COLOR = { kb: 0x60a5fa, dsplus: 0xc084fc, ds: 0xc084fc, radtan: 0x4ade80,
                      ucm: 0xfb923c, eucm: 0xfb923c, ocam: 0xf472b6 };
function modelColor(m) { return MODEL_COLOR[m] || 0x7dd3fc; }

// ---------------------------------------------------------------------------------------
// renderer / scene -- everything below is procedural (primitives + shaders), no external 3D
// assets of any kind. Real per-camera data drives only ONE thing, deliberately: each camera's
// depth in the pond, mapped from its own live reprojection error (see updateOrbs) -- the worse
// a camera's fit, the deeper it sits; the better, the closer to the surface. That is the whole
// visualization. Everything else (water, fish, pearls, particles) is atmosphere.
// ---------------------------------------------------------------------------------------
const canvas = document.getElementById('cv');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
function resize() { renderer.setSize(innerWidth, innerHeight); }
resize(); window.addEventListener('resize', () => { resize(); composer.setSize(innerWidth, innerHeight); });

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x03181d, 0.075);
const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 0.01, 100);
camera.position.set(2.6, 0.6, 3.4);

scene.add(new THREE.AmbientLight(0x2f6b78, 1.35));
const sun = new THREE.DirectionalLight(0xbdeeff, 1.5);
sun.position.set(1.5, 6, 2.5);
scene.add(sun);

// floating particulate motes for underwater depth cue
{
  const N = 500;
  const pos = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    pos[i*3] = (Math.random() - 0.5) * 14;
    pos[i*3+1] = -3 + Math.random() * 3.2;
    pos[i*3+2] = (Math.random() - 0.5) * 14;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const m = new THREE.PointsMaterial({ color: 0xdff6ff, size: 0.018, transparent: true, opacity: 0.4 });
  scene.add(new THREE.Points(g, m));
}

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.5, 0.65, 0.25);
composer.addPass(bloom);

// cinematic color grade -- vignette + chromatic aberration + film grain, with a reactive
// "punch" uniform flashKick()/shakeAmount below drive on real rms-delta events
const GRADE_SHADER = {
  uniforms: { tDiffuse: { value: null }, uTime: { value: 0 }, uVignette: { value: 0.34 },
             uAberration: { value: 0.0012 }, uGrain: { value: 0.024 }, uShake: { value: 0 } },
  vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `
    uniform sampler2D tDiffuse; uniform float uTime, uVignette, uAberration, uGrain, uShake;
    varying vec2 vUv;
    float rand(vec2 co) { return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453); }
    void main() {
      vec2 uv = vUv;
      vec2 dir = uv - 0.5;
      float ab = uAberration + uShake * 0.006;
      vec2 off = dir * ab;
      float r = texture2D(tDiffuse, uv - off).r;
      float g = texture2D(tDiffuse, uv).g;
      float b = texture2D(tDiffuse, uv + off).b;
      vec3 col = vec3(r, g, b);
      float vig = smoothstep(0.95, 0.25, length(dir));
      col *= mix(1.0 - uVignette, 1.0, vig);
      col += (rand(uv + uTime) - 0.5) * uGrain;
      gl_FragColor = vec4(col, 1.0);
    }
  `,
};
const gradePass = new ShaderPass(GRADE_SHADER);
composer.addPass(gradePass);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enabled = false;
controls.enableDamping = true;

// ---------------------------------------------------------------------------------------
// rig root -- the post-finale digital-twin reveal (real camera poses + board replay) is
// parented under this ONE group so it can be manually rotated as a single rigid body.
// Calibration only ever fixes ONE reference camera as the origin; that choice has no idea
// which way is "up" in the real room, so the rig's own frame is correct relative to itself
// but arbitrary relative to gravity. Letting the user spin the whole rigid assembly (never
// touching the underlying T_c_g/T_g_o data, purely a display-space transform) lets them
// re-orient it to match how the physical rig actually sat, without breaking the real,
// already-consistent relative geometry between cameras and the board.
// ---------------------------------------------------------------------------------------
const rigRoot = new THREE.Group();
scene.add(rigRoot);
const rigGizmo = new TransformControls(camera, renderer.domElement);
rigGizmo.setMode('rotate');
rigGizmo.size = 0.7;
rigGizmo.visible = false;
rigGizmo.enabled = false;
rigGizmo.addEventListener('dragging-changed', (e) => { controls.enabled = !e.value && manual; });
scene.add(rigGizmo);
let rigOrientMode = false;

// ---------------------------------------------------------------------------------------
// pond geometry -- fixed, stylized scale (not tied to the rig's real-world units, since orb
// depth now encodes error, not real position)
// ---------------------------------------------------------------------------------------
const POND_R = 2.6;
const FLOOR_Y = -2.1;
const SURFACE_Y = 0.0;
const POND_CENTER = new THREE.Vector3(0, (FLOOR_Y + SURFACE_Y) / 2, 0);

const floor = new THREE.Mesh(
  new THREE.CircleGeometry(POND_R * 1.7, 48),
  new THREE.MeshStandardMaterial({ color: 0x0a2e33, roughness: 0.95, metalness: 0.05 }));
floor.rotation.x = -Math.PI / 2; floor.position.y = FLOOR_Y;
scene.add(floor);

// animated caustic light patch on the floor -- pure procedural shader, no texture asset
const causticMat = new THREE.ShaderMaterial({
  uniforms: { uTime: { value: 0 } },
  transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
  fragmentShader: `
    varying vec2 vUv; uniform float uTime;
    void main() {
      vec2 uv = vUv * 7.0;
      float c = sin(uv.x + uTime*0.6) * sin(uv.y - uTime*0.5)
              + sin(uv.x*1.7 - uTime*0.8 + 1.5) * sin(uv.y*1.3 + uTime*0.4);
      c = smoothstep(0.55, 1.0, abs(c) * 0.5 + 0.5);
      float edge = 1.0 - smoothstep(0.6, 1.0, length(vUv - 0.5) * 2.0);
      gl_FragColor = vec4(vec3(0.35, 0.85, 1.0) * c * edge, c * edge * 0.3);
    }
  `,
});
const caustics = new THREE.Mesh(new THREE.CircleGeometry(POND_R * 1.65, 48), causticMat);
caustics.rotation.x = -Math.PI / 2; caustics.position.y = FLOOR_Y + 0.01;
scene.add(caustics);

// water surface -- vertex-displaced traveling waves + a fresnel-ish deep/shallow color blend,
// fully self-contained (no external texture/asset)
const WATER_VERT = `
  varying vec2 vUv; varying float vHeight;
  uniform float uTime;
  void main() {
    vUv = uv;
    vec3 pos = position;
    float h = sin(pos.x * 0.55 + uTime * 0.85) * 0.055
            + sin(pos.y * 0.8  - uTime * 1.15) * 0.04
            + sin((pos.x + pos.y) * 0.32 + uTime * 0.5) * 0.045;
    pos.z += h;
    vHeight = h;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
  }
`;
const WATER_FRAG = `
  varying vec2 vUv; varying float vHeight;
  uniform float uTime; uniform vec3 uDeep; uniform vec3 uShallow;
  void main() {
    vec3 col = mix(uDeep, uShallow, smoothstep(-0.06, 0.09, vHeight));
    float shimmer = sin(vUv.x * 46.0 + uTime * 1.6) * sin(vUv.y * 46.0 - uTime * 1.2);
    col += shimmer * 0.025;
    float edge = smoothstep(0.0, 0.3, vUv.x) * smoothstep(1.0, 0.7, vUv.x)
               * smoothstep(0.0, 0.3, vUv.y) * smoothstep(1.0, 0.7, vUv.y);
    gl_FragColor = vec4(col, mix(0.32, 0.8, edge));
  }
`;
const waterGeo = new THREE.PlaneGeometry(POND_R * 3.8, POND_R * 3.8, 56, 56);
const waterMat = new THREE.ShaderMaterial({
  uniforms: { uTime: { value: 0 }, uDeep: { value: new THREE.Color(0x0b4d55) },
             uShallow: { value: new THREE.Color(0x6fd6e0) } },
  vertexShader: WATER_VERT, fragmentShader: WATER_FRAG, transparent: true, side: THREE.DoubleSide,
});
const water = new THREE.Mesh(waterGeo, waterMat);
water.rotation.x = -Math.PI / 2; water.position.y = SURFACE_Y;
scene.add(water);

// ---------------------------------------------------------------------------------------
// particle bursts -- one generic system reused for splashes, breakthroughs, and the finale
// ---------------------------------------------------------------------------------------
let particleBursts = [];
function spawnBurst(pos, opts) {
  const { n = 80, colors = [0xdff6ff, 0x9fd8e8], speed = 1.2, spread = 1.0, up = true,
         size = 0.035, life = 1.1, gravity = -1.4 } = opts || {};
  const posArr = new Float32Array(n * 3);
  const vel = [];
  for (let i = 0; i < n; i++) {
    posArr[i * 3] = pos.x; posArr[i * 3 + 1] = pos.y; posArr[i * 3 + 2] = pos.z;
    const dir = new THREE.Vector3((Math.random() - 0.5), up ? Math.random() * 0.8 + 0.2 : (Math.random() - 0.5),
                                  (Math.random() - 0.5)).normalize();
    vel.push(dir.multiplyScalar(speed * (0.5 + Math.random() * spread)));
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
  const colArr = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const c = new THREE.Color(colors[i % colors.length]);
    colArr[i * 3] = c.r; colArr[i * 3 + 1] = c.g; colArr[i * 3 + 2] = c.b;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
  const mat = new THREE.PointsMaterial({ size, vertexColors: true, transparent: true });
  const points = new THREE.Points(geo, mat);
  scene.add(points);
  particleBursts.push({ points, vel, t: 0, life, gravity });
}
function spawnSplash(pos) {
  spawnBurst(pos, { n: 110, colors: [0xe0f7fa, 0xbae6fd, 0xffffff], speed: 1.7, up: true, size: 0.045, life: 1.3 });
  flashKick('#bae6fd', 0.16);
}
function updateParticleBursts(dt) {
  particleBursts = particleBursts.filter(pb => {
    pb.t += dt;
    const p = pb.points.geometry.attributes.position.array;
    for (let i = 0; i < pb.vel.length; i++) {
      pb.vel[i].y += pb.gravity * dt;
      p[i * 3] += pb.vel[i].x * dt; p[i * 3 + 1] += pb.vel[i].y * dt; p[i * 3 + 2] += pb.vel[i].z * dt;
    }
    pb.points.geometry.attributes.position.needsUpdate = true;
    pb.points.material.opacity = Math.max(0, 1 - pb.t / pb.life);
    if (pb.t > pb.life) { scene.remove(pb.points); pb.points.geometry.dispose(); return false; }
    return true;
  });
}

// ---------------------------------------------------------------------------------------
// fish -- real Reynolds boid flocking (separation/alignment/cohesion -- Reynolds 1987,
// "Flocks, Herds, and Schools: A Distributed Behavioral Model") PLUS a per-school "formation
// seek" force toward a slowly-evolving parametric curve (a Lissajous figure-8, a circling
// drift with a vertical sine bob, an expanding/contracting spiral -- one per school), which is
// what gives the whole group its own traveling wave-like path on top of the local boid
// jostling: this is the standard textbook way flocking sims layer a goal/migration urge onto
// the three core rules, not a departure from them. On top of that, one or two larger predator
// fish (see makeBigFish/updatePredators) roam the pond on their own slow wander path; any small
// fish within FEAR_RADIUS of a predator drops formation-seeking entirely and flees directly
// away at a higher speed cap (the real "confusion effect" scatter seen in prey fish schools),
// then resumes seeking its school's formation curve once the predator moves off -- this is the
// same fear/flee-then-regroup mechanism used in most predator-avoidance boid demos, just
// without needing a full state machine: "scared" is recomputed fresh every frame from real
// distance to the nearest predator, so regrouping falls out for free once that distance grows.
// ---------------------------------------------------------------------------------------
const FISH_COLORS = [0xfacc15, 0xfb923c, 0x38bdf8, 0x34d399, 0xf472b6];
const FEAR_RADIUS = 0.85;

function makeSmallFish(color, shapeIdx) {
  let bodyGeo;
  if (shapeIdx === 0) {                       // dart -- narrow cone
    bodyGeo = new THREE.ConeGeometry(0.045, 0.16, 4, 1);
    bodyGeo.rotateX(Math.PI / 2);
  } else if (shapeIdx === 1) {                // tetra -- flattened diamond
    bodyGeo = new THREE.OctahedronGeometry(0.075, 0);
    bodyGeo.scale(1, 0.55, 1.7);
  } else {                                    // guppy -- squashed teardrop
    bodyGeo = new THREE.SphereGeometry(0.06, 6, 5);
    bodyGeo.scale(0.7, 0.85, 1.9);
  }
  const mat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.35,
                                               flatShading: true, roughness: 0.55 });
  const mesh = new THREE.Mesh(bodyGeo, mat);
  const fin = new THREE.Mesh(new THREE.ConeGeometry(0.045, 0.07, 3), mat);
  fin.rotation.x = Math.PI / 2; fin.position.z = -0.1;
  mesh.add(fin);
  scene.add(mesh);
  return mesh;
}
function makeBigFish(color) {
  const bodyGeo = new THREE.ConeGeometry(0.12, 0.5, 6, 1);
  bodyGeo.rotateX(Math.PI / 2);
  const mat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.18,
                                               flatShading: true, roughness: 0.6 });
  const mesh = new THREE.Mesh(bodyGeo, mat);
  const fin = new THREE.Mesh(new THREE.ConeGeometry(0.09, 0.16, 3), mat);
  fin.rotation.x = Math.PI / 2; fin.position.z = -0.26;
  mesh.add(fin);
  scene.add(mesh);
  return mesh;
}

// per-school parametric "formation" target -- the mathematical wave pattern the group as a
// whole traces through the pond while individuals still jostle locally via boids
function schoolTarget(school, t) {
  const depthMid = FLOOR_Y + (SURFACE_Y - FLOOR_Y) * 0.4;
  if (school === 0) {                         // Lissajous figure-8
    return new THREE.Vector3(
      POND_CENTER.x + Math.sin(t * 0.25) * POND_R * 0.55,
      depthMid + Math.sin(t * 0.18) * 0.32,
      POND_CENTER.z + Math.sin(t * 0.5) * POND_R * 0.4);
  } else if (school === 1) {                  // circling drift with a vertical bob (traveling wave)
    const ang = t * 0.15;
    return new THREE.Vector3(
      POND_CENTER.x + Math.cos(ang) * POND_R * 0.58,
      depthMid - 0.35 + Math.sin(t * 0.4) * 0.22,
      POND_CENTER.z + Math.sin(ang) * POND_R * 0.58);
  }
  const ang = t * 0.22;                        // slow expanding/contracting spiral
  const r = POND_R * (0.22 + 0.25 * Math.sin(t * 0.1));
  return new THREE.Vector3(POND_CENTER.x + Math.cos(ang) * r, depthMid + 0.32 + Math.sin(t * 0.3) * 0.18,
                           POND_CENTER.z + Math.sin(ang) * r);
}

const FISH_N = 24, FISH_GROUPS = 3, PRED_N = 1;
const fish = [];
const predators = [];
function initFish() {
  for (let i = 0; i < FISH_N; i++) {
    const shapeIdx = i % 3;
    const mesh = makeSmallFish(FISH_COLORS[i % FISH_COLORS.length], shapeIdx);
    fish.push({
      mesh,
      pos: new THREE.Vector3((Math.random() - 0.5) * POND_R * 1.5,
                             FLOOR_Y + 0.2 + Math.random() * (SURFACE_Y - FLOOR_Y) * 0.7,
                             (Math.random() - 0.5) * POND_R * 1.5),
      vel: new THREE.Vector3((Math.random() - 0.5), (Math.random() - 0.5) * 0.15, (Math.random() - 0.5))
        .normalize().multiplyScalar(0.35),
      school: i % FISH_GROUPS,
      scared: false,
    });
  }
  for (let i = 0; i < PRED_N; i++) {
    predators.push({ mesh: makeBigFish(0x64748b), pos: null, phase: Math.random() * 20 });
  }
}
function updatePredators(t) {
  for (const p of predators) {
    const ang = t * 0.05 + p.phase;
    const r = POND_R * (0.5 + 0.35 * Math.sin(t * 0.04 + p.phase));
    const y = FLOOR_Y + (SURFACE_Y - FLOOR_Y) * (0.3 + 0.35 * Math.sin(t * 0.07 + p.phase * 1.3));
    const pos = new THREE.Vector3(POND_CENTER.x + Math.cos(ang) * r, y, POND_CENTER.z + Math.sin(ang) * r);
    const prev = p.pos ? p.pos.clone() : pos.clone();
    p.pos = pos;
    p.mesh.position.copy(pos);
    const fwd = pos.clone().sub(prev);
    if (fwd.lengthSq() > 1e-8) p.mesh.lookAt(pos.clone().add(fwd.normalize()));
  }
}
function updateFish(t, dt) {
  for (const f of fish) {
    const sep = new THREE.Vector3(), ali = new THREE.Vector3(), coh = new THREE.Vector3();
    let nSep = 0, nAli = 0;
    for (const o of fish) {
      if (o === f || o.school !== f.school) continue;
      const d = f.pos.distanceTo(o.pos);
      if (d < 0.32 && d > 1e-4) { sep.add(f.pos.clone().sub(o.pos).divideScalar(d)); nSep++; }
      if (d < 0.9) { ali.add(o.vel); coh.add(o.pos); nAli++; }
    }
    if (nSep) sep.divideScalar(nSep);
    if (nAli) { ali.divideScalar(nAli); coh.divideScalar(nAli).sub(f.pos); }

    let flee = new THREE.Vector3();
    let scared = false;
    for (const p of predators) {
      if (!p.pos) continue;
      const d = f.pos.distanceTo(p.pos);
      if (d < FEAR_RADIUS) {
        flee.add(f.pos.clone().sub(p.pos).normalize().multiplyScalar((FEAR_RADIUS - d) * 4));
        scared = true;
      }
    }
    f.scared = scared;

    const toCenter = POND_CENTER.clone().sub(f.pos);
    const distFromCenter = f.pos.distanceTo(POND_CENTER);
    const bound = distFromCenter > POND_R * 0.9 ? toCenter.normalize().multiplyScalar(0.9) : new THREE.Vector3();
    f.vel.addScaledVector(sep, 0.9 * dt).addScaledVector(ali, 0.4 * dt)
        .addScaledVector(coh, 0.25 * dt).addScaledVector(bound, dt);
    if (scared) {
      f.vel.addScaledVector(flee, dt);
      f.vel.clampLength(0, 1.3);                // panic burst -- faster than cruise speed
    } else {
      const seek = schoolTarget(f.school, t).sub(f.pos).multiplyScalar(0.18);
      f.vel.addScaledVector(seek, dt);
      f.vel.clampLength(0, 0.5);
    }
    if (f.pos.y > SURFACE_Y - 0.15) f.vel.y -= 0.6 * dt;
    if (f.pos.y < FLOOR_Y + 0.12) f.vel.y += 0.6 * dt;
    f.pos.addScaledVector(f.vel, dt);
    f.mesh.position.copy(f.pos);
    if (f.vel.lengthSq() > 1e-5) f.mesh.lookAt(f.pos.clone().add(f.vel));
  }
}

// ---------------------------------------------------------------------------------------
// camera orbs -- each camera is a faceted glowing gem in a ring cradle, colored and
// vertically positioned by its own real, currently-live reprojection error: worse error sits
// deeper, better error rises toward the surface. This is the entire visualization's core
// mechanic, and it is 100% real data, updated every render.
// ---------------------------------------------------------------------------------------
const orbs = new Map();

function makeLabelSprite() {
  const cvs = document.createElement('canvas'); cvs.width = 256; cvs.height = 64;
  const ctx = cvs.getContext('2d');
  const tex = new THREE.CanvasTexture(cvs);
  const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true });
  const spr = new THREE.Sprite(mat);
  spr.scale.set(0.85, 0.22, 1);
  return { sprite: spr, ctx, tex, cvs };
}
function drawLabel(lbl, text, color) {
  const { ctx, cvs, tex } = lbl;
  ctx.clearRect(0, 0, cvs.width, cvs.height);
  ctx.font = '600 20px monospace';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = 'rgba(4,20,26,.6)';
  const w = ctx.measureText(text).width + 20;
  ctx.beginPath(); ctx.roundRect((cvs.width - w) / 2, 18, w, 30, 8); ctx.fill();
  ctx.fillStyle = color;
  ctx.fillText(text, (cvs.width - w) / 2 + 10, 33);
  tex.needsUpdate = true;
}

// A camera "creature": a faceted gem body with small fin spikes (so it reads as alive, not a
// static jewel) in a ring cradle, always carrying its "CAM n" tag on its head (a billboard
// sprite -- Three.js Sprites always face the camera by construction, so the tag is legible
// from any angle without extra bookkeeping).
// Each camera model gets its own sea-creature silhouette -- purely procedural (primitives
// only, same "no borrowed 3D assets" rule as the rest of the scene). Every dangling part
// (tentacle/leg/arm) is built as a MESH with a FIXED local orientation nested inside its own
// PIVOT group; animation only ever rotates the pivot, never the mesh itself, so the part's
// base shape can never be fought or distorted by its own wave motion -- the same "separate the
// tracked transform from the cosmetic one" principle that fixed the spawn-in tween bug earlier
// in this file, generalized to every swaying part instead of just one wobble.
const MODEL_CREATURE = { kb: 'octopus', dsplus: 'squid', ds: 'squid', radtan: 'jellyfish',
                         ucm: 'hammerhead', eucm: 'hammerhead', ocam: 'starfish' };
function creatureKind(model) { return MODEL_CREATURE[model] || 'jellyfish'; }

function addRadialPivotPart(visual, n, ringR, pivotY, meshBuilder) {
  const parts = [];
  for (let i = 0; i < n; i++) {
    const ang = (i / n) * Math.PI * 2;
    const pivot = new THREE.Group();
    pivot.position.set(Math.cos(ang) * ringR, pivotY, Math.sin(ang) * ringR);
    pivot.add(meshBuilder(ang));
    visual.add(pivot);
    parts.push({ pivot, ang, i });
  }
  return parts;
}

function buildCreatureBody(kind, color) {
  const visual = new THREE.Group();
  const meshes = [];                            // every body mesh -- for uniform error-color tinting
  const mat = () => {
    const m = new THREE.MeshPhysicalMaterial({ color, metalness: 0.1, roughness: 0.18, transmission: 0.4,
                                               thickness: 0.35, emissive: color, emissiveIntensity: 0.32,
                                               flatShading: true });
    return m;
  };
  const track = (mesh) => { meshes.push(mesh); return mesh; };
  const anim = { dangle: [], bell: null };       // `dangle`: pivots animated with a wave-wag each frame

  if (kind === 'octopus') {
    visual.add(track(new THREE.Mesh(new THREE.SphereGeometry(0.13, 10, 8), mat())));
    anim.dangle = addRadialPivotPart(visual, 6, 0.08, -0.05, () => {
      const leg = track(new THREE.Mesh(new THREE.ConeGeometry(0.02, 0.17, 5), mat()));
      leg.rotation.x = Math.PI; leg.position.y = -0.08;
      return leg;
    });
  } else if (kind === 'squid') {
    const body = track(new THREE.Mesh(new THREE.ConeGeometry(0.09, 0.3, 8), mat()));
    body.position.y = 0.05; visual.add(body);
    anim.dangle = addRadialPivotPart(visual, 5, 0.04, -0.12, () => {
      const t = track(new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.016, 0.16, 4), mat()));
      t.rotation.x = Math.PI; t.position.y = -0.08;
      return t;
    });
    for (const s of [-1, 1]) {
      const fin = track(new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.1, 3), mat()));
      fin.rotation.z = s * Math.PI / 2.4; fin.position.set(s * 0.09, 0.14, 0);
      visual.add(fin);
    }
  } else if (kind === 'jellyfish') {
    const bell = track(new THREE.Mesh(new THREE.SphereGeometry(0.14, 12, 8, 0, Math.PI * 2, 0, Math.PI * 0.55), mat()));
    visual.add(bell);
    anim.bell = bell;
    anim.dangle = addRadialPivotPart(visual, 8, 0.08, -0.1, () => {
      const t = track(new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.009, 0.24, 3), mat()));
      t.rotation.x = Math.PI; t.position.y = -0.12;
      return t;
    });
  } else if (kind === 'hammerhead') {
    const body = track(new THREE.Mesh(new THREE.ConeGeometry(0.055, 0.32, 6), mat()));
    body.rotation.x = Math.PI / 2; visual.add(body);
    const hammer = track(new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.032, 0.05), mat()));
    hammer.position.z = 0.15; visual.add(hammer);
    const dorsal = track(new THREE.Mesh(new THREE.ConeGeometry(0.045, 0.08, 3), mat()));
    dorsal.position.y = 0.06; visual.add(dorsal);
    anim.dangle = addRadialPivotPart(visual, 1, 0, 0, () => {
      const tail = track(new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.1, 3), mat()));
      tail.rotation.x = -Math.PI / 2; tail.position.z = -0.19;
      return tail;
    });
  } else {                                       // starfish
    visual.add(track(new THREE.Mesh(new THREE.SphereGeometry(0.045, 8, 6), mat())));
    anim.dangle = addRadialPivotPart(visual, 5, 0, 0, (ang) => {
      const arm = track(new THREE.Mesh(new THREE.ConeGeometry(0.032, 0.16, 4), mat()));
      arm.rotation.x = Math.PI / 2; arm.position.z = 0.08;   // points outward along local +Z before pivot yaw
      return arm;
    });
    anim.dangle.forEach(p => { p.pivot.rotation.y = -p.ang; });   // orient each arm radially
  }
  const light = new THREE.PointLight(color, 1.3, 2.6, 2);
  visual.add(light);
  return { visual, meshes, anim, light };
}

function makeCameraOrb(color, model) {
  const group = new THREE.Group();
  const kind = creatureKind(model);
  const { visual, meshes, anim, light } = buildCreatureBody(kind, color);
  group.add(visual);
  const lbl = makeLabelSprite();
  lbl.sprite.position.y = 0.42;
  visual.add(lbl.sprite);                       // rides `visual`, not `group` -- always on the head regardless of strain wobble
  scene.add(group);
  return { group, visual, kind, meshes, anim, light, lbl, curPos: new THREE.Vector3(),
           phase: Math.random() * Math.PI * 2, spawned: false, finale: false, finalTargetY: null,
           surfaced: false, jump: null, jumpCooldown: 0, _lastLabel: '' };
}

// One real, un-scripted ballistic leap: a smoothstep ease up to `peakY` then either a
// symmetric fall back to `startY` (a preview leap -- any of the current top 3 can trigger
// these once the run is far enough along) or, for the true finale jump, an ease down to a
// victory-float height above the surface instead of falling back at all.
function startOrbJump(orb, isFinaleJump, finalY) {
  orb.jump = { t: 0, duration: isFinaleJump ? 2.2 : 0.9 + Math.random() * 0.3,
              startY: orb.curPos.y, peakY: isFinaleJump ? SURFACE_Y + 0.95 : SURFACE_Y + 0.15 + Math.random() * 0.15,
              finalY: finalY ?? null, x: orb.curPos.x, z: orb.curPos.z,
              isFinaleJump, splashedUp: false, splashedDown: false };
}
function updateOrbJump(orb, dt) {
  const j = orb.jump;
  j.t += dt;
  const k = Math.min(1, j.t / j.duration);
  let y;
  if (j.isFinaleJump) {
    if (k < 0.55) {
      y = THREE.MathUtils.lerp(j.startY, j.peakY, Math.sin((k / 0.55) * Math.PI / 2));
    } else {
      const kk = (k - 0.55) / 0.45;
      y = THREE.MathUtils.lerp(j.peakY, j.finalY, kk * kk * (3 - 2 * kk));
    }
  } else {
    y = j.startY + (j.peakY - j.startY) * (4 * k * (1 - k));   // parabola: launches and lands at startY
  }
  orb.curPos.set(j.x, y, j.z);
  orb.group.position.copy(orb.curPos);
  if (!j.splashedUp && y >= SURFACE_Y) { j.splashedUp = true; spawnSplash(orb.curPos.clone()); }
  if (j.splashedUp && !j.splashedDown && !j.isFinaleJump && y < SURFACE_Y && k > 0.5) {
    j.splashedDown = true; spawnSplash(orb.curPos.clone());
  }
  if (k >= 1) {
    if (j.isFinaleJump) { orb.finale = true; orb.finalTargetY = j.finalY; }
    orb.jump = null;
  }
}

// runConfidence ramps 0->1 over the run's real elapsed time -- everyone starts pinned down
// (struggling), and the *attainable* ceiling for both tiers loosens as the run goes on, so the
// scene reads as "straining upward, harder at first" rather than an instant snap to rank.
let errMin = Infinity, errMax = -Infinity, runConfidence = 0;
const _lookMat = new THREE.Matrix4();
const _lookTarget = new THREE.Vector3();
function updateOrbs(state, t, dt) {
  runConfidence = Math.min(1, runConfidence + dt * 0.02);
  const n = state.cameras.length || 1;
  const ranked = [...state.cameras].sort((a, b) => (a.err ?? 1e9) - (b.err ?? 1e9));
  const rankOf = new Map(ranked.map((c, i) => [c.id, i]));
  state.cameras.forEach(c => {
    if (c.err !== null) { errMin = Math.min(errMin, c.err); errMax = Math.max(errMax, c.err); }
  });
  const range = Math.max(errMax - errMin, 0.05);

  state.cameras.forEach((c, i) => {
    let orb = orbs.get(c.id);
    if (!orb) { orb = makeCameraOrb(modelColor(c.model), c.model); orbs.set(c.id, orb); }
    const ang = (i / n) * Math.PI * 2;
    const slotX = Math.cos(ang) * POND_R * 0.5;
    const slotZ = Math.sin(ang) * POND_R * 0.5;

    if (orb.realPose) {
      // post-finale "digital twin" reveal -- the camera's actual solved T_c_g pose, not the
      // pond's error-driven depth (see startRealGeometryReveal). Highest priority: once set,
      // this owns the orb until the page is closed. orb.group is now a child of rigRoot (which
      // the user can rotate via the gizmo), so Object3D.lookAt -- which expects a WORLD-space
      // target -- would be wrong here: curPos/fwd/up are all in rigRoot-LOCAL space. Build the
      // rotation directly from those local vectors with Matrix4.lookAt instead, which is a pure
      // relative computation with no world/parent context, so it stays correct under any rig
      // rotation the user dials in.
      orb.curPos.lerp(orb.realPose.pos, Math.min(1, dt * 0.8));
      orb.group.position.copy(orb.curPos);
      const upVec = orb.realPose.up.lengthSq() > 1e-6 ? orb.realPose.up : new THREE.Vector3(0, 1, 0);
      _lookMat.lookAt(orb.curPos, _lookTarget.copy(orb.curPos).add(orb.realPose.fwd), upVec);
      orb.group.quaternion.setFromRotationMatrix(_lookMat);
    } else if (orb.jump) {
      updateOrbJump(orb, dt);
    } else if (orb.finale) {
      const target = new THREE.Vector3(slotX, orb.finalTargetY, slotZ);
      orb.curPos.lerp(target, Math.min(1, dt * 0.9));
      orb.group.position.copy(orb.curPos);
    } else {
      const rank = rankOf.get(c.id);
      const isTop3 = rank < 3;
      const norm = c.err !== null ? THREE.MathUtils.clamp(1 - (c.err - errMin) / range, 0, 1) : 0.05;
      const desiredY = FLOOR_Y + 0.25 + norm * (SURFACE_Y - FLOOR_Y - 0.3);
      // rank -- not raw error -- decides who is even ALLOWED near the surface: 4th place and
      // below are held down regardless of how good their absolute error becomes.
      const ceilY = isTop3
        ? THREE.MathUtils.lerp(FLOOR_Y + 0.7, SURFACE_Y - 0.04, runConfidence)
        : THREE.MathUtils.lerp(FLOOR_Y + 0.15, FLOOR_Y + 1.0, runConfidence);
      const targetY = Math.min(desiredY, ceilY);
      const target = new THREE.Vector3(slotX, targetY, slotZ);
      if (!orb.spawned) { orb.curPos.copy(target); orb.spawned = true; }
      orb.curPos.lerp(target, Math.min(1, dt * 1.3));
      orb.group.position.copy(orb.curPos);

      // periodic preview leap -- only ever the current top 3, only once the ceiling is close
      // enough to the surface to make a breach plausible, each with its own cooldown so
      // several can compete to breach at different moments
      orb.jumpCooldown = Math.max(0, orb.jumpCooldown - dt);
      if (isTop3 && runConfidence > 0.45 && orb.jumpCooldown <= 0 && ceilY - orb.curPos.y < 0.25
          && Math.random() < dt * 0.2) {
        startOrbJump(orb, false);
        orb.jumpCooldown = 3 + Math.random() * 3;
      }
      // "straining against the ceiling" -- a small idle bob always, a stronger pulse when
      // actually pressed up against its current ceiling
      const pressing = Math.max(0, 1 - (ceilY - orb.curPos.y) / 0.2);
      orb.visual.position.y = Math.sin(t * 2.4 + orb.phase) * (0.012 + pressing * 0.03);
    }
    if (!orb.realPose) orb.group.rotation.y += dt * 0.6;   // gentle spin -- but real-pose orientation must be stable, not fought
    const col = errColor(c.err);
    orb.light.color = col;
    for (const m of orb.meshes) { m.material.color = col; m.material.emissive = col; }
    // dangling parts (tentacles/legs/arms) wave via their pivots only -- the mesh itself never
    // rotates, so this can never distort the creature's base silhouette (see buildCreatureBody)
    for (const p of orb.anim.dangle) {
      p.pivot.rotation.x = Math.sin(t * 2.6 + p.i * 0.9 + orb.phase) * 0.3;
      p.pivot.rotation.z = Math.cos(t * 2.2 + p.i * 0.9 + orb.phase) * 0.18;
    }
    if (orb.anim.bell) { orb.anim.bell.scale.y = 1 + Math.sin(t * 2.2 + orb.phase) * 0.16; }
    const label = `CAM ${c.id}` + (c.err !== null ? `  ${c.err.toFixed(2)}px` : '');
    if (orb._lastLabel !== label) { drawLabel(orb.lbl, label, '#e6e8ee'); orb._lastLabel = label; }
  });
}

// ---------------------------------------------------------------------------------------
// director (cinematic autopilot camera) -- orbits the fixed pond center; hard cuts to a
// fresh angle on each act change, same as before, just re-scaled to the pond's fixed geometry
// instead of the rig's real-world extent (no longer meaningful now that orb position encodes
// error, not real pose).
// ---------------------------------------------------------------------------------------
let manual = false;
let az = 0.6, el = 0.22, dist = 4.4;
let lastStage = null;
let cutPending = false, shakeAmount = 0;
let finaleFocusId = null;
const clock = new THREE.Clock();

const ACTS = [
  { re: /^\(0\)/, kicker: 'ACT 0', title: 'INTO THE DEEP', sub: 'Every camera starts at the bottom of the pond.', azCut: 0.3, distMul: 1.15 },
  { re: /^\(a\)/, kicker: 'ACT I', title: 'FINDING THE BOTTOM', sub: 'Rough poses settle out of raw detections.', azCut: 1.7, distMul: 1.0 },
  { re: /^\(b\)/, kicker: 'ACT II', title: 'THE CURRENTS ALIGN', sub: 'Camera extrinsics come into agreement with each other.', azCut: 3.0, distMul: 0.92 },
  { re: /^\(c\)/, kicker: 'ACT III', title: 'RISING TOGETHER', sub: 'Every camera climbs as its own error falls.', azCut: 4.4, distMul: 0.8 },
  { re: /^\(d\)/, kicker: 'ACT IV', title: 'THE FINAL ASCENT', sub: 'Structure refinement rounds polish the last details.', azCut: 5.6, distMul: 0.85 },
  { re: /^\(save\)/, kicker: 'ACT V', title: 'DRAWING WATER', sub: 'Reprojection and detection overlays saved to disk.', azCut: 2.2, distMul: 1.1 },
];
function actFor(stage) {
  for (const a of ACTS) if (a.re.test(stage || '')) return a;
  return null;
}
let curDistMul = 1.0;

function showCard(kicker, title, sub, holdMs) {
  const card = document.getElementById('card');
  document.getElementById('cardKicker').textContent = kicker;
  document.getElementById('cardTitle').textContent = title;
  document.getElementById('cardSub').textContent = sub || '';
  card.style.opacity = 1;
  document.getElementById('lbTop').classList.add('on');
  document.getElementById('lbBottom').classList.add('on');
  clearTimeout(showCard._t);
  showCard._t = setTimeout(() => {
    card.style.opacity = 0;
    document.getElementById('lbTop').classList.remove('on');
    document.getElementById('lbBottom').classList.remove('on');
  }, holdMs || 2600);
}
function flashKick(color, opacity) {
  const el = document.getElementById('flash');
  el.style.background = color;
  el.style.opacity = opacity;
  requestAnimationFrame(() => requestAnimationFrame(() => { el.style.opacity = 0; }));
}
function callout(text, color, holdMs) {
  const el = document.getElementById('callout');
  el.textContent = text; el.style.color = color;
  el.style.opacity = 1; el.style.transform = 'translate(-50%, -50%) scale(1)';
  clearTimeout(callout._t);
  callout._t = setTimeout(() => {
    el.style.opacity = 0; el.style.transform = 'translate(-50%, -50%) scale(.85)';
  }, holdMs || 1200);
}

let revealActive = false, revealFocus = null;
function updateCamera(t, dt) {
  // once the digital-twin reveal starts, frame ITS centroid, not the pond -- and look down
  // more steeply so the water (which the reveal floats well above) recedes toward the bottom
  // of frame instead of competing with it, per feedback that the rig should read as centered
  // with the water "pushed down", not the other way around.
  const focus = revealActive && revealFocus ? revealFocus
              : (finaleFocusId !== null && orbs.get(finaleFocusId) ? orbs.get(finaleFocusId).curPos : POND_CENTER);
  const elBias = revealActive ? 0.5 : 0;
  const distBias = revealActive ? 1.3 : 1.0;
  const d = Math.max(dist * curDistMul * distBias, 0.6);
  shakeAmount = Math.max(0, shakeAmount - dt * 2.2);
  gradePass.uniforms.uShake.value = shakeAmount;
  const jitterAz = shakeAmount * Math.sin(t * 47) * 0.02;
  if (!manual) {
    if (!rigOrientMode) az += dt * 0.08;    // hold the view steady while the user drags the gizmo
    const el2 = el + elBias + Math.sin(t * 0.12) * 0.1;
    const eye = new THREE.Vector3(
      focus.x + d * Math.cos(el2) * Math.sin(az + jitterAz),
      focus.y + d * Math.sin(el2),
      focus.z + d * Math.cos(el2) * Math.cos(az + jitterAz));
    if (cutPending) { camera.position.copy(eye); cutPending = false; }
    else camera.position.lerp(eye, Math.min(1, dt * 1.3));
    camera.lookAt(focus.x, focus.y, focus.z);
  } else {
    controls.target.copy(focus);
    controls.update();
  }
}

// ---------------------------------------------------------------------------------------
// reactive drama -- driven by the REAL rms delta between ticks, not scripted.
// ---------------------------------------------------------------------------------------
let prevRms = null, lastEventAt = -999;
function checkReactiveEvent(state) {
  if (state.status !== 'running' || state.rms === null) return;
  const t = clock.elapsedTime;
  if (prevRms !== null && t - lastEventAt > 3.0) {
    const rel = (state.rms - prevRms) / Math.max(prevRms, 0.05);
    if (rel < -0.12) {
      lastEventAt = t;
      flashKick('#4ade80', 0.14);
      callout('RISING', '#4ade80');
    } else if (rel > 0.15) {
      lastEventAt = t;
      flashKick('#f87171', 0.18);
      callout('SINKING BACK…', '#f87171');
      shakeAmount = 0.6;
    }
  }
  prevRms = state.rms;
}

// ---------------------------------------------------------------------------------------
// state polling + scene update
// ---------------------------------------------------------------------------------------
let latest = null, finished = false;

function updatePreflight(state) {
  const el = document.getElementById('preflight');
  if (state.status !== 'detecting') { el.style.display = 'none'; return; }
  el.style.display = 'flex';
  const detect = state.detect || {};
  const camIds = Object.keys(detect).map(Number).sort((a, b) => a - b);
  document.getElementById('preflightRows').innerHTML = camIds.map(c => {
    const { i, n } = detect[c];
    const pct = n > 0 ? Math.round((i / n) * 100) : 0;
    return `<div class="cam-row"><span class="lbl">cam ${c}</span>` +
      `<span class="bar"><div style="width:${pct}%"></div></span>` +
      `<span class="n">${i}/${n}</span></div>`;
  }).join('');
}

function updateVitals(state) {
  document.getElementById('rmsnum').innerHTML = state.rms !== null ? `${state.rms.toFixed(3)} <small>px rms</small>` : '-- <small>px rms</small>';
  let stageText = state.stage ? `▸ ${state.stage}` : '';
  if (state.save) {
    const totals = Object.values(state.save).reduce((a, s) => [a[0] + s.i, a[1] + s.n], [0, 0]);
    stageText += `  (${totals[0]}/${totals[1]} images)`;
  }
  document.getElementById('stagelabel').textContent = stageText;
  const cv = document.getElementById('spark'), ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  const h = state.history || [];
  if (h.length > 1) {
    const max = Math.max(...h), min = Math.min(...h);
    ctx.beginPath();
    h.forEach((v, i) => {
      const x = (i / (h.length - 1)) * cv.width;
      const y = cv.height - ((v - min) / (max - min || 1)) * cv.height;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    const cur = h[h.length - 1];
    ctx.strokeStyle = cur < 1.0 ? '#4ade80' : cur < 3.0 ? '#facc15' : '#f87171';
    ctx.lineWidth = 2; ctx.stroke();
  }
}

function updateLeaderboard(state) {
  const rows = [...state.cameras].sort((a, b) => (a.err ?? 1e9) - (b.err ?? 1e9));
  const medals = ['🥇', '🥈', '🥉'];
  document.getElementById('rows').innerHTML = rows.map((c, i) => {
    const badge = (c.err !== null && c.err < 0.5) ? '<span class="badge">🎯</span>' : '';
    return `<div class="row"><span class="rank">${medals[i] || i + 1}</span>` +
      `<span class="cam">cam ${c.id} · ${c.model}</span>` +
      `<span class="err ${errClass(c.err)}">${c.err !== null && c.err !== undefined ? c.err.toFixed(2) : '--'}px</span>${badge}</div>`;
  }).join('');
}

let introDone = false;
function maybeIntro(state) {
  if (introDone) return;
  introDone = true;
  showCard('MISSION', 'INTO THE POND', `${state.cameras.length} camera(s) begin at the bottom -- watch them rise as they converge.`, 3400);
}

function onState(state) {
  latest = state;
  updatePreflight(state);
  if (state.status === 'detecting') {
    document.getElementById('status').textContent = 'detecting corners...';
    return;
  }
  document.getElementById('status').textContent = state.status === 'finished' ? 'converged' : `optimizing · step ${state.step}`;
  maybeIntro(state);
  const act = actFor(state.stage);
  if (state.stage !== lastStage) {
    lastStage = state.stage;
    if (act) {
      showCard(act.kicker, act.title, act.sub);
      curDistMul = act.distMul;
      az = act.azCut; cutPending = true;
    }
  }
  updateVitals(state);
  updateLeaderboard(state);
  checkReactiveEvent(state);
  if (state.status === 'finished' && !finished) {
    finished = true;
    onFinale(state);
  }
}

function fmtNum(v) { return typeof v === 'number' ? v.toFixed(v > 100 ? 0 : 3) : '--'; }

function spawnFireworks(center) {
  const palette = [0x4ade80, 0xfacc15, 0x38bdf8, 0xf472b6, 0xf87171, 0xffffff];
  for (let i = 0; i < 6; i++) {
    setTimeout(() => {
      const pos = center.clone().add(new THREE.Vector3((Math.random() - 0.5) * 1.4, 0.3 + Math.random() * 0.7,
                                                        (Math.random() - 0.5) * 1.4));
      spawnBurst(pos, { n: 130, colors: palette, speed: 1.5, up: true, size: 0.05, life: 1.7, gravity: -0.7 });
    }, i * 320);
  }
}

// ---------------------------------------------------------------------------------------
// post-finale "digital twin" reveal -- after the pond drama settles, every camera creature
// swims to its ACTUAL solved T_c_g pose (not the pond's error-driven depth) and floats above
// the water, while the board plays back through every real captured frame's solved pose and
// corner points, start to end, on a loop. This is the same real geometry the static HTML
// report shows (final._final_payload reuses report._frame_payload verbatim) -- the pond is
// the fun wait, this is the real proof of what was actually solved.
// ---------------------------------------------------------------------------------------
function computeRealGeometryTransform(cams, frames) {
  const pts = [];
  for (const c of cams) if (c.pos) pts.push(c.pos);
  for (const f of frames) for (const p of f.points) pts.push([p[0], p[1], p[2]]);
  if (!pts.length) return null;
  // Y is mapped from `minY` (the single lowest real point), NOT the centroid: a centroid-plus-
  // fixed-offset mapping (what this used to do) lets any point below the centroid's own Y end
  // up mapped below the water surface whenever the real vertical spread exceeds that fixed
  // offset -- measured, not theoretical: real rig data routinely has cameras well below the
  // board's own Y (looking up at it), which is exactly the "some cameras still inside water"
  // bug. Anchoring to minY instead makes "the single lowest point maps to just above the
  // surface" a hard guarantee, so nothing can ever end up underwater regardless of the real
  // data's vertical distribution.
  let cx = 0, cz = 0, minY = Infinity;
  for (const p of pts) { cx += p[0]; cz += p[2]; minY = Math.min(minY, p[1]); }
  cx /= pts.length; cz /= pts.length;
  let maxR = 0.001;
  for (const p of pts) maxR = Math.max(maxR, Math.hypot(p[0] - cx, p[2] - cz));
  let ySpread = 0.001;
  for (const p of pts) ySpread = Math.max(ySpread, p[1] - minY);
  const scale = (POND_R * 0.85) / Math.max(maxR, ySpread * 0.6);
  return { cx, cz, minY, scale };
}
// Local to rigRoot -- NOT world space. rigRoot itself carries the one world anchor (fixed,
// at the pond's center/surface) plus whatever rotation the user dials in with the gizmo; every
// camera and board point below is a child of rigRoot expressed relative to that, so rotating
// rigRoot rigidly rotates the whole reveal together, exactly preserving their real relative
// geometry (rotating a rigid body doesn't change distances/angles within it).
function toRigLocalSpace(xform, p) {
  return new THREE.Vector3((p[0] - xform.cx) * xform.scale,
                           0.55 + (p[1] - xform.minY) * xform.scale,
                           (p[2] - xform.cz) * xform.scale);
}

let boardReplay = null;
function updateBoardReplay(dt) {
  if (!boardReplay) return;
  boardReplay.timer += dt;
  if (boardReplay.timer > 0.7) {
    boardReplay.timer = 0;
    boardReplay.idx = (boardReplay.idx + 1) % boardReplay.frames.length;
  }
  const pts = boardReplay.frames[boardReplay.idx].points;
  while (boardReplay.meshes.length < pts.length) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(1, 8, 8),
      new THREE.MeshStandardMaterial({ emissive: 0x222222, emissiveIntensity: 0.75, roughness: 0.3 }));
    rigRoot.add(m);                         // child of rigRoot -- rotates rigidly with the cameras
    boardReplay.meshes.push(m);
  }
  for (let i = 0; i < boardReplay.meshes.length; i++) boardReplay.meshes[i].visible = i < pts.length;
  pts.forEach((p, i) => {
    const m = boardReplay.meshes[i];
    m.position.copy(toRigLocalSpace(boardReplay.xform, p));
    m.scale.setScalar(0.028);
    const c = errColor(p[3]);
    m.material.color = c; m.material.emissive = c;
  });
}

function startRealGeometryReveal(f) {
  if (!f) return;
  const frames = f.frames || [];
  const xform = computeRealGeometryTransform(f.cameras, frames);
  if (!xform) return;
  rigRoot.position.set(POND_CENTER.x, SURFACE_Y, POND_CENTER.z);
  rigRoot.quaternion.identity();
  f.cameras.forEach(c => {
    if (!c.pos) return;
    const orb = orbs.get(c.id);
    if (!orb) return;
    // reparent world-position-preserving: convert the orb's CURRENT world position into
    // rigRoot-local space before switching parents, so it doesn't visually pop when the
    // realPose branch (see updateOrbs) starts lerping it toward its new local-space target.
    const worldPos = orb.group.getWorldPosition(new THREE.Vector3());
    rigRoot.add(orb.group);
    const startLocal = rigRoot.worldToLocal(worldPos);
    orb.curPos.copy(startLocal);
    orb.group.position.copy(startLocal);
    orb.realPose = {
      pos: toRigLocalSpace(xform, c.pos),
      fwd: new THREE.Vector3(c.ax_z[0], c.ax_z[1], c.ax_z[2]),
      up: new THREE.Vector3(c.ax_y[0], c.ax_y[1], c.ax_y[2]).multiplyScalar(-1),
    };
  });
  rigGizmo.attach(rigRoot);
  document.getElementById('orientBtn').style.display = 'inline-block';
  revealFocus = rigRoot.position.clone();   // a fixed world anchor -- doesn't drift as the user rotates the rig
  revealActive = true;                      // director camera now frames the reveal, not the pond
  if (frames.length) boardReplay = { frames, xform, idx: 0, timer: 0, meshes: [] };
  showCard('THE DIGITAL TWIN', 'WHAT WAS ACTUALLY SOLVED', 'Real poses, real corners, every captured frame.', 3400);
}

function onFinale(state) {
  const f = state.final;
  const mvpId = f ? f.mvp : null;
  showCard('THE POND', 'RECKONING', 'One camera breaks the surface. The rest return to the bottom.', 3200);
  orbs.forEach((orb, id) => {
    if (id === mvpId) {
      startOrbJump(orb, true, SURFACE_Y + 0.4);       // the real ballistic finale leap, not a lerp
    } else {
      orb.finale = true;
      orb.finalTargetY = FLOOR_Y + 0.12;
    }
  });
  finaleFocusId = mvpId;
  setTimeout(() => {
    // celebration and the digital-twin reveal start together, right when the pond drama
    // settles. #board/#vitals track the LIVE optimization (leaderboard, rms sparkline) --
    // once finished, that data is frozen and no longer the focus, and #finale's own
    // right-edge position (top:50%, vertically centered) can span up into #board's
    // top-right corner on a tall/narrow viewport, so explicitly hiding them here is what
    // actually keeps #finale from overlapping the leaderboard, not just #finale's own CSS.
    // The stats side panel (a persistent element, unlike the transient title cards) only
    // fades in a beat later so the fireworks read as the moment, not as backdrop to a table.
    document.getElementById('vitals').style.opacity = 0;
    document.getElementById('vitals').style.pointerEvents = 'none';
    document.getElementById('board').style.opacity = 0;
    document.getElementById('board').style.pointerEvents = 'none';
    showCard('MISSION', 'COMPLETE', '', 2400);
    spawnFireworks(new THREE.Vector3(POND_CENTER.x, SURFACE_Y + 0.3, POND_CENTER.z));
    startRealGeometryReveal(f);
    setTimeout(() => showStatsPanel(f), 1600);
  }, 4200);
}

function showStatsPanel(f) {
  if (f) {
    const o = f.overall;
    const lvl = (o.median ?? Infinity) <= 1.0 ? '#4ade80' : (o.median ?? Infinity) <= 3.0 ? '#facc15' : '#f87171';
    document.getElementById('finaleStats').innerHTML = [
      ['median px', o.median], ['mean px', o.mean], ['p95 px', o.p95], ['n obs', o.n],
    ].map(([k, v]) => `<div><div class="v" style="color:${k.includes('px') ? lvl : '#e6e8ee'}">${fmtNum(v)}</div><div class="k">${k}</div></div>`).join('');
    document.getElementById('finaleCredits').innerHTML = f.cameras.map(c =>
      `<div><span>${c.id === f.mvp ? '🏆 ' : ''}cam ${c.id} · ${c.model}</span><span>${fmtNum(c.median)}px median</span></div>`).join('');
  }
  document.getElementById('finale').style.opacity = 1;
  document.getElementById('finale').style.pointerEvents = 'auto';
}

async function poll() {
  try {
    const res = await fetch('state.json', { cache: 'no-store' });
    if (res.ok) onState(await res.json());
  } catch (e) { /* server may be gone after the run exits -- last state already shown */ }
  if (!finished) setTimeout(poll, 150);
}
initFish();
poll();

document.getElementById('modebtn').addEventListener('click', () => {
  manual = !manual;
  controls.enabled = manual;
  document.getElementById('modebtn').textContent = manual ? 'return to director mode' : 'take manual control';
  document.getElementById('hint').textContent = manual ? 'manual: drag to orbit, wheel to zoom' : 'director mode: watching the calibration';
});

document.getElementById('orientBtn').addEventListener('click', () => {
  rigOrientMode = !rigOrientMode;
  rigGizmo.enabled = rigOrientMode;
  rigGizmo.visible = rigOrientMode;
  document.getElementById('orientBtn').textContent = rigOrientMode ? 'done orienting' : 'orient rig';
  if (rigOrientMode) document.getElementById('hint').textContent = 'orient rig: drag the gizmo rings to match your real setup\'s up direction';
  else document.getElementById('hint').textContent = manual ? 'manual: drag to orbit, wheel to zoom' : 'director mode: watching the calibration';
});

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.1);
  const t = clock.elapsedTime;
  gradePass.uniforms.uTime.value = t;
  waterMat.uniforms.uTime.value = t;
  causticMat.uniforms.uTime.value = t;
  if (latest && latest.status !== 'detecting') updateOrbs(latest, t, dt);
  updateFish(t, dt);
  updatePredators(t);
  updateCamera(t, dt);
  updateParticleBursts(dt);
  updateBoardReplay(dt);
  composer.render();
}
animate();
</script>
</body>
</html>
"""
