"""A machine-readable progress log of a rig calibration (``--progress-file``).

:class:`ProgressLog` sits in front of the live view (or of nothing) and appends one JSON
object per line to a file as the calibration moves on, so another program can follow a run
without parsing terminal output. It only listens: every call is passed on unchanged and the
solve does not depend on it.

Each line has ``t`` (seconds since the log was opened) and ``event``:

* ``start`` -- the log was opened.
* ``detect`` -- corner detection: ``cameras`` maps a camera id to ``[images done, images]``.
* ``scene`` -- detection is finished: ``cameras`` and ``observations`` counts.
* ``stage`` -- a calibration stage began: ``name`` (the same labels the live view shows) and
  ``index`` (1 for the first stage).
* ``iter`` -- solver progress inside a stage: ``it``, ``max_iter``, ``rms`` (px), ``cost``.
* ``save`` -- overlay images being written: ``cameras`` as for ``detect``.
* ``done`` -- the calibration finished: ``rms`` (px, the worst camera's).

``detect``, ``iter`` and ``save`` are written at most every ``min_interval`` seconds, plus
the last one of each run of them.
"""
from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, Optional, Tuple


class ProgressLog:
    """Progress receiver that logs to ``path`` and forwards to ``inner`` (may be ``None``).

    It offers the hooks the rig pipeline calls on its ``on_iter`` object: the call itself,
    ``set_stage``, ``bind_scene``, ``detect_progress``, ``save_progress`` and ``finish``."""

    def __init__(self, path: str, inner: Any = None, *, min_interval: float = 0.5):
        self._inner = inner
        self._min_interval = float(min_interval)
        self._t0 = time.monotonic()
        self._last: Dict[str, float] = {}
        self._detect: Dict[int, Tuple[int, int]] = {}
        self._save: Dict[int, Tuple[int, int]] = {}
        self._stage_index = 0
        self._file = open(path, "w", encoding="utf-8")       # noqa: SIM115 - lives with the run
        self._emit("start")

    def __getattr__(self, name: str) -> Any:                 # anything else the view offers
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def _emit(self, event: str, **fields: Any) -> None:
        if self._file.closed:
            return
        rec = {"t": round(time.monotonic() - self._t0, 3), "event": event, **fields}
        self._file.write(json.dumps(rec, allow_nan=False) + "\n")
        self._file.flush()

    def _due(self, kind: str, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now - self._last.get(kind, -math.inf) < self._min_interval:
            return False
        self._last[kind] = now
        return True

    @staticmethod
    def _counts(table: Dict[int, Tuple[int, int]]) -> Dict[str, list]:
        return {str(c): [int(i), int(n)] for c, (i, n) in sorted(table.items())}

    # -- hooks ---------------------------------------------------------------------------
    def detect_progress(self, cam_id: int, i: int, n: int, path: str) -> None:
        self._detect[int(cam_id)] = (int(i), int(n))
        if self._due("detect", force=i == n):
            self._emit("detect", cameras=self._counts(self._detect))
        if self._inner is not None:
            self._inner.detect_progress(cam_id, i, n, path)

    def bind_scene(self, obj: Any, object_obs: Any) -> None:
        self._emit("scene", cameras=len({o.cam_id for o in object_obs}),
                   observations=len(object_obs))
        if self._inner is not None:
            self._inner.bind_scene(obj, object_obs)

    def set_stage(self, name: str) -> None:
        self._stage_index += 1
        self._emit("stage", name=str(name), index=self._stage_index)
        if self._inner is not None and hasattr(self._inner, "set_stage"):
            self._inner.set_stage(name)

    def save_progress(self, cam_id: int, i: int, n: int, frame_id: int) -> None:
        self._save[int(cam_id)] = (int(i), int(n))
        if self._due("save", force=i == n):
            self._emit("save", cameras=self._counts(self._save))
        if self._inner is not None and hasattr(self._inner, "save_progress"):
            self._inner.save_progress(cam_id, i, n, frame_id)

    def __call__(self, it: int, max_iter: int, rms: float, cost: float, rig: Any) -> None:
        if self._due("iter"):
            self._emit("iter", it=int(it), max_iter=int(max_iter), rms=_num(rms),
                       cost=_num(cost))
        if self._inner is not None:
            self._inner(it, max_iter, rms, cost, rig)

    def finish(self, final_rig: Any = None, **kw: Any) -> None:
        self._emit("done", rms=_num(kw.get("rms")))
        self._file.close()
        if self._inner is not None:
            self._inner.finish(final_rig, **kw)


def _num(x: Optional[float]) -> Optional[float]:
    return float(x) if x is not None and math.isfinite(float(x)) else None
