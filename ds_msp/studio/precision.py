"""How far a finished calibration is from the precision someone needs, and how many more photos
of the same kind that would take.

The comparison uses the values the calibration's own uncertainty gate compares: the camera-pair
uncertainty widened for the number of frames (``*_sigma_t``). The photo count assumes the
uncertainty shrinks with the square root of the number of frames. That holds for more photos
like the ones already taken; better placed photos can need far fewer.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

#: Most pairs returned, worst first; the counts always cover every pair.
MAX_ROWS = 12


def _t_factor(frames: int) -> float:
    from scipy import stats

    return float(stats.t.ppf(0.975, frames - 1) / stats.norm.ppf(0.975))


def frames_for(sigma: float, frames: int, limit: float) -> int:
    """The fewest frames, at least ``frames``, for which ``sigma`` (measured from ``frames``
    frames, not yet widened) meets ``limit`` once widened as the gate widens it."""
    need = max(frames, math.ceil(frames * (sigma / limit) ** 2))
    while sigma * math.sqrt(frames / need) * _t_factor(need) > limit:
        need += max(1, need // 200)
    return need


def _ratio(value: Optional[float], limit: Optional[float]) -> Optional[float]:
    return None if value is None or limit is None else value / limit


def precision_gap(unc: Optional[Mapping[str, Any]], max_rotation: Optional[float] = None,
                  max_centre: Optional[float] = None) -> Dict[str, Any]:
    """The gap between a result's ``extrinsic_uncertainty`` and the limits.

    A limit left as ``None`` falls back to the one the run was given. Returns ``ok``; ``limits``
    and whether they came ``from_run``; ``frames``; ``n_pairs``, ``n_over`` and
    ``n_undetermined``; ``rows``, the worst pairs first; ``worst_rotation`` and ``worst_centre``;
    and ``frames_needed``, the estimate, which is ``None`` when every pair is within the limits
    or a pair cannot be estimated."""
    pairs = (unc or {}).get("pairs") or {}
    if not pairs:
        return {"ok": False, "reason": "This result has no camera-pair uncertainty."}
    frames = int((unc or {}).get("n_frame_clusters") or 0)
    run = (unc or {}).get("limits") or {}
    given = max_rotation is not None or max_centre is not None
    rot = max_rotation if given else run.get("max_rotation_sigma_deg")
    ctr = max_centre if given else run.get("max_centre_sigma")
    out: Dict[str, Any] = {"ok": True, "frames": frames, "from_run": not given and (rot is not None or ctr is not None),
                           "limits": {"rotation": rot, "centre": ctr}, "n_pairs": len(pairs)}
    rows: List[Dict[str, Any]] = []
    need = 0
    undetermined = 0
    for key, p in pairs.items():
        r_t, c_t = p.get("rotation_sigma_deg_t"), p.get("centre_sigma_t")
        known = p.get("identifiable") is not False and r_t is not None and c_t is not None
        undetermined += not known
        ratios = [x for x in (_ratio(r_t, rot), _ratio(c_t, ctr)) if x is not None]
        over = (rot is not None or ctr is not None) and (not known or any(x > 1 for x in ratios))
        rows.append({"pair": key, "rotation": r_t, "centre": c_t, "rotation_ratio": _ratio(r_t, rot),
                     "centre_ratio": _ratio(c_t, ctr), "determined": known, "over": over,
                     "worst_ratio": max(ratios) if known and ratios else None})
        if known and frames >= 2:
            for sigma, limit, widened in ((p.get("rotation_sigma_deg"), rot, r_t), (p.get("centre_sigma"), ctr, c_t)):
                if limit is not None and limit > 0 and widened > limit:
                    need = max(need, frames_for(float(sigma), frames, float(limit)))
    rows.sort(key=lambda r: (r["determined"], -(r["worst_ratio"] or 0), -(r["rotation"] or 0)))
    out["n_over"] = sum(1 for r in rows if r["over"])
    out["n_undetermined"] = undetermined
    out["rows"] = rows[:MAX_ROWS]
    measured = [r for r in rows if r["determined"]]
    out["worst_rotation"] = max(measured, key=lambda r: r["rotation"]) if measured else None
    out["worst_centre"] = max(measured, key=lambda r: r["centre"]) if measured else None
    out["frames_needed"] = need if need and not undetermined else None
    out["times"] = (need / frames) if need and frames and not undetermined else None
    return out
