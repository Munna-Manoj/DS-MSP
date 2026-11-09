"""One-line answers for Studio's evidence views, computed from the FR-RIG-031 diagnostics.

Each view leads with an answer, not a plot the user has to read:

* :func:`error_pattern` -- "is there a pattern in the errors?" Residual size against distance
  from the image centre (8 rings), plus a 6 x 4 grid of mean residuals with a bias flag per
  region. Errors that grow toward the edges point at a lens model that is too stiff; a region
  whose errors lean one way points at detector bias or a model misfit there.
* :func:`lens_zones` -- "where can I trust this lens?" Each cell of the 24 x 18 lens sigma grid
  is *trusted* (the board was seen there and sigma is within the limit), *extrapolated* (no
  corner was ever detected there: the value is the model's guess, which a lean model
  under-reports) or *uncertain* (sigma over the limit).

The thresholds below are display rules for a first reading, not validated acceptance limits;
they never change a verdict.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

RINGS = 8
GRID = (6, 4)
#: An outer ring whose median error exceeds the centre's by this factor reads "grows toward edges".
EDGE_GROWTH = 1.5
#: A region is "biased" when its mean residual exceeds this many standard errors and this
#: share of the residual RMS (both needed: many corners make tiny biases significant).
BIAS_SE = 4.0
BIAS_FRAC = 0.3
MIN_PER_BIN = 20


def _where(x: float, y: float, w: float, h: float) -> str:
    """Plain words for a point's place in the image."""
    col = "left" if x < w / 3 else "right" if x > 2 * w / 3 else ""
    row = "top" if y < h / 3 else "bottom" if y > 2 * h / 3 else ""
    if row and col:
        return f"{row}-{col} corner"
    if row or col:
        return f"{row or col} edge"
    return "centre"


_AREAS = {(0, 0): "top-left corner", (1, 0): "top edge", (2, 0): "top-right corner",
          (0, 1): "left edge", (1, 1): "centre", (2, 1): "right edge",
          (0, 2): "bottom-left corner", (1, 2): "bottom edge", (2, 2): "bottom-right corner"}


def _areas(mask, u, v, w: float, h: float, base=None) -> str:
    """Plain words for where ``mask`` lies: the 3 x 3 image areas in which it holds most of the
    cells (of ``base``, default all), worst first, at most three."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    mask = np.asarray(mask, bool)
    base = np.ones(len(u), bool) if base is None else np.asarray(base, bool)
    ai = np.clip((u / w * 3).astype(int), 0, 2)
    aj = np.clip((v / h * 3).astype(int), 0, 2)
    hits = []
    for (i, j), name in _AREAS.items():
        m = base & (ai == i) & (aj == j)
        if m.any():
            share = float(mask[m].mean())
            if share >= 0.5:
                hits.append((share, name))
    hits.sort(reverse=True)
    names = [n for _s, n in hits[:3]]
    if not names:
        return "scattered across the image"
    return "the " + (names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1])


def error_pattern(u, v, dx, dy, w: float, h: float) -> Optional[Dict[str, Any]]:
    """Answer and charts for the valid residuals of one camera (``u, v`` detections, ``dx, dy``
    prediction minus detection, px)."""
    u, v, dx, dy = (np.asarray(a, float) for a in (u, v, dx, dy))
    if len(u) < MIN_PER_BIN:
        return None
    r = np.hypot(dx, dy)
    cx, cy = w / 2.0, h / 2.0
    rho = np.hypot(u - cx, v - cy) / np.hypot(cx, cy)
    edges = np.linspace(0.0, 1.0, RINGS + 1)
    rings: List[Dict[str, Any]] = []
    for k in range(RINGS):
        m = (rho >= edges[k]) & (rho < edges[k + 1] if k < RINGS - 1 else rho <= 1.0 + 1e-9)
        n = int(m.sum())
        rings.append({"from": float(edges[k]), "to": float(edges[k + 1]), "n": n,
                      "median_px": float(np.median(r[m])) if n >= MIN_PER_BIN else None,
                      "p90_px": float(np.percentile(r[m], 90)) if n >= MIN_PER_BIN else None})
    filled = [x for x in rings if x["median_px"] is not None]
    inner, outer = (filled[0], filled[-1]) if len(filled) >= 2 else (None, None)
    growth = (outer["median_px"] / inner["median_px"]) if inner and inner["median_px"] > 0 else None

    nx, ny = GRID
    ix = np.clip((u / w * nx).astype(int), 0, nx - 1)
    iy = np.clip((v / h * ny).astype(int), 0, ny - 1)
    rms_all = float(np.sqrt(np.mean(r ** 2)))
    cells = []
    for j in range(ny):
        for i in range(nx):
            m = (ix == i) & (iy == j)
            n = int(m.sum())
            if n < MIN_PER_BIN:
                cells.append({"i": i, "j": j, "n": n, "mx": None, "my": None, "biased": False})
                continue
            mx, my = float(dx[m].mean()), float(dy[m].mean())
            rms = float(np.sqrt(np.mean(r[m] ** 2)))
            se = rms / np.sqrt(n)
            mag = float(np.hypot(mx, my))
            cells.append({"i": i, "j": j, "n": n, "mx": mx, "my": my,
                          "biased": bool(mag > BIAS_SE * se and mag > BIAS_FRAC * rms)})
    biased = [c for c in cells if c["biased"]]

    if growth is not None and growth > EDGE_GROWTH:
        kind = "edges"
        answer = (f"Errors grow toward the edges: {growth:.1f}× larger at the rim than at the "
                  f"centre. The lens model may be too stiff for this lens.")
    elif biased:
        c = max(biased, key=lambda c: c["n"])
        where = _where((c["i"] + 0.5) * w / nx, (c["j"] + 0.5) * h / ny, w, h)
        kind = "regional"
        answer = (f"Errors lean one way in {len(biased)} of {sum(1 for x in cells if x['mx'] is not None)} "
                  f"regions, most at the {where}. Look for detector bias or a bent board there.")
    else:
        kind = "none"
        answer = "No pattern: errors are small and random across the image."
    return {"kind": kind, "answer": answer, "rms_px": rms_all, "typical_px": float(np.median(r)),
            "rings": rings,
            "edge_growth": growth, "grid": {"nx": nx, "ny": ny, "cells": cells},
            "rules": {"edge_growth": EDGE_GROWTH, "bias_se": BIAS_SE, "bias_frac": BIAS_FRAC}}


def lens_zones(lens: Dict[str, Any], w: float, h: float) -> Optional[Dict[str, Any]]:
    """Three-zone reading of a camera's lens sigma grid (as served by ``load_diagnostics``)."""
    s = lens.get("sigma_px")
    if not s:
        return None
    lim = lens.get("limit_px") or 1.0
    obs = np.asarray(lens["observed"], bool)
    sig = np.array([np.inf if x is None else x for x in s], float)
    projects = np.array([x is not None for x in s])
    if lens.get("unbounded"):
        zone = np.where(projects, "uncertain", "none")
    else:
        zone = np.where(~projects, "none", np.where(sig > lim, "uncertain",
                                                     np.where(obs, "trusted", "extrapolated")))
    n = int(projects.sum()) or 1
    share = {z: float(np.sum(zone == z)) / n for z in ("trusted", "extrapolated", "uncertain")}
    u, v = np.asarray(lens["u"], float), np.asarray(lens["v"], float)
    unseen = zone != "trusted"
    unseen &= projects
    if lens.get("unbounded"):
        answer = "Not determined anywhere: the photos do not pin this lens down."
    elif share["trusted"] > 0.97:
        answer = "Trusted across the whole image."
    else:
        where = _areas(unseen, u, v, w, h, base=projects) if unseen.any() else ""
        parts = [f"Trusted over {share['trusted']:.0%} of the image"]
        if share["uncertain"] >= 0.01:
            parts.append(f"{share['uncertain']:.0%} is over the {lim:g} px limit")
        if share["extrapolated"] >= 0.01:
            parts.append(f"{share['extrapolated']:.0%} was never covered by the board")
        answer = "; ".join(parts) + (f" (mostly {where})." if where else ".")
    finite = sig[projects & np.isfinite(sig)]
    return {"answer": answer, "zones": zone.tolist(), "share": share, "limit_px": float(lim),
            "sigma_median_px": float(np.median(finite)) if finite.size else None,
            "sigma_max_trusted_px": float(sig[zone == "trusted"].max()) if (zone == "trusted").any() else None}
