"""Verdict and next actions for one result, from checks the library itself records.

Studio shows, first, one verdict for the rig and a level per camera, then at most three
actions, each naming its target and the evidence view that supports it. Only checks recorded in
``calibration_acceptance.json`` set a level:

* the lens check (lens 1-sigma and parameters at a bound; FR-RIG-030),
* per-camera coverage below the configured minimum,
* the opt-in split check, when it ran,
* observability findings (each names its cameras and its fix),
* per-pair extrinsic uncertainty: unidentifiable, or over configured limits,
* the final solver status and frame-pose support,
* the share of corners outside the inlier band (FR-RIG-031), against mrcal's rule of thumb of
  3%. This limit is provisional: it has not been validated on the DS-MSP matrix, so it only
  raises *check*, never *improve* or *reject*.

Levels, worst last: ``accept`` < ``check`` (usable; look at the noted item) < ``improve``
(usable, but the listed actions should help) < ``reject`` (a hard check failed: something is
not determined by the data). Nothing here is tuned per dataset.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

LEVELS = ("accept", "check", "improve", "reject")
#: mrcal: "more than about 3% outliers should raise questions". Provisional for DS-MSP.
OUTLIER_CHECK_FRAC = 0.03
MAX_ACTIONS = 3


def _rank(level: str) -> int:
    return LEVELS.index(level)


def _cam(c: Any) -> str:
    return f"camera {c}"


class _Collector:
    def __init__(self, cams: List[str]):
        self.cams: Dict[str, Dict[str, Any]] = {c: {"level": "accept", "reasons": []} for c in cams}
        self.rig: List[Dict[str, Any]] = []
        self.actions: List[Dict[str, Any]] = []

    def add(self, level: str, reason: str, *, cams=(), action: Optional[str] = None,
            evidence: Optional[str] = None, check: str = "") -> None:
        cams = [str(c) for c in cams]
        for c in cams:
            e = self.cams.setdefault(c, {"level": "accept", "reasons": []})
            if _rank(level) > _rank(e["level"]):
                e["level"] = level
            e["reasons"].append(reason)
        if not cams:
            self.rig.append({"level": level, "reason": reason, "check": check})
        if action:
            self.actions.append({"level": level, "text": action, "cams": cams,
                                 "evidence": evidence, "check": check})


def run_verdict(report: Dict[str, Any], acceptance: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """``{"level", "headline", "cameras": {cam: {level, reasons}}, "rig_reasons", "actions",
    "n_actions"}`` for one result. ``actions`` holds every action, worst first; Studio shows
    the first :data:`MAX_ACTIONS`."""
    acc = acceptance or {}
    n = (report.get("capture") or {}).get("n_cameras") or len(report.get("intrinsics") or [])
    col = _Collector([str(i) for i in range(int(n or 0))])

    solver = acc.get("solver") or {}
    if solver.get("numerical_failures"):
        col.add("reject", "the solver hit a non-finite cost", check="solver",
                action="The solve failed numerically; check the observations and lens models for "
                       "points outside the model's valid range.", evidence="tech")
    status = solver.get("final_status")
    if status not in (None, "CONVERGED"):
        col.add("improve", f"the final solve ended {str(status).lower().replace('_', ' ')}",
                check="solver",
                action="The final solve did not converge; rerun it, then check the worst photos "
                       "for wrong detections.", evidence="frames")
    unresolved = [r for r in solver.get("frame_support") or [] if not r.get("supported", True)]
    if unresolved:
        col.add("check", f"{len(unresolved)} board pose(s) are weakly supported by their photos",
                check="frame_support")

    lens = (acc.get("lens") or {}).get("cameras") or {}
    limit = (acc.get("lens") or {}).get("limit_px")
    for c, v in sorted(lens.items(), key=lambda kv: int(kv[0])):
        wid = v.get("sigma_median_widened_px")
        if "sigma" in (v.get("reasons") or []):
            if wid is None:
                col.add("reject", f"lens not determined from {v.get('frames')} photo(s)", cams=[c],
                        check="lens", evidence="lens",
                        action=f"Add photos of {_cam(c)}: its lens is not determined by "
                               f"{v.get('frames')} photo(s). Cover the whole image, close and "
                               f"tilted about 45°.")
            else:
                col.add("improve", f"lens uncertain: at least ±{wid:.2f} px (limit {limit} px)",
                        cams=[c], check="lens", evidence="lens",
                        action=f"Add photos of {_cam(c)} with the board in the areas marked on "
                               f"its uncertainty map, close and tilted about 45°.")
        if v.get("params_at_bound"):
            col.add("improve", "lens parameter at its limit: " + ", ".join(v["params_at_bound"]),
                    cams=[c], check="lens", evidence="lens",
                    action=f"{_cam(c).capitalize()}: {', '.join(v['params_at_bound'])} stopped at "
                           f"its limit. Add photos near the image edges, or try another lens model.")

    cov = acc.get("coverage") or {}
    min_cov = acc.get("min_camera_coverage")
    for c, v in cov.items():
        if min_cov is not None and v.get("coverage", 1.0) < min_cov:
            col.add("reject", f"the fit explains only {v['coverage']:.0%} of its corners",
                    cams=[c], check="coverage", evidence="frames",
                    action=f"{_cam(c).capitalize()}: only {v['coverage']:.0%} of its corners fit. "
                           f"Check its worst photos for wrong detections or a moved camera.")

    cv = acc.get("crossval") or {}
    for c, v in (cv.get("cameras") or {}).items():
        if v.get("applicable") and v.get("ok") is False:
            col.add("improve", f"two halves of the capture disagree by ≈{v.get('error_estimate_px', 0):.2f} px",
                    cams=[c], check="crossval", evidence="residuals",
                    action=f"{_cam(c).capitalize()}: the two halves of the capture disagree. Look "
                           f"for a pattern in its residual map; the lens model may not fit.")

    ob = acc.get("observability") or {}
    if ob and ob.get("gauge_ok") is False:
        col.add("reject", "the rig's reference frame is not fixed by the data", check="observability")
    for f in ob.get("findings") or []:
        cams = f.get("cams") or ([f["cam"]] if f.get("cam") is not None else [])
        col.add("improve", "weakly tied to the rest of the rig", cams=cams, check="observability",
                evidence="rig", action=f.get("hint") or f.get("message"))

    unc = acc.get("extrinsic_uncertainty") or {}
    for k, p in (unc.get("pairs") or {}).items():
        a, b = k.split("-")
        if p.get("identifiable") is False:
            col.add("reject", f"relative pose of cameras {a} and {b} is not determined",
                    cams=[a, b], check="uncertainty", evidence="links",
                    action=f"Cameras {a} and {b}: their relative pose is not determined. Take "
                           f"photos where both see the board at the same time.")
    lim = unc.get("limits") or {}
    if unc and unc.get("within_limits") is False and lim:
        worst = unc.get("worst") or {}
        col.add("improve", f"pair {worst.get('pair', '?')} over the pose-uncertainty limit",
                cams=str(worst.get("pair", "")).split("-") if worst.get("pair") else (),
                check="uncertainty", evidence="links",
                action=f"Pair {worst.get('pair', '?')}: relative pose uncertainty is over the limit. "
                       f"Take photos both cameras see, at varied angles.")

    diag = acc.get("diagnostics") or {}
    for c, v in (diag.get("outliers") or {}).items():
        frac = v.get("outlier_frac")
        if frac is not None and frac > OUTLIER_CHECK_FRAC:
            col.add("check", f"{frac:.1%} of its corners disagree with the fit (rule of thumb 3%)",
                    cams=[c], check="outliers", evidence="frames",
                    action=f"{_cam(c).capitalize()}: {frac:.1%} of its corners disagree with the fit. "
                           f"Look at its worst photos for glare, blur or wrong detections.")

    levels = [e["level"] for e in col.cams.values()] + [r["level"] for r in col.rig]
    level = max(levels, key=_rank) if levels else "accept"
    col.actions.sort(key=lambda a: -_rank(a["level"]))
    bad = [c for c, e in col.cams.items() if e["level"] in ("improve", "reject")]
    headline = {
        "accept": "Your rig is calibrated",
        "check": "Calibrated; a few items to check",
        "improve": f"Usable, but {len(bad)} camera{'s' if len(bad) != 1 else ''} would gain from more photos"
                   if bad else "Usable, but the actions below should help",
        "reject": "Not ready: part of the rig is not determined by the photos",
    }[level]
    return {"level": level, "headline": headline, "cameras": col.cams, "rig_reasons": col.rig,
            "actions": col.actions, "n_actions": len(col.actions), "max_shown": MAX_ACTIONS,
            "outlier_check_frac": OUTLIER_CHECK_FRAC}
