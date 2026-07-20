"""Observability audit of a rig calibration: named weak directions + coverage diagnostics.

After (or before) the joint BA, eigen-analyse the gauge-fixed weighted normal matrix the
solver already assembles and turn near-null directions into **named, actionable findings**
("cam0: focal and xi move together — add tilted views") plus cheap coverage corroboration
(periphery occupancy, board-tilt diversity). This surfaces exactly the failure class the
solver itself silently absorbs: a near-singular Hessian is rescued by escalating Cholesky
jitter (:mod:`ds_msp.core.optimize`) and returns a confident-looking answer with no warning
that some parameter combination was never observable from the capture.

The linear algebra lives in :mod:`ds_msp.core.observability` (van der Sluis equilibration —
see there for why raw-Hessian eigenvalues would be unit artefacts); this module owns the
rig semantics: column labels, the fisheye degeneracy signatures, and the messages.

Known-degeneracy signatures (citations):

- Planar-target focal/distortion gauge coupling (DS ``xi``): Usenko-Demmel-Cremers 3DV 2018;
  Hartley-Zisserman MVG (planar gauge freedom). This repo's verified house fact: DS xi-focal
  coupling on coplanar points is a capture degeneracy, not a Jacobian bug.
- Outer-FOV shape parameters are only constrained by periphery observations — an empty outer
  annulus leaves them free.
- Degenerate capture motion (single-axis rotation / near-pure translation): the flag-and-warn
  discipline of visual-inertial observability analysis (OpenVINS; Yang et al.).
- Capture-guidance precedent: AprilCal (Richardson-Strom-Olson, IROS 2013) and Calibration
  Wizard (Peng-Sturm, ICCV 2019) pick the next view from the parameter covariance; DS-MSP
  instead *names* the weak directions inside the library report and can gate on them.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.observability import eigen_weakness, orientation_spread, radial_occupancy
from ..core.robust import auto_kernel_scale, robust_weight
from . import bundle
from .types import ObjectObs, RigState

# Parameter-name classes used by the signature matcher. Focal columns exist in every model
# but OCam (whose polynomial a0 plays the paraxial-focal role); the "primary" distortion
# parameter is the lowest-order shape parameter a planar low-tilt capture couples with focal;
# "outer-FOV" parameters are the ones only periphery observations constrain.
_FOCAL_PARAMS = ("fx", "fy", "a0")
_PRIMARY_DISTORTION = ("xi", "alpha", "k1", "lambda1", "a2")
_OUTER_FOV_PARAMS = ("xi", "alpha", "beta", "k3", "k4", "lambda2", "tau_x", "tau_y",
                     "a3", "a4")


def _column_labels(rig: RigState, *, fix_intrinsics: bool) -> List[Tuple]:
    """Per-column labels matching :func:`bundle.build_problem`'s tangent layout exactly:
    ``("cam_extr", cam_id, axis)``, ``("obj_pose", (object_id, frame_id), axis)``,
    ``("intr", cam_id, param_name)`` with ``axis`` in ``rx ry rz tx ty tz``."""
    axes = ("rx", "ry", "rz", "tx", "ty", "tz")
    labels: List[Tuple] = []
    for c in sorted(rig.cameras):
        if c == rig.ref_cam_id:
            continue
        labels.extend(("cam_extr", c, a) for a in axes)
    for key in sorted(rig.object_poses):
        labels.extend(("obj_pose", key, a) for a in axes)
    if not fix_intrinsics:
        for c in sorted(rig.cameras):
            labels.extend(("intr", c, p) for p in type(rig.cameras[c]).param_names)
    return labels


def _coverage(rig: RigState, object_obs: List[ObjectObs]) -> Dict[int, Dict]:
    """Per-camera radial occupancy + board-tilt diversity (cheap, Hessian-free)."""
    out: Dict[int, Dict] = {}
    for c in sorted(rig.cameras):
        uv = [o.pts_2d for o in object_obs
              if o.cam_id == c and (o.object_id, o.frame_id) in rig.object_poses]
        if not uv:
            continue
        uv = np.concatenate(uv, axis=0)
        K = rig.cameras[c].K
        center = np.array([K[0, 2], K[1, 2]])
        wh = rig.img_size.get(c)
        if wh is not None:
            w, h = wh
            corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
            R = float(np.max(np.linalg.norm(corners - center[None, :], axis=1)))
        else:
            R = None
        occ, periphery_frac = radial_occupancy(uv, center, R=R)

        normals, tilts = [], []
        R_cg = rig.T_c_g[c][:3, :3]
        t_cg = rig.T_c_g[c][:3, 3]
        for o in object_obs:
            if o.cam_id != c:
                continue
            key = (o.object_id, o.frame_id)
            T = rig.object_poses.get(key)
            if T is None:
                continue
            n = R_cg @ T[:3, :3] @ np.array([0.0, 0.0, 1.0])
            pts = rig.objects[o.object_id].pts_3d[o.point_rows]
            centroid_g = T[:3, :3] @ pts.mean(axis=0) + T[:3, 3]
            d = R_cg @ centroid_g + t_cg
            d = d / max(np.linalg.norm(d), 1e-12)
            normals.append(n)
            tilts.append(np.degrees(np.arccos(min(abs(float(n @ d)), 1.0))))
        if normals:
            _eig, tilt_diversity = orientation_spread(np.asarray(normals))
            tilts = np.asarray(tilts)
            tilt_range = float(tilts.max() - tilts.min())
        else:
            tilt_diversity, tilt_range = 0.0, 0.0
        out[c] = {"periphery_frac": periphery_frac, "occ": occ,
                  "tilt_diversity": float(tilt_diversity), "tilt_range_deg": tilt_range}
    return out


def _coobservation_count(object_obs: List[ObjectObs], cam: int) -> int:
    """Frames where ``cam`` and at least one other camera both observed the object."""
    by_frame: Dict[Tuple[int, int], set] = {}
    for o in object_obs:
        by_frame.setdefault((o.object_id, o.frame_id), set()).add(o.cam_id)
    return sum(1 for cams in by_frame.values() if cam in cams and len(cams) > 1)


def _name_direction(energy: np.ndarray, labels: List[Tuple], Hh_pairs: List[Tuple],
                    coverage: Dict[int, Dict], object_obs: List[ObjectObs], *,
                    tilt_deg_min: float, periphery_frac_min: float,
                    corr_thresh: float) -> Dict:
    """Map one weak eigenvector's participation energy to a named finding dict."""
    by_block: Dict[Tuple, float] = {}
    for j, lab in enumerate(labels):
        by_block[lab[:2]] = by_block.get(lab[:2], 0.0) + float(energy[j])
    e_cam = sum(v for k, v in by_block.items() if k[0] == "cam_extr")
    e_obj = sum(v for k, v in by_block.items() if k[0] == "obj_pose")
    e_intr = sum(v for k, v in by_block.items() if k[0] == "intr")
    top_block, top_e = max(by_block.items(), key=lambda kv: kv[1])
    n_spread = sum(1 for v in by_block.values() if v > 0.01)

    # 1. global gauge: energy spread over pose blocks (cameras and/or object poses), none
    #    dominant — only possible when the datum is (wrongly) unfixed; the shipped layout
    #    pins it. Measured on the unfixed-ref positive control: all 6 gauge modes sit at
    #    machine-zero eigenvalues with pose-block energy > 0.85; a global rotation can load
    #    mostly the camera blocks, so no minimum object-block energy is required.
    if (e_cam + e_obj) > 0.85 and e_cam > 0.05 and top_e < 0.5 and n_spread >= 3:
        return {"kind": "global_gauge", "cam": None, "frames": None, "params": (),
                "message": ("a global gauge freedom is present (a joint rigid move of all "
                            "cameras and object poses leaves the fit unchanged) — the datum "
                            "is not pinned; this should be impossible in the shipped layout")}

    # 2. degenerate capture motion: one motion axis jointly weak across most frames.
    if e_obj > 0.8 and top_e < 0.35:
        axis_dom: Dict[str, int] = {}
        for (kind, key), _v in by_block.items():
            if kind != "obj_pose":
                continue
            idx = [j for j, lab in enumerate(labels) if lab[:2] == (kind, key)]
            best = max(idx, key=lambda j: energy[j])
            axis_dom[labels[best][2]] = axis_dom.get(labels[best][2], 0) + 1
        axis, cnt = max(axis_dom.items(), key=lambda kv: kv[1])
        n_obj = sum(1 for k in by_block if k[0] == "obj_pose")
        if n_obj and cnt / n_obj >= 0.7:
            what = "rotation" if axis.startswith("r") else "translation"
            return {"kind": "degenerate_motion", "cam": None, "frames": None,
                    "params": (axis,),
                    "message": (f"capture motion is degenerate: the board's {what} about "
                                f"{axis} is weakly excited across nearly all frames — vary "
                                "board orientation about all 3 axes")}

    # 3. intrinsics-dominated weak direction.
    if e_intr > 0.5:
        cam = max((k for k in by_block if k[0] == "intr"),
                  key=lambda k: by_block[k])[1]
        p_energy: Dict[str, float] = {}
        for j, lab in enumerate(labels):
            if lab[:2] == ("intr", cam):
                p_energy[lab[2]] = p_energy.get(lab[2], 0.0) + float(energy[j])
        focal_e = sum(p_energy.get(p, 0.0) for p in _FOCAL_PARAMS)
        dist_params = [p for p in p_energy
                       if p in _PRIMARY_DISTORTION or p in _OUTER_FOV_PARAMS]
        dist_e = sum(p_energy[p] for p in dist_params)
        cov = coverage.get(cam, {})
        top_dist = max(dist_params, key=lambda p: p_energy[p]) if dist_params else None

        if focal_e > 0.1 and dist_e > 0.1 and top_dist is not None:
            tilt_note = ""
            if cov.get("tilt_range_deg", 90.0) < tilt_deg_min:
                tilt_note = (f" (board tilt range is only "
                             f"{cov['tilt_range_deg']:.0f} deg)")
            periph_note = ""
            if cov.get("periphery_frac", 1.0) < periphery_frac_min:
                periph_note = (", and capture the board near the image periphery "
                               f"(outer-annulus coverage is "
                               f"{100 * cov.get('periphery_frac', 0.0):.0f}%)")
            return {"kind": "focal_distortion_coupling", "cam": cam, "frames": None,
                    "params": ("fx", "fy", top_dist),
                    "message": (f"cam{cam}: focal (fx,fy) and {top_dist} move together — "
                                "the planar-target focal/distortion coupling"
                                f"{tilt_note}; add board views with out-of-plane tilt "
                                f"(>~20 deg) to break it{periph_note}")}
        outer = [p for p in p_energy if p in _OUTER_FOV_PARAMS]
        outer_e = sum(p_energy[p] for p in outer)
        if outer and outer_e > 0.5 and cov.get("periphery_frac", 1.0) < periphery_frac_min:
            return {"kind": "periphery_underobserved", "cam": cam, "frames": None,
                    "params": tuple(sorted(outer, key=lambda p: -p_energy[p])),
                    "message": (f"cam{cam}: outer-FOV distortion "
                                f"({', '.join(sorted(outer, key=lambda p: -p_energy[p]))}) "
                                f"weakly observed — only "
                                f"{100 * cov.get('periphery_frac', 0.0):.0f}% of corners "
                                "fall in the outer radial annulus; capture the board near "
                                "the image periphery")}
        top_params = tuple(sorted(p_energy, key=lambda p: -p_energy[p])[:3])
        return {"kind": "intrinsic_weak", "cam": cam, "frames": None, "params": top_params,
                "message": (f"cam{cam}: intrinsic combination {top_params} is weakly "
                            "constrained by this capture")}

    # 4. one object pose dominates.
    if top_block[0] == "obj_pose" and top_e > 0.5:
        key = top_block[1]
        return {"kind": "frame_weak", "cam": None, "frames": (key[1],), "params": (),
                "message": (f"frame {key[1]}: object pose weakly constrained — its "
                            "geometry is near-unobservable; drop the frame or add coverage")}

    # 5. one camera extrinsic dominates.
    if top_block[0] == "cam_extr" and top_e > 0.5:
        cam = top_block[1]
        m = _coobservation_count(object_obs, cam)
        return {"kind": "extrinsic_weak", "cam": cam, "frames": None, "params": (),
                "message": (f"cam{cam}: extrinsic weakly constrained — it co-observes only "
                            f"{m} frame(s) with the rest of the rig; add frames where "
                            f"cam{cam} and its neighbours see the board together")}

    labs = [labels[j] for j in np.argsort(energy)[::-1][:4]]
    return {"kind": "weak_direction", "cam": None, "frames": None,
            "params": tuple(str(lab) for lab in labs),
            "message": f"weak parameter combination across {labs}"}


def _merge_frame_findings(findings: List[Dict]) -> List[Dict]:
    """Group contiguous single-frame ``frame_weak`` findings into one ranged finding."""
    frame_ws = sorted((f for f in findings if f["kind"] == "frame_weak"),
                      key=lambda f: f["frames"][0])
    rest = [f for f in findings if f["kind"] != "frame_weak"]
    merged: List[Dict] = []
    for f in frame_ws:
        if merged and f["frames"][0] == merged[-1]["frames"][-1] + 1:
            prev = merged[-1]
            prev["frames"] = tuple(list(prev["frames"]) + [f["frames"][0]])
            a, b = prev["frames"][0], prev["frames"][-1]
            prev["message"] = (f"frames {a}-{b}: object poses weakly constrained — thin "
                               "coverage over this stretch; drop them or add corners")
        else:
            merged.append(dict(f))
    return rest + merged


def audit_rig(rig: RigState, object_obs: List[ObjectObs], *,
              fix_intrinsics: bool = False,
              kernel: str = "cauchy", scale: Optional[float] = None,
              tau_rel: float = 1e-6, soft_rel: float = 1e-3,
              corr_thresh: float = 0.95,
              tilt_deg_min: float = 15.0, periphery_frac_min: float = 0.02) -> Dict:
    """Observability audit of a (converged) rig fit: named weak directions + coverage.

    Assembles the same gauge-fixed dense Jacobian the BA uses
    (:func:`bundle.build_problem`), weights it with the same robust-kernel weights the final
    BA stage applies (``kernel``/``scale``, auto-estimated from the final residuals when
    ``scale=None`` — so the audit reflects the information matrix of the solve actually
    performed), equilibrates ``H = JᵀWJ`` to remove the mixed-units artefact
    (:func:`ds_msp.core.observability.eigen_weakness`), and maps every near-null eigenvector
    to a named finding with an actionable capture suggestion.

    **Two-tier thresholds, set from measured spectra, not theory.** Characterization on
    synthetic captures (characterization runs of 2026-07-18) measured: structural degeneracies (true
    gauge modes, the planar focal/xi coupling) collapse to equilibrated-eigenvalue ratios
    ``<= 1e-10``, while the softest directions a healthy 3D-target capture carries (e.g. the
    ubiquitous RadTan k2/k3 near-collinearity) sit at ``>= 1e-4`` — a ~6-order empty gap.
    ``tau_rel`` (default ``1e-6``) sits mid-gap and flags only **structural** findings (these
    gate); directions in ``[tau_rel, soft_rel)`` are counted as ``n_soft`` and summarized
    without individual alarms, so a normal calibration is not drowned in notices.

    Returns ``{"cond", "cond_raw", "n_weak", "n_soft", "gap", "gauge_ok", "findings": [...],
    "soft": [...], "coverage": {cam: {...}}, "pairs": [...]}`` where each finding is
    ``{"kind", "cam", "frames", "params", "message", "ratio"}``. ``gauge_ok`` is ``True``
    when no global-gauge weak direction is present — always expected for the shipped layout,
    which pins the datum by construction (reference camera excluded from the state, metric
    board fixed); a ``global_gauge`` finding therefore indicates a bug, and the audit's own
    test suite verifies it *would* fire by deliberately unfixing the reference camera.

    Cost: one dense ``eigh`` of a K x K matrix (K ~ 200-500 at calibration scale) — well
    under a single BA iteration, which factors same-size systems repeatedly.
    """
    state0, residual, jacobian, _retract, K = bundle.build_problem(
        rig, object_obs, fix_intrinsics=fix_intrinsics)
    J = np.asarray(jacobian(state0), float)
    r = np.asarray(residual(state0), float)

    bn = np.linalg.norm(r.reshape(-1, 2), axis=1)
    if kernel != "none":
        if scale is None:
            scale = auto_kernel_scale(bn, kernel)
        w = robust_weight(bn * bn, kernel, float(scale))
    else:
        scale, w = 1.0, np.ones_like(bn)
    W_row = np.repeat(w, 2)
    H = (J * W_row[:, None]).T @ J

    ew = eigen_weakness(H, tau_rel=soft_rel, corr_thresh=corr_thresh)
    labels = _column_labels(rig, fix_intrinsics=fix_intrinsics)
    if len(labels) != K:
        raise AssertionError(f"column labels ({len(labels)}) out of sync with tangent "
                             f"dimension ({K})")
    coverage = _coverage(rig, object_obs)

    critical = [w for w in ew["weak"] if w["ratio"] < tau_rel]
    soft_dirs = [w for w in ew["weak"] if w["ratio"] >= tau_rel]

    findings = []
    for wdir in critical:
        f = _name_direction(wdir["energy"], labels, ew["pairs"], coverage, object_obs,
                            tilt_deg_min=tilt_deg_min,
                            periphery_frac_min=periphery_frac_min,
                            corr_thresh=corr_thresh)
        f["ratio"] = wdir["ratio"]
        findings.append(f)
    findings = _merge_frame_findings(findings)
    gauge_ok = not any(f["kind"] == "global_gauge" for f in findings)

    soft = []
    for wdir in soft_dirs:
        top = [labels[j] for j in wdir["participating"][:3]]
        soft.append({"ratio": wdir["ratio"], "top": top})

    named_pairs = [(labels[i], labels[j], corr) for i, j, corr in ew["pairs"]]
    return {"cond": ew["cond"], "cond_raw": ew["cond_raw"], "n_weak": len(critical),
           "n_soft": len(soft), "gap": ew["gap"], "gauge_ok": gauge_ok,
           "findings": findings, "soft": soft, "coverage": coverage,
           "pairs": named_pairs}
