"""Drive the rig calibration on a real MC-Calib dataset, MC-Calib style.

This is the orchestration MC-Calib's ``apps/calibrate`` performs, in DS-MSP terms:
choose a model per camera (the ``camera_models`` config field), optionally optimize
extrinsics-only or intrinsics+extrinsics (``fix_intrinsic``), then write the result in
MC-Calib's exact OpenCV-YAML format. Used by ``scripts/calibrate_rig.py`` and the
cross-dataset validation.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..io.mccalib import (
    Scenario,
    save_detection_images,
    save_mccalib_results,
    save_reprojection_images,
)
from ..models.registry import FISHEYE_MODELS, PINHOLE_MODELS
from . import bundle
from .accuracy import AccuracyProfile, accuracy_metrics
from .calibrate import (
    _gated_pnp,
    calibrate_rig,
    make_bundle_front_end,
    paraxial_focal,
    resolve_structure_mode,
)
from .types import RigState


def random_model_assignment(cam_ids: List[int], *, kind: str = "pinhole",
                            seed: int = 0, allowed=None) -> Dict[int, str]:
    """Assign each camera a random *valid* model name.

    ``kind="pinhole"`` draws from {radtan, ucm, eucm, ds, ocam} (KB excluded — it is
    fisheye-only); ``kind="fisheye"`` from {kb, ucm, eucm, ds, ocam} (RadTan excluded — it
    cannot represent a fisheye). The sphere/polynomial models handle both. Pass ``allowed``
    to restrict the pool.
    """
    pool = list(allowed) if allowed is not None else list(
        PINHOLE_MODELS if kind == "pinhole" else FISHEYE_MODELS)
    rng = np.random.default_rng(seed)
    return {c: str(rng.choice(pool)) for c in sorted(cam_ids)}


def make_fixed_intrinsic_front_end(cameras: Dict[int, object]):
    """Front-end for ``fix_intrinsic=true``: use the *given* per-camera intrinsics and only
    estimate per-frame object poses (robust gated PnP). No intrinsic calibration — the
    global BA then refines extrinsics with these intrinsics held fixed, mirroring MC-Calib's
    fixed-intrinsic mode (which needs initial intrinsics and never refines them)."""
    def front_end(obj, obs_by_cam, img_size):
        """Seed per-frame object poses by robust PnP; intrinsics are the fixed ``cameras``.

        Matches the ``front_end(obj, obs_by_cam, img_size) -> {cam_id: CameraModel}``
        signature :func:`~ds_msp.rig.calibrate.calibrate_rig` expects. ``obj`` may be a single
        :class:`Object3D` or a ``{object_id: Object3D}`` map (multi-object rig) — each pose is
        resected against the object *its* observation saw.
        """
        objs = obj if isinstance(obj, dict) else {obj.object_id: obj}
        for cam_id, obs in obs_by_cam.items():
            model = cameras[cam_id]
            for o in obs:
                o.T_c_o = _gated_pnp(model, objs[o.object_id].pts_3d[o.point_rows], o.pts_2d)
        return {c: cameras[c] for c in obs_by_cam}
    return front_end


def calibrate_scenario(scn: Scenario, model_spec, *, fix_intrinsics: bool = False,
                       init_cameras: Optional[Dict[int, object]] = None,
                       init_K: Optional[Dict[int, object]] = None,
                       save_dir: Optional[str] = None,
                       camera_params_file_name: str = "",
                       image_root: Optional[str] = None,
                       cam_prefix: str = "Cam_", he_approach: int = 0,
                       refine_structure: bool = False,
                       structure_prior_sigma: Optional[float] = None,
                       structure_mode: Optional[str] = None,
                       noise_bound: Optional[float] = None,
                       verbose: bool = False, on_iter=None,
                       objects: Optional[list] = None,
                       reproj_gate_px: Optional[float] = None,
                       report_covariance: bool = False,
                       audit_gate: str = "warn", accuracy_gate: str = "off",
                       certify: bool = False,
                       accuracy_profile: Optional[AccuracyProfile] = None,
                       uncertainty_gate: str = "warn",
                       max_rotation_sigma_deg: Optional[float] = None,
                       max_centre_sigma: Optional[float] = None,
                       config_warnings: Optional[List[str]] = None,
                       profile: str = "default") -> Dict:
    """Calibrate one loaded :class:`Scenario` and (optionally) write MC-Calib output.

    ``model_spec`` is a single model or a ``{cam_id: model}`` map (names or classes).
    ``fix_intrinsics=True`` requires ``init_cameras`` (per-camera models with the intrinsics
    to hold fixed) and optimizes extrinsics only. ``init_K`` (``{cam_id: 3x3}``) seeds the
    refining front-end's focal / principal point from a known intrinsics file (MC-Calib's
    ``cam_params_path`` initialization) — essential for a real strong-fisheye / mixed-resolution
    rig. When ``image_root`` is given, MC-Calib-style reprojection overlay images are written
    too. ``verbose=True`` prints per-stage progress (front-end, groups, each BA pass) as it
    runs instead of staying silent until the final result. ``on_iter(it, max_iter, rms, cost,
    rig_snapshot)`` fires once per solver iteration across every stage, with a real
    ``RigState`` snapshot of the mid-solve geometry — the hook a live progress display (or a
    3D terminal animator) drives from. ``report_covariance=True`` attaches a ``"covariance"`` key
    (:func:`ds_msp.rig.bundle.parameter_covariance`) to the returned dict with per-camera
    extrinsic/intrinsic parameter uncertainty from the frame-clustered sandwich estimator.

    ``audit_gate`` controls the observability audit (:func:`ds_msp.rig.audit.audit_rig`):
    ``"warn"`` (default) runs it and attaches an ``"audit"`` key — default-on because its
    entire value is catching a silently under-constrained capture the user did not suspect,
    and its cost is one eigendecomposition, well under a single BA iteration; ``"refuse"``
    additionally raises :class:`RuntimeError` when a structural weak direction is found (for
    CI/production pipelines, like ``reproj_gate_px``); ``"off"`` skips it.

    ``certify=True`` attaches a ``"certificate"`` key
    (:func:`ds_msp.rig.certify.certify_rotations`): the a-posteriori global-optimality
    certificate of the extrinsic-rotation backbone against the per-view PnP measurement
    graph — opt-in like ``report_covariance``.
    ``accuracy_gate="refuse"`` requires a profile with limits and independent truth;
    incomplete or failing comparisons refuse export. Both refusal gates run before saving.
    ``structure_mode`` (``rigid`` default | ``bend`` | ``free`` | ``off``) selects multi-board target
    refinement (see :func:`~ds_msp.rig.calibrate.calibrate_rig`); legacy
    ``refine_structure=True`` means ``free``. ``structure_prior_sigma`` specifies free-mode
    target geometry uncertainty in scene length units, fixed across refinement rounds.
    ``uncertainty_gate`` (``warn`` default | ``refuse`` | ``off``) evaluates the
    frame-clustered sandwich covariance of the solved rig (:func:`bundle.shared_covariance`,
    frame and board poses marginalised) and propagates it to every camera pair's 1-sigma
    relative rotation (deg) and camera-centre (scene units). The result is attached as
    ``"uncertainty"`` and recorded in the acceptance file; ``refuse`` blocks export when any
    pair exceeds ``max_rotation_sigma_deg`` / ``max_centre_sigma``. Observations hard-dropped
    by ``reproj_gate_px`` are recorded there too, one ledger entry each.
    Returns ``{rig, models, paths, metrics}`` (``+ "covariance"``/``"audit"``/
    ``"certificate"``/``"uncertainty"`` when enabled).
    """
    if uncertainty_gate not in ("off", "warn", "refuse"):
        raise ValueError(f"uncertainty_gate must be 'off', 'warn' or 'refuse', got {uncertainty_gate!r}")
    if uncertainty_gate == "refuse" and (max_rotation_sigma_deg is None or max_centre_sigma is None):
        raise ValueError("uncertainty_gate='refuse' needs max_rotation_sigma_deg and max_centre_sigma")
    if audit_gate not in ("off", "warn", "refuse"):
        raise ValueError(f"audit_gate must be 'off', 'warn' or 'refuse', got {audit_gate!r}")
    if accuracy_gate not in ("off", "refuse"):
        raise ValueError("accuracy_gate must be off or refuse")
    if structure_prior_sigma is not None and (
            not np.isfinite(structure_prior_sigma) or structure_prior_sigma <= 0):
        raise ValueError("structure_prior_sigma must be positive in target length units")
    structure_mode = resolve_structure_mode(structure_mode, refine_structure)
    if accuracy_gate == "refuse" and accuracy_profile is None:
        raise ValueError("accuracy_gate='refuse' requires an accuracy profile and independent truth")
    if init_cameras is not None:
        # Start from the provided per-camera models (MC-Calib's cam_params_path init): the
        # front-end uses them directly instead of a from-scratch re-fit — essential for a
        # wide-FOV fisheye, whose from-scratch model fit can diverge. ``fix_intrinsics`` then
        # decides only whether the joint BA refines these intrinsics or holds them.
        front_end = make_fixed_intrinsic_front_end(init_cameras)
    elif fix_intrinsics:
        raise ValueError("fix_intrinsics=True needs init_cameras (initial intrinsics)")
    else:
        front_end = make_bundle_front_end(model_spec, init_K=init_K)
    rig = calibrate_rig(scn.object, scn.object_obs, scn.img_size,
                        fix_intrinsics=fix_intrinsics, front_end=front_end,
                        he_approach=he_approach, structure_mode=structure_mode,
                        structure_prior_sigma=structure_prior_sigma,
                        noise_bound=noise_bound, verbose=verbose, on_iter=on_iter,
                        objects=objects, reproj_gate_px=reproj_gate_px)
    # keep the (possibly structure-refined) object the rig actually solved with
    refined_object = rig.objects.get(scn.object.object_id, scn.object)

    metrics = _scenario_metrics(rig, scn, accuracy_profile=accuracy_profile)
    out = {"rig": rig, "models": {c: rig.cameras[c].name for c in rig.cameras},
          "paths": {}, "metrics": metrics}
    if report_covariance:
        out["covariance"] = bundle.parameter_covariance(
            rig, scn.object_obs, fix_intrinsics=fix_intrinsics)
    if certify:
        from .certify import certify_rotations
        out["certificate"] = certify_rotations(rig, scn.object_obs)
    if audit_gate != "off":
        from .audit import audit_rig
        audit = audit_rig(rig, scn.object_obs, fix_intrinsics=fix_intrinsics)
        out["audit"] = audit
        if audit_gate == "refuse" and (audit["n_weak"] > 0 or not audit["gauge_ok"]):
            details = "; ".join(f["message"] for f in audit["findings"]) or \
                f"{audit['n_weak']} unnamed weak direction(s)"
            raise RuntimeError(
                f"observability audit refused this calibration (audit_gate='refuse'): "
                f"{details}")
    uncertainty = None
    if uncertainty_gate != "off":
        uncertainty = extrinsic_uncertainty(rig, scn.object_obs, fix_intrinsics=fix_intrinsics,
                                            structure_mode=structure_mode,
                                            max_rotation_sigma_deg=max_rotation_sigma_deg,
                                            max_centre_sigma=max_centre_sigma)
        out["uncertainty"] = uncertainty
        if not uncertainty["identifiable"]:
            bad = [k for k, p in uncertainty["pairs"].items() if not p["identifiable"]]
            warnings.warn(f"extrinsic uncertainty is unbounded (not identifiable) for camera "
                          f"pair(s) {', '.join(bad) or 'all'}: the data or the chosen camera "
                          f"models do not constrain these extrinsics", RuntimeWarning,
                          stacklevel=2)
        if uncertainty_gate == "refuse" and not uncertainty["within_limits"]:
            worst = uncertainty["worst"]
            raise RuntimeError(
                f"uncertainty gate refused this calibration (uncertainty_gate='refuse'): pair "
                f"{worst['pair']} 1-sigma rotation {worst['rotation_sigma_deg_t'] or float('inf'):.4f} "
                f"deg / centre {worst['centre_sigma_t'] or float('inf'):.4g} (small-sample widened) "
                f"exceed limits "
                f"{max_rotation_sigma_deg} deg / {max_centre_sigma}; capture more varied views")
    truth = metrics["accuracy"]["ground_truth"]
    physical_pass = (truth.get("status") == "EVALUATED"
                     and truth.get("limit_checks", {}).get("all_pass") is True)
    out["acceptance"] = {
        "accuracy_status": "PASS" if physical_pass else "NOT_ACCEPTED",
        "accuracy_gate": accuracy_gate, "observability_gate": audit_gate,
        "fit_scope": "training reprojection only; not physical accuracy",
        "structure_mode": structure_mode,
        "structure_refined": structure_mode != "off",
        "structure_prior_sigma": structure_prior_sigma,
        "accuracy_evaluation_status": truth.get("status"),
        "limit_checks": truth.get("limit_checks"),
        "accuracy_profile": accuracy_profile.to_dict() if accuracy_profile is not None else None,
        "profile": profile,
        "uncertainty_gate": uncertainty_gate,
        "extrinsic_uncertainty": uncertainty,
        "outlier_gate_px": reproj_gate_px,
        "rejected_observations": list(getattr(rig, "rejected", [])),
        "config_warnings": list(config_warnings or []),
    }
    if accuracy_gate == "refuse" and not physical_pass:
        raise RuntimeError("accuracy gate refused calibration: independent truth must cover "
                           "required pairs and pass every configured physical limit")
    paths = {}
    if save_dir is not None:
        if on_iter is not None and hasattr(on_iter, "set_stage"):
            on_iter.set_stage("(save) writing MC-Calib output")
        paths = save_mccalib_results(rig, save_dir, object3d=refined_object,
                                     object_obs=scn.object_obs,
                                     camera_params_file_name=camera_params_file_name,
                                     image_root=image_root, cam_prefix=cam_prefix)
        if image_root is not None:
            # both used to run fully serially, silently, *after* the BA had already
            # converged -- the live view's most visible "it's just sitting there" gap. Now
            # parallel (see io/mccalib.py) and, if the caller carries a save_progress hook
            # (WebLive3DAnimator), reported live instead of going dark.
            save_cb = getattr(on_iter, "save_progress", None)
            if on_iter is not None and hasattr(on_iter, "set_stage"):
                on_iter.set_stage("(save) reprojection overlays")
            nr = save_reprojection_images(rig, scn.object_obs, image_root, save_dir,
                                          cam_prefix=cam_prefix, progress_cb=save_cb)
            if on_iter is not None and hasattr(on_iter, "set_stage"):
                on_iter.set_stage("(save) detection overlays")
            nd = save_detection_images(scn.object_obs, image_root, save_dir,
                                       cam_prefix=cam_prefix, progress_cb=save_cb)
            paths["reprojection_images"] = f"Reprojection/ ({nr} images)"
            paths["detection_images"] = f"Detection/ ({nd} images)"
    if save_dir is not None:
        acceptance_path = Path(save_dir) / "calibration_acceptance.json"
        acceptance_path.write_text(json.dumps(out["acceptance"], indent=2, allow_nan=False) + "\n")
        paths["acceptance"] = str(acceptance_path)
    out["paths"] = paths
    return out


def extrinsic_uncertainty(rig: RigState, object_obs, *, fix_intrinsics: bool,
                          structure_mode: str = "rigid",
                          max_rotation_sigma_deg: Optional[float] = None,
                          max_centre_sigma: Optional[float] = None) -> Dict:
    """Per camera-pair 1-sigma extrinsic uncertainty from the frame-clustered sandwich
    covariance (:func:`bundle.shared_covariance` -> :func:`bundle.pair_uncertainty`), with
    board poses/bows marginalised when the target was refined rigidly. JSON-safe; pair keys
    are ``"a-b"``. Free per-point (``structure_mode="free"``) target motion is not modelled,
    so its sigmas are optimistic and flagged.

    With ``G`` frame clusters the sigma is itself estimated from ~``G`` effective samples, so
    error/sigma follows Student-t with ``G - 1`` degrees of freedom: the reported ``k``-sigma
    intervals are widened by ``t_factor = t_{0.975, G-1} / z_{0.975}`` (1.15 at 10 frames, 1.04 at
    30) and the gate compares the widened sigma with the limits. Measured on 200-capture Monte
    Carlo per condition: 2-sigma coverage 0.913 -> 0.942 at 10 frames, 0.94-0.97 elsewhere."""
    from scipy import stats

    unc = bundle.shared_covariance(rig, object_obs, fix_intrinsics=fix_intrinsics,
                                   refine_boards=structure_mode in ("rigid", "bend"),
                                   bend_boards=structure_mode == "bend")
    pairs = bundle.pair_uncertainty(rig, unc)
    dof = unc["n_clusters"] - 1
    # one frame cluster leaves no degrees of freedom: no interval, and any limit is exceeded
    t_factor = float(stats.t.ppf(0.975, dof) / stats.norm.ppf(0.975)) if dof >= 1 else None
    table = {}
    for (a, b), p in pairs.items():
        row = dict(p)
        row["rotation_sigma_deg_t"] = None if t_factor is None else p["rotation_sigma_deg"] * t_factor
        row["centre_sigma_t"] = None if t_factor is None else p["centre_sigma"] * t_factor
        table[f"{a}-{b}"] = row

    def unbounded(x):                    # unidentified (inf) or missing; NaN never passes a gate
        return x is None or not np.isfinite(x)

    def over(p):
        if t_factor is None or not p["identifiable"]:
            return max_rotation_sigma_deg is not None or max_centre_sigma is not None
        return ((max_rotation_sigma_deg is not None
                 and (unbounded(p["rotation_sigma_deg_t"])
                      or p["rotation_sigma_deg_t"] > max_rotation_sigma_deg))
                or (max_centre_sigma is not None
                    and (unbounded(p["centre_sigma_t"]) or p["centre_sigma_t"] > max_centre_sigma)))

    def rank(k):
        x = table[k]["rotation_sigma_deg"]
        return np.inf if unbounded(x) else x
    within = not any(over(p) for p in table.values())
    worst_key = max(table, key=rank) if table else None
    # strict JSON for the acceptance file: an unidentified sigma is recorded as null
    for row in table.values():
        for k, x in row.items():
            if isinstance(x, float) and not np.isfinite(x):
                row[k] = None
    return {
        "estimator": "frame-clustered M-estimator sandwich (Schur-marginalised)",
        "units": {"rotation": "deg", "centre": "scene length units (square_size)"},
        "n_frame_clusters": unc["n_clusters"],
        "dof": dof,
        "t_factor": t_factor,
        "interval_note": "k-sigma intervals use sigma * t_factor (Student-t, dof = frames - 1)",
        "structure_mode": structure_mode,
        "structure_modelled": structure_mode != "free",
        "pairs": table,
        "limits": {"max_rotation_sigma_deg": max_rotation_sigma_deg,
                   "max_centre_sigma": max_centre_sigma},
        "identifiable": bool(unc["identifiable"]) and all(p["identifiable"] for p in table.values()),
        "within_limits": within,
        "worst": None if worst_key is None else {"pair": worst_key, **{
            k: table[worst_key][k] for k in ("rotation_sigma_deg", "centre_sigma",
                                             "rotation_sigma_deg_t", "centre_sigma_t")}},
    }


def intrinsics_error(rig: RigState, ref_K: Dict[int, np.ndarray]) -> Dict[int, Dict[str, float]]:
    """Per-camera intrinsic error vs a reference ``K`` per camera, model-independent.

    Compares the **paraxial focal** ``f_eff`` (the focal that means the same in every model,
    so a camera calibrated with DS / UCM / EUCM is checked against the same physical lens) and
    the principal point. Returns ``{ref_fx, ref_fy, opt_fx, opt_fy, focal_pct, cx_pct, cy_pct}``.
    """
    out = {}
    for c in rig.cameras:
        if c not in ref_K or ref_K[c] is None:
            continue
        Kr = np.asarray(ref_K[c], float)
        rfx, rfy, rcx, rcy = Kr[0, 0], Kr[1, 1], Kr[0, 2], Kr[1, 2]
        fx, fy = paraxial_focal(rig.cameras[c])
        Kc = rig.cameras[c].K
        out[c] = {
            "ref_fx": float(rfx), "ref_fy": float(rfy), "opt_fx": float(fx), "opt_fy": float(fy),
            "focal_pct": 100.0 * max(abs(fx - rfx) / rfx, abs(fy - rfy) / rfy),
            "cx_pct": 100.0 * abs(Kc[0, 2] - rcx) / abs(rcx) if rcx else float("nan"),
            "cy_pct": 100.0 * abs(Kc[1, 2] - rcy) / abs(rcy) if rcy else float("nan"),
        }
    return out


def _rel(Tref, Ti):
    return Ti @ np.linalg.inv(Tref)


def baseline_error_per_camera(rig: RigState, scn: Scenario) -> Dict[int, float]:
    """Deprecated baseline-length-only diagnostic vs GT, retained for compatibility.

    It is not an extrinsic-accuracy metric: it cannot observe rotation or translation
    direction.  New callers should use ``metrics["accuracy"]["ground_truth"]``.
    Blender GT poses are world->camera, matching ``T_c_g``.
    """
    ref = rig.ref_cam_id
    gt = scn.gt
    if ref not in gt:
        return {}
    out = {}
    for c in rig.T_c_g:
        if c == ref or c not in gt:
            continue
        mine = _rel(rig.T_c_g[ref], rig.T_c_g[c])
        other = _rel(gt[ref].pose, gt[c].pose)
        b_other = np.linalg.norm(other[:3, 3])
        out[c] = 100.0 * abs(np.linalg.norm(mine[:3, 3]) - b_other) / b_other if b_other > 1e-9 \
            else float("nan")
    return out


def _scenario_metrics(rig: RigState, scn: Scenario,
                      accuracy_profile: Optional[AccuracyProfile] = None) -> Dict:
    """Training-fit metrics plus full-pose accuracy vs available references.

    Blender GT stores world->camera while MC-Calib output stores camera->world. Legacy baseline
    fields remain temporarily as deprecated compatibility output; they must not be interpreted
    as pose accuracy.
    """
    rms = bundle.reprojection_rms(rig, scn.object_obs)
    ref = rig.ref_cam_id

    def worst_baseline(ref_poses: Dict[int, np.ndarray], invert: bool) -> Optional[float]:
        """Largest inter-camera baseline error (%) vs a reference pose set.

        Compares each non-reference camera's baseline relative to ``ref`` (from
        ``rig.T_c_g``) against the same relative baseline in ``ref_poses``.
        ``invert=True`` treats ``ref_poses`` as camera->world (GT/MC-Calib
        convention) and inverts before comparing. Returns ``None`` if ``ref`` has
        no entry in ``ref_poses`` or there is no overlapping camera to compare.
        """
        common = [c for c in rig.T_c_g if c in ref_poses and c != ref] if ref in ref_poses else []
        if not common:
            return None
        out = 0.0
        for c in common:
            mine = _rel(rig.T_c_g[ref], rig.T_c_g[c])
            R = ref_poses
            other = _rel(R[ref], R[c]) if not invert else _rel(np.linalg.inv(R[ref]),
                                                               np.linalg.inv(R[c]))
            b_mine = np.linalg.norm(mine[:3, 3])
            b_other = np.linalg.norm(other[:3, 3])
            if b_other > 1e-9:
                out = max(out, abs(b_mine - b_other) / b_other)
        return 100.0 * out

    gt_poses = {c: g.pose for c, g in scn.gt.items()} if scn.gt else {}
    mc_poses = {c: m.pose for c, m in scn.mccalib.items()} if scn.mccalib else {}
    full_accuracy = accuracy_metrics(
        rig.T_c_g,
        ground_truth_world_to_camera=gt_poses,
        mccalib_camera_to_world=mc_poses,
        profile=accuracy_profile,
    )
    return {
        "rms_px": {c: float(rms[c]) for c in rms},
        "max_rms_px": float(max(rms.values())) if rms else float("nan"),
        # Blender GroundTruth.yml P_i is world->camera, matching T_c_g.
        "worst_baseline_pct_vs_gt": worst_baseline(gt_poses, invert=False),
        "worst_baseline_pct_vs_mccalib": worst_baseline(mc_poses, invert=True),
        "accuracy": full_accuracy,
        "compatibility": {
            "deprecated_fields": {
                "worst_baseline_pct_vs_gt":
                    "baseline length only; use accuracy.ground_truth",
                "worst_baseline_pct_vs_mccalib":
                    "baseline length only; use accuracy.mccalib_reference",
            },
            "removal": "next major release",
        },
    }
