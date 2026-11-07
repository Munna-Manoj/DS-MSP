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

from ..core.optimize import CONVERGED, NUMERICAL_FAILURE
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


#: 2-DoF 99% chi-squared radius in sigmas: the same inlier band the GNC-TLS solve uses.
TLS_BAND_SIGMAS = 3.03
#: Lowest share of a camera's corners the estimate may leave unexplained before the coverage
#: gate acts. On the real-data benchmark the worst camera of 140 model trials is at 0.868 (a
#: wide lens fitted with RadTan), and of the injected faults that changed the data used, at most
#: 0.503. These are few lenses, so it only warns by default, also under the production profile,
#: until it has been checked across fields of view.
DEFAULT_MIN_CAMERA_COVERAGE = 0.75


def random_model_assignment(cam_ids: List[int], *, kind: str = "pinhole",
                            seed: int = 0, allowed=None) -> Dict[int, str]:
    """Assign each camera a random *valid* model name.

    ``kind="pinhole"`` draws from {radtan, ucm, eucm, ds} (KB excluded — it is
    fisheye-only); ``kind="fisheye"`` from {kb, ucm, eucm, ds} (RadTan excluded — it
    cannot represent a fisheye). The sphere models handle both. Pass ``allowed``
    to restrict the pool.
    """
    pool = list(allowed) if allowed is not None else list(
        PINHOLE_MODELS if kind == "pinhole" else FISHEYE_MODELS)
    rng = np.random.default_rng(seed)
    return {c: str(rng.choice(pool)) for c in sorted(cam_ids)}


def make_fixed_intrinsic_front_end(cameras: Dict[int, object]):
    """Use given per-camera intrinsics and estimate object poses with robust gated PnP.

    This front-end never recalibrates the supplied lens models. Subsequent bundle
    adjustment may hold or refine them according to the caller's ``fix_intrinsics``.
    """
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


def make_initialized_front_end(cameras: Dict[int, object], model_spec, *, init_K=None):
    """Use complete supplied lenses; calibrate only cameras missing from the file.

    Supplying a lens is initialization, not a constraint on subsequent bundle
    adjustment. The caller still controls that with ``fix_intrinsics``. Keeping
    the two subsets separate also prevents a focal-consensus reset from replacing
    an independently calibrated lens in a heterogeneous rig.
    """
    supplied = make_fixed_intrinsic_front_end(cameras)
    estimated = make_bundle_front_end(model_spec, init_K=init_K)

    def front_end(obj, obs_by_cam, img_size):
        known = {c: obs for c, obs in obs_by_cam.items() if c in cameras}
        missing = {c: obs for c, obs in obs_by_cam.items() if c not in cameras}
        result = supplied(obj, known, img_size) if known else {}
        report = {c: {"source": "supplied"} for c in known}
        if missing:
            result.update(estimated(obj, missing, img_size))
            report.update(getattr(estimated, "report", {}))
        front_end.report = report
        return result

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
                       uncertainty_gate: str = "warn", convergence_gate: str = "warn",
                       coverage_gate: str = "warn",
                       min_camera_coverage: float = DEFAULT_MIN_CAMERA_COVERAGE,
                       max_rotation_sigma_deg: Optional[float] = None,
                       max_centre_sigma: Optional[float] = None,
                       config_warnings: Optional[List[str]] = None,
                       profile: str = "default",
                       lens_gate: str = "warn", crossval_gate: str = "warn",
                       max_lens_sigma_px: Optional[float] = None) -> Dict:
    """Calibrate one loaded :class:`Scenario` and (optionally) write MC-Calib output.

    ``model_spec`` is a single model or a ``{cam_id: model}`` map (names or classes).
    ``fix_intrinsics=True`` requires ``init_cameras`` for every observed camera and
    optimizes extrinsics only. Otherwise a partial ``init_cameras`` map uses the supplied
    full lens models and calibrates missing cameras from scratch. ``init_K`` (``{cam_id: 3x3}``) seeds the
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

    Solver termination (:func:`solver_acceptance`) is recorded as ``"solver"`` in the acceptance
    file. A ``NUMERICAL_FAILURE`` in any BA stage always raises :class:`RuntimeError`, because a
    non-finite cost has no meaning. ``convergence_gate`` (``warn`` default | ``refuse`` |
    ``off``) judges the final stage, the one that produced the returned estimate: ``warn``
    raises a ``RuntimeWarning`` when it did not converge, ``refuse`` blocks export. Front-end
    fallbacks and unlinked hand-eye groups always warn.

    ``coverage_gate`` (``warn`` default | ``refuse`` | ``off``) judges each camera's share of
    observed corners that project validly within the GNC-TLS inlier band
    (:func:`bundle.camera_coverage`), against ``min_camera_coverage``. Solver status and RMS
    cannot show a camera or frames lost to the robust solve; this can. Recorded as
    ``"coverage"`` in the acceptance file.

    ``lens_gate`` (``warn`` default | ``refuse`` | ``off``) judges each camera's lens
    (:func:`ds_msp.rig.lens_check.lens_uncertainty`): the widened median pixel sigma of its
    projection against ``max_lens_sigma_px`` (default ``noise_bound``, else 1 px), and lens
    parameters left at an optimizer bound. ``crossval_gate`` (same values) re-solves the rig on
    the even and the odd frames and compares each camera's two lenses
    (:func:`ds_msp.rig.lens_check.crossval_lens`) against the same limit; it costs about one
    extra joint BA. Both are skipped when intrinsics are fixed, and recorded as ``"lens"`` and
    ``"crossval"``. The observability audit's findings are recorded as ``"observability"`` and,
    under ``audit_gate: warn``, raised as a ``RuntimeWarning``.
    Returns ``{rig, models, paths, metrics}`` (``+ "covariance"``/``"audit"``/
    ``"certificate"``/``"uncertainty"``/``"lens"``/``"crossval"`` when enabled).
    """
    for name, gate in (("lens_gate", lens_gate), ("crossval_gate", crossval_gate)):
        if gate not in ("off", "warn", "refuse"):
            raise ValueError(f"{name} must be 'off', 'warn' or 'refuse', got {gate!r}")
    if max_lens_sigma_px is not None and not (np.isfinite(max_lens_sigma_px)
                                              and max_lens_sigma_px > 0):
        raise ValueError("max_lens_sigma_px must be positive")
    if uncertainty_gate not in ("off", "warn", "refuse"):
        raise ValueError(f"uncertainty_gate must be 'off', 'warn' or 'refuse', got {uncertainty_gate!r}")
    if uncertainty_gate == "refuse" and (max_rotation_sigma_deg is None or max_centre_sigma is None):
        raise ValueError("uncertainty_gate='refuse' needs max_rotation_sigma_deg and max_centre_sigma")
    if audit_gate not in ("off", "warn", "refuse"):
        raise ValueError(f"audit_gate must be 'off', 'warn' or 'refuse', got {audit_gate!r}")
    if accuracy_gate not in ("off", "refuse"):
        raise ValueError("accuracy_gate must be off or refuse")
    if convergence_gate not in ("off", "warn", "refuse"):
        raise ValueError(f"convergence_gate must be 'off', 'warn' or 'refuse', got {convergence_gate!r}")
    if coverage_gate not in ("off", "warn", "refuse"):
        raise ValueError(f"coverage_gate must be 'off', 'warn' or 'refuse', got {coverage_gate!r}")
    if not 0.0 <= min_camera_coverage <= 1.0:
        raise ValueError("min_camera_coverage must be between 0 and 1")
    if structure_prior_sigma is not None and (
            not np.isfinite(structure_prior_sigma) or structure_prior_sigma <= 0):
        raise ValueError("structure_prior_sigma must be positive in target length units")
    structure_mode = resolve_structure_mode(structure_mode, refine_structure)
    if accuracy_gate == "refuse" and accuracy_profile is None:
        raise ValueError("accuracy_gate='refuse' requires an accuracy profile and independent truth")
    if fix_intrinsics:
        missing = sorted({o.cam_id for o in scn.object_obs} - set(init_cameras or {}))
        if init_cameras is None or missing:
            raise ValueError("fix_intrinsics=True needs init_cameras (initial intrinsics) "
                             f"for every observed camera; missing: {missing}")
        front_end = make_fixed_intrinsic_front_end(init_cameras)
    elif init_cameras:
        # Start from the provided per-camera models (MC-Calib's cam_params_path init): the
        # front-end uses them directly instead of a from-scratch re-fit — essential for a
        # wide-FOV fisheye, whose from-scratch model fit can diverge. ``fix_intrinsics`` then
        # decides only whether the joint BA refines these intrinsics or holds them.
        front_end = make_initialized_front_end(init_cameras, model_spec, init_K=init_K)
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
    solver = solver_acceptance(rig)
    _gate_solver(solver, convergence_gate)
    band_px = TLS_BAND_SIGMAS * (float(noise_bound) if noise_bound else 1.0)
    coverage = bundle.camera_coverage(rig, scn.object_obs, band_px,
                                      rejected=getattr(rig, "rejected", []))
    _gate_coverage(coverage, band_px, min_camera_coverage, coverage_gate)

    metrics = _scenario_metrics(rig, scn, accuracy_profile=accuracy_profile)
    out = {"rig": rig, "models": {c: rig.cameras[c].name for c in rig.cameras},
          "paths": {}, "metrics": metrics}
    if report_covariance:
        out["covariance"] = bundle.parameter_covariance(
            rig, scn.object_obs, fix_intrinsics=fix_intrinsics)
    if certify:
        from .certify import certify_rotations
        out["certificate"] = certify_rotations(rig, scn.object_obs)
    observability = None
    if audit_gate != "off":
        from .audit import audit_rig
        audit = audit_rig(rig, scn.object_obs, fix_intrinsics=fix_intrinsics)
        out["audit"] = audit
        observability = _observability_record(audit)
        if audit["n_weak"] > 0 or not audit["gauge_ok"]:
            details = "; ".join(f["message"] for f in audit["findings"]) or \
                f"{audit['n_weak']} unnamed weak direction(s)"
            if audit_gate == "refuse":
                raise RuntimeError(
                    f"observability audit refused this calibration (audit_gate='refuse'): "
                    f"{details}")
            warnings.warn(f"observability audit: {audit['n_weak']} weakly determined parameter "
                          f"direction(s): {details}", RuntimeWarning, stacklevel=2)
    lens_limit = float(max_lens_sigma_px if max_lens_sigma_px is not None
                       else noise_bound if noise_bound else 1.0)
    unc = None
    if uncertainty_gate != "off" or (lens_gate != "off" and not fix_intrinsics):
        unc = bundle.shared_covariance(rig, scn.object_obs, fix_intrinsics=fix_intrinsics,
                                       refine_boards=structure_mode in ("rigid", "bend"),
                                       bend_boards=structure_mode == "bend")
    lens = None
    if lens_gate != "off" and not fix_intrinsics:
        from .lens_check import lens_uncertainty
        assert unc is not None
        lens = lens_uncertainty(rig, scn.object_obs, unc, limit_px=lens_limit)
        out["lens"] = lens
        _gate_lens(lens, lens_gate)
    crossval = None
    if crossval_gate != "off" and not fix_intrinsics:
        from .lens_check import crossval_lens
        crossval = crossval_lens(rig, scn.object_obs, structure_mode=structure_mode,
                                 limit_px=lens_limit, lens=lens)
        out["crossval"] = crossval
        _gate_crossval(crossval, crossval_gate)
    uncertainty = None
    if uncertainty_gate != "off":
        uncertainty = extrinsic_uncertainty(rig, scn.object_obs, fix_intrinsics=fix_intrinsics,
                                            structure_mode=structure_mode,
                                            max_rotation_sigma_deg=max_rotation_sigma_deg,
                                            max_centre_sigma=max_centre_sigma, unc=unc)
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
        "convergence_gate": convergence_gate,
        "solver": solver,
        "coverage_gate": coverage_gate,
        "min_camera_coverage": min_camera_coverage,
        "coverage_band_px": band_px,
        "coverage": {str(c): v for c, v in coverage.items()},
        "outlier_gate_px": reproj_gate_px,
        "rejected_observations": list(getattr(rig, "rejected", [])),
        "config_warnings": list(config_warnings or []),
        "observability": observability,
        "lens_gate": lens_gate,
        "lens": lens,
        "crossval_gate": crossval_gate,
        "crossval": crossval,
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


def solver_acceptance(rig: RigState) -> Dict:
    """Acceptance summary of how the calibration's solves ended, from ``rig.solver_report``
    (:func:`~ds_msp.rig.calibrate.calibrate_rig`).

    Returns the report (``stages``, ``front_end``, ``links``) plus ``final_stage`` and
    ``final_status`` (the last BA solve, which produced the estimate),
    ``numerical_failures`` (stages that hit a non-finite cost or linearisation),
    ``front_end_fallbacks`` (``{cam: reason}``) and ``unlinked_groups``."""
    rep = getattr(rig, "solver_report", None) or {}
    stages = list(rep.get("stages", []))
    final = stages[-1] if stages else None
    front = rep.get("front_end")
    return {
        "final_stage": final.get("stage") if final else None,
        "final_status": final["status"] if final else None,
        "numerical_failures": [s.get("stage") for s in stages if s["status"] == NUMERICAL_FAILURE],
        "front_end_fallbacks": {c: f["fallback"] for c, f in (front or {}).items()
                                if f.get("fallback")},
        "unlinked_groups": [ln["group"] for ln in rep.get("links", []) if not ln["linked"]],
        "stages": stages, "front_end": front, "links": list(rep.get("links", [])),
        "pose_recovery": list(rep.get("pose_recovery", [])),
        "frame_support": list(rep.get("frame_support", [])),
    }


def _gate_solver(solver: Dict, convergence_gate: str) -> None:
    """Apply decision rules to :func:`solver_acceptance`: numerical failure always refuses;
    a non-converged final stage warns or refuses per ``convergence_gate``; fallbacks warn."""
    if solver["numerical_failures"]:
        raise RuntimeError(
            f"calibration failed: non-finite cost or linearisation in BA stage(s) "
            f"{', '.join(map(str, solver['numerical_failures']))}; check the observations and "
            f"camera models for NaN/inf or points outside the model's valid domain")
    if solver["front_end_fallbacks"]:
        warnings.warn("front-end intrinsics fell back for camera(s) "
                      + ", ".join(f"{c} ({r})" for c, r in solver["front_end_fallbacks"].items())
                      + "; the global BA starts these cameras from a seed, not a fit",
                      RuntimeWarning, stacklevel=3)
    unresolved = [r for r in solver.get("frame_support", []) if not r["supported"]]
    if unresolved:
        warnings.warn(f"{len(unresolved)} object-frame pose(s) remain below the recovery "
                      "support threshold; inspect solver.frame_support before accepting "
                      "the rig. A small global median does not validate these frames.",
                      RuntimeWarning, stacklevel=3)
    if solver["final_status"] not in (None, CONVERGED) and convergence_gate != "off":
        msg = (f"the final BA stage ({solver['final_stage']}) ended {solver['final_status']} "
               f"after {solver['stages'][-1]['iterations']} iterations "
               f"(gradient ratio {solver['stages'][-1]['gradient_ratio']}); the estimate may "
               f"not be at the optimum")
        if convergence_gate == "refuse":
            raise RuntimeError(f"convergence gate refused this calibration "
                               f"(convergence_gate='refuse'): {msg}")
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _gate_coverage(coverage: Dict[int, dict], band_px: float, minimum: float, gate: str) -> None:
    """Warn or refuse when a camera's data is not explained by the estimate."""
    low = {c: v for c, v in coverage.items() if v["coverage"] < minimum}
    if not low or gate == "off":
        return
    detail = ", ".join(f"camera {c} {v['coverage']:.1%} ({v['n_within']}/{v['n_observed']})"
                       for c, v in sorted(low.items()))
    msg = (f"too few observed corners project within {band_px:.2f} px: {detail}; the minimum is "
           f"{minimum:.0%}. The solve dropped this data or diverged, so its status and RMS do "
           f"not describe these cameras")
    if gate == "refuse":
        raise RuntimeError(f"coverage gate refused this calibration (coverage_gate='refuse'): {msg}")
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _observability_record(audit: Dict) -> Dict:
    """The JSON-safe part of an audit for the acceptance file."""
    def num(x):
        return None if x is None or not np.isfinite(x) else float(x)
    findings = []
    for f in audit.get("findings", []):
        findings.append({"kind": f.get("kind"), "cam": f.get("cam"), "cams": f.get("cams"),
                         "frames": list(f["frames"]) if f.get("frames") else None,
                         "params": [str(p) for p in f.get("params", ())],
                         "variance_share": f.get("variance_share"),
                         "ratio": num(f.get("ratio")), "message": f.get("message")})
    return {"n_weak": int(audit.get("n_weak", 0)), "n_soft": int(audit.get("n_soft", 0)),
            "gauge_ok": bool(audit.get("gauge_ok", True)), "cond": num(audit.get("cond")),
            "gap": num(audit.get("gap")), "findings": findings}


def _gate_lens(lens: Dict, gate: str) -> None:
    """Warn or refuse when a camera's lens is not determined to the pixel limit."""
    if not lens["failed"] or gate == "off":
        return
    parts = []
    for c in lens["failed"]:
        v = lens["cameras"][c]
        why = []
        if "sigma" in v["reasons"]:
            s = v["sigma_median_widened_px"]
            why.append(f"lens 1-sigma {'unbounded' if s is None else f'{s:.2f} px'} "
                       f"from {v['frames']} frame(s)")
        if "at_bound" in v["reasons"]:
            why.append(f"{', '.join(v['params_at_bound'])} at an optimizer bound")
        parts.append(f"camera {c}: {'; '.join(why)}")
    msg = (f"lens not determined to {lens['limit_px']:.2f} px: {' | '.join(parts)}. Add views "
           f"of these cameras that cover their image")
    if gate == "refuse":
        raise RuntimeError(f"lens gate refused this calibration (lens_gate='refuse'): {msg}")
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _gate_crossval(cv: Dict, gate: str) -> None:
    """Warn or refuse when the two half-data solves disagree on a camera's lens."""
    if not cv.get("failed") or gate == "off":
        return
    parts = []
    for c in cv["failed"]:
        v = cv["cameras"][c]
        d = v["diff_median_observed_px"]
        exp = v.get("expected_from_noise_px")
        parts.append(f"camera {c}: halves differ by {'unbounded' if d is None else f'{d:.2f} px'}"
                     + (f", noise alone predicts {exp:.2f} px" if exp is not None else ""))
    msg = (f"lens cross-validation: re-solving on even and on odd frames gives lenses whose "
           f"difference implies a full-data lens error above {cv['limit_px']:.2f} px over the "
           f"observed image ({'; '.join(parts)}). The data do not determine these lenses; add "
           f"views, or check the camera model and the detections")
    if gate == "refuse":
        raise RuntimeError(f"cross-validation gate refused this calibration "
                           f"(crossval_gate='refuse'): {msg}")
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def extrinsic_uncertainty(rig: RigState, object_obs, *, fix_intrinsics: bool,
                          structure_mode: str = "rigid",
                          max_rotation_sigma_deg: Optional[float] = None,
                          max_centre_sigma: Optional[float] = None,
                          unc: Optional[Dict] = None) -> Dict:
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

    if unc is None:
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
