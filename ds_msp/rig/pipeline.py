"""Drive the rig calibration on a real MC-Calib dataset, MC-Calib style.

This is the orchestration MC-Calib's ``apps/calibrate`` performs, in DS-MSP terms:
choose a model per camera (the ``camera_models`` config field), optionally optimize
extrinsics-only or intrinsics+extrinsics (``fix_intrinsic``), then write the result in
MC-Calib's exact OpenCV-YAML format. Used by ``scripts/calibrate_rig.py`` and the
cross-dataset validation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..io.mccalib import (Scenario, save_detection_images, save_mccalib_results,
                          save_reprojection_images)
from ..models.registry import (FISHEYE_MODELS, PINHOLE_MODELS)
from . import bundle
from .calibrate import (_gated_pnp, calibrate_rig, make_bundle_front_end,
                            paraxial_focal)
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
                       noise_bound: Optional[float] = None,
                       verbose: bool = False, on_iter=None,
                       objects: Optional[list] = None,
                       reproj_gate_px: Optional[float] = None,
                       report_covariance: bool = False) -> Dict:
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
    Returns ``{rig, models, paths, metrics}`` (``+ "covariance"`` when requested).
    """
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
                        he_approach=he_approach, refine_structure=refine_structure,
                        noise_bound=noise_bound, verbose=verbose, on_iter=on_iter,
                        objects=objects, reproj_gate_px=reproj_gate_px)
    # keep the (possibly structure-refined) object the rig actually solved with
    refined_object = rig.objects.get(scn.object.object_id, scn.object)

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
    metrics = _scenario_metrics(rig, scn)
    out = {"rig": rig, "models": {c: rig.cameras[c].name for c in rig.cameras},
          "paths": paths, "metrics": metrics}
    if report_covariance:
        out["covariance"] = bundle.parameter_covariance(
            rig, scn.object_obs, fix_intrinsics=fix_intrinsics)
    return out


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
    """Per-camera inter-camera baseline error (%) vs GT. GT poses are camera->world;
    ``T_c_g`` is world->camera, so invert before forming the relative transform."""
    ref = rig.ref_cam_id
    gt = scn.gt
    if ref not in gt:
        return {}
    out = {}
    for c in rig.T_c_g:
        if c == ref or c not in gt:
            continue
        mine = _rel(rig.T_c_g[ref], rig.T_c_g[c])
        other = _rel(np.linalg.inv(gt[ref].pose), np.linalg.inv(gt[c].pose))
        b_other = np.linalg.norm(other[:3, 3])
        out[c] = 100.0 * abs(np.linalg.norm(mine[:3, 3]) - b_other) / b_other if b_other > 1e-9 \
            else float("nan")
    return out


def _scenario_metrics(rig: RigState, scn: Scenario) -> Dict:
    """Reprojection RMS per camera + worst inter-camera baseline error vs GT / MC-Calib.

    GT/MC-Calib store camera->world poses; ``rig.T_c_g`` is world->camera, so invert before
    comparing baselines."""
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
    return {
        "rms_px": {c: float(rms[c]) for c in rms},
        "max_rms_px": float(max(rms.values())) if rms else float("nan"),
        # GT P_i is camera->world; T_c_g is world->camera -> invert to compare baselines.
        "worst_baseline_pct_vs_gt": worst_baseline(gt_poses, invert=True),
        "worst_baseline_pct_vs_mccalib": worst_baseline(mc_poses, invert=True),
    }
