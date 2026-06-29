"""Top-level N-camera rig calibration orchestrator — the ``runCalibrationWorkflow``
analogue (apps/calibrate/src/calibrate.cpp:9), trimmed to DS-MSP's scope.

Stages: per-camera intrinsics + object poses (front-end) -> camera-group covisibility
+ extrinsics init -> object-in-group pose averaging -> staged global BA (poses, then
full joint incl. intrinsics). Non-overlapping groups are linked by hand-eye when
present (``rig.handeye``).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..calib.bundle import calibrate as _calibrate_single
from ..geometry.resection import intrinsics_seed, ransac_pnp_normalized
from ..core.contracts import CameraModel
from ..models.radtan import RadTanModel
from . import bundle
from .extrinsics import init_camera_groups
from .pose_init import (average_object_pose_in_group, robust_pose_irls)
from .types import Object3D, ObjectObs, RigState


def _robust_pinhole(objpts, imgpts, w, h):
    """From-scratch robust pinhole intrinsic seed — RANSAC DLT resection, **no OpenCV**.

    Plain ``cv2.calibrateCamera`` is L2: a few 40 px blunders drag the focal to garbage,
    and the post-hoc residual gate keys off that already-corrupted fit, so robustness
    collapses past ~6-10 % gross outliers. Here the robustness lives in the *seed*: each
    view's 3x4 camera matrix is fit by RANSAC over a linear DLT on the genuinely-3D target
    (:func:`calib.robust_init.intrinsics_seed`), which rejects blunders by construction, and
    the focal/principal-point seed is the robust median of the inlier views' RQ-decomposed
    ``K``. The downstream per-model ``calibrate`` jointly refines focal+distortion under
    IRLS. Returns ``(K, dist)`` with ``dist`` zeroed (distortion is fit per-model later).
    """
    op = [np.asarray(o, float) for o in objpts]
    ip = [np.asarray(p, float) for p in imgpts]
    K, _poses = intrinsics_seed(op, ip, w, h)              # robust focal/pp seed (DLT RANSAC)
    # Refine the pinhole seed with a robust RadTan bundle on the same correspondences. The
    # linear DLT fits *no* distortion, so its focal is biased for anything but a pinhole;
    # a Brown/RadTan refine (which `calibrate` does from-scratch under a Cauchy kernel)
    # removes that bias — matching what cv2.calibrateCamera gave but without its L2 fragility,
    # since the pose seeds are RANSAC and the kernel down-weights the surviving blunders.
    seed = RadTanModel(K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0.0, 0.0, 0.0, 0.0, 0.0)
    vis = [np.ones(len(o), bool) for o in op]
    try:
        res = _calibrate_single(seed, op, ip, vis, loss="cauchy", f_scale=1.0, max_nfev=80)
        Kr = res["model"].K
        if np.isfinite(Kr).all() and Kr[0, 0] > 0 and Kr[1, 1] > 0:
            K = Kr
    except (np.linalg.LinAlgError, ValueError):
        pass
    return K, np.zeros(5)


def _T_from_rt(rvec, tvec) -> np.ndarray:
    """4x4 object->camera transform from an OpenCV (rvec, tvec) pair."""
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float))[0]
    T[:3, 3] = np.asarray(tvec, float).ravel()
    return T


def _set_poses(obs: List[ObjectObs], rvecs, tvecs) -> None:
    for o, rv, tv in zip(obs, rvecs, tvecs):
        o.T_c_o = _T_from_rt(rv, tv)


def _front_end_opencv(obj: Object3D, obs_by_cam: Dict[int, List[ObjectObs]],
                      img_size: Dict[int, Tuple[int, int]]):
    """Per-camera intrinsics + per-frame object poses via ``cv2.calibrateCamera``
    (Brown/RadTan, distortion_type 0). Fills each ObjectObs' ``T_c_o``.

    A camera that only ever sees a single near-planar board (and at low tilt) suffers
    the focal/distance ambiguity, so ``calibrateCamera`` can return an implausible focal.
    Such cameras are detected (focal far outside the image-plausible range) and re-seeded
    from the consensus of the well-constrained cameras; the global BA then refines their
    intrinsics through the rigid-rig constraint. The seed only needs to land the camera's
    pose in the right basin.
    """
    raw: Dict[int, dict] = {}
    for cam_id, obs in obs_by_cam.items():
        objpts = [obj.pts_3d[o.point_rows].astype(np.float32) for o in obs]
        imgpts = [o.pts_2d.astype(np.float32) for o in obs]
        w, h = img_size[cam_id]
        K0 = np.array([[float(w), 0.0, w / 2.0],
                       [0.0, float(w), h / 2.0],
                       [0.0, 0.0, 1.0]])
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpts, imgpts, img_size[cam_id], K0, None,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        diag = float(np.hypot(w, h))
        plausible = 0.2 * diag < K[0, 0] < 4.0 * diag and 0.2 * diag < K[1, 1] < 4.0 * diag
        raw[cam_id] = dict(K=K, dist=dist.ravel(), rvecs=rvecs, tvecs=tvecs,
                           objpts=objpts, imgpts=imgpts, ok=plausible)

    good = [r["K"] for r in raw.values() if r["ok"]]
    consensus = np.median(np.stack(good), axis=0) if good else None

    cameras: Dict[int, CameraModel] = {}
    for cam_id, obs in obs_by_cam.items():
        r = raw[cam_id]
        if r["ok"] or consensus is None:
            K, d, rvecs, tvecs = r["K"], r["dist"], r["rvecs"], r["tvecs"]
        else:
            # degenerate camera: adopt the consensus intrinsics and re-solve its poses
            K = consensus.copy()
            d = np.zeros(5)
            rvecs, tvecs = [], []
            for op, ip in zip(r["objpts"], r["imgpts"]):
                ok, rv, tv = cv2.solvePnP(op, ip, K, d)
                rvecs.append(rv if ok else np.zeros(3))
                tvecs.append(tv if ok else np.array([0.0, 0.0, 1.0]))
        cameras[cam_id] = RadTanModel(K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                                      d[0], d[1], d[2], d[3], d[4])
        _set_poses(obs, rvecs, tvecs)
    return cameras


# Neutral seed values per intrinsic parameter name — a generic, GT-free starting point
# for from-scratch single-camera calibration of any model. Distortion/shape params not
# listed here seed to 0.0 (see ``_seed_from_K``), so any model — including DS+/EUCM+ whose
# extra terms (lambda1/lambda2 division-distortion, tau_x/tau_y tilt) are neutral at 0 —
# is supported without enumerating every parameter name.
_NEUTRAL = {"alpha": 0.5, "xi": 0.0, "beta": 1.0, "k1": 0.0, "k2": 0.0, "k3": 0.0,
            "k4": 0.0, "p1": 0.0, "p2": 0.0,
            "lambda1": 0.0, "lambda2": 0.0, "tau_x": 0.0, "tau_y": 0.0}


def _seed_from_K(model_cls, K: np.ndarray) -> CameraModel:
    """Build a seed instance of ``model_cls`` from a pinhole ``K`` (focal + principal
    point) with neutral distortion — the from-scratch starting point for any model."""
    if not set(model_cls.param_names) >= {"fx", "fy"}:
        # Non-pinhole-parameterized model (e.g. OCam: cx,cy + projection polynomial). It
        # has no fx/fy; seed cx,cy + a focal-scaled leading polynomial term and let
        # `initialize_from_correspondences` fit the rest from rays.
        return model_cls(K[0, 2], K[1, 2])
    vals = {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2], **_NEUTRAL}
    # Unknown shape/distortion params seed to 0.0 (neutral) so any model is supported.
    vec = np.array([vals.get(n, 0.0) for n in model_cls.param_names], float)
    return model_cls.from_params(vec)


def paraxial_focal(model: CameraModel) -> Tuple[float, float]:
    """The *model-independent* focal length f_eff = dr/dθ|₀ (paraxial focal), in pixels.

    A model's stored ``fx`` is model-relative — `fx` means different things in different
    projections, so two correct calibrations of the same lens have different `fx`
    (docs/learn/are_two_models_the_same_camera.md). For Double Sphere the axial sphere
    shift gives ``f_eff = fx / (1 + xi)``; KB/UCM/EUCM/RadTan/pinhole have ``f_eff = fx``
    at the optical axis. Compare cameras (and seed focals) by this, never by raw ``fx``.
    """
    p = dict(zip(model.param_names, model.params))
    if "fx" not in p or "fy" not in p:  # model without explicit fx/fy (e.g. OCam): use K
        K = model.K
        return float(K[0, 0]), float(K[1, 1])
    fx, fy = p["fx"], p["fy"]
    if "xi" in p:                       # Double Sphere: paraxial = fx / (1 + xi)
        s = 1.0 + p["xi"]
        if abs(s) > 1e-9:
            return fx / s, fy / s
    return fx, fy


def _model_aware_seed(model_cls, Kp, ge6, obj) -> CameraModel:
    """Seed ``model_cls`` using each model's OWN intrinsic geometry.

    The pinhole pre-calibration gives the paraxial focal + principal point in ``Kp``. Each
    model then solves its native distortion from true ray↔pixel correspondences via
    ``initialize_from_correspondences`` (KB: LS on the θ-polynomial; DS/UCM/EUCM: linear
    α-solve from the projection equation), rather than a generic ``alpha=0.5`` guess that
    would seed a sub-optimal basin. The per-view pose + **inlier mask** come from the
    from-scratch :func:`calib.robust_init.ransac_pnp_normalized` (pixels unprojected through
    the pinhole ``Kp``), so gross outliers neither corrupt the linear α/k solve nor poison
    the downstream ``calibrate`` pose seeding. The seed only needs to be good enough; the
    downstream robust ``calibrate`` refines from it.
    """
    fx, fy, cx, cy = Kp[0, 0], Kp[1, 1], Kp[0, 2], Kp[1, 2]
    foc = 0.5 * (fx + fy)
    rays, pix, Xcal, uvcal = [], [], [], []
    for o in ge6:
        X = obj.pts_3d[o.point_rows].astype(np.float64)
        uv = o.pts_2d.astype(np.float64)
        pn = np.column_stack([(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy])
        T, inl = ransac_pnp_normalized(X, pn, focal=foc, thresh_px=3.0)
        if T is None or inl.sum() < 6:
            continue
        R, t = T[:3, :3], T[:3, 3]
        Xc = X[inl] @ R.T + t
        rays.append(Xc / np.linalg.norm(Xc, axis=1, keepdims=True))
        pix.append(uv[inl])
        Xcal.append(X[inl])
        uvcal.append(uv[inl])         # inlier set for calibrate
    seed = _seed_from_K(model_cls, Kp)
    if rays and sum(len(r) for r in rays) >= 6:
        seed.initialize_from_correspondences(Kp, np.vstack(rays), np.vstack(pix))
    return seed, Xcal, uvcal


def _resolve_model_map(model_spec, cam_ids) -> Dict[int, type]:
    """Resolve ``model_spec`` to a ``{cam_id: model_cls}`` map.

    Accepts a single model class / name string (every camera uses it — the model-agnostic
    case), or a dict ``{cam_id: class-or-name}`` to give each camera its **own** model, the
    MC-Calib ``camera_models`` / ``distortion_per_camera`` behaviour (camera 0 can be DS,
    camera 1 KB, ...). Name strings are resolved through :mod:`ds_msp.models.registry`, so
    MC-Calib spellings (``double_sphere``) and DS-MSP spellings (``ds``) both work.
    """
    from ..models.registry import model_class
    if isinstance(model_spec, dict):
        return {c: (m if isinstance(m, type) else model_class(m))
                for c, m in ((c, model_spec[c]) for c in cam_ids)}
    cls = model_spec if isinstance(model_spec, type) else model_class(model_spec)
    return {c: cls for c in cam_ids}


def make_bundle_front_end(model_spec, *, loss: str = "cauchy", f_scale: float = 1.0,
                          max_nfev: int = 150, init_K: Optional[Dict[int, np.ndarray]] = None):
    """Build a front-end that calibrates each camera with its chosen model.

    ``model_spec`` is either one model (class or name) used for **every** camera, or a
    ``{cam_id: model}`` map giving each camera its own model — the MC-Calib per-camera
    ``camera_models`` behaviour (camera 0 → DS, camera 1 → KB, ...). Seeding is data-driven
    and model-independent: a from-scratch robust pinhole pre-calibration (RANSAC DLT) gives
    a focal / principal-point seed, then the DS-MSP single-camera bundle adjuster
    (``calib.bundle.calibrate``) refines the *chosen model's* full parameter vector from
    that seed. Avoiding a blind focal sweep matters for models whose distortion can absorb a
    focal-seed error (e.g. KB). Returns a callable with the ``front_end`` signature used by
    :func:`calibrate_rig`.

    ``init_K`` (``{cam_id: 3x3}``) supplies a per-camera focal / principal-point **seed** from
    a known intrinsics file — MC-Calib's ``cam_params_path`` initialization. When given for a
    camera it replaces the from-scratch pinhole pre-calibration (which is biased on a real
    strong-fisheye lens) and that camera bypasses the cross-camera focal-consensus reset (the
    consensus median is meaningless across a mixed-resolution rig, e.g. 2592 + 640 px cameras).
    The chosen model still fits its native distortion and the global BA still refines unless
    intrinsics are fixed.
    """
    def front_end(obj, obs_by_cam, img_size):
        model_map = _resolve_model_map(model_spec, list(obs_by_cam))
        seeded = set(init_K) if init_K else set()
        raw = {}
        for cam_id, obs in obs_by_cam.items():
            model_cls = model_map[cam_id]
            w, h = img_size[cam_id]
            ge6 = [o for o in obs if len(o.point_rows) >= 6]   # views usable for intrinsics
            objpts = [obj.pts_3d[o.point_rows].astype(np.float32) for o in ge6]
            imgpts = [o.pts_2d.astype(np.float32) for o in ge6]
            # Focal / principal-point seed: the provided intrinsics if given (MC-Calib's
            # cam_params_path init), else a from-scratch robust pinhole pre-calibration.
            if cam_id in seeded:
                Kp = np.asarray(init_K[cam_id], float)
            else:
                Kp, _distp = _robust_pinhole(objpts, imgpts, w, h)
            # Model-aware seed + RANSAC-inlier correspondences (so gross outliers neither
            # corrupt the per-model distortion solve nor wreck calibrate()'s pose seeding).
            seed_ma, Xcal, uvcal = _model_aware_seed(model_cls, Kp, ge6, obj)
            if len(Xcal) >= 3:
                vis = [np.ones(len(x), bool) for x in Xcal]
                # Two-start: the model-aware distortion solve helps low-order models (DS/
                # UCM/EUCM α) but a high-order fit (KB's 4-term θ-polynomial) from slightly
                # biased pinhole bearings can seed *worse* than neutral. Calibrate from both
                # and keep the lower-RMS fit, so the geometry-aware seed is used only when it
                # actually helps and never degrades a model whose fx is already paraxial.
                cands = [_calibrate_single(s, Xcal, uvcal, vis, loss=loss, f_scale=f_scale,
                                           max_nfev=max_nfev)
                         for s in (seed_ma, _seed_from_K(model_cls, Kp))]
                # Focal-collapse anchor: the robust pinhole pre-calibration (RANSAC DLT) gives
                # a reliable paraxial focal even under gross outliers, but the per-model Cauchy
                # refine can still slide into a tiny-focal local minimum that absorbs surviving
                # blunders into distortion (a low-RMS but wrong fit). Reject any candidate whose
                # paraxial focal departs the seed by >2x; if both collapse, fall back to the
                # seed focal with neutral distortion and let the global BA fit distortion
                # through the rigid-rig constraint. The band is wide enough never to fire on a
                # genuine fisheye (its f_eff still tracks the perspective seed near the axis),
                # and being per-camera + absolute it survives an all-cameras collapse that the
                # median-based consensus guard below cannot.
                foc_seed = 0.5 * (Kp[0, 0] + Kp[1, 1])

                def _focal_ok(m):
                    fx, fy = paraxial_focal(m)
                    return (0.5 * foc_seed < fx < 2.0 * foc_seed
                            and 0.5 * foc_seed < fy < 2.0 * foc_seed)

                ok = [r for r in cands if _focal_ok(r["model"])]
                model = (min(ok, key=lambda r: r["rms_px"])["model"] if ok
                         else _seed_from_K(model_cls, Kp))
            else:
                model = seed_ma
            raw[cam_id] = dict(model=model, obs=obs, cls=model_cls)

        # Consensus guard: a camera that views the target near-planar (e.g. an obliquely
        # angled camera seeing one board) hits the focal ambiguity and calibrates to a wrong
        # / anamorphic focal. Detect such cameras by deviation from the per-rig median
        # *paraxial* focal (the model-independent f_eff — comparing raw fx across a model
        # with non-trivial axial terms is meaningless), reset to the consensus, and let the
        # global BA refine them through the rigid-rig constraint.
        feff = {c: paraxial_focal(raw[c]["model"]) for c in raw}
        med_fx = float(np.median([feff[c][0] for c in raw]))
        med_fy = float(np.median([feff[c][1] for c in raw]))

        cameras = {}
        for cam_id, r in raw.items():
            fx, fy = feff[cam_id]
            if (cam_id not in seeded and len(raw) >= 2
                    and (abs(fx - med_fx) > 0.25 * med_fx
                         or abs(fy - med_fy) > 0.25 * med_fy)):
                w, h = img_size[cam_id]
                Kc = np.array([[med_fx, 0, w / 2.0], [0, med_fy, h / 2.0], [0, 0, 1.0]])
                model = _seed_from_K(r["cls"], Kc)
            else:
                model = r["model"]
            cameras[cam_id] = model
            # all object poses via robust gated PnP (keeps every point downstream; the
            # global BA does the IRLS weighting, no per-point rejection in the answer).
            for o in r["obs"]:
                o.T_c_o = _gated_pnp(model, obj.pts_3d[o.point_rows], o.pts_2d)
        return cameras
    return front_end


def _gated_pnp(model, X, uv, max_rms_px: float = 2.0):
    """Per-view pose by **robust reweighting, not rejection** (the down-weight-don't-drop
    philosophy, matching the global BA and the robust PnP path).

    A RANSAC P3P warm-start seeds a redescending IRLS refinement over *every* corner
    (:func:`pose_init.robust_pose_irls`): gross outliers get a vanishing weight instead of
    being discarded, so a partly-corrupted view still contributes its good corners to the
    extrinsics graph rather than being thrown away. Returns ``None`` only when the view has
    too few unprojectable points to define any pose (insufficient data, MC-Calib's <4 rule)
    — not as outlier rejection. ``max_rms_px`` is accepted for signature compatibility and
    no longer gates the result."""
    return robust_pose_irls(model, X, uv, kernel="cauchy", gnc_iters=5, gnc_start=4.0,
                            studentize=True)


def calibrate_rig(obj: Object3D, object_obs: List[ObjectObs],
                  img_size: Dict[int, Tuple[int, int]],
                  *, fix_intrinsics: bool = False, verbose: bool = False,
                  front_end: Optional[Callable] = None, he_approach: int = 0,
                  refine_structure: bool = False, structure_rounds: int = 6,
                  gnc_iters: int = 5, gnc_start: float = 4.0) -> RigState:
    """Calibrate a multi-camera rig from fused-object observations.

    Returns a :class:`RigState` with per-camera intrinsics, ``T_c_g`` extrinsics
    (reference camera = identity), and per-frame object poses.
    """
    obs_by_cam: Dict[int, List[ObjectObs]] = defaultdict(list)
    for o in object_obs:
        obs_by_cam[o.cam_id].append(o)
    cam_ids = sorted(obs_by_cam)

    # 1. per-camera intrinsics + object poses (T_c_o). Default to the robust from-scratch
    #    front-end (RANSAC-DLT seed + per-model Cauchy refine), not the plain-L2
    #    cv2.calibrateCamera path (`_front_end_opencv`), which collapses the focal under
    #    gross outliers — so a direct calibrate_rig caller gets the robust behaviour the
    #    high-level entry points already use, without having to know to pass front_end.
    fe = front_end or make_bundle_front_end(RadTanModel)
    cameras = fe(obj, obs_by_cam, img_size)
    if verbose:
        print(f"[front-end] calibrated {len(cameras)} cameras")

    # 2. camera-group covisibility -> extrinsics init (T_c_g; ref cam = identity)
    groups, extr = init_camera_groups(object_obs, cam_ids)
    if verbose:
        print(f"[groups] {len(groups)} group(s): {groups}")
    if len(groups) > 1:
        # Non-overlapping groups: link with hand-eye, then re-base every camera to the
        # global reference (group 0's reference camera).
        from .handeye import link_groups
        extr = link_groups(groups, extr, object_obs, he_approach=he_approach)
    ref_cam = groups[0][0]

    # 3. per-frame object-in-group poses (average over the cameras that see it)
    by_fo: Dict[Tuple[int, int], List[Tuple[int, np.ndarray]]] = defaultdict(list)
    for o in object_obs:
        if o.T_c_o is not None and o.cam_id in extr:
            by_fo[(o.object_id, o.frame_id)].append((o.cam_id, o.T_c_o))
    object_poses: Dict[Tuple[int, int], np.ndarray] = {}
    for key, lst in by_fo.items():
        object_poses[key] = average_object_pose_in_group(lst, extr, ref_cam)

    rig = RigState(cameras=cameras, T_c_g=extr, ref_cam_id=ref_cam,
                   object_poses=object_poses, objects={0: obj}, img_size=img_size)

    # 4. hierarchical refinement, MC-Calib's staged structure, every stage an analytic-
    #    Jacobian BA (no autodiff), robust IRLS weighting (no rejection):
    #    (a) per-object — refine each frame's object pose with cameras+intrinsics fixed
    #        (estimatePoseAllObjects / computeAllObjPoseInCameraGroup): a metric BA warm-up
    #        of the closed-form averaged object poses before any extrinsic moves.
    rig = bundle.refine(rig, object_obs, fix_intrinsics=True, fix_extrinsics=True,
                    robust_kernel="huber", robust_scale="auto", verbose=verbose)
    #    (b) per-camera-group — refine each group's extrinsics + its object poses, intrinsics
    #        fixed (calibrateCameraGroup / refineAllCameraGroupAndObjects). Single group ->
    #        whole-rig poses-only; multiple groups -> each independently before the joint pass.
    rig = bundle.refine_groups(rig, object_obs, groups,
                           robust_kernel="huber", robust_scale="auto", verbose=verbose)
    #    (c) global joint — full rig + (optionally) intrinsics with a redescending Cauchy
    #        kernel and a short GNC anneal (refineAllCameraGroupAndObjectsAndIntrinsics).
    # Graduated non-convexity: anneal the robust scale from ``gnc_start``×MAD down. A wider
    # start + more steps escapes the bad-data basin under heavy (≈50%) gross-outlier
    # contamination, where a MAD scale is itself corrupted — measurably more robust at the
    # breakdown point with negligible cost on clean data.
    rig = bundle.refine(rig, object_obs, fix_intrinsics=fix_intrinsics, robust_kernel="cauchy",
                    robust_scale="auto", gnc_iters=gnc_iters, gnc_start=gnc_start,
                    verbose=verbose)

    #    (d) object-structure refinement (MC-Calib's refineObject), multi-board only: alternate
    #        re-triangulating the fused object's 3-D points (gauge anchored on the reference
    #        board) with the joint BA. A reconstructed nominal object carries inter-board pose
    #        and physical-board error that no camera/pose update can absorb; freeing the
    #        structure removes it — the dominant reprojection win on a real multi-board target.
    if refine_structure and len(obj.board_ids) > 1:
        prev = float("inf")
        for _ in range(structure_rounds):
            rig = bundle.refine_object_structure(rig, object_obs)
            # a light joint polish each round (no GNC, capped iters): the structure step did
            # the heavy lifting, so the BA only has to absorb it — full-length GNC passes here
            # were ~80% of total runtime for <0.5% extra accuracy.
            rig = bundle.refine(rig, object_obs, fix_intrinsics=fix_intrinsics,
                            robust_kernel="cauchy", robust_scale="auto", max_iter=25,
                            verbose=verbose)
            cur = max(bundle.reprojection_rms(rig, object_obs).values())
            if prev - cur < 0.003 * prev:           # converged -> stop early
                break
            prev = cur
    return rig
