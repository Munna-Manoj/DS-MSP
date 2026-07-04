"""Top-level N-camera rig calibration orchestrator — the ``runCalibrationWorkflow``
analogue (apps/calibrate/src/calibrate.cpp:9), trimmed to DS-MSP's scope.

Stages: per-camera intrinsics + object poses (front-end) -> camera-group covisibility
+ extrinsics init -> object-in-group pose averaging -> staged global BA (poses, then
full joint incl. intrinsics). Non-overlapping groups are linked by hand-eye when
present (``rig.handeye``).
"""

from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..calib.bundle import calibrate as _calibrate_single
from ..geometry.resection import intrinsics_seed, ransac_pnp_normalized
from ..core.contracts import CameraModel
from ..models.radtan import RadTanModel
from ..models.registry import seed_from_K as _seed_from_K
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
        # force_multistart=True: RadTan's own accuracy doesn't need multi-start (that's
        # exactly what lets the front-end's Two-start step skip it — NFR-PERF-001), but THIS
        # seed feeds every downstream model, including the sphere models whose fit is
        # documented to be chaotically sensitive to which "equally good" focal a restart
        # lands on (measured: a skipped restart here flipped a DS fit between the UCM-collapse
        # basin and a genuine xi solution on real synthetic data). Keep the full sweep.
        res = _calibrate_single(seed, op, ip, vis, loss="cauchy", f_scale=1.0, max_nfev=80,
                                force_multistart=True)
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


def _fit_one_camera(model_cls, obj, obs, w, h, init_K_cam, loss, f_scale, max_nfev):
    """One camera's independent intrinsic pre-calibration — the front-end's parallelizable
    unit of work (no dependency on any other camera). Returns ``dict(model=.., cls=..)``,
    the exact per-camera result the serial loop used to build inline. Pulled out to a
    top-level function so it is picklable for :class:`~concurrent.futures.ProcessPoolExecutor`
    (see :func:`make_bundle_front_end`'s ``n_jobs``)."""
    ge6 = [o for o in obs if len(o.point_rows) >= 6]   # views usable for intrinsics
    objpts = [obj.pts_3d[o.point_rows].astype(np.float32) for o in ge6]
    imgpts = [o.pts_2d.astype(np.float32) for o in ge6]
    # Focal / principal-point seed: the provided intrinsics if given (MC-Calib's
    # cam_params_path init), else a from-scratch robust pinhole pre-calibration.
    if init_K_cam is not None:
        Kp = np.asarray(init_K_cam, float)
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
        # A model whose ``initialize_from_correspondences`` is a neutral-distortion
        # no-op (RadTan, OCam's affine terms) makes ``seed_ma`` byte-identical to
        # ``_seed_from_K`` — solving that seed twice is deterministic, wasted work,
        # not a second opinion, so skip the duplicate candidate entirely.
        start_seeds = [seed_ma]
        seed_k = _seed_from_K(model_cls, Kp)
        if not np.array_equal(seed_ma.params, seed_k.params):
            start_seeds.append(seed_k)
        cands = [_calibrate_single(s, Xcal, uvcal, vis, loss=loss, f_scale=f_scale,
                                   max_nfev=max_nfev)
                 for s in start_seeds]
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
    return dict(model=model, cls=model_cls)


def _init_pool_worker(obj: Object3D) -> None:
    """``ProcessPoolExecutor`` initializer for the front-end's per-camera pool.

    Two jobs, both done once per **worker process**, not once per camera:
    1. Force single-threaded BLAS (same fix as CI's ``OMP_NUM_THREADS=1`` for pytest-xdist —
       see ``.github/workflows/ci.yml``) so N worker processes don't each also fan out into M
       BLAS threads and oversubscribe the machine. Must happen before any BLAS-backed NumPy
       call in this fresh interpreter, which is exactly what an initializer guarantees.
    2. Stash the (potentially large, shared, read-only) fused ``Object3D`` in a module global
       so it is pickled to each worker **once**, not once per camera — the difference between
       O(workers) and O(cameras) serialization at 100s-1000s of cameras.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    global _POOL_OBJ
    _POOL_OBJ = obj


_POOL_OBJ: Optional[Object3D] = None


def _fit_one_camera_pooled(cam_id, model_cls, obs, w, h, init_K_cam, loss, f_scale, max_nfev):
    """Pool task: identical to :func:`_fit_one_camera` but reads the shared ``Object3D`` from
    the worker-local global set by :func:`_init_pool_worker` instead of a per-task argument."""
    r = _fit_one_camera(model_cls, _POOL_OBJ, obs, w, h, init_K_cam, loss, f_scale, max_nfev)
    return cam_id, r


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
                          max_nfev: int = 150, init_K: Optional[Dict[int, np.ndarray]] = None,
                          n_jobs: Optional[int] = -1):
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

    ``n_jobs`` — every camera's intrinsic pre-calibration (and, once intrinsics are known,
    every camera's per-frame pose seeding) is independent of every other camera, so it runs
    across a :class:`~concurrent.futures.ProcessPoolExecutor`: ``-1`` (default) uses all CPU
    cores (capped at the camera count — never more workers than cameras), a positive int caps
    the worker count, ``1`` forces the original serial loop (useful for debugging or a
    single-camera rig, where pool startup would only add overhead). This is **pure
    parallelism, not an approximation** — same seeds, same solver, same deterministic result,
    it just runs concurrently instead of one-camera-at-a-time; it therefore speeds up *every*
    camera model and distortion level equally, and the benefit grows with the camera count
    (unlike the model-specific redundancy fixes elsewhere in this module).
    """
    def front_end(obj, obs_by_cam, img_size):
        """Per-camera intrinsic calibration + pose seeding front-end.

        Matches the ``front_end(obj, obs_by_cam, img_size) -> {cam_id: CameraModel}``
        signature :func:`calibrate_rig` expects; see :func:`make_bundle_front_end`'s
        docstring for the seeding/parallelism strategy.
        """
        model_map = _resolve_model_map(model_spec, list(obs_by_cam))
        seeded = set(init_K) if init_K else set()
        cam_ids = list(obs_by_cam)
        workers = (os.cpu_count() or 1) if n_jobs is None or n_jobs < 0 else max(1, n_jobs)
        workers = min(workers, len(cam_ids))

        def _args(cam_id):
            obs = obs_by_cam[cam_id]
            w, h = img_size[cam_id]
            init_K_cam = np.asarray(init_K[cam_id], float) if cam_id in seeded else None
            return model_map[cam_id], obs, w, h, init_K_cam, loss, f_scale, max_nfev

        # One pool spans both independent-per-camera stages below (intrinsic fit, then pose
        # seeding once intrinsics are known) so process start-up is paid once, not twice.
        pool = (ProcessPoolExecutor(max_workers=workers, initializer=_init_pool_worker,
                                    initargs=(obj,)) if workers > 1 else nullcontext())
        with pool as ex:
            raw = {}
            if ex is not None:
                futs = [ex.submit(_fit_one_camera_pooled, cam_id, *_args(cam_id))
                       for cam_id in cam_ids]
                for fut in futs:
                    cam_id, r = fut.result()
                    raw[cam_id] = dict(r, obs=obs_by_cam[cam_id])
            else:
                for cam_id in cam_ids:
                    model_cls, obs, w, h, init_K_cam, *rest = _args(cam_id)
                    r = _fit_one_camera(model_cls, obj, obs, w, h, init_K_cam, *rest)
                    raw[cam_id] = dict(r, obs=obs_by_cam[cam_id])

            # Consensus guard: a camera that views the target near-planar (e.g. an obliquely
            # angled camera seeing one board) hits the focal ambiguity and calibrates to a
            # wrong / anamorphic focal. Detect such cameras by deviation from the per-rig
            # median *paraxial* focal (the model-independent f_eff — comparing raw fx across a
            # model with non-trivial axial terms is meaningless), reset to the consensus, and
            # let the global BA refine them through the rigid-rig constraint.
            feff = {c: paraxial_focal(raw[c]["model"]) for c in raw}
            med_fx = float(np.median([feff[c][0] for c in raw]))
            med_fy = float(np.median([feff[c][1] for c in raw]))

            cameras = {}
            pnp_args = {}
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
                pnp_args[cam_id] = [(o.point_rows, o.pts_2d) for o in r["obs"]]

            # All object poses via robust gated PnP (keeps every point downstream; the global
            # BA does the IRLS weighting, no per-point rejection in the answer) — independent
            # per camera exactly like the intrinsic fit above, so it shares the same pool.
            if ex is not None:
                futs = [ex.submit(_pnp_poses_pooled, cam_id, cameras[cam_id], pnp_args[cam_id])
                       for cam_id in cam_ids]
                poses = dict(fut.result() for fut in futs)
            else:
                poses = {cam_id: _pnp_poses_one_camera(cameras[cam_id], obj, pnp_args[cam_id])
                         for cam_id in cam_ids}
        for cam_id in cam_ids:
            for o, T in zip(raw[cam_id]["obs"], poses[cam_id]):
                o.T_c_o = T
        return cameras
    return front_end


def _gated_pnp(model, X, uv, max_rms_px: float = 2.0):
    """Per-view pose by **high-breakdown RANSAC PnP**, falling back to redescending IRLS.

    A RANSAC PnP (:func:`ds_msp.ops.solve_pnp_ransac`) rejects gross-outlier corners by
    consensus and refines the pose on the inlier set, so a view stays correct *past* the 50%
    breakdown of a median-scaled reweighting — the front-end seed that lets the rig survive
    heavy per-frame contamination (the MAD-based IRLS warm-started here used to soften the seed
    back toward outliers above 50%). Falls back to the keep-every-corner IRLS refinement
    (:func:`pose_init.robust_pose_irls`) when RANSAC finds no consensus (too few points /
    extreme contamination). On clean views RANSAC keeps every corner, so the pose is unchanged.
    Returns ``T_cam_obj`` (4x4), or ``None`` only when the view has too few unprojectable
    points. ``max_rms_px`` is the RANSAC inlier gate in pixels.

    RANSAC is used only to **identify** gross-outlier corners; the pose is then refined by the
    accurate IRLS on the clean inlier subset (seeded from the RANSAC pose). When RANSAC finds
    no outliers — every clean view — this is bit-identical to the plain IRLS over all corners,
    so there is no clean-data regression; only contaminated views drop their blunders before
    the refine."""
    from ..ops.pose import solve_pnp_ransac
    ok, rvec, tvec, inliers = solve_pnp_ransac(model, X, uv, thresh_px=max_rms_px,
                                               max_iters=1000)
    X = np.asarray(X, float)
    uv = np.asarray(uv, float)
    if ok and inliers is not None and 4 <= int(inliers.sum()) < len(uv):
        T0 = np.eye(4)
        T0[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float))[0]
        T0[:3, 3] = np.asarray(tvec, float).ravel()
        return robust_pose_irls(model, X[inliers], uv[inliers], T0=T0, kernel="cauchy",
                                gnc_iters=5, gnc_start=4.0, studentize=True)
    return robust_pose_irls(model, X, uv, kernel="cauchy", gnc_iters=5, gnc_start=4.0,
                            studentize=True)


def _pnp_poses_one_camera(model, obj: Object3D, obs_light):
    """Per-frame pose seeding for one camera's observations — independent of every other
    camera once ``model`` (its calibrated intrinsics) is known. ``obs_light`` is
    ``[(point_rows, pts_2d), ...]``; returns the matching list of ``T_cam_obj`` (or ``None``).
    """
    X_all = obj.pts_3d
    return [_gated_pnp(model, X_all[point_rows], uv) for point_rows, uv in obs_light]


def _pnp_poses_pooled(cam_id, model, obs_light):
    """Pool task counterpart of :func:`_pnp_poses_one_camera`, reading the shared
    ``Object3D`` from the worker-local global set by :func:`_init_pool_worker`."""
    return cam_id, _pnp_poses_one_camera(model, _POOL_OBJ, obs_light)


def calibrate_rig(obj: Object3D, object_obs: List[ObjectObs],
                  img_size: Dict[int, Tuple[int, int]],
                  *, fix_intrinsics: bool = False, verbose: bool = False,
                  front_end: Optional[Callable] = None, he_approach: int = 0,
                  refine_structure: bool = False, structure_rounds: int = 6,
                  gnc_iters: int = 5, gnc_start: float = 4.0,
                  noise_bound: Optional[float] = None, on_iter=None) -> RigState:
    """Calibrate a multi-camera rig from fused-object observations.

    Returns a :class:`RigState` with per-camera intrinsics, ``T_c_g`` extrinsics
    (reference camera = identity), and per-frame object poses.

    ``noise_bound`` (per-corner reprojection σ in pixels) makes the **global joint** BA run the
    median-free **GNC-TLS** solver (``barc = 3.03·σ`` inlier band), which rejects gross-outlier
    corners past the 50% breakdown of the MAD auto-scale; the cheaper stages (group refine,
    structure polish) stay on the legacy reweighting so the cost stays bounded. The per-frame
    pose seed (:func:`_gated_pnp`) always uses a high-breakdown RANSAC PnP. This low-level
    primitive defaults to ``None`` (legacy path); the config-driven product entry
    :func:`~ds_msp.rig.calib_param.calibrate_from_config` supplies a ``noise_bound`` (config
    default ``1.0`` px) so a real calibration is robust by default.

    ``on_iter(it, max_iter, rms, cost, rig_snapshot)`` — optional live-progress callback,
    forwarded to every stage's underlying solver (:func:`bundle.refine` / ``core.optimize.lm_solve`` /
    ``schur_lm``); fires once per solver iteration across every stage in sequence.
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
    #    This has no on_iter callback of its own (multi-start seed dispersion runs 2 full
    #    per-camera solves each -- NFR-PERF-001 -- measured ~19s on an 8-camera real rig) so
    #    a live view would otherwise sit frozen on the previous stage's label for that whole
    #    stretch; set_stage still fires here even without per-iteration progress inside it.
    if on_iter is not None and hasattr(on_iter, "set_stage"):
        on_iter.set_stage("(0) per-camera front-end intrinsics")
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
    if on_iter is not None and hasattr(on_iter, "set_stage"):
        on_iter.set_stage("(a) per-object pose warm-up")
    rig = bundle.refine(rig, object_obs, fix_intrinsics=True, fix_extrinsics=True,
                    robust_kernel="huber", robust_scale="auto", verbose=verbose,
                    on_iter=on_iter)
    #    (b) per-camera-group — refine each group's extrinsics + its object poses, intrinsics
    #        fixed (calibrateCameraGroup / refineAllCameraGroupAndObjects). Single group ->
    #        whole-rig poses-only; multiple groups -> each independently before the joint pass.
    if on_iter is not None and hasattr(on_iter, "set_stage"):
        on_iter.set_stage("(b) per-camera-group extrinsics" if len(groups) > 1
                           else "(b) rig extrinsics")
    rig = bundle.refine_groups(rig, object_obs, groups,
                           robust_kernel="huber", robust_scale="auto", verbose=verbose,
                           on_iter=on_iter)
    #    (c) global joint — full rig + (optionally) intrinsics with a redescending Cauchy
    #        kernel and a short GNC anneal (refineAllCameraGroupAndObjectsAndIntrinsics).
    # Graduated non-convexity: anneal the robust scale from ``gnc_start``×MAD down. A wider
    # start + more steps escapes the bad-data basin under heavy (≈50%) gross-outlier
    # contamination, where a MAD scale is itself corrupted — measurably more robust at the
    # breakdown point with negligible cost on clean data.
    if on_iter is not None and hasattr(on_iter, "set_stage"):
        on_iter.set_stage("(c) global joint BA" + (" + intrinsics" if fix_intrinsics is False
                                                     else ""))
    rig = bundle.refine(rig, object_obs, fix_intrinsics=fix_intrinsics, robust_kernel="cauchy",
                    robust_scale="auto", gnc_iters=gnc_iters, gnc_start=gnc_start,
                    noise_bound=noise_bound, verbose=verbose, on_iter=on_iter)

    #    (d) object-structure refinement (MC-Calib's refineObject), multi-board only: alternate
    #        re-triangulating the fused object's 3-D points (gauge anchored on the reference
    #        board) with the joint BA. A reconstructed nominal object carries inter-board pose
    #        and physical-board error that no camera/pose update can absorb; freeing the
    #        structure removes it — the dominant reprojection win on a real multi-board target.
    if refine_structure and len(obj.board_ids) > 1:
        prev = float("inf")
        for round_i in range(structure_rounds):
            rig = bundle.refine_object_structure(rig, object_obs)
            # a light joint polish each round (no GNC, capped iters): the structure step did
            # the heavy lifting, so the BA only has to absorb it — full-length GNC passes here
            # were ~80% of total runtime for <0.5% extra accuracy.
            if on_iter is not None and hasattr(on_iter, "set_stage"):
                on_iter.set_stage(f"(d) structure refinement {round_i + 1}/{structure_rounds}")
            rig = bundle.refine(rig, object_obs, fix_intrinsics=fix_intrinsics,
                            robust_kernel="cauchy", robust_scale="auto", max_iter=25,
                            verbose=verbose, on_iter=on_iter)
            cur = max(bundle.reprojection_rms(rig, object_obs).values())
            if prev - cur < 0.003 * prev:           # converged -> stop early
                break
            prev = cur
    return rig
