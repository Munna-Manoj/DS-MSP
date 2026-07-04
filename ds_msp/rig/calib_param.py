"""Single-file, MC-Calib-compatible entry point — ``calibrate_from_config(calib_param.yml)``.

MC-Calib is driven by exactly one config file: you edit ``calib_param.yml`` and run
``./calibrate calib_param.yml``. This module reads that *same* file (every key MC-Calib
reads, parsed with ``cv2.FileStorage`` so ``%YAML:1.0`` works) and runs the whole DS-MSP[rig]
pipeline from it — raw images or pre-detected keypoints in, MC-Calib-format results out —
so the tool is as easy to use as MC-Calib, with one extension: a *per-camera camera model*.

Config keys (MC-Calib's, plus the ``camera_models`` extension):

* board geometry — ``number_x_square``, ``number_y_square``, ``length_square``,
  ``length_marker``, ``square_size``, ``number_board`` (+ ``*_per_board`` overrides);
* models — ``distortion_model`` (0 Brown→``radtan``, 1 Kannala→``kb``; a mode *selector*, not
  a flag, so it stays numeric), ``distortion_per_camera`` (per-camera selector, same 0/1
  encoding as ``distortion_model``), and the DS-MSP extension ``camera_models`` (per-camera
  model name from {radtan,ucm,eucm,ds,kb,ocam}, highest precedence);
* input — ``number_camera``, ``root_path`` + ``cam_prefix`` (raw images) or
  ``keypoints_path`` (pre-detected; ``"None"`` ⇒ detect from images);
* intrinsics — ``fix_intrinsic`` (``true``/``false``), ``cam_params_path`` (initial
  intrinsics);
* output — ``save_path``, ``camera_params_file_name``, ``save_detection``,
  ``save_reprojection`` (``true``/``false``);
* live view — ``webviewer`` (``true``/``false``, default ``true``): launch the live browser
  3D view during the run. Independent of ``verbose`` (terminal progress lines) so either can
  be silenced without the other, e.g. a headless/CI run wants ``webviewer: false`` but often
  still wants ``verbose: true``.

Every genuinely boolean field above (``fix_intrinsic``, ``save_detection``,
``save_reprojection``, ``verbose``, ``webviewer``) accepts ``true``/``false`` (also
``yes``/``no``) in the YAML, and, for backward compatibility with older config files, still
accepts the previous ``0``/``1`` numeric form (see :func:`_bool_field`). Fields that pick
between more than two named modes (``distortion_model``, ``distortion_per_camera``,
``he_approach``) are not booleans and stay as documented integer selectors.

Relative paths resolve against the config file's directory; pass ``overrides`` to retarget
them (e.g. point ``root_path`` at the real dataset). Single-board objects are built from
config. Multi-board *fused-object* geometry (MC-Calib's board-group reconstruction) is
**reconstructed from the detections** (:mod:`ds_msp.rig.reconstruct`) when no pre-built
``object_path`` / ``calibrated_objects_data.yml`` is supplied — so a multi-board rig also
calibrates straight from a raw image folder.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy.optimize import least_squares

from ..detect.charuco import BoardSpec, detect_rig, single_board_object
from ..io.mccalib import (Scenario, _load_cameras, _load_detections, _load_groundtruth,
                          _load_object, radtan_from_cameragt)
from ..models.kb import KannalaBrandtModel
from ..models.registry import canonical_name, model_class
from ..rig.types import ObjectObs
from .calibrate import _seed_from_K
from .pipeline import calibrate_scenario

_BROWN, _KANNALA = 0, 1
_DIST_TO_MODEL = {_BROWN: "radtan", _KANNALA: "kb"}


def _provided_model_name(cam) -> str:
    """Canonical model the provided intrinsics are stored in.

    Uses the model the file explicitly states (``camera_model`` string or ``distortion_type``
    int, parsed into ``cam.model_name``) when present — this is the only reliable signal for the
    sphere/poly models (ucm/eucm/ds/dsplus/eucmplus) whose distortion length overlaps. For a
    plain MC-Calib file that states nothing, falls back to the historical length heuristic:
    4 coeffs ⇒ KB (fisheye), anything else ⇒ RadTan/Brown."""
    if getattr(cam, "model_name", None):
        return canonical_name(cam.model_name)
    d = np.asarray(cam.dist, float).ravel() if cam.dist is not None else np.zeros(5)
    return "kb" if d.size == 4 else "radtan"


def _model_from_K_dist(name: str, K, dist):
    """Reconstruct a camera model from an MC-Calib ``camera_matrix`` + ``distortion_vector``.

    The inverse of how :func:`ds_msp.io.mccalib.save_mccalib_cameras` serializes a model, so a
    DS-MSP-written intrinsics file round-trips to the exact same parameters. All models except
    OCam store their intrinsic vector as ``[fx, fy, cx, cy, *distortion]``; OCam stores
    ``[cx, cy]`` in ``K`` and ``[c, d, e, a0..a4]`` in the distortion vector."""
    name = canonical_name(name)
    cls = model_class(name)
    d = np.asarray(dist, float).ravel() if dist is not None else np.zeros(0)
    K = np.asarray(K, float)
    if name == "ocam":
        if d.size != 8:
            raise ValueError(f"ocam intrinsics need 8 distortion coeffs [c,d,e,a0..a4], "
                             f"file has {d.size}")
        return cls.from_params([K[0, 2], K[1, 2], *d])
    n_dist = len(cls.param_names) - 4
    if d.size != n_dist:
        raise ValueError(f"{name} intrinsics need {n_dist} distortion coeffs "
                         f"{cls.param_names[4:]}, file has {d.size}")
    return cls.from_params([K[0, 0], K[1, 1], K[0, 2], K[1, 2], *d])


def _source_model(cam):
    """The camera model the provided intrinsics are stored in, built exactly from the file.

    Honors the model the file states (any of the DS-MSP models); falls back to the KB/RadTan
    length heuristic for a plain MC-Calib file. If the stated model and the distortion-vector
    length disagree (a malformed / mismatched intrinsics entry) the reconstruction raises, and
    for the safe default models we degrade to the heuristic rather than fail hard."""
    name = _provided_model_name(cam)
    try:
        return _model_from_K_dist(name, cam.K, cam.dist)
    except (ValueError, KeyError):
        # A plain file labeled only by distortion_type can still be consistently read by length;
        # keep the legacy KB/RadTan behavior so existing MC-Calib outputs never regress.
        d = np.asarray(cam.dist, float).ravel() if cam.dist is not None else np.zeros(5)
        if d.size == 4:
            return KannalaBrandtModel(cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2], *d[:4])
        return radtan_from_cameragt(cam)


def _init_model(cam, model_name, wh):
    """Build the chosen camera model from the provided intrinsics for the fixed-intrinsic path.

    If the chosen model is the file's native one (KB↔fisheye, RadTan↔Brown) the exact stored
    parameters are used. Otherwise the *same physical lens* is converted into the chosen model:
    a pixel grid is unprojected through the provided model and the chosen model fits its native
    distortion to those identical ray↔pixel pairs (the "two models, one camera" identity), so a
    DS / UCM / EUCM camera is held at intrinsics that represent the same lens as the reference.
    """
    name = canonical_name(model_name)
    src = _source_model(cam)
    if name == src.name:
        return src
    w, h = wh
    uu, vv = np.meshgrid(np.linspace(0.04 * w, 0.96 * w, 48),
                         np.linspace(0.04 * h, 0.96 * h, 48))
    pix = np.column_stack([uu.ravel(), vv.ravel()])
    rays, valid = src.unproject(pix)
    rays, pix = np.asarray(rays, float), np.asarray(pix, float)
    if valid is not None:
        valid = np.asarray(valid).ravel().astype(bool)
        rays, pix = rays[valid], pix[valid]
    cls = model_class(name)
    tgt = _seed_from_K(cls, cam.K)
    # Refine the chosen model's full intrinsic vector to reproduce the source lens (minimize the
    # reprojection of the source rays onto the source pixels). Bounded TRF keeps the sphere /
    # polynomial shape parameters in their valid range (an unbounded LM lets DS slide to a
    # non-physical xi and diverge). A wide-FOV lens is only approximated by a 1-2 parameter model
    # — that irreducible residual is a model-expressiveness limit, surfaced by the caller.
    lo, hi = _param_bounds(cls)

    def _resid(p):
        uv, ok = cls.from_params(p).project(rays)
        r = uv - pix
        if ok is not None:
            r = r * np.asarray(ok, float).reshape(-1, 1)
        return np.nan_to_num(r, nan=1e3).ravel()

    try:
        sol = least_squares(_resid, tgt.params.astype(float), bounds=(lo, hi),
                            method="trf", x_scale="jac", max_nfev=300)
        cand = cls.from_params(sol.x)
        if np.isfinite(cand.params).all():
            tgt = cand
    except (np.linalg.LinAlgError, ValueError):
        pass
    return tgt


def _param_bounds(cls):
    """(lower, upper) bounds for a model's intrinsic vector, used when converting a provided
    lens into the model. Focal / principal point are left free; the distortion-shape parameters
    are held to their physically valid ranges so the bounded fit cannot diverge."""
    rng = {"fx": (1.0, 1e5), "fy": (1.0, 1e5), "cx": (-1e4, 1e4), "cy": (-1e4, 1e4),
           "alpha": (0.0, 1.0), "beta": (0.05, 20.0), "xi": (-1.0, 1.0)}
    lo, hi = [], []
    for n in cls.param_names:
        a, b = rng.get(n, (-10.0, 10.0))            # radtan k/p terms: a generous symmetric band
        lo.append(a)
        hi.append(b)
    return lo, hi


def _node(fs, name, default=None):
    n = fs.getNode(name)
    return n if not n.empty() else None


def _scalar(fs, name, default):
    n = fs.getNode(name)
    if n is None or n.empty():
        return default
    if n.isString():
        return n.string()
    return n.real()


def _bool_field(ov: Dict, fs, name: str, default: int) -> bool:
    """Parse a config boolean. ``cv2.FileStorage`` has no native YAML boolean type -- a bare
    ``true``/``false`` token comes back from ``_scalar`` as the *string* ``"true"``/``"false"``
    (verified against the real parser: ``isString()`` is true, ``.real()`` is a DBL_MAX
    sentinel, not 0/1) -- so this accepts that string form, plus ``yes``/``no``, plus the older
    ``0``/``1`` numeric form still used by some config files, from both the YAML file itself and
    a ``--set name=value`` override (which always arrives as a raw string).

    Two more ``cv2.FileStorage`` quirks verified directly against the real parser, both only for
    an *unquoted* bareword value (quoting sidesteps both -- ``fix_intrinsic: "false"`` is always
    safe regardless of what the trailing comment says, which is why every DS-MSP-authored
    template quotes it):

    1. A trailing ``# comment`` on the same line is NOT stripped the way it is for a numeric 0/1
       value -- the whole rest of the line becomes part of the string. Stripped here explicitly.
    2. If that comment contains a ``:`` (e.g. ``# false: refine // true: hold fixed``), the
       parser misreads the *value* as a nested YAML mapping (``isMap()`` true, not
       ``isString()``) and ``_scalar`` silently falls through to ``.real()``'s DBL_MAX sentinel
       -- which ``bool(int(...))`` then turns into a wrong ``True`` with **no error at all**.
       Detected explicitly below and raised as a clear, actionable error instead of a silent
       wrong answer.
    """
    if name in ov:
        raw = ov[name]
    else:
        n = fs.getNode(name)
        if n is None or n.empty():
            raw = default
        elif n.isString():
            raw = n.string()
        elif n.isMap():
            raise ValueError(
                f"config field {name!r}: could not be parsed as a plain value. This usually "
                f"means an unquoted true/false has a ':' inside its trailing comment, which "
                f"this YAML parser misreads as a nested mapping -- quote the value instead, "
                f"e.g. {name}: \"true\"  # a colon: in a comment is safe once quoted.")
        else:
            raw = n.real()
    if isinstance(raw, str):
        s = raw.split("#", 1)[0].strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
        raise ValueError(f"config field {name!r}: expected true/false, got {raw!r}")
    return bool(int(raw))


def _int_seq(fs, name) -> List[int]:
    n = fs.getNode(name)
    if n is None or n.empty() or not n.isSeq():
        return []
    return [int(n.at(i).real()) for i in range(n.size())]


def _str_seq(fs, name) -> List[str]:
    n = fs.getNode(name)
    if n is None or n.empty() or not n.isSeq():
        return []
    return [n.at(i).string() for i in range(n.size())]


@dataclass
class RigConfig:
    """Parsed MC-Calib ``calib_param.yml`` (paths already resolved to absolute)."""
    boards: List[BoardSpec]
    number_camera: int
    camera_models: List[str]                    # one per camera, canonical names
    root_path: Optional[str]
    cam_prefix: str
    keypoints_path: Optional[str]
    fix_intrinsic: bool
    cam_params_path: Optional[str]
    save_path: Optional[str]
    camera_params_file_name: str
    save_detection: bool
    save_reprojection: bool
    ransac_threshold: float
    number_iterations: int
    he_approach: int
    noise_bound: float = 1.0
    verbose: bool = True
    webviewer: bool = True                      # launch the live browser view (independent of `verbose`)
    object_path: Optional[str] = None
    raw: Dict = field(default_factory=dict)

    @property
    def number_board(self) -> int:
        return len(self.boards)


def _resolve(base_dir: str, path: Optional[str]) -> Optional[str]:
    if path is None or str(path).strip() in ("", "None"):
        return None
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir, path))


def _board_specs(fs) -> List[BoardSpec]:
    nb = int(_scalar(fs, "number_board", 1))
    nx = int(_scalar(fs, "number_x_square", 5))
    ny = int(_scalar(fs, "number_y_square", 5))
    ls = float(_scalar(fs, "length_square", 0.04))
    lm = float(_scalar(fs, "length_marker", 0.03))
    sq = float(_scalar(fs, "square_size", 1.0))
    nx_pb, ny_pb = _int_seq(fs, "number_x_square_per_board"), _int_seq(fs, "number_y_square_per_board")
    sq_pb = _int_seq(fs, "square_size_per_board")            # rarely set; falls back to sq
    specs = []
    for b in range(nb):
        specs.append(BoardSpec(
            n_x=nx_pb[b] if b < len(nx_pb) else nx,
            n_y=ny_pb[b] if b < len(ny_pb) else ny,
            length_square=ls, length_marker=lm,
            square_size=float(sq_pb[b]) if b < len(sq_pb) else sq))
    return specs


def _camera_models(fs, n_cam: int) -> List[str]:
    """Per-camera model: ``camera_models`` (extension) > ``distortion_per_camera`` > global
    ``distortion_model``. Names are canonicalized through the model registry."""
    names = _str_seq(fs, "camera_models")
    if names:
        if len(names) == 1:
            names = names * n_cam
        return [canonical_name(s) for s in names]
    per = _int_seq(fs, "distortion_per_camera")
    glob = int(_scalar(fs, "distortion_model", _BROWN))
    return [_DIST_TO_MODEL.get(per[c] if c < len(per) else glob, "radtan") for c in range(n_cam)]


def load_config(config_path: str, overrides: Optional[Dict] = None) -> RigConfig:
    """Parse a ``calib_param.yml`` into a :class:`RigConfig`. ``overrides`` replaces raw
    config values *before* path resolution (e.g. ``{"root_path": "/abs/Images"}``)."""
    base = os.path.dirname(os.path.abspath(config_path))
    fs = cv2.FileStorage(config_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(config_path)
    n_cam = int(_scalar(fs, "number_camera", 1))
    ov = overrides or {}

    def path_key(name, default=None):
        if name in ov:
            return _resolve(base, ov[name])
        return _resolve(base, _scalar(fs, name, default))

    cfg = RigConfig(
        boards=_board_specs(fs),
        number_camera=n_cam,
        camera_models=ov.get("camera_models") or _camera_models(fs, n_cam),
        root_path=path_key("root_path", "None"),
        cam_prefix=str(_scalar(fs, "cam_prefix", "Cam_")),
        keypoints_path=path_key("keypoints_path", "None"),
        fix_intrinsic=_bool_field(ov, fs, "fix_intrinsic", 0),
        cam_params_path=path_key("cam_params_path", "None"),
        save_path=path_key("save_path", "None"),
        camera_params_file_name=str(_scalar(fs, "camera_params_file_name", "")),
        save_detection=_bool_field(ov, fs, "save_detection", 0),
        save_reprojection=_bool_field(ov, fs, "save_reprojection", 0),
        ransac_threshold=float(_scalar(fs, "ransac_threshold", 10.0)),
        number_iterations=int(_scalar(fs, "number_iterations", 1000)),
        he_approach=int(_scalar(fs, "he_approach", 0)),
        noise_bound=float(_scalar(fs, "noise_bound", 1.0)),
        verbose=_bool_field(ov, fs, "verbose", 1),
        webviewer=_bool_field(ov, fs, "webviewer", 1),
        object_path=path_key("object_path", "None"),
        raw={"path": config_path},
    )
    fs.release()
    return cfg


def _try_load_object(cfg: RigConfig):
    """Return a pre-built fused object if one is available (``object_path`` or a
    ``calibrated_objects_data.yml`` next to the keypoints / save path), else ``None``.
    A pre-built object is the fast path; when absent, multi-board geometry is reconstructed
    from the detections (:func:`rig.reconstruct.reconstruct_object`)."""
    for cand in (cfg.object_path,
                 os.path.join(os.path.dirname(cfg.keypoints_path or ""), "calibrated_objects_data.yml")
                 if cfg.keypoints_path else None,
                 os.path.join(cfg.save_path or "", "calibrated_objects_data.yml")
                 if cfg.save_path else None):
        if cand and os.path.exists(cand):
            return _load_object(cand)
    return None


def _detect_obs(cfg: RigConfig, obj, *, animator=None):
    """Object observations from a known object — keypoints if given, else raw images."""
    if cfg.keypoints_path:
        return _obs_from_keypoints(cfg, obj)
    if cfg.root_path:
        cam_ids = list(range(cfg.number_camera))
        progress_cb = _compose_progress(cfg, animator)
        return detect_rig(cfg.root_path, cam_ids, cfg.boards, obj,
                          cam_prefix=cfg.cam_prefix, min_corners=8, progress_cb=progress_cb)
    raise ValueError("config has neither keypoints_path nor root_path")


def _detect_progress_printer():
    """A ``detect_rig`` ``progress_cb`` that live-updates one terminal line per image —
    the per-image OpenCV detection pass otherwise runs silently for minutes on a large
    image set (the #1 source of "the script tells the user nothing")."""
    from .report import live_line

    def cb(cam_id: int, i: int, n: int, path: str) -> None:
        live_line(f"[detect] cam {cam_id}: {i}/{n}  {os.path.basename(path)}")
    return cb


def _compose_progress(cfg: RigConfig, animator):
    """Combine the terminal detect-progress printer with the live web view's own
    ``detect_progress`` (if an animator is given) into one ``progress_cb`` — both should see
    every image as it completes, not just one or the other."""
    cbs = []
    if cfg.verbose:
        cbs.append(_detect_progress_printer())
    if animator is not None:
        cbs.append(animator.detect_progress)
    if not cbs:
        return None
    if len(cbs) == 1:
        return cbs[0]

    def cb(cam_id: int, i: int, n: int, path: str) -> None:
        for c in cbs:
            c(cam_id, i, n, path)
    return cb


def _reconstruct(cfg: RigConfig, *, animator=None):
    """Reconstruct the fused multi-board object from raw detections, then map observations
    onto it — MC-Calib's ``calibrate3DObjects`` (no pre-built object file needed).

    When ``cam_params_path`` is given, boards are resected with each camera's **native model**
    (built from the provided intrinsics) so a wide-FOV fisheye is reconstructed correctly — the
    default Brown bootstrap cannot model it and corrupts the fused geometry."""
    from .reconstruct import reconstruct_from_images, reconstruct_from_keypoints
    init_models = None
    if cfg.cam_params_path and os.path.exists(cfg.cam_params_path):
        cams = _load_cameras(cfg.cam_params_path)[0]
        init_models = {c: _source_model(cams[c]) for c in range(cfg.number_camera)
                       if c in cams and cams[c].K is not None}
    if cfg.keypoints_path:
        obj, obs, img_size = reconstruct_from_keypoints(
            cfg.keypoints_path, cfg.boards, init_models=init_models)
    elif cfg.root_path:
        cam_ids = list(range(cfg.number_camera))
        progress_cb = _compose_progress(cfg, animator)
        obj, obs, img_size = reconstruct_from_images(
            cfg.root_path, cam_ids, cfg.boards, cam_prefix=cfg.cam_prefix,
            init_models=init_models, progress_cb=progress_cb)
    else:
        raise ValueError("config has neither keypoints_path nor root_path")
    return obj, obs, img_size


def _obs_from_keypoints(cfg: RigConfig, obj):
    per_cam, img_size = _load_detections(cfg.keypoints_path, obj)
    obs: List[ObjectObs] = []
    for cam_id, frames in per_cam.items():
        for frame_id, (rows, uvs) in frames.items():
            if rows:
                obs.append(ObjectObs(cam_id=cam_id, frame_id=frame_id, object_id=0,
                                     point_rows=np.array(rows, int),
                                     pts_2d=np.array(uvs, float)))
    return obs, img_size


def _gt_and_mccalib(cfg: RigConfig):
    """Best-effort GroundTruth.yml / calibrated_cameras for metrics; empty if absent."""
    anchor = cfg.root_path or cfg.keypoints_path or cfg.save_path or ""
    scn_dir = anchor
    for _ in range(3):                              # walk up to the scenario root
        scn_dir = os.path.dirname(scn_dir.rstrip("/"))
        if os.path.exists(os.path.join(scn_dir, "GroundTruth.yml")):
            break
    gt_path = os.path.join(scn_dir, "GroundTruth.yml")
    gt = _load_groundtruth(gt_path) if os.path.exists(gt_path) else {}
    mc_path = os.path.join(scn_dir, "Results", "calibrated_cameras_data.yml")
    mccalib = _load_cameras(mc_path)[0] if os.path.exists(mc_path) else {}
    return gt, mccalib


def _warn(msg: str) -> None:
    print(f"[ds-msp][rig] WARNING: {msg}", file=sys.stderr)


def _resolve_init_intrinsics(cfg: RigConfig, cam_ids: List[int], spec: Dict[int, str],
                             img_size: Dict[int, tuple]):
    """Turn ``cam_params_path`` (if any) into ``(init_cameras, init_K)`` for the BA, enforcing the
    intrinsics policy:

    * **No intrinsics file** — ``fix_intrinsic=true`` is an error (nothing to hold fixed);
      otherwise every camera initializes from scratch (front-end estimate, MC-Calib-style with
      DS-MSP's wider model set), returning ``(None, None)``.
    * **Intrinsics given** — each camera's stored model is read from the file (explicit
      ``camera_model``/``distortion_type``, else inferred from the distortion length). For every
      camera:

      - *missing from the file* — error if ``fix_intrinsic`` (the intrinsics it must hold are
        absent); otherwise that camera initializes from scratch.
      - *model matches the config's chosen model* — built natively and used as-is.
      - *model differs from the config's chosen model* — error if ``fix_intrinsic`` (you cannot
        hold a camera fixed in a model it was not provided in); otherwise a WARNING is shown and
        the same physical lens is carried into the chosen model via the ``convert()`` identity
        (unproject a pixel grid through the provided model, fit the chosen model to the identical
        ray↔pixel pairs) so the BA still starts from the real lens.
    """
    if not (cfg.cam_params_path and os.path.exists(cfg.cam_params_path)):
        if cfg.fix_intrinsic:
            raise FileNotFoundError(
                "fix_intrinsic=true needs cam_params_path with initial intrinsics for every camera")
        return None, None

    cams = _load_cameras(cfg.cam_params_path)[0]
    init_K: Dict[int, np.ndarray] = {}
    init_cameras: Dict[int, object] = {}
    for c in cam_ids:
        cam = cams.get(c)
        if cam is None or cam.K is None:
            if cfg.fix_intrinsic:
                raise ValueError(
                    f"fix_intrinsic=true but camera {c} has no intrinsics in {cfg.cam_params_path}")
            _warn(f"camera {c}: no intrinsics in {os.path.basename(cfg.cam_params_path)}; "
                  f"initializing '{spec[c]}' from scratch")
            continue
        provided = _provided_model_name(cam)
        chosen = canonical_name(spec[c])
        if provided != chosen:
            if cfg.fix_intrinsic:
                raise ValueError(
                    f"camera {c}: config selects '{chosen}' but provided intrinsics are '{provided}'. "
                    f"With fix_intrinsic=true the model must match — set camera_models[{c}]='{provided}' "
                    f"to hold them fixed, or set fix_intrinsic=false to convert+refine into '{chosen}'.")
            _warn(f"camera {c}: config selects '{chosen}' but provided intrinsics are '{provided}'. "
                  f"Using convert() to seed '{chosen}' from the '{provided}' lens, then refining "
                  f"it in the bundle adjustment.")
        init_K[c] = cam.K
        init_cameras[c] = _init_model(cam, chosen, img_size[c])
    return (init_cameras or None), (init_K or None)


def calibrate_from_config(config_path: str, overrides: Optional[Dict] = None) -> Dict:
    """Run the full rig calibration from one MC-Calib ``calib_param.yml``.

    Detects ChArUco corners from ``root_path`` (or loads ``keypoints_path``), builds the
    per-camera model map, optimizes extrinsics-only (``fix_intrinsic=true``) or
    intrinsics+extrinsics, and writes MC-Calib-format results to ``save_path``. Returns the
    :func:`~ds_msp.rig.pipeline.calibrate_scenario` result dict plus the parsed ``config``.
    """
    cfg = load_config(config_path, overrides)

    # Launch the live view *before* detection starts, not once the geometry already exists:
    # on this repo's real MC-Calib dataset, corner detection (~5s) and the front-end's
    # per-camera intrinsic fit (~19s, no progress callback of its own) together can dominate
    # the first minute of a run with nothing to show for it — see WebLive3DAnimator's class
    # docstring. detect_progress/bind_scene/save_progress below stream real progress through
    # every phase instead of leaving the browser blank until the first BA iteration exists.
    from .web3d import WebLive3DAnimator
    animator = WebLive3DAnimator(verbose=cfg.webviewer)

    if cfg.number_board == 1:
        obj = single_board_object(cfg.boards[0])
        object_obs, img_size = _detect_obs(cfg, obj, animator=animator)
    else:
        obj = _try_load_object(cfg)
        if obj is not None:                         # pre-built fused object: fast path
            object_obs, img_size = _detect_obs(cfg, obj, animator=animator)
        else:                                       # reconstruct the fused object (MC-Calib
            obj, object_obs, img_size = _reconstruct(cfg, animator=animator)  # calibrate3DObjects analogue)
    animator.bind_scene(obj, object_obs)

    cam_ids = sorted({o.cam_id for o in object_obs})
    spec = {c: cfg.camera_models[c] if c < len(cfg.camera_models) else cfg.camera_models[-1]
            for c in cam_ids}

    # Initial intrinsics from cam_params_path (MC-Calib's intrinsic init), with the policy in
    # _resolve_init_intrinsics: native when the provided model matches the chosen one, convert()
    # (+warning) when it differs and intrinsics are being refined, and a hard error when
    # fix_intrinsic demands an intrinsic that is absent or of the wrong model. The file also
    # seeds each camera's focal / principal point (init_K) so a strong-fisheye / mixed-resolution
    # rig starts in the right basin even when intrinsics are refined.
    init_cameras, init_K = _resolve_init_intrinsics(cfg, cam_ids, spec, img_size)

    gt, mccalib = _gt_and_mccalib(cfg)
    scn = Scenario(name=os.path.basename(os.path.dirname(config_path)) or "rig", object=obj,
                   object_obs=object_obs, cam_ids=cam_ids, img_size=img_size,
                   gt=gt, mccalib=mccalib, mccalib_rms={})
    image_root = cfg.root_path if (cfg.save_reprojection and cfg.root_path) else None
    res = calibrate_scenario(scn, spec, fix_intrinsics=cfg.fix_intrinsic,
                             init_cameras=init_cameras, init_K=init_K, save_dir=cfg.save_path,
                             camera_params_file_name=cfg.camera_params_file_name,
                             image_root=image_root, cam_prefix=cfg.cam_prefix,
                             he_approach=cfg.he_approach,
                             refine_structure=(cfg.number_board > 1),
                             noise_bound=cfg.noise_bound, verbose=cfg.verbose,
                             on_iter=animator)
    animator.finish(res["rig"], rms=res["metrics"]["max_rms_px"])
    res["config"] = cfg
    res["scenario"] = scn
    return res
