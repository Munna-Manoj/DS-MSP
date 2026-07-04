"""
Kalibr camchain YAML I/O.

Reads/writes the standard Kalibr per-camera format so DS-MSP interoperates with
the calibration ecosystem. Field orderings are model-specific and verified
against Kalibr's C++ headers:

  ===========  ==============  ================  ==============================  ====================
  model        camera_model    distortion_model  intrinsics order                distortion_coeffs
  ===========  ==============  ================  ==============================  ====================
  DS           ds              none              [xi, alpha, fx, fy, cx, cy]     []
  EUCM         eucm            none              [alpha, beta, fx, fy, cx, cy]   []
  KB           pinhole         equidistant       [fx, fy, cx, cy]                [k1, k2, k3, k4]
  RadTan       pinhole         radtan            [fx, fy, cx, cy]                [k1, k2, p1, p2]  (no k3)
  UCM          omni            none              [xi_mei, fx, fy, cx, cy]        []
  ===========  ==============  ================  ==============================  ====================

Notes:
- Kalibr's omni mirror parameter ``xi_mei = alpha / (1 - alpha)`` (Mei), not the
  unified ``alpha``; converted on the fly.
- Kalibr radtan has only 4 coeffs: a non-zero ``k3`` is dropped on export (with a
  warning), since the on-disk format cannot represent it.

DS-MSP EXTENSION (DS⁺ / EUCM⁺ have no Kalibr-native equivalent -- these are DS-MSP's own
published models, not part of Kalibr's camera-model set, the same way ``ds_msp.rig`` already
extends MC-Calib's own format with ``camera_models``). Round-trips through this module, but a
real Kalibr installation will not recognize these two ``camera_model`` strings:

  ===========  ==============  ==================  ==============================  ==============================
  model        camera_model    distortion_model    intrinsics order                distortion_coeffs
  ===========  ==============  ==================  ==============================  ==============================
  DS⁺          ds_plus         ds_plus_div_tilt    [alpha, fx, fy, cx, cy]         [lambda1, lambda2, tau_x, tau_y]
  EUCM⁺        eucm_plus       eucm_plus_div_tilt  [alpha, beta, fx, fy, cx, cy]   [lambda1, tau_x, tau_y]
  ===========  ==============  ==================  ==============================  ==============================
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import yaml

from ..models.double_sphere import DoubleSphereModel
from ..models.dsplus import DSPlusModel
from ..models.eucm import EUCMModel
from ..models.eucmplus import EUCMPlusModel
from ..models.kb import KannalaBrandtModel
from ..models.radtan import RadTanModel
from ..models.ucm import UCMModel


def to_kalibr_cam(model, width: int, height: int) -> dict:
    """Serialize a model into a Kalibr ``camN`` stanza (dict)."""
    name = model.name
    if name == "ds":
        block = dict(camera_model="ds",
                     intrinsics=[model.xi, model.alpha, model.fx, model.fy, model.cx, model.cy],
                     distortion_model="none", distortion_coeffs=[])
    elif name == "eucm":
        block = dict(camera_model="eucm",
                     intrinsics=[model.alpha, model.beta, model.fx, model.fy, model.cx, model.cy],
                     distortion_model="none", distortion_coeffs=[])
    elif name == "kb":
        block = dict(camera_model="pinhole",
                     intrinsics=[model.fx, model.fy, model.cx, model.cy],
                     distortion_model="equidistant",
                     distortion_coeffs=[model.k1, model.k2, model.k3, model.k4])
    elif name == "radtan":
        if abs(model.k3) > 1e-12:
            warnings.warn("Kalibr radtan has no k3; dropping k3=%.3g on export." % model.k3)
        block = dict(camera_model="pinhole",
                     intrinsics=[model.fx, model.fy, model.cx, model.cy],
                     distortion_model="radtan",
                     distortion_coeffs=[model.k1, model.k2, model.p1, model.p2])
    elif name == "ucm":
        xi_mei = model.alpha / (1.0 - model.alpha)
        block = dict(camera_model="omni",
                     intrinsics=[xi_mei, model.fx, model.fy, model.cx, model.cy],
                     distortion_model="none", distortion_coeffs=[])
    elif name == "dsplus":
        block = dict(camera_model="ds_plus",
                     intrinsics=[model.alpha, model.fx, model.fy, model.cx, model.cy],
                     distortion_model="ds_plus_div_tilt",
                     distortion_coeffs=[model.lambda1, model.lambda2, model.tau_x, model.tau_y])
    elif name == "eucmplus":
        block = dict(camera_model="eucm_plus",
                     intrinsics=[model.alpha, model.beta, model.fx, model.fy, model.cx, model.cy],
                     distortion_model="eucm_plus_div_tilt",
                     distortion_coeffs=[model.lambda1, model.tau_x, model.tau_y])
    else:
        raise ValueError(f"No Kalibr mapping for model '{name}'")
    block["resolution"] = [int(width), int(height)]
    return block


def from_kalibr_cam(block: dict):
    """Reconstruct a model from a Kalibr ``camN`` stanza (dict)."""
    cm = block["camera_model"]
    dm = block.get("distortion_model", "none")
    intr = [float(v) for v in block["intrinsics"]]
    D = [float(v) for v in block.get("distortion_coeffs", [])]

    if cm == "ds":
        xi, alpha, fx, fy, cx, cy = intr
        return DoubleSphereModel(fx, fy, cx, cy, xi, alpha)
    if cm == "eucm":
        alpha, beta, fx, fy, cx, cy = intr
        return EUCMModel(fx, fy, cx, cy, alpha, beta)
    if cm == "omni":
        xi_mei, fx, fy, cx, cy = intr
        if dm not in ("none", None, ""):
            raise NotImplementedError("omni + distortion is not representable by UCMModel")
        alpha = xi_mei / (1.0 + xi_mei)
        return UCMModel(fx, fy, cx, cy, alpha)
    if cm == "ds_plus":
        alpha, fx, fy, cx, cy = intr
        lambda1, lambda2, tau_x, tau_y = D
        return DSPlusModel(fx, fy, cx, cy, alpha, lambda1, lambda2, tau_x, tau_y)
    if cm == "eucm_plus":
        alpha, beta, fx, fy, cx, cy = intr
        lambda1, tau_x, tau_y = D
        return EUCMPlusModel(fx, fy, cx, cy, alpha, beta, lambda1, tau_x, tau_y)
    if cm == "pinhole":
        fx, fy, cx, cy = intr
        if dm == "equidistant":
            k1, k2, k3, k4 = D
            return KannalaBrandtModel(fx, fy, cx, cy, k1, k2, k3, k4)
        if dm == "radtan":
            k1, k2, p1, p2 = D
            return RadTanModel(fx, fy, cx, cy, k1, k2, p1, p2, 0.0)
        if dm in ("none", None, ""):
            return RadTanModel(fx, fy, cx, cy)  # plain pinhole (zero distortion)
        raise NotImplementedError(f"Unsupported pinhole distortion_model '{dm}'")
    raise NotImplementedError(f"Unsupported camera_model '{cm}'")


def save_kalibr(model, path: str, width: int, height: int, cam: str = "cam0") -> None:
    """Write a single-camera Kalibr camchain YAML file."""
    data = {cam: to_kalibr_cam(model, width, height)}
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)


def save_kalibr_rig(rig, path: str, *, cam_order=None) -> None:
    """Write a full multi-camera Kalibr camchain: per-camera intrinsics + the chained
    extrinsics ``T_cn_cnm1`` (transform from the previous camera into this one).

    ``rig`` is a ``ds_msp.rig.RigState``; its ``T_c_g[c]`` is the projection extrinsic
    (group-ref -> camera). Kalibr wants ``T_cam_n_cam_{n-1} = T_c_g[n] @ inv(T_c_g[n-1])``.
    ``cam_order`` defaults to the sorted camera ids.
    """
    order = list(cam_order) if cam_order is not None else sorted(rig.cameras)
    data = {}
    for i, cid in enumerate(order):
        w, h = rig.img_size.get(cid, (0, 0))
        block = to_kalibr_cam(rig.cameras[cid], w, h)
        if i > 0:
            T = rig.T_c_g[cid] @ np.linalg.inv(rig.T_c_g[order[i - 1]])
            block["T_cn_cnm1"] = [list(map(float, row)) for row in T]
        data[f"cam{i}"] = block
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)


def load_kalibr(path: str, cam: str = "cam0"):
    """Read a model from a Kalibr camchain YAML file.

    Returns the model. Also accessible as ``(model, (width, height))`` via
    :func:`load_kalibr_with_resolution`.
    """
    model, _ = load_kalibr_with_resolution(path, cam)
    return model


def load_kalibr_with_resolution(path: str, cam: str = "cam0") -> Tuple[object, Tuple[int, int]]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if cam not in data:
        cam = sorted(k for k in data if k.startswith("cam"))[0]
    block = data[cam]
    model = from_kalibr_cam(block)
    res = block.get("resolution", [0, 0])
    return model, (int(res[0]), int(res[1]))


def load_kalibr_extrinsics(path: str, cam: str = "cam1") -> np.ndarray:
    """Load the stereo extrinsic ``T_cn_cnm1`` (4x4) for ``cam`` from a camchain.

    Kalibr stores, per camera, the transform from the *previous* camera into this
    one (so ``cam1``'s ``T_cn_cnm1`` is ``T_cam1_cam0``). Raises ``KeyError`` if the
    camera has no extrinsic block (e.g. the first camera in the chain).
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    T = data.get(cam, {}).get("T_cn_cnm1")
    if T is None:
        raise KeyError(f"{cam!r} has no T_cn_cnm1 in {path}")
    return np.asarray(T, dtype=np.float64)
