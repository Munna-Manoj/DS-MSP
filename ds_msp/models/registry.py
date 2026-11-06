"""Model-name <-> class registry, with MC-Calib name aliases.

MC-Calib selects a camera's model by a string in its config (``camera_models: [radtan,
double_sphere, ...]`` or the legacy ``distortion_per_camera`` ints 0=radtan / 1=kb) and
writes that string back as ``camera_model`` in ``calibrated_cameras_data.yml``. DS-MSP's
own model ``.name`` is ``"ds"`` where MC-Calib uses ``"double_sphere"``; everything else
matches. This module is the single place that bridges the two so the rig front-end and the
MC-Calib reader/writer stay byte-compatible.
"""

from __future__ import annotations

from typing import Dict, Type

import numpy as np

from ..core.contracts import CameraModel
from .double_sphere import DoubleSphereModel
from .eucm import EUCMModel
from .kb import KannalaBrandtModel
from .radtan import RadTanModel
from .ucm import UCMModel

#: Canonical DS-MSP name -> class.
_BY_NAME: Dict[str, Type[CameraModel]] = {
    "radtan": RadTanModel,
    "ds": DoubleSphereModel,
    "ucm": UCMModel,
    "eucm": EUCMModel,
    "kb": KannalaBrandtModel,
}

#: Accepted aliases (MC-Calib strings + legacy ints) -> canonical DS-MSP name.
_ALIAS: Dict[str, str] = {
    "double_sphere": "ds", "doublesphere": "ds",
    "kannala": "kb", "kannala_brandt": "kb", "fisheye": "kb",
    "brown": "radtan", "perspective": "radtan", "opencv": "radtan",
    "0": "radtan", "1": "kb",                      # legacy distortion_model ints
}

#: DS-MSP name -> the string MC-Calib writes in calibrated_cameras_data.yml.
_TO_MCCALIB: Dict[str, str] = {
    "radtan": "radtan", "ds": "double_sphere", "ucm": "ucm",
    "eucm": "eucm", "kb": "kb",
}

#: Models valid for a (forward-facing) pinhole camera, and for a fisheye camera, used by the
#: random model-of-choice assignment. KB is fisheye-only; RadTan/Brown is pinhole-only; UCM /
#: EUCM / Double-Sphere handle both.
PINHOLE_MODELS = ("radtan", "ucm", "eucm", "ds")
FISHEYE_MODELS = ("kb", "ucm", "eucm", "ds")


#: Models removed by ADR-0031: both leave rig extrinsics unobservable. Named here so
#: an old config or calibration file fails with the reason instead of "unknown model".
_REMOVED = {
    "ocam": "OCam", "ocam_calib": "OCam", "ocamcalib": "OCam", "scaramuzza": "OCam",
    "dsplus": "DS+", "ds+": "DS+", "ds_plus": "DS+", "doublespheres_plus": "DS+",
}


def canonical_name(name) -> str:
    """Normalize any accepted model spec (DS-MSP name, MC-Calib name, or legacy int)."""
    s = str(name).strip().lower()
    if s in _BY_NAME:
        return s
    if s in _ALIAS:
        return _ALIAS[s]
    if s in _REMOVED:
        raise KeyError(f"camera model {name!r} ({_REMOVED[s]}) was removed from DS-MSP: "
                       f"its extra terms are indistinguishable from camera rotation in a rig "
                       f"(ADR-0031). Use one of {sorted(_BY_NAME)}; KB or DS fit the same "
                       f"wide-angle lenses.")
    raise KeyError(f"unknown camera model {name!r}; "
                   f"known: {sorted(_BY_NAME)} (+ aliases {sorted(_ALIAS)})")


def model_class(name) -> Type[CameraModel]:
    """Resolve a model spec to its DS-MSP class."""
    return _BY_NAME[canonical_name(name)]


def mccalib_name(name) -> str:
    """The string MC-Calib uses for this model (``ds`` -> ``double_sphere``)."""
    return _TO_MCCALIB[canonical_name(name)]


# Neutral seed values per intrinsic parameter name — a generic, GT-free starting point for
# from-scratch calibration of any model. Distortion/shape params not listed here seed to 0.0
# (see seed_from_K), so any model is supported without enumerating every parameter name. Moved here (from ds_msp.rig.calibrate, its original home) rather than
# ds_msp.geometry.resection: this is trivial, mechanical plumbing (instantiate model_cls with
# neutral defaults, no cv2/scipy) that belongs with the other model-registry helpers.
_NEUTRAL = {"alpha": 0.5, "xi": 0.0, "beta": 1.0, "k1": 0.0, "k2": 0.0, "k3": 0.0,
            "k4": 0.0, "p1": 0.0, "p2": 0.0}


def seed_from_K(model_cls: Type[CameraModel], K: np.ndarray) -> CameraModel:
    """Build a seed instance of ``model_cls`` from a pinhole ``K`` (focal + principal point)
    with neutral distortion — the from-scratch starting point for any model. Shared by
    ``ds_msp.rig`` (per-camera intrinsics front-end) and ``ds_msp.calib`` (single-camera
    calibration)."""
    if not set(model_cls.param_names) >= {"fx", "fy"}:
        # A model without fx/fy (none ship today): seed the principal point only and let
        # `initialize_from_correspondences` fit the rest from rays.
        return model_cls(K[0, 2], K[1, 2])
    vals = {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2], **_NEUTRAL}
    # Unknown shape/distortion params seed to 0.0 (neutral) so any model is supported.
    vec = np.array([vals.get(n, 0.0) for n in model_cls.param_names], float)
    return model_cls.from_params(vec)
