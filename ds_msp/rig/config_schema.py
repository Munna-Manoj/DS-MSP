"""JSON Schema of the rig ``calib_param.yml`` keys, for forms and validation.

One schema serves the CLI file, the Studio forms and any other front end, so the three cannot
drift apart. Keys are the names written in ``calib_param.yml`` (MC-Calib's plus the DS-MSP
extensions in :data:`ds_msp.rig.calib_param._KNOWN_KEYS`). ``default`` is the value
:func:`ds_msp.rig.calib_param.load_config` uses when the key is absent;
``tests/rig/test_config_schema.py`` checks both against the loader.

Each property also carries ``x-group`` (the section a form shows it in) and ``x-level``
(``basic`` or ``expert``).
"""
from __future__ import annotations

from typing import Any, Dict, List

SCHEMA_ID = "ds-msp-rig-config"
SCHEMA_VERSION = 1

GATE_VALUES = ["off", "warn", "refuse"]
MODEL_VALUES = ["radtan", "ucm", "eucm", "ds", "kb"]

_NUM = "number"
_INT = "integer"
_STR = "string"
_BOOL = "boolean"


def _p(type_: Any, default: Any, group: str, description: str, *, level: str = "basic",
       enum: List[Any] | None = None, items: Dict[str, Any] | None = None,
       minimum: float | None = None) -> Dict[str, Any]:
    prop: Dict[str, Any] = {"type": type_, "default": default, "description": description,
                            "x-group": group, "x-level": level}
    if enum is not None:
        prop["enum"] = enum
    if items is not None:
        prop["items"] = items
    if minimum is not None:
        prop["minimum"] = minimum
    return prop


def _gate(default: str, group: str, description: str, level: str = "expert") -> Dict[str, Any]:
    return _p(_STR, default, group, description, level=level, enum=GATE_VALUES)


_OPT_STR = [_STR, "null"]
_OPT_NUM = [_NUM, "null"]
_NUM_LIST = {"type": _NUM}
_INT_LIST = {"type": _INT}

#: Every key, in the order a form lists them.
PROPERTIES: Dict[str, Dict[str, Any]] = {
    # Boards
    "number_x_square": _p(_INT, 5, "boards", "Squares along X on each ChArUco board.", minimum=2),
    "number_y_square": _p(_INT, 5, "boards", "Squares along Y on each ChArUco board.", minimum=2),
    "length_square": _p(_NUM, 0.04, "boards", "Square length used to generate the board "
                        "(sets the marker geometry).", level="expert"),
    "length_marker": _p(_NUM, 0.03, "boards", "Marker length used to generate the board; "
                        "smaller than length_square.", level="expert"),
    "number_board": _p(_INT, 1, "boards", "Physical boards in the rigid target.", minimum=1),
    "boards_index": _p("array", [], "boards", "Marker-id offset of each board; empty means "
                       "boards 0 to number_board - 1.", level="expert", items=_INT_LIST),
    "square_size": _p(_NUM, 1.0, "boards", "Printed square size in your length unit; sets the "
                      "metric scale of every output.", minimum=0),
    "number_x_square_per_board": _p("array", [], "boards", "Per-board squares along X when the "
                                    "boards differ; empty means all equal.", level="expert",
                                    items=_INT_LIST),
    "number_y_square_per_board": _p("array", [], "boards", "Per-board squares along Y when the "
                                    "boards differ.", level="expert", items=_INT_LIST),
    "square_size_per_board": _p("array", [], "boards", "Per-board square size when the boards "
                                "differ.", level="expert", items=_NUM_LIST),
    # Cameras
    "number_camera": _p(_INT, None, "cameras", "Cameras in the rig. Required.", minimum=1),
    "camera_models": _p("array", [], "cameras", "Lens model per camera; overrides "
                        "distortion_model when given.", items={"type": _STR, "enum": MODEL_VALUES}),
    "distortion_model": _p(_INT, 0, "cameras", "MC-Calib lens model for all cameras: "
                           "0 = radtan, 1 = kb.", level="expert", enum=[0, 1]),
    "distortion_per_camera": _p("array", [], "cameras", "MC-Calib lens model per camera "
                                "(0 or 1).", level="expert", items={"type": _INT, "enum": [0, 1]}),
    "cam_params_path": _p(_OPT_STR, None, "cameras", "Initial intrinsics file; empty to "
                          "estimate the lenses from the images."),
    "fix_intrinsic": _p(_BOOL, False, "cameras", "Hold the intrinsics from cam_params_path "
                        "fixed and solve only the extrinsics."),
    # Images
    "root_path": _p(_OPT_STR, None, "images", "Folder that holds one subfolder of images per "
                    "camera."),
    "cam_prefix": _p(_STR, "Cam_", "images", "Camera subfolder prefix; camera 0 is "
                     "<prefix>001."),
    "keypoints_path": _p(_OPT_STR, None, "images", "Pre-detected keypoints file; empty to "
                         "detect the boards in the images.", level="expert"),
    "exclude_views": _p("array", [], "images", "Views to leave out, as cam:frame or "
                        "cam:frame:board (camera counted from 0; frame and board numbers as in "
                        "the result files).", level="expert", items={"type": _STR}),
    "frame_pattern": _p(_OPT_STR, None, "images", "Regex whose group 'frame' is the frame "
                        "number in a file name.", level="expert"),
    "keypoints_pixel_convention": _p(_OPT_STR, None, "images", "Pixel convention of a "
                                     "keypoints file; empty to use the file's label.",
                                     level="expert"),
    "object_path": _p(_OPT_STR, None, "images", "Pre-built target file; empty to build the "
                      "target from the detections.", level="expert"),
    # Solver
    "ransac_threshold": _p(_NUM, 10.0, "solver", "Pixel threshold for dropping a board "
                           "observation when reject_outliers is on.", level="expert", minimum=0),
    "reject_outliers": _p(_BOOL, False, "solver", "Hard-drop board observations above "
                          "ransac_threshold.", level="expert"),
    "number_iterations": _p(_INT, 1000, "solver", "Iteration cap of the bundle adjustment.",
                            level="expert", minimum=1),
    "he_approach": _p(_INT, 0, "solver", "Hand-eye method: 0 = bootstrapped, 1 = traditional.",
                      level="expert", enum=[0, 1]),
    "noise_bound": _p(_OPT_NUM, 1.0, "solver", "Expected corner noise in pixels; also the "
                      "default lens limit.", level="expert"),
    "structure_mode": _p(_STR, "rigid", "solver", "Target refinement: rigid, bend, free or off.",
                         level="expert", enum=["rigid", "bend", "free", "off"]),
    "refine_structure": _p(_BOOL, False, "solver", "Legacy: true means structure_mode free.",
                           level="expert"),
    "structure_prior_sigma": _p(_OPT_NUM, None, "solver", "Prior on target shape in target "
                                "units (free mode).", level="expert"),
    # Checks
    "profile": _p(_STR, "default", "checks", "production sets the strict gates and refuses "
                  "unknown keys.", enum=["default", "production"]),
    "audit_gate": _gate("warn", "checks", "Observability audit: weakly determined directions."),
    "uncertainty_gate": _gate("warn", "checks", "Extrinsic 1-sigma limits."),
    "max_rotation_sigma_deg": _p(_OPT_NUM, None, "checks", "Rotation 1-sigma limit per pair, "
                                 "degrees.", level="expert"),
    "max_centre_sigma": _p(_OPT_NUM, None, "checks", "Centre 1-sigma limit per pair, square_size "
                           "units.", level="expert"),
    "convergence_gate": _gate("warn", "checks", "Final bundle-adjustment stage converged."),
    "coverage_gate": _gate("warn", "checks", "Share of each camera's corners the fit explains."),
    "min_camera_coverage": _p(_NUM, 0.75, "checks", "Lowest allowed share of a camera's "
                              "corners.", level="expert", minimum=0),
    "lens_gate": _gate("warn", "checks", "Per-camera lens 1-sigma in pixels and parameters at "
                       "a bound."),
    "max_lens_sigma_px": _p(_OPT_NUM, None, "checks", "Lens limit in pixels; empty to use "
                            "noise_bound.", level="expert"),
    "crossval_gate": _gate("off", "checks", "Split cross-validation of each lens on two halves "
                           "of one capture. Opt-in; a second capture validates a lens better."),
    "accuracy_gate": _p(_STR, "off", "checks", "Refuse when independent ground truth exceeds the "
                        "accuracy profile limits.", level="expert", enum=["off", "refuse"]),
    "accuracy_profile_path": _p(_OPT_STR, None, "checks", "Accuracy profile: unit, region of "
                                "interest, pairs and limits.", level="expert"),
    "report_covariance": _p(_BOOL, False, "checks", "Report the parameter covariance.",
                            level="expert"),
    "certify": _p(_BOOL, False, "checks", "Compute the rotation certificate.", level="expert"),
    # Output
    "save_path": _p(_OPT_STR, None, "output", "Results folder."),
    "camera_params_file_name": _p(_STR, "", "output", "Cameras file name; empty for "
                                  "calibrated_cameras_data.yml.", level="expert"),
    "save_detection": _p(_BOOL, False, "output", "Write images with the detected corners.",
                         level="expert"),
    "save_reprojection": _p(_BOOL, False, "output", "Write images with detected and reprojected "
                            "corners.", level="expert"),
    "verbose": _p(_BOOL, True, "output", "Print progress while calibrating.", level="expert"),
    "webviewer": _p(_BOOL, True, "output", "Open the live 3D view during a CLI run.",
                    level="expert"),
}

GROUPS = [
    {"id": "boards", "title": "Boards"},
    {"id": "cameras", "title": "Cameras"},
    {"id": "images", "title": "Images"},
    {"id": "solver", "title": "Solver"},
    {"id": "checks", "title": "Checks"},
    {"id": "output", "title": "Output"},
]


def config_schema() -> Dict[str, Any]:
    """The JSON Schema (draft 2020-12) of ``calib_param.yml``, with form hints."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"{SCHEMA_ID}/v{SCHEMA_VERSION}",
        "title": "DS-MSP rig calibration config (calib_param.yml)",
        "type": "object",
        "required": ["number_camera"],
        "properties": {k: dict(v) for k, v in PROPERTIES.items()},
        "x-groups": [dict(g) for g in GROUPS],
        "x-schema-version": SCHEMA_VERSION,
    }
