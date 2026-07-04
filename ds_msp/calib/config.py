"""Config-driven single-camera intrinsics calibration — board type, board geometry, and
camera model as YAML, mirroring ``ds_msp.rig.calib_param``'s config-driven pattern.

Plain ``yaml.safe_load``, not ``cv2.FileStorage``: ``ds_msp.rig.calib_param``'s
``cv2.FileStorage``-based parser (and its hardened workarounds for two real ``%YAML:1.0``
parsing quirks — see that module's ``_bool_field``) exists solely to stay byte-compatible with
MC-Calib's own config format. This config has no legacy format to mirror, ``pyyaml`` is already
a hard dependency, and it's already used for plain YAML elsewhere (``ds_msp/io/kalibr.py``) —
``true``/``false`` need no special-casing at all with ``yaml.safe_load``.

``BoardConfig.rows``/``cols`` always mean **interior corner counts** (matching OpenCV's native
checkerboard convention and ``CheckerboardSpec``), regardless of board type — for ChArUco,
:func:`build_board` converts corners to MC-Calib's square-count convention
(``n_x = cols + 1``) when building a :class:`~ds_msp.detect.charuco.BoardSpec`, so a user only
ever configures the physical thing they can count off a printed board's interior corners.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from ..detect.charuco import BoardSpec
from ..detect.checkerboard import CheckerboardSpec
from .board import AprilGridBoard, Board, CharucoBoard, CheckerboardBoard
from .targets import AprilGridTarget

_BOARD_TYPES = ("checkerboard", "charuco", "aprilgrid")


@dataclass
class BoardConfig:
    type: str = "checkerboard"                  # "checkerboard" | "charuco" | "aprilgrid"

    # checkerboard / charuco (single-board) shared geometry -- interior corner counts
    rows: int = 0
    cols: int = 0
    square_size: float = 0.0                     # metric 3-D spacing

    # charuco-only
    boards: List[Dict] = field(default_factory=list)   # multi-board override, see module docstring
    length_square: float = 0.0                   # marker-generation square size; 0 -> square_size
    length_marker: float = 0.0                   # marker-generation inner marker size
    legacy: bool = True
    tuned: bool = False
    min_corners: int = 4

    # checkerboard-only
    larger: bool = False
    marker: bool = False

    # aprilgrid-only
    tag_rows: int = 6
    tag_cols: int = 6
    tag_size: float = 0.088
    tag_spacing: float = 0.3
    family: str = "t36h11"
    scales: List[float] = field(default_factory=lambda: [1, 2, 3])
    recover: bool = False


@dataclass
class CalibConfig:
    board: BoardConfig = field(default_factory=BoardConfig)
    camera_model: str = "ds"                     # ds_msp.models.registry.model_class(name)
    images_path: Optional[str] = None
    pattern: str = "*"
    save_path: Optional[str] = None
    output_format: str = "kalibr"                # "kalibr" (v1)
    robust: str = "cauchy"
    robust_scale: "float | str" = "auto"
    gnc: bool = False
    multi_start: bool = True
    n_restarts: int = 4
    seed: int = 0
    max_nfev: int = 200
    verbose: bool = True
    raw: Dict = field(default_factory=dict)


def _coerce(value, default):
    """``--set`` values always arrive as strings; coerce to match the field's own default
    type (bool BEFORE int -- bool is an int subclass in Python)."""
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
        raise ValueError(f"expected true/false, got {value!r}")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def _apply_overrides(data: Dict, overrides: Optional[Dict]) -> Dict:
    """Shallow-dotted overrides, e.g. ``{"board.rows": "7", "save_path": "/abs/out"}``."""
    if not overrides:
        return data
    for key, value in overrides.items():
        parts = key.split(".")
        d = data
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value
    return data


def _build_dataclass(cls, raw: Dict):
    defaults = cls()
    kwargs = {}
    for k in vars(defaults):
        if k in raw:
            kwargs[k] = _coerce(raw[k], getattr(defaults, k))
    return cls(**kwargs)


def load_config(config_path: str, overrides: Optional[Dict] = None) -> CalibConfig:
    """Parse a ``calib_config.yml`` into a :class:`CalibConfig`. ``overrides`` (e.g. from
    ``--set``) are applied as shallow-dotted keys before path resolution. Relative
    ``images_path``/``save_path`` resolve against the config file's own directory."""
    base = os.path.dirname(os.path.abspath(config_path))
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    raw = _apply_overrides(dict(raw), overrides)

    board_cfg = _build_dataclass(BoardConfig, raw.get("board") or {})
    cfg = _build_dataclass(CalibConfig, {k: v for k, v in raw.items() if k != "board"})
    cfg.board = board_cfg
    cfg.raw = {"path": config_path}

    def resolve(p):
        if p is None or str(p).strip() in ("", "None"):
            return None
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

    cfg.images_path = resolve(cfg.images_path)
    cfg.save_path = resolve(cfg.save_path)
    return cfg


def build_board(cfg: CalibConfig) -> Board:
    """``cfg.board.type`` -> the matching native :class:`~ds_msp.calib.board.Board`."""
    b = cfg.board
    if b.type == "checkerboard":
        spec = CheckerboardSpec(cols=b.cols, rows=b.rows, square_size=b.square_size,
                               larger=b.larger, marker=b.marker)
        return CheckerboardBoard(spec)
    if b.type == "charuco":
        # OpenCV's CharucoBoard requires markerLength > 0 and squareLength > markerLength;
        # an unspecified length_marker must not silently become the invalid default 0.0, so
        # fall back to the same 0.75 square:marker ratio this repo's own rig templates use
        # (configs/calib_param.template.yml: length_square 0.04 / length_marker 0.03).
        def _board_spec(rows, cols, square_size, length_square=None, length_marker=None):
            ls = length_square or b.length_square or square_size
            lm = length_marker or b.length_marker or ls * 0.75
            return BoardSpec(n_x=cols + 1, n_y=rows + 1, length_square=ls,
                             length_marker=lm, square_size=square_size)

        if b.boards:
            specs = [_board_spec(d["rows"], d["cols"], d["square_size"],
                                 d.get("length_square"), d.get("length_marker"))
                    for d in b.boards]
        else:
            specs = [_board_spec(b.rows, b.cols, b.square_size)]
        return CharucoBoard(specs, legacy=b.legacy, tuned=b.tuned, min_corners=b.min_corners)
    if b.type == "aprilgrid":
        target = AprilGridTarget(tag_rows=b.tag_rows, tag_cols=b.tag_cols,
                                tag_size=b.tag_size, tag_spacing=b.tag_spacing)
        return AprilGridBoard(target, family=b.family, scales=tuple(b.scales),
                              recover=b.recover)
    raise ValueError(f"unknown board.type {b.type!r}; expected one of {_BOARD_TYPES}")
