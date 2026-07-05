"""``ds_msp.calib.config`` — YAML-driven single-camera calibration config, mirroring
``ds_msp.rig.calib_param``'s pattern with a plain ``yaml.safe_load`` parser (see the module
docstring for why no ``cv2.FileStorage`` quirks apply here)."""
import importlib.resources
import textwrap

import pytest

from ds_msp.calib.board import AprilGridBoard, CharucoBoard, CheckerboardBoard
from ds_msp.calib.config import BoardConfig, CalibConfig, build_board, load_config

_TEMPLATE = importlib.resources.files("ds_msp.calib") / "configs" / "calib_config.template.yml"


def _write_cfg(tmp_path, body):
    p = tmp_path / "calib_config.yml"
    p.write_text(textwrap.dedent(body))
    return str(p)


def test_template_exists_and_parses():
    assert _TEMPLATE.exists(), f"missing base config template at {_TEMPLATE}"
    cfg = load_config(str(_TEMPLATE))
    assert cfg.board.type == "checkerboard"
    assert cfg.board.rows > 0 and cfg.board.cols > 0
    assert cfg.camera_model


def test_load_config_types_and_path_resolution(tmp_path):
    cfgp = _write_cfg(tmp_path, """
        board:
          type: charuco
          rows: 5
          cols: 6
          square_size: 0.192
        camera_model: kb
        images_path: "./images"
        save_path: "./out"
        gnc: true
        max_nfev: 150
        verbose: false
    """)
    cfg = load_config(cfgp)
    assert cfg.board.type == "charuco"
    assert cfg.camera_model == "kb"
    assert cfg.images_path == str(tmp_path / "images")
    assert cfg.save_path == str(tmp_path / "out")
    assert cfg.gnc is True and isinstance(cfg.gnc, bool)
    assert cfg.max_nfev == 150 and isinstance(cfg.max_nfev, int)
    assert cfg.verbose is False


def test_load_config_overrides(tmp_path):
    cfgp = _write_cfg(tmp_path, """
        board:
          type: checkerboard
          rows: 5
          cols: 6
          square_size: 0.025
    """)
    cfg = load_config(cfgp, overrides={"board.rows": "7", "save_path": "/abs/out",
                                       "gnc": "true"})
    assert cfg.board.rows == 7
    assert cfg.save_path == "/abs/out"
    assert cfg.gnc is True


def test_build_board_checkerboard():
    cfg = CalibConfig(board=BoardConfig(type="checkerboard", rows=5, cols=6, square_size=0.025))
    board = build_board(cfg)
    assert isinstance(board, CheckerboardBoard)
    assert board.spec.rows == 5 and board.spec.cols == 6 and board.spec.square_size == 0.025


def test_build_board_charuco_single_converts_corners_to_squares():
    cfg = CalibConfig(board=BoardConfig(type="charuco", rows=5, cols=6, square_size=0.192,
                                        length_marker=0.03))
    board = build_board(cfg)
    assert isinstance(board, CharucoBoard)
    spec = board.specs[0]
    assert (spec.n_x, spec.n_y) == (7, 6)          # squares = corners + 1
    assert spec.n_corners == 30
    assert spec.length_square == 0.192 and spec.length_marker == 0.03


def test_build_board_charuco_length_marker_defaults_to_075_ratio():
    """OpenCV's CharucoBoard requires markerLength > 0 and squareLength > markerLength -- an
    unspecified length_marker must not silently become the invalid default 0.0."""
    cfg = CalibConfig(board=BoardConfig(type="charuco", rows=5, cols=6, square_size=0.192))
    board = build_board(cfg)
    assert board.specs[0].length_marker == pytest.approx(0.192 * 0.75)


def test_build_board_charuco_multi_board():
    cfg = CalibConfig(board=BoardConfig(type="charuco", boards=[
        {"rows": 4, "cols": 4, "square_size": 0.192},
        {"rows": 3, "cols": 3, "square_size": 0.1, "length_marker": 0.07},
    ]))
    board = build_board(cfg)
    assert len(board.specs) == 2
    assert board.specs[0].square_size == 0.192 and board.specs[1].square_size == 0.1
    assert board.specs[1].length_marker == 0.07


def test_build_board_aprilgrid():
    cfg = CalibConfig(board=BoardConfig(type="aprilgrid", tag_rows=6, tag_cols=6,
                                        tag_size=0.088, tag_spacing=0.3))
    board = build_board(cfg)
    assert isinstance(board, AprilGridBoard)
    assert board.target.tag_rows == 6 and board.target.tag_cols == 6
    assert board.target.tag_size == 0.088


def test_build_board_unknown_type_raises():
    cfg = CalibConfig(board=BoardConfig(type="not-a-board"))
    with pytest.raises(ValueError, match="unknown board.type"):
        build_board(cfg)


# Traceability: links this suite to the requirement(s) it verifies.
pytestmark = pytest.mark.req("FR-CALIB-007")
