"""``ds_msp.calib.cli`` — the console-script CLI that ships with ``pip install ds-msp`` (no
repo clone needed). Verifies FR-CALIB-007: the ``ds-msp-calibrate`` entry point,
``--init-config`` template generation (resolved via ``importlib.resources`` package data, the
same mechanism a real wheel install relies on), and a real end-to-end run on rendered images.
"""
import sys

import cv2
import numpy as np
import pytest
import yaml

from ds_msp.calib import cli
from ds_msp.models.double_sphere import DoubleSphereModel


@pytest.mark.req("FR-CALIB-007")
def test_main_is_a_real_console_script_target():
    assert callable(cli.main)


@pytest.mark.req("FR-CALIB-007")
def test_init_config_writes_a_valid_parseable_template(tmp_path, monkeypatch):
    out = tmp_path / "calib_config.yml"
    monkeypatch.setattr(sys, "argv", ["ds-msp-calibrate", "--init-config", str(out)])
    cli.main()
    assert out.exists()
    data = yaml.safe_load(out.read_text())
    assert data["board"]["type"] == "checkerboard"


@pytest.mark.req("FR-CALIB-007")
def test_no_images_dir_or_config_is_a_clear_argument_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ds-msp-calibrate"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "images" in capsys.readouterr().err.lower()


@pytest.mark.req("FR-INTEROP-003")
def test_cli_can_request_isaac_lut_during_calibration(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cfg, **_thresholds):
        captured["cfg"] = cfg
        return 0

    monkeypatch.setattr(cli, "_run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "ds-msp-calibrate", str(tmp_path), "--rows", "5", "--cols", "6",
        "--square-size", "0.025", "--save-dir", str(tmp_path / "out"),
        "--isaac-lut", "--lut-texture-size", "320", "240", "--lut-overwrite",
    ])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    cfg = captured["cfg"]
    assert cfg.isaac_lut.enabled is True
    assert (cfg.isaac_lut.texture_width, cfg.isaac_lut.texture_height) == (320, 240)
    assert cfg.isaac_lut.overwrite is True


def _render_checkerboard(model, R, t, rows, cols, square, w, h, supersample=4):
    W, H = w * supersample, h * supersample
    img = np.full((H, W), 255, dtype=np.uint8)
    for i in range(rows + 1):
        for j in range(cols + 1):
            if (i + j) % 2 != 0:
                continue
            sq = np.array([[j, i, 0], [j + 1, i, 0], [j + 1, i + 1, 0], [j, i + 1, 0]],
                          dtype=np.float64) * square
            Xc = (R @ sq.T).T + t
            uv, valid = model.project(Xc)
            if not valid.all():
                continue
            pts = (uv * supersample).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(img, [pts], color=0)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


@pytest.mark.req("FR-CALIB-007")
def test_end_to_end_checkerboard_run_via_config(tmp_path, monkeypatch):
    """A real run: render real checkerboard images from a known camera, drive the CLI through
    --config, and confirm it detects, calibrates, and writes a real Kalibr output file. Not a
    sub-pixel-accuracy assertion (that's already covered, with controlled correspondences, by
    tests/calib/test_single_camera.py) -- this is about the CLI's own plumbing end to end."""
    rows, cols, square, w, h = 5, 6, 0.2, 960, 640
    truth = DoubleSphereModel(400, 400, 480, 320, 0.18, 0.62)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(10):
        rvec = rng.uniform(-0.3, 0.3, 3)
        tvec = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.3, 0.3), rng.uniform(1.0, 1.8)])
        R, _ = cv2.Rodrigues(rvec)
        img = _render_checkerboard(truth, R, tvec, rows, cols, square, w, h)
        cv2.imwrite(str(images_dir / f"view_{i:02d}.png"), img)

    save_dir = tmp_path / "out"
    config_path = tmp_path / "calib_config.yml"
    config_path.write_text(f"""
board:
  type: checkerboard
  rows: {rows}
  cols: {cols}
  square_size: {square}
camera_model: ds
images_path: "{images_dir}"
save_path: "{save_dir}"
max_nfev: 100
isaac_lut:
  enabled: true
  texture_width: 32
  texture_height: 24
""")
    monkeypatch.setattr(sys, "argv", ["ds-msp-calibrate", "--config", str(config_path), "--quiet"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code in (0, 1)          # PASS/WARN -> 0, FAIL -> 1; either way it ran

    out_file = save_dir / "camchain.yaml"
    assert out_file.exists()
    data = yaml.safe_load(out_file.read_text())
    assert data["cam0"]["camera_model"] == "ds"
    assert data["cam0"]["resolution"] == [w, h]
    xi, alpha, fx, fy, cx, cy = data["cam0"]["intrinsics"]      # ds_msp.io.kalibr's DS ordering
    assert all(np.isfinite(v) for v in (xi, alpha, fx, fy, cx, cy))
    assert fx > 0 and fy > 0

    import ds_msp.calib as calib
    loaded = calib.load_camera(str(out_file))
    assert isinstance(loaded, DoubleSphereModel)
    assert np.isclose(loaded.fx, fx)

    lut_dir = save_dir / "isaac_lut"
    assert len(list(lut_dir.glob("*_ray_enter_direction.exr"))) == 1
    assert len(list(lut_dir.glob("*_ray_exit_position.exr"))) == 1
    assert len(list(lut_dir.glob("*_isaac_lut.json"))) == 1
