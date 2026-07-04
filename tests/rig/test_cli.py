"""``ds_msp.rig.cli`` -- the console-script CLI that ships with ``pip install ds-msp`` (no repo
clone needed). Verifies FR-RIG-016: the ``ds-msp-calibrate-rig`` entry point and its
``--init-config``/``--init-intrinsics`` template generation actually work end to end, resolving
the bundled templates via ``importlib.resources`` (package data) rather than a path relative to
this module's location on disk -- the same mechanism a real wheel install relies on.
"""
import sys

import cv2
import pytest

from ds_msp.rig import cli


@pytest.mark.req("FR-RIG-016")
def test_main_is_a_real_console_script_target():
    assert callable(cli.main)


@pytest.mark.req("FR-RIG-016")
def test_init_config_writes_a_valid_parseable_template(tmp_path, monkeypatch):
    out = tmp_path / "calib_param.yml"
    monkeypatch.setattr(sys, "argv", ["ds-msp-calibrate-rig", "--init-config", str(out)])
    cli.main()
    assert out.exists()
    assert out.read_text().lstrip().startswith("%YAML:1.0")
    fs = cv2.FileStorage(str(out), cv2.FILE_STORAGE_READ)
    assert fs.isOpened()
    fs.release()


@pytest.mark.req("FR-RIG-016")
def test_init_intrinsics_writes_a_valid_parseable_template(tmp_path, monkeypatch):
    out = tmp_path / "camera_intrinsics.yml"
    monkeypatch.setattr(sys, "argv", ["ds-msp-calibrate-rig", "--init-intrinsics", str(out)])
    cli.main()
    assert out.exists()
    assert out.read_text().lstrip().startswith("%YAML:1.0")
    fs = cv2.FileStorage(str(out), cv2.FILE_STORAGE_READ)
    assert fs.isOpened()
    fs.release()


@pytest.mark.req("FR-RIG-016")
def test_no_scenario_or_config_is_a_clear_argument_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ds-msp-calibrate-rig"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "scenario" in capsys.readouterr().err.lower()
