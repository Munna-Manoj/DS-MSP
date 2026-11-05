"""The scenario CLI keeps supplied calibration independent from evaluation truth."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ds_msp.io.mccalib import CameraGT
from ds_msp.rig import cli


def _scenario_cli(tmp_path, monkeypatch, cameras):
    result_dir = tmp_path / "Results"
    result_dir.mkdir()
    (result_dir / "calibrated_cameras_data.yml").touch()
    scenario = SimpleNamespace(
        name="native-lens", object=None, object_obs=[], cam_ids=[0], mccalib=cameras,
        # Different truth parameters ensure the evaluation source is never used.
        gt={0: CameraGT(K=np.diag([900., 900., 1.]), dist=np.zeros(5),
                        pose=np.eye(4), model_name="radtan")},
    )
    monkeypatch.setattr(cli, "load_scenario", lambda _: scenario)
    monkeypatch.setattr("ds_msp.rig.web3d.WebLive3DAnimator",
                        lambda **_: SimpleNamespace(bind_scene=lambda *args: None))
    monkeypatch.setattr(sys, "argv", ["ds-msp-calibrate-rig", str(tmp_path),
                                     "--fix-intrinsics", "--quiet", "--no-webviewer"])
    return scenario


def test_scenario_cli_passes_exact_native_kb_intrinsics(tmp_path, monkeypatch):
    camera = CameraGT(
        K=np.array([[420., 0., 320.], [0., 421., 240.], [0., 0., 1.]]),
        dist=np.array([-.02, .004, -.001, .0003]), pose=np.eye(4), model_name="kb")
    scenario = _scenario_cli(tmp_path, monkeypatch, {0: camera})
    captured = {}

    class CalibrationReached(Exception):
        pass

    def calibrate(scn, spec, **kwargs):
        captured.update(scenario=scn, spec=spec, **kwargs)
        raise CalibrationReached

    monkeypatch.setattr(cli, "calibrate_scenario", calibrate)
    with pytest.raises(CalibrationReached):
        cli.main()

    assert captured["scenario"] is scenario
    assert captured["spec"] == {0: "kb"}
    assert captured["fix_intrinsics"] is True
    supplied = captured["init_cameras"][0]
    assert supplied.name == "kb"
    np.testing.assert_array_equal(supplied.params, [420., 421., 320., 240., *camera.dist])


def test_scenario_cli_rejects_missing_camera_intrinsics(tmp_path, monkeypatch, capsys):
    _scenario_cli(tmp_path, monkeypatch, {})
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "missing cameras: [0]" in capsys.readouterr().err


def test_scenario_cli_rejects_missing_calibrated_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ds-msp-calibrate-rig", str(tmp_path), "--fix-intrinsics"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "needs Results/calibrated_cameras_data.yml" in capsys.readouterr().err
