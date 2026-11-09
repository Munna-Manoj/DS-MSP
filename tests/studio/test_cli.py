"""Opening a result in Studio: `ds-msp studio --open`, the hint a calibration ends with, and
`ds-msp-calibrate-rig --studio` (FR-UI-006)."""
import socket
import sys

import pytest

from ds_msp.rig import cli as rig_cli
from ds_msp.studio import cli as studio_cli

pytestmark = pytest.mark.req("FR-UI-006")


def test_open_serves_the_parent_and_targets_the_run(results_root):
    root, run_id = studio_cli._result_target(str(results_root / "lab" / "run_a"))
    assert root == str((results_root / "lab").resolve()) and run_id == "0:run_a"
    with pytest.raises(NotADirectoryError):
        studio_cli._result_target(str(results_root / "empty_dir"))
    assert studio_cli.run(["--open", str(results_root / "empty_dir")]) == 2


def test_a_taken_port_moves_to_the_next_free_one():
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        port = busy.getsockname()[1]
        assert studio_cli._free_port("127.0.0.1", port) != port


def test_the_hint_names_the_command_and_the_extra_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(rig_cli, "_ui_available", lambda: True)
    hint = rig_cli._studio_hint(str(tmp_path))
    assert hint == f'view in DS-MSP Studio:  ds-msp studio --open "{tmp_path}"'
    monkeypatch.setattr(rig_cli, "_ui_available", lambda: False)
    assert 'pip install "ds-msp[ui]"' in rig_cli._studio_hint(str(tmp_path))


def test_studio_flag_runs_studio_in_a_child_process(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(rig_cli, "_ui_available", lambda: True)
    monkeypatch.setattr("subprocess.run", lambda argv, check: calls.append(argv))
    rig_cli._open_studio(str(tmp_path), 8800)
    assert calls == [[sys.executable, "-m", "ds_msp.cli", "studio", "--open", str(tmp_path),
                      "--port", "8800"]]
    rig_cli._open_studio("", 8800)                              # nothing saved: message only
    monkeypatch.setattr(rig_cli, "_ui_available", lambda: False)
    rig_cli._open_studio(str(tmp_path), 8800)                   # no ui extra: message only
    assert len(calls) == 1
    err = capsys.readouterr().err
    assert "saved no result folder" in err and 'pip install "ds-msp[ui]"' in err
