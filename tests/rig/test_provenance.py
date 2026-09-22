"""NFR-REPRO-002: attributable rig outputs, reference runtime, and replay contract."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ds_msp.rig import provenance as prov
from ds_msp.rig.types import ObjectObs

from ._synth import make_object

pytestmark = pytest.mark.req("NFR-REPRO-002")


def _scenario(tmp_path):
    source = tmp_path / "points.yml"
    source.write_text("points: test\n", encoding="utf-8")
    obj = make_object({0: np.eye(4)})
    obs = ObjectObs(cam_id=0, frame_id=3, object_id=0,
                    point_rows=np.array([0, 2, 4]),
                    pts_2d=np.array([[10.25, 20.5], [30.0, 40.0], [50.0, 60.0]]))
    scn = SimpleNamespace(name="manifest-test", object=obj, object_obs=[obs],
                          img_size={0: (640, 480)})
    return scn, obj, source


def test_manifest_records_effective_inputs_environment_and_source_hash(tmp_path):
    scn, obj, source = _scenario(tmp_path)
    manifest = prov.build_manifest(
        scn, configuration={"models": ["radtan"], "seed": 7}, source_paths=[source],
        replay_argv=["--config", str(source)], seed=7, input_objects=obj)

    assert manifest["schema"] == prov.SCHEMA
    assert manifest["schema_version"] == 1
    assert len(manifest["run_id"]) == 20
    assert manifest["inputs"]["observations"]["n_corners"] == 3
    assert len(manifest["inputs"]["observations"]["sha256"]) == 64
    assert len(manifest["inputs"]["target_geometry"]["sha256"]) == 64
    assert manifest["inputs"]["files"][0]["sha256"] == prov.sha256_file(source)
    assert manifest["environment"]["packages"]["numpy"]
    assert "matches" in manifest["environment"]["reference"]
    assert manifest["invocation"]["seed"] == 7


def test_effective_input_digest_changes_for_one_pixel_or_target_coordinate(tmp_path):
    scn, obj, _source = _scenario(tmp_path)
    obs_before = prov.observation_digest(scn)["sha256"]
    target_before = prov.target_geometry_digest(obj)["sha256"]
    scn.object_obs[0].pts_2d[0, 0] += 1e-6
    assert prov.observation_digest(scn)["sha256"] != obs_before
    obj.pts_3d[0, 0] += 1e-9
    assert prov.target_geometry_digest(obj)["sha256"] != target_before


def test_manifest_writer_hashes_outputs_and_excludes_itself(tmp_path):
    scn, obj, source = _scenario(tmp_path)
    artifact = tmp_path / "calibrated.yml"
    artifact.write_text("answer: 42\n", encoding="utf-8")
    path = tmp_path / prov.MANIFEST_NAME
    manifest = prov.build_manifest(scn, source_paths=[source], input_objects=obj)
    prov.write_manifest(path, manifest, {"calibrated": artifact, "manifest": path})

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["outputs"]["calibrated"]["sha256"] == prov.sha256_file(artifact)
    assert "manifest" not in loaded["outputs"]
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_reference_lock_and_shipped_profile_cannot_drift():
    root = Path(__file__).resolve().parents[2]
    pins = {}
    for line in (root / "requirements-production.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            pins[name] = version
    assert pins == prov.load_reference_environment()["packages"]


def test_git_dirty_means_tracked_change_while_untracked_paths_are_separate(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("committed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run([
        "git", "-C", str(tmp_path), "-c", "user.name=Test", "-c",
        "user.email=test@example.invalid", "commit", "-qm", "initial",
    ], check=True)
    (tmp_path / "dataset.bin").write_bytes(b"input")
    package_subdir = tmp_path / "package" / "module"
    package_subdir.mkdir(parents=True)
    state = prov._git_state(package_subdir)
    assert state["dirty"] is False
    assert state["n_changed_paths"] == 0
    assert state["n_untracked_paths"] == 1
    tracked.write_text("modified\n", encoding="utf-8")
    assert prov._git_state(tmp_path)["dirty"] is True
