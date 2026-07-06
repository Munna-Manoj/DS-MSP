#!/usr/bin/env python3
"""
Record every WebLive3DAnimator state snapshot from a real rig calibration run into one gzip-
compressed JSON array, for a static "replay" of the pond visualization that doesn't need a live
Python backend (embedded in the published docs / GitHub Pages, which cannot run a server per
visitor).

    python scripts/make_rig_pond_replay.py     # writes docs/assets/rig_pond_replay/frames.json.gz

Floats are rounded to 4 decimal places before serializing (0.1mm precision on meter-scale
positions -- far finer than anything a viewer could perceive) and the JSON is minified, then
gzip-compressed. Measured on this exact recording: 16.3 MB raw -> 8.0 MB rounded+minified ->
0.38 MB gzip-compressed, with all frames kept (no subsampling). A browser can decompress this
client-side with the native ``DecompressionStream('gzip')`` Web API -- no external JS library.

Uses the cameras' real, pre-provided original intrinsics (KB for the 6 fisheye cameras,
RadTan for the 2 pinhole cameras) seeded from the dataset's own ground-truth
calibrated_cameras_data.yml, converting kb->dsplus for the fisheye cameras (radtan stays
native), then refining (fix_intrinsic=0, intrinsics optimization on) --
calib_param.seeded_dsplus_fisheye_radtan_pinhole.yml, the "convert kb->dsplus (warn+convert)"
scenario in RIG_CALIBRATION_GUIDE.md §7 (re-measured: 0.5470px, from a genuinely fresh
detection + object reconstruction -- see _try_load_object's fix in ds_msp/rig/calib_param.py:
it used to silently reuse a stray calibrated_objects_data.yml left in save_path by an earlier
run, which is exactly what this script's save_path pointed at; that's fixed now, so a plain
run of this config -- no keypoints_path/object_path override -- is a genuine from-scratch
recording, not an accidental warm start). Needs the real 8-camera capture present locally to
re-run (2026_06_26_MC-Calib/, git-ignored -- see datasets/README.md); only the output
frames.json (not the raw images) gets committed.
"""
from __future__ import annotations

import copy
import gzip
import json
import os

import ds_msp.rig.calib_param as calib_param_mod
import ds_msp.rig.web3d as web3d_mod
from ds_msp.rig.bundle import reprojection_metrics
from ds_msp.rig.calib_param import calibrate_from_config

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(HERE, "2026_06_26_MC-Calib")
CONFIG_PATH = os.path.join(DATASET_DIR, "calib_param.seeded_dsplus_fisheye_radtan_pinhole.yml")
OUT_PATH = os.path.join(HERE, "docs", "assets", "rig_pond_replay", "frames.json.gz")
_ROUND_DP = 4


def _round_floats(obj):
    """Recursively round every float in a JSON-safe structure to _ROUND_DP places -- 0.1mm on
    meter-scale positions, far finer than a viewer could ever perceive, but it roughly halves
    the serialized size (17-significant-digit doubles as text are the single biggest source of
    bloat in this recording) and makes the remaining digits far more repetitive across frames,
    which is exactly what gzip exploits."""
    if isinstance(obj, float):
        return round(obj, _ROUND_DP)
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v) for v in obj]
    return obj


def main() -> None:
    if not os.path.isdir(DATASET_DIR):
        raise SystemExit(
            f"real 8-camera capture not found at {DATASET_DIR} -- this script replays a real "
            "run, it does not fabricate one; see RIG_CALIBRATION_GUIDE.md §7"
        )

    frames: list = []
    orig_write_state = web3d_mod.WebLive3DAnimator._write_state

    def recording_write_state(self, state):
        frames.append(copy.deepcopy(web3d_mod._json_safe(state)))
        orig_write_state(self, state)

    # Capture the exact object_obs calibrate_scenario used, so the final RMS/median can be
    # recomputed directly with ds_msp.rig.bundle.reprojection_metrics (the codebase's own,
    # unmodified formula: per-camera sqrt(mean(error**2)) plus np.median(error)) -- not a
    # formula I invented for this script.
    captured = {}
    orig_calibrate_scenario = calib_param_mod.calibrate_scenario

    def capturing_calibrate_scenario(scn, *args, **kwargs):
        captured["object_obs"] = scn.object_obs
        return orig_calibrate_scenario(scn, *args, **kwargs)

    web3d_mod.WebLive3DAnimator._write_state = recording_write_state
    calib_param_mod.calibrate_scenario = capturing_calibrate_scenario
    try:
        # No overrides beyond webviewer/verbose: root_path/keypoints_path/object_path stay as
        # the config's own ("images" / "None" / "None"), so this is a genuine from-scratch
        # detection + object reconstruction, exactly like a real user's plain
        # `ds-msp-calibrate-rig --config calib_param.seeded_dsplus_fisheye_radtan_pinhole.yml`
        # invocation. save_path is left at the config's own value too -- it's just this run's
        # output location now, no longer an implicit input (see _try_load_object's fix).
        res = calibrate_from_config(
            CONFIG_PATH,
            overrides={"webviewer": "true", "verbose": "true"},
        )
    finally:
        web3d_mod.WebLive3DAnimator._write_state = orig_write_state
        calib_param_mod.calibrate_scenario = orig_calibrate_scenario

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    payload = json.dumps(_round_floats(frames), separators=(",", ":")).encode()
    with gzip.open(OUT_PATH, "wb", compresslevel=9) as f:
        f.write(payload)

    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"recorded {len(frames)} frames -> {OUT_PATH} ({size_mb:.3f} MB gzip-compressed)")
    per_cam = reprojection_metrics(res["rig"], captured["object_obs"])
    means = [v["rms"] for v in per_cam.values()]
    medians = [v["median"] for v in per_cam.values()]
    for c, v in sorted(per_cam.items()):
        print(f"  cam {c}: rms={v['rms']:.4f}px  median={v['median']:.4f}px  "
              f"inlier_frac={v['inlier_frac']:.3f}")
    print(f"mean-of-per-cam RMS:    {sum(means) / len(means):.4f} px "
          f"(cross-check vs RIG_CALIBRATION_GUIDE.md §7: 0.5470 px)")
    print(f"mean-of-per-cam median: {sum(medians) / len(medians):.4f} px")


if __name__ == "__main__":
    main()
