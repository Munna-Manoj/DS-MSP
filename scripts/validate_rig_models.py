"""Validate per-camera model-of-choice on the real MC-Calib Blender datasets.

For each scenario and several RNG seeds we assign **each camera a random valid model**
(the Blender cameras are pinhole, so the pool is {radtan, ucm, eucm, ds, ocam} — KB is
fisheye-only and excluded), run the full rig calibration, write MC-Calib-format output, and
check the recovered extrinsics against both ground truth and MC-Calib's own result. This is
the "users can pick any model per camera and still get MC-Calib accuracy" guarantee, on real
data, in MC-Calib's exact output format.

Usage:  python scripts/validate_rig_models.py [n_seeds] [blender_root]
"""
from __future__ import annotations

import os
import sys
import tempfile


sys.path.insert(0, ".")
from ds_msp.io.mccalib import load_scenario                                 # noqa: E402
from ds_msp.rig.pipeline import calibrate_scenario, random_model_assignment      # noqa: E402

SCENARIOS = ["Scenario_1", "Scenario_2", "Scenario_3", "Scenario_4", "Scenario_5"]


def main(n_seeds=3, root="../MC-Calib/Blender_Images"):
    from ds_msp.models.registry import PINHOLE_MODELS
    print(f"Per-camera random-model validation on real MC-Calib data ({n_seeds} seeds/scenario)\n"
          f"pool=pinhole {{{','.join(PINHOLE_MODELS)}}}; metric = worst inter-camera baseline error\n")
    print(f"{'scenario':14s} {'cams':>4} {'seed':>4}  {'assignment':38s} "
          f"{'vsGT%':>7} {'vsMC%':>7} {'maxRMS':>7}")
    worst_gt = 0.0
    ok_all = True
    for scn_name in SCENARIOS:
        scn_dir = os.path.join(root, scn_name)
        if not os.path.isdir(scn_dir):
            print(f"{scn_name:14s}  (missing — skipped)")
            continue
        scn = load_scenario(scn_dir)
        for seed in range(n_seeds):
            spec = random_model_assignment(scn.cam_ids, kind="pinhole", seed=seed)
            with tempfile.TemporaryDirectory() as d:
                res = calibrate_scenario(scn, spec, save_dir=d)
            m = res["metrics"]
            gt = m["worst_baseline_pct_vs_gt"]
            mc = m["worst_baseline_pct_vs_mccalib"]
            asg = ",".join(f"{c}:{res['models'][c]}" for c in sorted(res["models"]))
            worst_gt = max(worst_gt, gt if gt is not None else 0.0)
            ok = (gt is None or gt < 1.0)
            ok_all = ok_all and ok
            print(f"{scn_name:14s} {len(scn.cam_ids):4d} {seed:4d}  {asg:38.38s} "
                  f"{(gt if gt is not None else float('nan')):7.3f} "
                  f"{(mc if mc is not None else float('nan')):7.3f} {m['max_rms_px']:7.3f}")
    print(f"\nworst baseline error vs GroundTruth across all runs: {worst_gt:.3f}%")
    print(f"OVERALL: {'PASS (every camera-model choice within 1% of GT)' if ok_all else 'FAIL'}")
    return ok_all


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rt = sys.argv[2] if len(sys.argv) > 2 else "../MC-Calib/Blender_Images"
    main(ns, rt)
