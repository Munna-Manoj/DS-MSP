"""Validate ds_msp.rig against MC-Calib's Blender benchmark.

Runs the rig calibrator on a scenario's detected keypoints and compares recovered
extrinsics + reprojection RMS to (a) the synthetic ground truth and (b) MC-Calib's own
result. Reports per-camera baseline / rotation error and pass/fail at a 2% threshold.

Usage:  python scripts/validate_rig.py <Scenario_dir>
"""

from __future__ import annotations

import sys

import numpy as np

from ds_msp.io.mccalib import load_scenario
from ds_msp.rig import calibrate_rig
from ds_msp.rig import bundle


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def rel(pose_ref, pose_i):
    """Relative transform ref-frame -> i-frame from two 'group-ref->cam' style poses."""
    return pose_i @ np.linalg.inv(pose_ref)


def main(scn_dir):
    scn = load_scenario(scn_dir)
    print(f"=== {scn.name}: {len(scn.cam_ids)} cameras, "
          f"{len(scn.object_obs)} object-observations, "
          f"{scn.object.pts_3d.shape[0]} object points ===")

    rig = calibrate_rig(scn.object, scn.object_obs, scn.img_size,
                        fix_intrinsics=False, verbose=True)

    rms = bundle.reprojection_rms(rig, scn.object_obs)
    print("\nper-camera reprojection RMS (px):")
    for c in sorted(rms):
        print(f"  cam {c}: {rms[c]:.4f}")

    ref = rig.ref_cam_id
    print(f"\nextrinsics vs references (ref cam = {ref}):")
    print(f"{'cam':>4} {'baseline_m':>11} {'gt_base':>9} {'mc_base':>9} "
          f"{'t_err%':>7} {'rot_gt°':>8} {'rot_mc°':>8}  verdict")
    worst = 0.0
    for c in sorted(rig.T_c_g):
        if c == ref:
            continue
        # rig.T_c_g is the projection extrinsic (ref-frame -> camera, i.e. world-to-cam);
        # GT/MC-Calib store camera-to-world. Invert to compare in their convention.
        T_mine = np.linalg.inv(rel(rig.T_c_g[ref], rig.T_c_g[c]))
        base = np.linalg.norm(T_mine[:3, 3])
        row = [f"{c:>4}", f"{base:>11.5f}"]
        # ground truth
        terr = rot_gt = float("nan")
        if scn.gt:
            T_gt = rel(scn.gt[ref].pose, scn.gt[c].pose)
            gtb = np.linalg.norm(T_gt[:3, 3])
            terr = 100.0 * abs(base - gtb) / max(gtb, 1e-9)
            rot_gt = rot_angle_deg(T_mine[:3, :3] @ T_gt[:3, :3].T)
            row += [f"{gtb:>9.5f}"]
        else:
            row += [f"{'—':>9}"]
        # mc-calib
        rot_mc = float("nan")
        if scn.mccalib:
            T_mc = rel(scn.mccalib[ref].pose, scn.mccalib[c].pose)
            mcb = np.linalg.norm(T_mc[:3, 3])
            rot_mc = rot_angle_deg(T_mine[:3, :3] @ T_mc[:3, :3].T)
            row += [f"{mcb:>9.5f}"]
        else:
            row += [f"{'—':>9}"]
        row += [f"{terr:>7.2f}", f"{rot_gt:>8.3f}", f"{rot_mc:>8.3f}"]
        verdict = "OK" if (terr < 2.0 and rot_gt < 1.0) else "FAIL"
        worst = max(worst, terr)
        print(" ".join(row) + f"  {verdict}")

    print(f"\nworst translation error vs GT: {worst:.2f}%  "
          f"({'PASS' if worst < 2.0 else 'FAIL'} @ 2%)")
    return worst


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: validate_rig.py <scenario_dir>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
