"""Speed benchmark for the rig calibration pipeline — breaks total wall-clock into
front-end (per-camera intrinsics) vs global BA, so optimization work has a baseline.

Usage:  python scripts/benchmark_rig.py [repeats]
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
from tests.rig._synth import make_rig                          # noqa: E402
from ds_msp.rig import bundle                                      # noqa: E402
from ds_msp.rig.calibrate import (                         # noqa: E402
    _front_end_opencv, calibrate_rig)
from ds_msp.rig.extrinsics import init_camera_groups           # noqa: E402
from ds_msp.rig.pose_init import average_object_pose_in_group  # noqa: E402
from ds_msp.rig.types import RigState                          # noqa: E402


def _median(fn, repeats):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def bench(n_cam, n_frame, repeats, label):
    obj, obs, img, gt, _ = make_rig(n_cam=n_cam, n_frame=n_frame, noise_px=0.3, seed=0)
    n_obs = len(obs)
    n_pts = sum(len(o.point_rows) for o in obs)

    # full pipeline
    t_total = _median(lambda: calibrate_rig(obj, obs, img, fix_intrinsics=False), repeats)

    # stage breakdown (front-end, then BA on its output)
    obs_by_cam = defaultdict(list)
    for o in obs:
        obs_by_cam[o.cam_id].append(o)
    t_fe = _median(lambda: _front_end_opencv(obj, obs_by_cam, img), repeats)
    cameras = _front_end_opencv(obj, obs_by_cam, img)
    groups, extr = init_camera_groups(obs, sorted(obs_by_cam))
    ref = groups[0][0]
    by_fo = defaultdict(list)
    for o in obs:
        if o.T_c_o is not None and o.cam_id in extr:
            by_fo[(o.object_id, o.frame_id)].append((o.cam_id, o.T_c_o))
    op = {k: average_object_pose_in_group(v, extr, ref) for k, v in by_fo.items()}
    rig0 = RigState(cameras=dict(cameras), T_c_g=dict(extr), ref_cam_id=ref,
                    object_poses=dict(op), objects={0: obj}, img_size=img)

    def _ba():
        r = bundle.refine(rig0, obs, fix_intrinsics=True)
        bundle.refine(r, obs, fix_intrinsics=False)
    t_ba = _median(_ba, repeats)

    K = 6 * (n_cam - 1) + 6 * len(op) + sum(len(c.param_names) for c in cameras.values())
    print(f"{label}: {n_cam} cam x {n_frame} frame | {n_obs} obs, {n_pts} pts, "
          f"BA dim K={K}")
    print(f"  total      {t_total*1000:8.1f} ms")
    print(f"  front-end  {t_fe*1000:8.1f} ms  ({100*t_fe/t_total:4.0f}%)")
    print(f"  global BA  {t_ba*1000:8.1f} ms  ({100*t_ba/t_total:4.0f}%)")
    return dict(total=t_total, fe=t_fe, ba=t_ba, K=K)


def main(repeats=3):
    print(f"=== rig pipeline benchmark (median of {repeats}) ===")
    bench(3, 40, repeats, "small ")
    bench(4, 80, repeats, "medium")
    bench(5, 120, repeats, "large ")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
