"""Front-end parallelism benchmark — true original vs. current, independently reproducible.

Checks out the actual pre-fix code via ``git worktree`` (not a reconstruction) and times
``make_bundle_front_end`` on synthetic multi-camera rigs, same seed, same data, for every
camera-model family this library ships: RadTan (pinhole) and the wide-FOV sphere models
DS / UCM / EUCM. Prints a before/after table plus a bit-identical-accuracy check.

Run it yourself rather than take any before/after numbers on faith — this script is the exact,
reproducible method behind them.

Usage:  python scripts/benchmark_front_end_parallel.py [--cams 16,40] [--baseline-ref origin/main]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, ".")


def _run(root: str, n_cam: int, model_name: str, extra_kwargs: dict):
    """Import ``ds_msp`` fresh from ``root`` and time one ``front_end`` call."""
    sys.path.insert(0, root)
    for mod in list(sys.modules):
        if mod == "ds_msp" or mod.startswith("ds_msp."):
            del sys.modules[mod]
    from ds_msp.models.double_sphere import DoubleSphereModel
    from ds_msp.models.eucm import EUCMModel
    from ds_msp.models.radtan import RadTanModel
    from ds_msp.models.ucm import UCMModel
    from ds_msp.rig.calibrate import make_bundle_front_end

    if "tests.rig._synth" in sys.modules:
        del sys.modules["tests.rig._synth"]
    from tests.rig._synth import make_rig

    W, H = 1280, 960
    facs = {
        "radtan": lambda cam_id, rng: RadTanModel(
            800.0 * rng.uniform(0.98, 1.02), 800.0, W / 2, H / 2, -0.05, 0.01, 0.0, 0.0, 0.0),
        "ds": lambda cam_id, rng: DoubleSphereModel(350.0, 350.0, W / 2, H / 2, 0.2, 0.6),
        "ucm": lambda cam_id, rng: UCMModel(400.0, 400.0, W / 2, H / 2, 0.65),
        "eucm": lambda cam_id, rng: EUCMModel(400.0, 400.0, W / 2, H / 2, 0.6, 1.1),
    }
    obj, obs, img, gt_ext, gtm = make_rig(n_cam=n_cam, n_frame=20, noise_px=0.3, seed=0,
                                          w=W, h=H, model_factory=facs[model_name])
    obs_by_cam: dict = {}
    for o in obs:
        obs_by_cam.setdefault(o.cam_id, []).append(o)

    fe = make_bundle_front_end(model_name, **extra_kwargs)
    t0 = time.time()
    cams = fe(obj, obs_by_cam, img)
    dt = time.time() - t0
    sys.path.remove(root)
    return dt, {c: cams[c].params.copy() for c in cams}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cams", default="16,40", help="comma-separated camera counts")
    ap.add_argument("--models", default="radtan,ds,ucm,eucm",
                    help="comma-separated model names")
    ap.add_argument("--baseline-ref", default="origin/main",
                    help="git ref to check out as the 'before' baseline")
    args = ap.parse_args()

    current_root = "."
    baseline_root = tempfile.mkdtemp(prefix="dsmsp-baseline-")
    print(f"checking out {args.baseline_ref} into {baseline_root} for a true before/after "
         f"comparison (not a reconstruction)...")
    subprocess.run(["git", "worktree", "add", "--detach", baseline_root, args.baseline_ref],
                   check=True, capture_output=True, text=True)
    try:
        print(f"\n{'model':7s} {'n_cam':>6s} {'ORIGINAL':>10s} {'FIXED':>10s} "
             f"{'reduction':>10s}  {'accuracy'}")
        print("-" * 62)
        for model_name in args.models.split(","):
            for n_cam in (int(x) for x in args.cams.split(",")):
                t_orig, p_orig = _run(baseline_root, n_cam, model_name, {})
                t_new, p_new = _run(current_root, n_cam, model_name, {"n_jobs": -1})
                reduction = 100 * (1 - t_new / t_orig)
                import numpy as np
                same = all(np.allclose(p_orig[c], p_new[c], atol=1e-6)
                          for c in p_orig if c in p_new)
                print(f"{model_name:7s} {n_cam:6d} {t_orig:9.3f}s {t_new:9.3f}s "
                     f"{reduction:9.1f}%  {'unchanged' if same else 'DIFFERS -- check!'}")
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", baseline_root],
                       check=False, capture_output=True)
        shutil.rmtree(baseline_root, ignore_errors=True)


if __name__ == "__main__":
    main()
