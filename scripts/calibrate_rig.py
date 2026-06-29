"""Calibrate a multi-camera rig on an MC-Calib dataset and write MC-Calib-format output.

The DS-MSP analogue of MC-Calib's ``apps/calibrate``: pick a camera model per camera (like
the config's ``camera_models`` / ``distortion_per_camera``), optimize extrinsics-only or
intrinsics+extrinsics (``--fix-intrinsics``), and save ``calibrated_cameras_data.yml`` +
``calibrated_objects_data.yml`` + ``calibrated_objects_pose_data.yml`` in MC-Calib's exact
schema. Reuses MC-Calib's detected keypoints (identical 2D inputs) so the rig *math* is what
is exercised.

Examples
--------
  # one model for all cameras
  python scripts/calibrate_rig.py ../MC-Calib/Blender_Images/Scenario_1 --model ds

  # a different model per camera (camera 0 -> radtan, 1 -> double_sphere, 2 -> ucm, ...)
  python scripts/calibrate_rig.py <scn> --models radtan,double_sphere,ucm

  # random valid model per camera (pinhole pool); extrinsics + intrinsics
  python scripts/calibrate_rig.py <scn> --random pinhole --seed 0

  # MC-Calib style: drive everything from one config file (raw images or keypoints)
  python scripts/calibrate_rig.py --config calib_param.yml
  python scripts/calibrate_rig.py --config calib_param.yml --set root_path=/abs/Images --set save_path=/abs/out
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, ".")
from ds_msp.io.mccalib import load_scenario, radtan_from_cameragt          # noqa: E402
from ds_msp.rig.pipeline import calibrate_scenario, random_model_assignment     # noqa: E402


def _run_config(config_path, sets):
    """MC-Calib-compatible single-file entry: parse calib_param.yml and run."""
    from ds_msp.rig.calib_param import calibrate_from_config
    overrides = {}
    for kv in sets or []:
        k, _, v = kv.partition("=")
        overrides[k.strip()] = [s.strip() for s in v.split(",")] if k.strip() == "camera_models" \
            else v.strip()
    res = calibrate_from_config(config_path, overrides or None)
    cfg = res["config"]
    print(f"=== {os.path.basename(config_path)}: {cfg.number_camera} cameras, "
          f"{cfg.number_board} board(s), {len(res['rig'].cameras)} calibrated ===")
    print(f"per-camera model: {res['models']}")
    m = res["metrics"]
    if m.get("worst_baseline_pct_vs_gt") is not None:
        print(f"worst baseline error vs GroundTruth : {m['worst_baseline_pct_vs_gt']:.3f}%")
    print(f"max reprojection RMS: {m['max_rms_px']:.4f} px")
    if cfg.save_path:
        print(f"wrote MC-Calib-format output to: {cfg.save_path}")
    return res


def _resolve_spec(args, cam_ids):
    if args.random:
        return random_model_assignment(cam_ids, kind=args.random, seed=args.seed)
    if args.models:
        names = [s.strip() for s in args.models.split(",")]
        if len(names) != len(cam_ids):
            raise SystemExit(f"--models has {len(names)} entries but rig has {len(cam_ids)} cameras")
        return {c: names[i] for i, c in enumerate(sorted(cam_ids))}
    return args.model            # single model name for all cameras


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", nargs="?", help="path to a Blender_Images/Scenario_* directory")
    ap.add_argument("--config", help="MC-Calib calib_param.yml — drive the whole run from it")
    ap.add_argument("--init-config", metavar="PATH", dest="init_config",
                    help="write a base MC-Calib-compatible calib_param.yml template to PATH and exit")
    ap.add_argument("--init-intrinsics", metavar="PATH", dest="init_intrinsics",
                    help="write a camera_intrinsics.yml template (all models documented) to PATH and exit")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="override a config value (repeatable), e.g. --set save_path=/abs/out")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--model", default="radtan", help="one model for all cameras")
    g.add_argument("--models", help="comma-separated model per camera (cam 0,1,2,...)")
    g.add_argument("--random", choices=["pinhole", "fisheye"],
                   help="assign a random valid model per camera")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --random")
    ap.add_argument("--fix-intrinsics", action="store_true",
                    help="optimize extrinsics only; seed intrinsics from GroundTruth (RadTan)")
    ap.add_argument("--save-dir", default=None,
                    help="output dir (default <scenario>/Results_dsmsp)")
    ap.add_argument("--save-reprojection", action="store_true",
                    help="also write MC-Calib-style reprojection overlay images (needs Images/)")
    args = ap.parse_args()

    if args.init_config or args.init_intrinsics:
        import shutil
        cfgdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
        if args.init_config:
            shutil.copyfile(os.path.join(cfgdir, "calib_param.template.yml"), args.init_config)
            print(f"wrote base calib_param template to: {args.init_config}")
        if args.init_intrinsics:
            shutil.copyfile(os.path.join(cfgdir, "camera_intrinsics.template.yml"),
                            args.init_intrinsics)
            print(f"wrote camera intrinsics template to: {args.init_intrinsics}")
        return

    if args.config:
        _run_config(args.config, args.set)
        return
    if not args.scenario:
        ap.error("provide a scenario directory or --config calib_param.yml")

    scn = load_scenario(args.scenario)
    spec = _resolve_spec(args, scn.cam_ids)
    save_dir = args.save_dir or os.path.join(args.scenario, "Results_dsmsp")

    init_cameras = None
    if args.fix_intrinsics:
        # fixed-intrinsic mode needs prior intrinsics; use MC-Calib's calibrated values
        # (the realistic "intrinsics already known, solve extrinsics only" case), RadTan/Brown.
        init_cameras = {c: radtan_from_cameragt(scn.mccalib[c]) for c in scn.cam_ids}

    print(f"=== {scn.name}: {len(scn.cam_ids)} cameras, {len(scn.object_obs)} object-obs ===")
    print(f"model spec: {spec if not args.fix_intrinsics else 'fixed-intrinsic (RadTan, MC-Calib K)'}")
    image_root = os.path.join(args.scenario, "Images") if args.save_reprojection else None
    res = calibrate_scenario(scn, spec, fix_intrinsics=args.fix_intrinsics,
                             init_cameras=init_cameras, save_dir=save_dir,
                             image_root=image_root)

    m = res["metrics"]
    print("\nper-camera model -> reprojection RMS (px):")
    for c in sorted(res["models"]):
        print(f"  cam {c}: {res['models'][c]:13s}  rms={m['rms_px'][c]:.4f}")
    if m["worst_baseline_pct_vs_gt"] is not None:
        print(f"\nworst baseline error vs GroundTruth : {m['worst_baseline_pct_vs_gt']:.3f}%")
    if m["worst_baseline_pct_vs_mccalib"] is not None:
        print(f"worst baseline error vs MC-Calib    : {m['worst_baseline_pct_vs_mccalib']:.3f}%")
    print(f"\nwrote MC-Calib-format output to: {save_dir}")
    for k, v in res["paths"].items():
        print(f"  {k}: {os.path.basename(v)}")


if __name__ == "__main__":
    main()
