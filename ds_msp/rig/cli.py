"""Calibrate a multi-camera rig on an MC-Calib dataset and write MC-Calib-format output.

The DS-MSP analogue of MC-Calib's ``apps/calibrate``: pick a camera model per camera (like
the config's ``camera_models`` / ``distortion_per_camera``), optimize extrinsics-only or
intrinsics+extrinsics (``--fix-intrinsics``), and save ``calibrated_cameras_data.yml`` +
``calibrated_objects_data.yml`` + ``calibrated_objects_pose_data.yml`` in MC-Calib's exact
schema. Reuses MC-Calib's detected keypoints (identical 2D inputs) so the rig *math* is what
is exercised.

This is the real CLI logic behind two entry points:

* ``pip install ds-msp`` gives you the ``ds-msp-calibrate-rig`` console command directly
  (a ``[project.scripts]`` entry point registered in ``pyproject.toml``, ``main`` below).
* A git-clone checkout can still run ``python scripts/calibrate_rig.py ...`` -- that file is
  now a thin wrapper delegating here, kept for repo-root convenience.

Only ``ds_msp*`` ships in the wheel (``scripts/`` does not), so this had to live inside the
package for pip-only users to have a CLI at all -- ``--init-config``/``--init-intrinsics``
below resolve their templates via :mod:`importlib.resources` against ``ds_msp.rig.configs``
(shipped as package data) rather than a path relative to this file's location on disk, so it
works identically whether installed from a wheel, an sdist, or an editable checkout.

Examples
--------
  # one model for all cameras
  ds-msp-calibrate-rig ../MC-Calib/Blender_Images/Scenario_1 --model ds

  # a different model per camera (camera 0 -> radtan, 1 -> double_sphere, 2 -> ucm, ...)
  ds-msp-calibrate-rig <scn> --models radtan,double_sphere,ucm

  # random valid model per camera (pinhole pool); extrinsics + intrinsics
  ds-msp-calibrate-rig <scn> --random pinhole --seed 0

  # MC-Calib style: drive everything from one config file (raw images or keypoints)
  ds-msp-calibrate-rig --init-config calib_param.yml   # write a starter config
  ds-msp-calibrate-rig --config calib_param.yml
  ds-msp-calibrate-rig --config calib_param.yml --set root_path=/abs/Images --set save_path=/abs/out
"""
from __future__ import annotations

import argparse
import importlib.resources
import os

from ..io.mccalib import load_scenario, radtan_from_cameragt
from . import report as rpt
from .pipeline import calibrate_scenario, random_model_assignment


def _report_and_exit_code(rig, scn, models, save_dir, *, pass_px, warn_px, html_path):
    """Shared tail for both entry points: distribution-stats table, verdict, and the
    self-contained HTML digital-twin report. Returns the process exit code (0 PASS/WARN,
    1 FAIL) so a CI/cron caller can tell success from failure without parsing text."""
    per_cam, overall = rpt.camera_and_overall_stats(rig, scn.object_obs)
    level, message = rpt.verdict(overall, pass_px=pass_px, warn_px=warn_px)
    print()
    rpt.print_report(models, per_cam, overall, level, message)
    if html_path:
        rpt.write_html_report(html_path, rig, scn, models, per_cam, overall, level, message)
        print(f"\ninteractive report: {html_path}  (open in any browser, no server needed)")
    if save_dir:
        print(f"wrote MC-Calib-format output to: {save_dir}")
    return 0 if level in ("PASS", "WARN") else 1


def _run_config(config_path, sets, *, pass_px, warn_px, report_path):
    """MC-Calib-compatible single-file entry: parse calib_param.yml and run."""
    from .calib_param import calibrate_from_config
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
    html_path = report_path or (os.path.join(cfg.save_path, "report.html") if cfg.save_path
                                else None)
    code = _report_and_exit_code(res["rig"], res["scenario"], res["models"], cfg.save_path,
                                 pass_px=pass_px, warn_px=warn_px, html_path=html_path)
    res["exit_code"] = code
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
    ap.add_argument("--quiet", action="store_true",
                    help="suppress live per-stage progress (only the final report prints)")
    ap.add_argument("--no-webviewer", action="store_true",
                    help="don't launch the live browser 3D view (independent of --quiet)")
    ap.add_argument("--report", metavar="PATH", default=None,
                    help="path for the self-contained interactive HTML report "
                         "(default <save-dir>/report.html; --report='' disables it)")
    ap.add_argument("--pass-px", type=float, default=rpt.DEFAULT_PASS_PX,
                    help=f"median-px bar for a PASS verdict (default {rpt.DEFAULT_PASS_PX})")
    ap.add_argument("--warn-px", type=float, default=rpt.DEFAULT_WARN_PX,
                    help=f"median-px bar for a WARN (above this is FAIL) (default {rpt.DEFAULT_WARN_PX})")
    args = ap.parse_args()

    if args.init_config or args.init_intrinsics:
        import shutil
        cfgdir = importlib.resources.files("ds_msp.rig") / "configs"
        if args.init_config:
            with importlib.resources.as_file(cfgdir / "calib_param.template.yml") as src:
                shutil.copyfile(src, args.init_config)
            print(f"wrote base calib_param template to: {args.init_config}")
        if args.init_intrinsics:
            with importlib.resources.as_file(cfgdir / "camera_intrinsics.template.yml") as src:
                shutil.copyfile(src, args.init_intrinsics)
            print(f"wrote camera intrinsics template to: {args.init_intrinsics}")
        return

    if args.config:
        sets = list(args.set or [])
        if args.quiet:
            sets.append("verbose=false")
        if args.no_webviewer:
            sets.append("webviewer=false")
        res = _run_config(args.config, sets, pass_px=args.pass_px, warn_px=args.warn_px,
                          report_path=args.report or None)
        raise SystemExit(res["exit_code"])
    if not args.scenario:
        ap.error("provide a scenario directory or --config calib_param.yml")

    # Launch the live view before loading the scenario, not once its geometry already exists
    # -- see WebLive3DAnimator's class docstring for why (this mirrors the --config entry
    # point's ordering for consistency, even though this path's keypoints are usually
    # pre-baked rather than freshly detected, so there's less silent time here to fill).
    from .web3d import WebLive3DAnimator
    animator = WebLive3DAnimator(verbose=not (args.quiet or args.no_webviewer))

    scn = load_scenario(args.scenario)
    animator.bind_scene(scn.object, scn.object_obs)
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
    with rpt.Stage("calibrating", verbose=not args.quiet):
        res = calibrate_scenario(scn, spec, fix_intrinsics=args.fix_intrinsics,
                                 init_cameras=init_cameras, save_dir=save_dir,
                                 image_root=image_root, verbose=not args.quiet,
                                 on_iter=animator)
    m = res["metrics"]
    animator.finish(res["rig"], rms=m["max_rms_px"])

    if m["worst_baseline_pct_vs_gt"] is not None:
        print(f"worst baseline error vs GroundTruth : {m['worst_baseline_pct_vs_gt']:.3f}%")
    if m["worst_baseline_pct_vs_mccalib"] is not None:
        print(f"worst baseline error vs MC-Calib    : {m['worst_baseline_pct_vs_mccalib']:.3f}%")
    for k, v in res["paths"].items():
        print(f"  {k}: {os.path.basename(v)}")
    html_path = args.report if args.report is not None else os.path.join(save_dir, "report.html")
    code = _report_and_exit_code(res["rig"], scn, res["models"], save_dir,
                                 pass_px=args.pass_px, warn_px=args.warn_px,
                                 html_path=html_path or None)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
