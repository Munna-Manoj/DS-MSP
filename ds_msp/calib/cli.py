"""Config-driven single-camera intrinsics calibration.

The ``ds_msp.calib`` analogue of ``ds_msp.rig.cli`` — gives ``pip install ds-msp`` a real
``ds-msp-calibrate`` console command with no repo clone needed. Board type (checkerboard /
ChArUco / AprilGrid), board geometry, and camera model are just configuration; detection is a
swappable front end (:mod:`ds_msp.calib.board`) feeding the one shared, model-agnostic
bundle-adjustment backend (:func:`ds_msp.calib.single_camera.calibrate_camera`).

Examples
--------
  # one-off CLI flags, no config file
  ds-msp-calibrate ./images --board checkerboard --rows 5 --cols 6 --square-size 0.025 --model ds

  # config-driven, MC-Calib-style
  ds-msp-calibrate --init-config calib_config.yml   # write a starter config
  ds-msp-calibrate --config calib_config.yml
  ds-msp-calibrate --config calib_config.yml --set board.rows=7 --set save_path=/abs/out
"""

from __future__ import annotations

import argparse
import glob
import importlib.resources
import os
import shutil

from . import report as rpt
from .config import BoardConfig, CalibConfig, build_board, load_config
from .single_camera import calibrate_camera

_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _find_images(images_path, pattern):
    return sorted(f for f in glob.glob(os.path.join(images_path, pattern))
                 if f.lower().endswith(_IMG_EXT))


def _image_size(path):
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    h, w = img.shape[:2]
    return w, h


def _run(cfg: CalibConfig, *, pass_px: float, warn_px: float) -> int:
    """Shared tail for both entry points: find images, detect + calibrate (live progress,
    one enclosing Stage banner), report, save. Returns the process exit code (0 PASS/WARN,
    1 FAIL) so a CI/cron caller can tell success from failure without parsing text."""
    if not cfg.images_path:
        raise SystemExit("no images directory given (config's images_path, or the CLI's "
                         "positional argument)")
    files = _find_images(cfg.images_path, cfg.pattern)
    if not files:
        raise SystemExit(f"no images found under {cfg.images_path} matching {cfg.pattern!r}")
    w, h = _image_size(files[0])
    board = build_board(cfg)

    print(f"=== {cfg.board.type} / {cfg.camera_model}: {len(files)} images, {w}x{h} ===")
    progress_cb = rpt.make_detect_progress(verbose=cfg.verbose)
    with rpt.Stage("calibrating", verbose=cfg.verbose):
        result = calibrate_camera(board, files, cfg.camera_model, w, h, progress_cb=progress_cb,
                                  robust=cfg.robust, robust_scale=cfg.robust_scale, gnc=cfg.gnc,
                                  multi_start=cfg.multi_start, n_restarts=cfg.n_restarts,
                                  seed=cfg.seed, max_nfev=cfg.max_nfev, verbose=0)

    overall = rpt.ErrorStats.from_result(result)
    level, message = rpt.verdict(overall, pass_px=pass_px, warn_px=warn_px)
    print()
    rpt.print_report(cfg.board.type, cfg.camera_model, result["n_images_detected"],
                     result["n_images_total"], overall, level, message)

    if cfg.save_path:
        os.makedirs(cfg.save_path, exist_ok=True)
        if cfg.output_format == "kalibr":
            from ..io.kalibr import save_kalibr
            out_path = os.path.join(cfg.save_path, "camchain.yaml")
            save_kalibr(result["model"], out_path, w, h)
            print(f"\nwrote {out_path}")
        else:
            raise SystemExit(f"unknown output_format {cfg.output_format!r} (only 'kalibr' "
                             f"is supported today)")
    return 0 if level in ("PASS", "WARN") else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images_dir", nargs="?", help="folder of calibration images")
    ap.add_argument("--config", help="calib_config.yml — drive the whole run from it")
    ap.add_argument("--init-config", metavar="PATH", dest="init_config",
                    help="write a starter calib_config.yml template to PATH and exit")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="override a config value (repeatable, dotted keys), "
                         "e.g. --set board.rows=7")
    ap.add_argument("--board", choices=["checkerboard", "charuco", "aprilgrid"],
                    default="checkerboard",
                    help="board type (non-config mode; not recommended for extreme fisheye "
                         "— prefer charuco/aprilgrid beyond ~120 degrees FOV)")
    ap.add_argument("--rows", type=int, default=0,
                    help="interior corner rows (checkerboard/charuco)")
    ap.add_argument("--cols", type=int, default=0,
                    help="interior corner cols (checkerboard/charuco)")
    ap.add_argument("--square-size", type=float, default=0.0, dest="square_size")
    ap.add_argument("--model", default="ds", help="camera model name (default ds)")
    ap.add_argument("--pattern", default="*", help="glob pattern for image files")
    ap.add_argument("--save-dir", default=None, dest="save_dir",
                    help="output dir for the calibrated intrinsics (Kalibr camchain.yaml)")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress live detection progress + stage banners")
    ap.add_argument("--pass-px", type=float, default=rpt.DEFAULT_PASS_PX, dest="pass_px",
                    help=f"median-px verdict PASS threshold (default {rpt.DEFAULT_PASS_PX})")
    ap.add_argument("--warn-px", type=float, default=rpt.DEFAULT_WARN_PX, dest="warn_px",
                    help=f"median-px verdict WARN threshold (default {rpt.DEFAULT_WARN_PX})")
    args = ap.parse_args()

    if args.init_config:
        cfgdir = importlib.resources.files("ds_msp.calib") / "configs"
        with importlib.resources.as_file(cfgdir / "calib_config.template.yml") as src:
            shutil.copyfile(src, args.init_config)
        print(f"wrote base calib_config template to: {args.init_config}")
        return

    if args.config:
        sets = {}
        for kv in args.set or []:
            k, _, v = kv.partition("=")
            sets[k.strip()] = v.strip()
        cfg = load_config(args.config, sets or None)
        if args.quiet:
            cfg.verbose = False
        raise SystemExit(_run(cfg, pass_px=args.pass_px, warn_px=args.warn_px))

    if not args.images_dir:
        ap.error("provide an images directory or --config calib_config.yml")

    cfg = CalibConfig(board=BoardConfig(type=args.board, rows=args.rows, cols=args.cols,
                                        square_size=args.square_size),
                      camera_model=args.model, images_path=args.images_dir,
                      pattern=args.pattern, save_path=args.save_dir, verbose=not args.quiet)
    raise SystemExit(_run(cfg, pass_px=args.pass_px, warn_px=args.warn_px))


if __name__ == "__main__":
    main()
