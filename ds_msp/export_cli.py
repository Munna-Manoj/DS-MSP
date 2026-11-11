"""``ds-msp export``: LDC meshes and Isaac Sim LUTs for cameras you already have.

Writes the same folder a rig run with ``export_ldc``/``export_isaac_lut`` writes:
``ldc/``, ``isaac_lut/`` and ``exports.json`` (:mod:`ds_msp.exports`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

from .exports import SUMMARY_FILE, ExportCamera, LdcSettings, LutSettings, write_exports
from .io.camera_file import LoadedCalibration, list_cameras, load_calibration, model_from_params


def build_parser(prog: str = "ds-msp export") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Write a TI LDC mesh and/or an Isaac Sim LUT for each camera of a calibration "
            "file, or for one camera from typed lens values, into one folder with an "
            "exports.json summary."
        ),
    )
    parser.add_argument(
        "--input", "--path", dest="input_path", type=Path,
        help=(
            "MC-Calib or DS-MSP rig result YAML, Kalibr camchain, COLMAP cameras.txt "
            "(or its folder), nerfstudio transforms.json, or DS-MSP model JSON/YAML"
        ),
    )
    parser.add_argument(
        "--camera", action="append", metavar="NAME",
        help="camera to export; repeat it, or say 'all' (default: all cameras in the file)",
    )
    parser.add_argument("--model", help="camera model; required with --params")
    parser.add_argument("--params", "--param", dest="params", nargs="+", metavar="VALUE",
                        help="ordered values or name=value entries")
    parser.add_argument("--resolution", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                        help="the camera's image size; required with --params")
    parser.add_argument("--ldc", action="store_true", help="write a TI LDC mesh per camera")
    parser.add_argument("--isaac-lut", action="store_true",
                        help="write an Isaac Sim LUT per camera")
    parser.add_argument("--ldc-output-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                        help="corrected image size the board outputs (default: camera size)")
    parser.add_argument("--ldc-downsample-factor", type=int, default=4,
                        help="mesh node every 2**N output pixels (default: 4)")
    parser.add_argument("--ldc-balance", type=float, default=0.5,
                        help="0 keeps the widest view, 1 crops tightest (default: 0.5)")
    parser.add_argument("--lut-texture-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                        help="LUT texture size (default: camera size)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="folder for ldc/, isaac_lut/ and exports.json")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an earlier export in that folder")
    return parser


def _stem(name: str) -> str:
    return f"camera_{name}" if name.isdigit() else name


def _cameras(args, parser) -> List[ExportCamera]:
    if args.input_path:
        if args.params:
            parser.error("--params cannot be combined with --input")
        resolution = tuple(args.resolution) if args.resolution else None
        wanted = args.camera or ["all"]
        try:
            if "all" in wanted:
                if len(wanted) > 1:
                    parser.error("'all' cannot be combined with named cameras")
                found = list_cameras(args.input_path, model_name=args.model)
                bad = [c for c in found if c.loaded is None]
                if bad:
                    parser.error("; ".join(f"{c.camera}: {c.problem}" for c in bad))
                wanted = [c.camera for c in found]
            loaded = [(name, load_calibration(args.input_path, camera=name,
                                              model_name=args.model, resolution=resolution))
                      for name in wanted]
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            parser.error(str(exc).strip("'\""))
        source_file = args.input_path.name
    else:
        if not args.model or not args.params:
            parser.error("provide --input, or --model with --params and --resolution")
        if not args.resolution:
            parser.error("--resolution WIDTH HEIGHT is required with --params")
        if args.camera and len(args.camera) > 1:
            parser.error("typed values describe one camera; give --camera once")
        try:
            model = model_from_params(args.model, args.params)
        except (KeyError, TypeError, ValueError) as exc:
            parser.error(str(exc).strip("'\""))
        name = (args.camera or ["cam0"])[0]
        w, h = args.resolution
        loaded = [(name, LoadedCalibration(model, (int(w), int(h)), "typed", name))]
        source_file = None
    return [
        ExportCamera(_stem(name), cal.model, cal.resolution,
                     {"file": source_file, "format": cal.source_format, "camera": cal.camera})
        for name, cal in loaded
    ]


def run(argv: Sequence[str] | None = None, *, prog: str = "ds-msp export") -> int:
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    if not (args.ldc or args.isaac_lut):
        parser.error("choose what to write: --ldc, --isaac-lut, or both")
    if not 0 <= args.ldc_downsample_factor <= 7:
        parser.error("--ldc-downsample-factor must be between 0 and 7")
    if not 0.0 <= args.ldc_balance <= 1.0:
        parser.error("--ldc-balance must be between 0 and 1")
    for flag in ("ldc_output_size", "lut_texture_size"):
        size = getattr(args, flag)
        if size and min(size) <= 0:
            parser.error(f"--{flag.replace('_', '-')} must be positive")
    cameras = _cameras(args, parser)
    out = args.output_dir.expanduser()
    if (out / SUMMARY_FILE).exists() and not args.overwrite:
        parser.error(f"{out} already holds an export; choose another folder or add --overwrite")
    ldc = (LdcSettings(tuple(args.ldc_output_size or ()), args.ldc_downsample_factor,
                       args.ldc_balance) if args.ldc else None)
    lut = LutSettings(tuple(args.lut_texture_size or ())) if args.isaac_lut else None
    try:
        write_exports(cameras, str(out), ldc=ldc, lut=lut, overwrite=args.overwrite)
    except OSError as exc:
        parser.error(f"cannot write to {out}: {exc}")
    summary = json.loads((out / SUMMARY_FILE).read_text())
    failed = sum("problem" in row for part in summary.values() if part
                 for row in part["cameras"])
    print(f"export summary: {out / SUMMARY_FILE}")
    if failed:
        print(f"{failed} export(s) could not be written; exports.json says why")
    return 1 if failed else 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
