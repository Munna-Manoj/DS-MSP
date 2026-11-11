"""Command-line export of TI Jacinto LDC displacement meshes (``ds-msp ldc``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

from .io.camera_file import LoadedCalibration, list_cameras, load_calibration, model_from_params
from .ldc import export_ldc
from .models.registry import _BY_NAME


def build_parser(prog: str = "ds-msp-ldc") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Write a TI Jacinto LDC displacement mesh for each camera, from a calibration "
            "file or from typed lens values, and report how accurate each mesh is."
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
        help="camera to export, such as cam0 or camera_2; repeat it, or say 'all' "
             "(default: all cameras in the file)",
    )
    parser.add_argument("--list-cameras", action="store_true",
                        help="list the cameras in --input and exit")
    parser.add_argument("--model", help="camera model; required with --params")
    parser.add_argument(
        "--params", "--param", dest="params", nargs="+", metavar="VALUE",
        help="ordered values or name=value entries (run --list-models for each order)",
    )
    parser.add_argument(
        "--resolution", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
        help="the camera's image size; required with --params, overrides the file's value",
    )
    parser.add_argument(
        "--output-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
        help="size of the corrected image the board outputs (default: the camera's size)",
    )
    parser.add_argument(
        "--downsample-factor", type=int, default=4,
        help="mesh nodes every 2**N output pixels (default: 4, so every 16 pixels)",
    )
    parser.add_argument(
        "--balance", type=float, default=0.5,
        help="0 keeps the widest view, 1 crops tightest (default: 0.5)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="output folder (default: <input folder>/ldc, or ./ldc with --params)",
    )
    parser.add_argument("--stem", help="file name start (default: the camera name)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing files")
    parser.add_argument("--list-models", action="store_true",
                        help="print models and parameter order, then exit")
    return parser


def _cameras(args, parser) -> List[Tuple[str, LoadedCalibration]]:
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
                if resolution is None:
                    return [(c.camera, c.loaded) for c in found if c.loaded is not None]
                wanted = [c.camera for c in found]
            return [
                (name, load_calibration(args.input_path, camera=name, model_name=args.model,
                                        resolution=resolution))
                for name in wanted
            ]
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            parser.error(str(exc).strip("'\""))
    if not args.model:
        parser.error("provide --input, or provide --model with --params and --resolution")
    if not args.resolution:
        parser.error("--resolution WIDTH HEIGHT is required with --params")
    if args.camera and len(args.camera) > 1:
        parser.error("typed values describe one camera; give --camera once")
    name = (args.camera or ["cam0"])[0]
    try:
        model = model_from_params(args.model, args.params or [])
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc).strip("'\""))
    width, height = args.resolution
    return [(name, LoadedCalibration(model, (int(width), int(height)), "typed", name))]


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def run(argv: Sequence[str] | None = None, *, prog: str = "ds-msp-ldc") -> int:
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    if args.list_models:
        print("Available camera models (ordered --params):")
        for name, cls in sorted(_BY_NAME.items()):
            print(f"  {name:<8} {' '.join(cls.param_names)}")
        return 0
    if args.list_cameras:
        if not args.input_path:
            parser.error("--list-cameras needs --input")
        try:
            found = list_cameras(args.input_path, model_name=args.model)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            parser.error(str(exc).strip("'\""))
        for cam in found:
            if cam.loaded is None:
                print(f"{cam.camera}: cannot read: {cam.problem}")
            else:
                w, h = cam.loaded.resolution
                print(f"{cam.camera}: {cam.loaded.model.name} {w}x{h}")
        return 0
    if not 0 <= args.downsample_factor <= 7:
        parser.error("--downsample-factor must be between 0 and 7")
    if not 0.0 <= args.balance <= 1.0:
        parser.error("--balance must be between 0 and 1")
    if args.output_size and min(args.output_size) <= 0:
        parser.error("--output-size must be positive")

    cameras = _cameras(args, parser)
    if args.output_dir:
        output_dir = args.output_dir
    elif args.input_path:
        base = args.input_path.expanduser().resolve()
        output_dir = (base if base.is_dir() else base.parent) / "ldc"
    else:
        output_dir = Path.cwd() / "ldc"

    source_file = args.input_path.name if args.input_path else None
    for name, loaded in cameras:
        stem = name if not args.stem else (args.stem if len(cameras) == 1 else f"{args.stem}_{name}")
        try:
            done = export_ldc(
                loaded.model, loaded.resolution, output_dir, stem,
                output_size=tuple(args.output_size) if args.output_size else None,
                downsample_factor=args.downsample_factor, balance=args.balance,
                overwrite=args.overwrite,
                source={"file": source_file, "format": loaded.source_format,
                        "camera": loaded.camera},
            )
        except (FileExistsError, OSError, TypeError, ValueError) as exc:
            parser.error(f"{name}: {exc}")
        cfg, acc = done.config, done.accuracy
        h, w = cfg["mesh_size"][:2]
        print(f"{name}: {loaded.model.name} {loaded.resolution[0]}x{loaded.resolution[1]} "
              f"-> {cfg['output_width']}x{cfg['output_height']}, mesh {w}x{h} nodes")
        print(f"  mesh error px: median {_fmt(acc['median_px'])}, "
              f"99% {_fmt(acc['p99_px'])}, worst {_fmt(acc['max_px'])}")
        if acc["in_view_fraction"] is not None:
            print(f"  output that shows the scene: {100 * acc['in_view_fraction']:.1f} %")
        for note in done.notes:
            print(f"  note: {note}", file=sys.stderr)
        print(f"  mesh:   {done.files.mesh}")
        print(f"  record: {done.files.record}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
