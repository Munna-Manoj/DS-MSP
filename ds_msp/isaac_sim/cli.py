"""Command-line export of DS-MSP camera intrinsics to Isaac Sim RTX LUTs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..models.registry import _BY_NAME, canonical_name
from .artifacts import load_calibration, model_from_params
from .lut import export_lut


def build_parser(prog: str = "ds-msp-lut") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Generate an OmniLensDistortionLutAPI EXR pair from any DS-MSP camera model. "
            "Use either an existing calibration artifact or explicit model parameters."
        ),
    )
    parser.add_argument(
        "--input", "--path", dest="input_path", type=Path,
        help="Kalibr camchain, MC-Calib camera YAML, or DS-MSP model JSON/YAML",
    )
    parser.add_argument(
        "--camera", default="cam0",
        help="camera stanza/index in a multi-camera artifact (default: cam0)",
    )
    parser.add_argument(
        "--model", help="camera model; required with --params, optional artifact check/override",
    )
    parser.add_argument(
        "--params", "--param", dest="params", nargs="+", metavar="VALUE",
        help="ordered values or name=value entries (run --list-models for each order)",
    )
    parser.add_argument(
        "--resolution", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
        help="nominal calibration resolution; overrides or supplies the artifact value",
    )
    parser.add_argument(
        "--texture-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
        help="LUT resolution (default: nominal calibration resolution)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="output directory (default: <artifact-dir>/isaac_lut or ./isaac_lut)",
    )
    parser.add_argument("--stem", help="filename stem (default: content-addressed)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing bundle")
    parser.add_argument(
        "--chunk-rows", type=int, default=128,
        help="working-set row count for large LUTs (default: 128)",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="print models and parameter order, then exit"
    )
    return parser


def _print_models() -> None:
    print("Available camera models (ordered --params):")
    for name, cls in sorted(_BY_NAME.items()):
        print(f"  {name:<8} {' '.join(cls.param_names)}")


def run(argv: Sequence[str] | None = None, *, prog: str = "ds-msp-lut") -> int:
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    if args.list_models:
        _print_models()
        return 0

    requested_resolution = tuple(args.resolution) if args.resolution else None
    if args.input_path:
        if args.params:
            parser.error("--params cannot be combined with --input/--path")
        try:
            loaded = load_calibration(
                args.input_path,
                camera=args.camera,
                model_name=args.model,
                resolution=requested_resolution,
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        model = loaded.model
        width, height = loaded.resolution
        source = f"{loaded.source_format}:{loaded.camera} from {args.input_path}"
        output_dir = args.output_dir or args.input_path.expanduser().resolve().parent / "isaac_lut"
    else:
        if not args.model:
            parser.error("provide --input/--path, or provide --model with --params")
        if requested_resolution is None:
            parser.error("--resolution WIDTH HEIGHT is required with explicit --params")
        try:
            model = model_from_params(args.model, args.params or [])
        except (KeyError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        width, height = requested_resolution
        source = "explicit parameters"
        output_dir = args.output_dir or Path.cwd() / "isaac_lut"

    if args.model:
        try:
            canonical_name(args.model)
        except KeyError as exc:
            parser.error(str(exc))
    texture_width, texture_height = args.texture_size or (width, height)
    try:
        bundle = export_lut(
            model,
            width,
            height,
            output_dir,
            texture_width=texture_width,
            texture_height=texture_height,
            stem=args.stem,
            overwrite=args.overwrite,
            chunk_rows=args.chunk_rows,
        )
    except (FileExistsError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"camera:    {bundle.model_name} {width}x{height} ({source})")
    print(f"LUT size:  {texture_width}x{texture_height}")
    print(f"ray enter: {bundle.paths.ray_enter}")
    print(f"ray exit:  {bundle.paths.ray_exit}")
    print(f"manifest:  {bundle.manifest}")
    print(f"USD camera:{bundle.camera_usda}")
    print(f"max round-trip error: {bundle.round_trip_max_error_px:.3e} px")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
