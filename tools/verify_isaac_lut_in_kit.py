"""Render every supplied DS-MSP LUT bundle in a real Isaac/Kit process.

Run with Isaac's application launcher, not regular Python::

    DS_MSP_ISAAC_LUT_MANIFEST_DIR=/path/to/luts \
    DS_MSP_ISAAC_LUT_OUTPUT_DIR=/tmp/lut_render_test \
      isaacsim isaacsim.exp.full --no-window \
      --exec tools/verify_isaac_lut_in_kit.py

The environment variables avoid differences in how Kit launchers forward
arguments to ``--exec`` scripts. Direct script arguments are also supported.

The script authors each manifest on a real USD Camera, verifies both assets
resolve, captures a frame, and reports whether RTX produced an image materially
different from the same camera's pinhole baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import numpy as np


def _parse_args():
    # Kit keeps its own command line in sys.argv. Arguments after a standalone
    # `--` belong to this script; direct Python execution also remains convenient.
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser()
    manifest_default = os.environ.get("DS_MSP_ISAAC_LUT_MANIFEST_DIR")
    output_default = os.environ.get("DS_MSP_ISAAC_LUT_OUTPUT_DIR")
    parser.add_argument(
        "--manifest-dir", type=Path, default=manifest_default, required=not manifest_default
    )
    parser.add_argument(
        "--output-dir", type=Path, default=output_default, required=not output_default
    )
    parser.add_argument("--resolution", nargs=2, type=int, default=(320, 240))
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=int(os.environ.get("DS_MSP_ISAAC_LUT_SETTLE_FRAMES", "24")),
    )
    parser.add_argument("--effect-threshold", type=float, default=1.0)
    return parser.parse_args(argv)


ARGS = _parse_args()

# Imports below require a fully started Kit application; --exec evaluates this
# file after extension startup and the coroutine yields until the viewport exists.
import carb  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

from ds_msp.isaac_sim.lut import load_manifest  # noqa: E402
from ds_msp.isaac_sim.usd import apply_lut_manifest, verify_lut_camera  # noqa: E402


LOG = "[DS_MSP_ISAAC_LUT_TEST]"


async def _frames(count):
    app = omni.kit.app.get_app()
    for _ in range(count):
        await app.next_update_async()


def _scene(stage):
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    camera = UsdGeom.Camera.Define(stage, "/World/TestCamera")
    camera.CreateClippingRangeAttr((0.05, 1000.0))
    camera.CreateFocalLengthAttr(24.0)

    # An asymmetric, colorful depth layout makes different projections easy to
    # distinguish without relying on external scene assets or network access.
    specs = [
        ("Backdrop", (0.0, 0.0, -8.0), (4.5, 3.2, 0.08), (0.18, 0.18, 0.22)),
        ("Red", (-1.8, 0.9, -4.0), (0.45, 0.45, 0.45), (0.9, 0.05, 0.03)),
        ("Green", (1.5, 0.5, -3.2), (0.35, 0.65, 0.35), (0.05, 0.9, 0.08)),
        ("Blue", (-0.8, -1.2, -2.7), (0.3, 0.3, 0.55), (0.05, 0.15, 0.95)),
        ("Yellow", (2.4, -1.1, -5.0), (0.6, 0.25, 0.25), (0.95, 0.8, 0.05)),
    ]
    for name, position, scale, color in specs:
        cube = UsdGeom.Cube.Define(stage, f"/World/{name}")
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        xform = UsdGeom.XformCommonAPI(cube)
        xform.SetTranslate(Gf.Vec3d(*position))
        xform.SetScale(Gf.Vec3f(*scale))
    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(1200.0)
    light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    return camera.GetPrim()


async def _capture(annotator, path: Path, settle_frames: int):
    """Read a render-product annotator without relying on a visible viewport."""
    import cv2

    image = None
    # Explicit Replicator steps are required in a headless --exec process. RTX
    # subframes let lighting settle while producing one fresh annotated frame.
    for _ in range(3):
        await rep.orchestrator.step_async(
            rt_subframes=max(1, settle_frames), delta_time=0.0, pause_timeline=False
        )
        candidate = annotator.get_data()
        if candidate is not None and getattr(candidate, "ndim", 0) == 3 and candidate.size:
            image = np.asarray(candidate)
            break
    if image is None:
        raise RuntimeError("RGB render-product annotator produced no image")
    if image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif image.shape[2] == 3:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        raise RuntimeError(f"unexpected RGB annotator shape {image.shape}")
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"could not write captured image {path}")
    return bgr


def _digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def run():
    output = ARGS.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifests = sorted(ARGS.manifest_dir.expanduser().resolve().glob("*_isaac_lut.json"))
    if not manifests:
        raise RuntimeError(f"no *_isaac_lut.json files under {ARGS.manifest_dir}")

    context = omni.usd.get_context()
    await context.new_stage_async()
    stage = context.get_stage()
    _scene(stage)
    render_product = rep.create.render_product(
        "/World/TestCamera", resolution=tuple(ARGS.resolution)
    )
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([render_product.path])

    baseline_path = output / "pinhole_baseline.png"
    baseline = (await _capture(rgb, baseline_path, ARGS.settle_frames)).astype(np.float32)

    results = []
    for manifest_path in manifests:
        manifest = load_manifest(manifest_path)
        model_name = manifest["model"]["name"]
        apply_lut_manifest(stage, "/World/TestCamera", manifest_path)
        inspection = verify_lut_camera(stage, "/World/TestCamera")
        image_path = output / f"{model_name}.png"
        image = (await _capture(rgb, image_path, ARGS.settle_frames)).astype(np.float32)
        if image.shape != baseline.shape:
            raise RuntimeError(f"capture shape changed for {model_name}: {image.shape}")
        delta = np.abs(image - baseline)
        assets_valid = inspection["has_lut_api"] and all(
            asset["exists"] for asset in inspection["assets"].values()
        )
        mean_difference = float(delta.mean())
        result = {
            "model": model_name,
            "manifest": str(manifest_path),
            "capture": str(image_path),
            "capture_sha256": _digest(image_path),
            "schema_assets_valid": bool(assets_valid),
            "mean_abs_difference_from_pinhole_8bit": mean_difference,
            "max_abs_difference_from_pinhole_8bit": float(delta.max()),
            "renderer_effect_observed": mean_difference >= ARGS.effect_threshold,
            "inspection": inspection,
        }
        results.append(result)
        carb.log_info(
            f"{LOG} model={model_name} assets={assets_valid} mean_delta={mean_difference:.4f}"
        )

    try:
        isaac_version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError:
        isaac_version = "unknown"
    report = {
        "isaac_sim_version": isaac_version,
        "resolution": list(ARGS.resolution),
        "pinhole_baseline": str(baseline_path),
        "pinhole_baseline_sha256": _digest(baseline_path),
        "effect_threshold_mean_8bit": ARGS.effect_threshold,
        "models": results,
        "schema_asset_gate_passed": all(item["schema_assets_valid"] for item in results),
        "renderer_effect_gate_passed": all(item["renderer_effect_observed"] for item in results),
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    carb.log_info(f"{LOG} report={report_path}")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    # A nonzero process result is represented by a marker file because Kit's
    # post_quit does not provide a portable application exit-code parameter.
    if not report["schema_asset_gate_passed"] or not report["renderer_effect_gate_passed"]:
        (output / "FAILED").write_text("See report.json\n")
    rgb.detach([render_product.path])
    render_product.destroy()
    omni.kit.app.get_app().post_quit()


async def _run_with_report():
    try:
        await run()
    except Exception as exc:
        ARGS.output_dir.mkdir(parents=True, exist_ok=True)
        (ARGS.output_dir / "ERROR.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        carb.log_error(f"{LOG} {type(exc).__name__}: {exc}")
        omni.kit.app.get_app().post_quit()
        raise


asyncio.ensure_future(_run_with_report())
