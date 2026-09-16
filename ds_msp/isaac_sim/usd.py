"""Optional Isaac/Omniverse helpers for assigning a generated LUT bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .lut import load_manifest


def apply_lut_manifest(stage, camera_path: str, manifest_path: str | Path):
    """Apply a generated manifest to a USD Camera prim.

    This function imports ``pxr`` lazily, so normal DS-MSP installation and LUT
    generation do not require Isaac Sim. Run it from Isaac's Script Editor or
    Isaac Python environment.
    """
    try:
        from pxr import Gf, Sdf, UsdGeom
    except ImportError as exc:  # pragma: no cover - exercised only inside Isaac/Kit
        raise RuntimeError("apply_lut_manifest must run inside Isaac Sim/Omniverse Kit") from exc

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest: Dict[str, Any] = load_manifest(manifest_file)
    prim = stage.GetPrimAtPath(camera_path)
    if not prim or not prim.IsValid():
        prim = UsdGeom.Camera.Define(stage, camera_path).GetPrim()
    elif not prim.IsA(UsdGeom.Camera):
        raise TypeError(f"{camera_path} exists but is not a USD Camera prim")
    prim.ApplyAPI("OmniLensDistortionLutAPI")

    attrs = manifest["isaac_sim"]["attributes"]
    prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set("lut")
    prim.CreateAttribute(
        "omni:lensdistortion:lut:nominalWidth", Sdf.ValueTypeNames.Float
    ).Set(float(attrs["omni:lensdistortion:lut:nominalWidth"]))
    prim.CreateAttribute(
        "omni:lensdistortion:lut:nominalHeight", Sdf.ValueTypeNames.Float
    ).Set(float(attrs["omni:lensdistortion:lut:nominalHeight"]))
    center = attrs["omni:lensdistortion:lut:opticalCenter"]
    prim.CreateAttribute(
        "omni:lensdistortion:lut:opticalCenter", Sdf.ValueTypeNames.Float2
    ).Set(Gf.Vec2f(float(center[0]), float(center[1])))
    for name in (
        "omni:lensdistortion:lut:rayEnterDirectionTexture",
        "omni:lensdistortion:lut:rayExitPositionTexture",
    ):
        asset = (manifest_file.parent / attrs[name]).resolve()
        if not asset.is_file():
            raise FileNotFoundError(asset)
        prim.CreateAttribute(name, Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(asset)))
    return prim


def verify_lut_camera(stage, camera_path: str) -> Dict[str, Any]:
    """Inspect a camera's authored LUT API and resolve both texture files."""
    prim = stage.GetPrimAtPath(camera_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"invalid camera prim {camera_path!r}")
    names = (
        "omni:lensdistortion:lut:rayEnterDirectionTexture",
        "omni:lensdistortion:lut:rayExitPositionTexture",
    )
    assets = {}
    for name in names:
        value = prim.GetAttribute(name).Get()
        asset_path = getattr(value, "resolvedPath", "") or getattr(value, "path", "")
        assets[name] = {"path": str(asset_path), "exists": Path(str(asset_path)).is_file()}
    return {
        "camera_path": camera_path,
        "has_lut_api": "OmniLensDistortionLutAPI" in prim.GetAppliedSchemas(),
        "model": prim.GetAttribute("omni:lensdistortion:model").Get(),
        "assets": assets,
    }


__all__ = ["apply_lut_manifest", "verify_lut_camera"]
