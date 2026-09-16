"""Export any DS-MSP camera model as an Isaac Sim RTX camera LUT pair.

The RTX generalized camera contract uses two 32-bit floating-point EXR textures:

* ``rayEnterDirectionTexture`` maps image NDC to a camera-local unit ray.
* ``rayExitPositionTexture`` maps an octahedrally encoded ray to image NDC.

DS-MSP models use the OpenCV camera frame (+X right, +Y down, +Z forward), while
RTX uses +X right, +Y up, -Z forward.  :class:`IsaacProjectionAdapter` owns that
coordinate conversion so model implementations remain renderer-independent.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, NamedTuple, Tuple

import numpy as np

from ..core.contracts import CameraModel
from .exr import read_rgb32_exr, write_rgb32_exr


FORMAT_VERSION = "ds-msp.isaac-sim-lut/v1"
INVALID_NDC = -1.0


class LutPaths(NamedTuple):
    """Filesystem paths for the two renderer textures."""

    ray_enter: Path
    ray_exit: Path


@dataclass(frozen=True)
class IsaacLutBundle:
    """All artifacts emitted by :func:`export_lut`."""

    paths: LutPaths
    manifest: Path
    camera_usda: Path
    model_name: str
    nominal_resolution: Tuple[int, int]
    texture_resolution: Tuple[int, int]
    round_trip_max_error_px: float
    valid_pixel_fraction: float


class IsaacProjectionAdapter:
    """Adapt a DS-MSP ``CameraModel`` to the NVIDIA RTX LUT coordinate contract."""

    def __init__(self, model: CameraModel, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("nominal camera width and height must be positive")
        self.model = model
        self.width = int(width)
        self.height = int(height)

    def unproject(self, ndc_u: np.ndarray, ndc_v: np.ndarray):
        """Return ``(rtx_rays, valid)`` for normalized image coordinates.

        Invalid pixels use RTX's documented ``(0, 0, +1)`` sentinel, which
        points backward and is clipped by the renderer.
        """
        ndc_u, ndc_v = np.broadcast_arrays(
            np.asarray(ndc_u, dtype=np.float64), np.asarray(ndc_v, dtype=np.float64)
        )
        pixels = np.stack((ndc_u * self.width, ndc_v * self.height), axis=-1)
        cv_rays, valid = self.model.unproject(pixels)
        cv_rays = np.asarray(cv_rays, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool) & np.all(np.isfinite(cv_rays), axis=-1)

        # OpenCV (+X right, +Y down, +Z forward) -> RTX (+X right, +Y up, -Z forward).
        rtx = np.empty_like(cv_rays)
        rtx[..., 0] = cv_rays[..., 0]
        rtx[..., 1] = -cv_rays[..., 1]
        rtx[..., 2] = -cv_rays[..., 2]
        norms = np.linalg.norm(rtx, axis=-1)
        valid &= np.isfinite(norms) & (norms > 0.0)
        np.divide(rtx, norms[..., None], out=rtx, where=(norms > 0.0)[..., None])
        rtx[~valid] = (0.0, 0.0, 1.0)
        return rtx, valid

    def project(self, rtx_rays: np.ndarray):
        """Return ``(ndc, valid)`` for camera-local RTX directions.

        Invalid/non-projectable directions are deliberately placed off-screen.
        The ray-exit texture has no sentinel value; writing zero would incorrectly
        make an invalid direction land on the top-left image pixel.
        """
        rtx = np.asarray(rtx_rays, dtype=np.float64)
        if rtx.shape[-1] != 3:
            raise ValueError(f"expected (..., 3) RTX rays, got shape {rtx.shape}")
        cv_rays = np.empty_like(rtx)
        cv_rays[..., 0] = rtx[..., 0]
        cv_rays[..., 1] = -rtx[..., 1]
        cv_rays[..., 2] = -rtx[..., 2]
        pixels, valid = self.model.project(cv_rays)
        pixels = np.asarray(pixels, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool) & np.all(np.isfinite(pixels), axis=-1)
        ndc = np.empty_like(pixels)
        ndc[..., 0] = pixels[..., 0] / self.width
        ndc[..., 1] = pixels[..., 1] / self.height
        ndc[~valid] = INVALID_NDC
        return ndc, valid


def texel_centers(size: int) -> np.ndarray:
    """Normalized coordinates at texture texel centers."""
    if size <= 0:
        raise ValueError("texture dimensions must be positive")
    return (np.arange(size, dtype=np.float64) + 0.5) / float(size)


def octahedral_directions(
    width: int, height: int, *, row_start: int = 0, row_stop: int | None = None
) -> np.ndarray:
    """Decode a row range of an octahedral texture into unit directions.

    The returned directions use the RTX LUT layout, with camera-forward ``-Z``
    at the texture center.  Row ranges let large production LUTs be generated
    without allocating several full-resolution float64 work arrays.
    """
    if width <= 0 or height <= 0:
        raise ValueError("texture dimensions must be positive")
    stop = height if row_stop is None else int(row_stop)
    start = int(row_start)
    if not (0 <= start <= stop <= height):
        raise ValueError(f"invalid row range [{start}, {stop}) for height {height}")

    oct_x = texel_centers(width) * 2.0 - 1.0
    oct_y = ((np.arange(start, stop, dtype=np.float64) + 0.5) / height) * 2.0 - 1.0
    direction_x, direction_y = np.meshgrid(oct_x, oct_y)
    direction_z = 1.0 - np.abs(direction_x) - np.abs(direction_y)
    sign_x = np.where(direction_x >= 0.0, 1.0, -1.0)
    sign_y = np.where(direction_y >= 0.0, 1.0, -1.0)
    original_x = direction_x.copy()
    folded = direction_z <= 0.0
    direction_x = np.where(folded, (1.0 - np.abs(direction_y)) * sign_x, direction_x)
    direction_y = np.where(folded, (1.0 - np.abs(original_x)) * sign_y, direction_y)
    inverse_norm = 1.0 / np.sqrt(
        direction_x * direction_x + direction_y * direction_y + direction_z * direction_z
    )
    # Conventional octahedral decoding has +Z at the center. RTX flips Z
    # before encoding, so negate it here to put camera-forward -Z at center.
    return np.stack(
        (direction_x * inverse_norm, direction_y * inverse_norm, -direction_z * inverse_norm),
        axis=-1,
    )


def _model_signature(model: CameraModel) -> Dict[str, Any]:
    return {
        "name": str(model.name),
        "param_names": list(model.param_names),
        "params": [float(v) for v in np.asarray(model.params, dtype=np.float64).ravel()],
    }


def default_stem(
    model: CameraModel,
    nominal_resolution: Tuple[int, int],
    texture_resolution: Tuple[int, int],
) -> str:
    """Stable, content-addressed filename stem for a model and sampling grid."""
    signature = {
        "format": FORMAT_VERSION,
        "model": _model_signature(model),
        "nominal_resolution": list(nominal_resolution),
        "texture_resolution": list(texture_resolution),
    }
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    width, height = nominal_resolution
    return f"{model.name}_{width}x{height}_{digest}"


def validate_projection(
    adapter: IsaacProjectionAdapter, *, sample_width: int = 33, sample_height: int = 25
) -> Tuple[float, float]:
    """Validate model/axis round-trip and return ``(max_error_px, valid_fraction)``."""
    u, v = np.meshgrid(texel_centers(sample_width), texel_centers(sample_height))
    rays, valid_unproject = adapter.unproject(u, v)
    norms = np.linalg.norm(rays, axis=-1)
    if not np.all(np.isfinite(rays)) or not np.allclose(norms, 1.0, atol=2e-10, rtol=0.0):
        raise ValueError("camera unproject did not produce finite unit RTX directions")
    ndc, valid_project = adapter.project(rays)
    valid = valid_unproject & valid_project
    if not np.any(valid):
        raise ValueError("camera model has no valid pixels on the nominal sensor")
    dx = (ndc[..., 0] - u) * adapter.width
    dy = (ndc[..., 1] - v) * adapter.height
    max_error = float(np.max(np.hypot(dx[valid], dy[valid])))
    if not np.isfinite(max_error) or max_error > 1e-5:
        raise ValueError(
            "project/unproject are inconsistent after RTX axis conversion: "
            f"maximum round-trip error is {max_error:.6g} pixels"
        )
    return max_error, float(np.mean(valid_unproject))


def _write_rgb32_exr(path: Path, rgb: np.ndarray) -> None:
    # DS-MSP's own scanline writer (ds_msp.isaac_sim.exr): the project's OpenCV
    # dependency range includes wheels built without the OpenEXR codec.
    write_rgb32_exr(path, np.ascontiguousarray(rgb, dtype=np.float32), compression="zip")


def _read_rgb32_exr(path: Path) -> np.ndarray:
    try:
        image = read_rgb32_exr(path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not read generated EXR texture: {path}: {exc}") from exc
    if not np.all(np.isfinite(image)):
        raise RuntimeError(f"EXR contains non-finite values: {path}")
    return image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_camera_usda(path: Path, manifest: Dict[str, Any]) -> None:
    attributes = manifest["isaac_sim"]["attributes"]
    enter = attributes["omni:lensdistortion:lut:rayEnterDirectionTexture"]
    exit_ = attributes["omni:lensdistortion:lut:rayExitPositionTexture"]
    center_x, center_y = attributes["omni:lensdistortion:lut:opticalCenter"]
    content = f'''#usda 1.0
(
    defaultPrim = "Camera"
)

def Camera "Camera" (
    prepend apiSchemas = ["OmniLensDistortionLutAPI"]
)
{{
    token omni:lensdistortion:model = "lut"
    float omni:lensdistortion:lut:nominalWidth = {float(attributes['omni:lensdistortion:lut:nominalWidth'])}
    float omni:lensdistortion:lut:nominalHeight = {float(attributes['omni:lensdistortion:lut:nominalHeight'])}
    float2 omni:lensdistortion:lut:opticalCenter = ({float(center_x)}, {float(center_y)})
    asset omni:lensdistortion:lut:rayEnterDirectionTexture = @{enter}@
    asset omni:lensdistortion:lut:rayExitPositionTexture = @{exit_}@
}}
'''
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def load_manifest(path: str | Path) -> Dict[str, Any]:
    """Load and minimally validate a generated LUT manifest."""
    manifest_path = Path(path)
    value = json.loads(manifest_path.read_text())
    if value.get("format") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported Isaac LUT manifest format {value.get('format')!r}; "
            f"expected {FORMAT_VERSION!r}"
        )
    return value


def export_lut(
    model: CameraModel,
    width: int,
    height: int,
    output_dir: str | Path,
    *,
    texture_width: int | None = None,
    texture_height: int | None = None,
    stem: str | None = None,
    overwrite: bool = False,
    chunk_rows: int = 128,
) -> IsaacLutBundle:
    """Generate an Isaac Sim ``OmniLensDistortionLutAPI`` artifact bundle.

    Parameters are measured against ``width`` x ``height``.  The texture size
    defaults to that nominal resolution; lowering it is useful for quick tests,
    while production textures should be at least as large as the render product.
    """
    width, height = int(width), int(height)
    texture_width = width if texture_width is None else int(texture_width)
    texture_height = height if texture_height is None else int(texture_height)
    if min(width, height, texture_width, texture_height, chunk_rows) <= 0:
        raise ValueError("nominal size, texture size, and chunk_rows must be positive")

    adapter = IsaacProjectionAdapter(model, width, height)
    max_error, valid_fraction = validate_projection(adapter)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = stem or default_stem(model, (width, height), (texture_width, texture_height))
    paths = LutPaths(
        output / f"{stem}_ray_enter_direction.exr",
        output / f"{stem}_ray_exit_position.exr",
    )
    manifest_path = output / f"{stem}_isaac_lut.json"
    camera_path = output / f"{stem}_camera.usda"
    artifacts = (*paths, manifest_path, camera_path)
    existing = [p for p in artifacts if p.exists()]
    if existing and not overwrite:
        if len(existing) != len(artifacts):
            raise FileExistsError(
                f"incomplete existing LUT bundle for stem {stem!r}; use --overwrite"
            )
        manifest = load_manifest(manifest_path)
        expected = manifest.get("checksums", {})
        for texture in paths:
            if _sha256(texture) != expected.get(texture.name):
                raise RuntimeError(f"existing LUT checksum mismatch: {texture}")
            image = _read_rgb32_exr(texture)
            if image.shape[:2] != (texture_height, texture_width):
                raise RuntimeError(
                    f"existing LUT has shape {image.shape[:2]}, expected "
                    f"{(texture_height, texture_width)}: {texture}"
                )
        return IsaacLutBundle(
            paths, manifest_path, camera_path, str(model.name), (width, height),
            (texture_width, texture_height), max_error, valid_fraction,
        )

    enter_rgb = np.empty((texture_height, texture_width, 3), dtype=np.float32)
    u = texel_centers(texture_width)
    for start in range(0, texture_height, chunk_rows):
        stop = min(start + chunk_rows, texture_height)
        v = (np.arange(start, stop, dtype=np.float64) + 0.5) / texture_height
        grid_u, grid_v = np.meshgrid(u, v)
        rays, _ = adapter.unproject(grid_u, grid_v)
        enter_rgb[start:stop] = rays.astype(np.float32)
    _write_rgb32_exr(paths.ray_enter, enter_rgb)
    del enter_rgb
    gc.collect()

    # The documented texture is RG. The bundle keeps the RGB layout that was
    # validated in Kit, so B is present and zero; R/G hold the exact NDC values.
    exit_rgb = np.zeros((texture_height, texture_width, 3), dtype=np.float32)
    for start in range(0, texture_height, chunk_rows):
        stop = min(start + chunk_rows, texture_height)
        directions = octahedral_directions(
            texture_width, texture_height, row_start=start, row_stop=stop
        )
        ndc, _ = adapter.project(directions)
        exit_rgb[start:stop, :, :2] = ndc.astype(np.float32)
    _write_rgb32_exr(paths.ray_exit, exit_rgb)
    del exit_rgb
    gc.collect()

    for texture in paths:
        image = _read_rgb32_exr(texture)
        if image.shape[:2] != (texture_height, texture_width):
            raise RuntimeError(f"unexpected generated EXR size at {texture}: {image.shape}")

    # The principal point is already baked into both maps. NVIDIA requires the
    # USD optical center to be the nominal image center in that case, otherwise
    # the calibrated offset would be applied a second time.
    attributes: Dict[str, Any] = {
        "omni:lensdistortion:model": "lut",
        "omni:lensdistortion:lut:nominalWidth": float(width),
        "omni:lensdistortion:lut:nominalHeight": float(height),
        "omni:lensdistortion:lut:opticalCenter": [width / 2.0, height / 2.0],
        "omni:lensdistortion:lut:rayEnterDirectionTexture": paths.ray_enter.name,
        "omni:lensdistortion:lut:rayExitPositionTexture": paths.ray_exit.name,
    }
    manifest: Dict[str, Any] = {
        "format": FORMAT_VERSION,
        "model": _model_signature(model),
        "nominal_resolution": [width, height],
        "texture_resolution": [texture_width, texture_height],
        "isaac_sim": {
            "api_schema": "OmniLensDistortionLutAPI",
            "attributes": attributes,
            "asset_paths_are_relative_to": "manifest",
        },
        "validation": {
            "round_trip_max_error_px": max_error,
            "valid_pixel_fraction": valid_fraction,
            "exr_dtype": "float32",
            "ray_enter_channels": "RGB = RTX camera-local XYZ unit direction",
            "ray_exit_channels": "RG = image NDC UV; B = 0 (unused)",
        },
        "checksums": {
            paths.ray_enter.name: _sha256(paths.ray_enter),
            paths.ray_exit.name: _sha256(paths.ray_exit),
        },
    }
    _write_json(manifest_path, manifest)
    _write_camera_usda(camera_path, manifest)
    return IsaacLutBundle(
        paths, manifest_path, camera_path, str(model.name), (width, height),
        (texture_width, texture_height), max_error, valid_fraction,
    )


__all__ = [
    "FORMAT_VERSION",
    "INVALID_NDC",
    "IsaacLutBundle",
    "IsaacProjectionAdapter",
    "LutPaths",
    "default_stem",
    "export_lut",
    "load_manifest",
    "octahedral_directions",
    "texel_centers",
    "validate_projection",
]
