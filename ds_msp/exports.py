"""Write LDC meshes and Isaac Sim LUTs for a set of cameras, in one folder layout.

A rig run (``export_ldc``/``export_isaac_lut`` in ``calib_param.yml``) and ``ds-msp export``
both call :func:`write_exports`, so their folders look the same: ``ldc/`` and ``isaac_lut/``
with one file set per camera, and ``exports.json`` listing each file, its measured accuracy,
or why it could not be written.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .core.contracts import CameraModel

SUMMARY_FILE = "exports.json"


@dataclass(frozen=True)
class LdcSettings:
    """How to build each LDC mesh. An empty ``output_size`` uses each camera's size."""

    output_size: Tuple[int, ...] = ()
    downsample_factor: int = 4
    balance: float = 0.5


@dataclass(frozen=True)
class LutSettings:
    """How to build each Isaac LUT. An empty ``texture_size`` uses each camera's size."""

    texture_size: Tuple[int, ...] = ()


@dataclass(frozen=True)
class ExportCamera:
    """One camera to export: the file-name stem, the model and its image size."""

    name: str
    model: CameraModel
    size: Tuple[int, int]
    source: Optional[Dict[str, Any]] = None


def write_exports(
    cameras: Sequence[ExportCamera],
    folder: str,
    *,
    ldc: Optional[LdcSettings] = None,
    lut: Optional[LutSettings] = None,
    overwrite: bool = True,
) -> Dict[str, str]:
    """Export every camera; a camera that fails is reported and the others still export.

    Returns an output listing for reports and manifests. Writes ``exports.json`` in
    ``folder`` even when every export failed, so the reason is always on disk.
    """
    paths: Dict[str, str] = {}
    summary: Dict[str, Any] = {"ldc": None, "isaac_lut": None}
    if ldc is not None:
        from .ldc import export_ldc

        out = os.path.join(folder, "ldc")
        rows: List[Dict[str, Any]] = []
        for cam in cameras:
            row: Dict[str, Any] = {"camera": cam.name}
            try:
                done = export_ldc(
                    cam.model, cam.size, out, cam.name,
                    output_size=tuple(ldc.output_size) or None,  # type: ignore[arg-type]
                    downsample_factor=ldc.downsample_factor, balance=ldc.balance,
                    overwrite=overwrite, source=cam.source,
                )
            except (FileExistsError, OSError, TypeError, ValueError) as exc:
                row["problem"] = str(exc)
                print(f"LDC mesh {cam.name}: not written: {exc}", file=sys.stderr)
            else:
                acc = done.accuracy
                row.update(record=os.path.basename(done.files.record),
                           mesh=os.path.basename(done.files.mesh),
                           output_size=[int(done.config["output_width"]),
                                        int(done.config["output_height"])],
                           q3_overflow=bool(done.config["q3_overflow"]),
                           n_invalid_nodes=int(done.config["n_invalid_nodes"]),
                           median_px=acc["median_px"], p99_px=acc["p99_px"],
                           max_px=acc["max_px"], in_view_fraction=acc["in_view_fraction"],
                           notes=list(done.notes))
                worst = "n/a" if acc["max_px"] is None else f"{acc['max_px']:.3f}"
                print(f"LDC mesh {cam.name}: worst mesh error {worst} px -> {done.files.record}")
                for note in done.notes:
                    print(f"  note: {note}")
            rows.append(row)
        summary["ldc"] = {"folder": "ldc", "cameras": rows,
                          "settings": {"output_size": list(ldc.output_size),
                                       "downsample_factor": ldc.downsample_factor,
                                       "balance": ldc.balance}}
        paths["ldc_meshes"] = (f"{out}/ ({sum('record' in r for r in rows)} of "
                               f"{len(cameras)} cameras)")
    if lut is not None:
        from .isaac_sim import export_lut

        out = os.path.join(folder, "isaac_lut")
        rows = []
        for cam in cameras:
            row = {"camera": cam.name}
            w, h = (int(v) for v in cam.size)
            tw, th = lut.texture_size or (w, h)
            try:
                bundle = export_lut(cam.model, w, h, out, texture_width=int(tw),
                                    texture_height=int(th), stem=cam.name, overwrite=overwrite)
            except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
                row["problem"] = str(exc)
                print(f"Isaac LUT {cam.name}: not written: {exc}", file=sys.stderr)
            else:
                row.update(manifest=os.path.basename(bundle.manifest),
                           texture_size=[int(tw), int(th)],
                           round_trip_max_px=bundle.round_trip_max_error_px)
                print(f"Isaac LUT {cam.name}: {bundle.manifest}")
            rows.append(row)
        summary["isaac_lut"] = {"folder": "isaac_lut", "cameras": rows,
                                "settings": {"texture_size": list(lut.texture_size)}}
        paths["isaac_luts"] = (f"{out}/ ({sum('manifest' in r for r in rows)} of "
                               f"{len(cameras)} cameras)")
    os.makedirs(folder, exist_ok=True)
    summary_path = os.path.join(folder, SUMMARY_FILE)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    paths["exports_summary"] = summary_path
    return paths
