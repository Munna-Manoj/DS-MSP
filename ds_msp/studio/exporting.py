"""Studio's export page: read the cameras of a source, check the settings, build the command.

A source is one of:

* ``{"kind": "file", "path": ...}``: any calibration file :mod:`ds_msp.io.camera_file` reads;
* ``{"kind": "result", "id": ...}``: a Studio result (its ``calibrated_cameras_data.yml``);
* ``{"kind": "typed", "cameras": [...]}``: lens values typed on the page, one entry per
  camera, in the form :func:`ds_msp.studio.lineup.write_lens_values` takes.

Every export runs ``ds-msp export`` (ADR-0035), so the page and a terminal give the same files.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..exports import SUMMARY_FILE

CAMERAS_FILE = "calibrated_cameras_data.yml"
TYPED_FILE = "typed_cameras.yml"


class ExportProblem(ValueError):
    """Settings an export cannot run with, as ``{"key", "message"}`` rows."""

    def __init__(self, message: str, problems: Optional[List[Dict[str, str]]] = None):
        super().__init__(message)
        self.problems = problems or [{"key": "", "message": message}]


def _typed_models(cameras: Sequence[Mapping[str, Any]]) -> List[Tuple[str, Any, Tuple[int, int]]]:
    """Check typed values exactly as the lens file writer does, without writing anything."""
    from ..models.registry import model_class
    from .lineup import lens_fields

    if not cameras:
        raise ExportProblem("Add at least one camera.", [{"key": "cameras",
                                                          "message": "no camera"}])
    out = []
    for k, cam in enumerate(cameras):
        model = str(cam.get("model") or "")
        try:
            fields = lens_fields(model)
            params = {key: float(cam[key]) for key in fields}
            width, height = float(cam["width"]), float(cam["height"])
        except (KeyError, TypeError, ValueError):
            raise ExportProblem(f"Camera {k + 1}: fill in every value.",
                                [{"key": f"cameras.{k}", "message": "has an empty value"}]) from None
        if params["fx"] <= 0 or params["fy"] <= 0:
            raise ExportProblem(f"Camera {k + 1}: fx and fy must be above 0.",
                                [{"key": f"cameras.{k}", "message": "fx and fy must be above 0"}])
        if width < 1 or height < 1 or width != int(width) or height != int(height):
            raise ExportProblem(f"Camera {k + 1}: the image size must be whole pixels.",
                                [{"key": f"cameras.{k}", "message": "image size"}])
        try:
            out.append((f"camera_{k}", model_class(model)(**params), (int(width), int(height))))
        except (TypeError, ValueError) as exc:
            raise ExportProblem(f"Camera {k + 1}: {exc}",
                                [{"key": f"cameras.{k}", "message": str(exc)}]) from None
    return out


def source_file(source: Mapping[str, Any], inside: Callable[[str], Path],
                result_folder: Callable[[str], Path]) -> Path:
    """The calibration file a file or result source names, checked to exist."""
    kind = source.get("kind")
    if kind == "file":
        path = inside(str(source.get("path") or ""))
    elif kind == "result":
        path = result_folder(str(source.get("id") or "")) / CAMERAS_FILE
    else:
        raise ExportProblem("Choose where the camera values come from.",
                            [{"key": "source", "message": "unknown source"}])
    if not (path.is_file() or (path.is_dir() and (path / "cameras.txt").is_file())):
        raise ExportProblem(f"{path} is not a calibration file Studio can read.",
                            [{"key": "source", "message": "no such file"}])
    return path


def list_source(source: Mapping[str, Any], inside: Callable[[str], Path],
                result_folder: Callable[[str], Path]) -> Dict[str, Any]:
    """Every camera of a source: name, model, image size, or why it cannot be read."""
    if source.get("kind") == "typed":
        rows = [{"camera": name, "model": m.name, "width": size[0], "height": size[1],
                 "problem": None}
                for name, m, size in _typed_models(list(source.get("cameras") or []))]
        return {"format": "typed", "file": None, "cameras": rows}
    from ..io.camera_file import list_cameras

    path = source_file(source, inside, result_folder)
    try:
        found = list_cameras(path)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ExportProblem(f"Studio could not read {path.name}: {exc}",
                            [{"key": "source", "message": str(exc)}]) from None
    rows = []
    fmt = None
    for c in found:
        if c.loaded is None:
            rows.append({"camera": c.camera, "model": None, "width": None, "height": None,
                         "problem": c.problem})
        else:
            fmt = fmt or c.loaded.source_format
            rows.append({"camera": c.camera, "model": c.loaded.model.name,
                         "width": c.loaded.resolution[0], "height": c.loaded.resolution[1],
                         "problem": None})
    if not rows:
        raise ExportProblem(f"{path.name} lists no camera.", [{"key": "source",
                                                              "message": "no camera"}])
    return {"format": fmt, "file": str(path), "cameras": rows}


def _settings(spec: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """The checked LDC and LUT settings of a spec; ``None`` for a target not chosen."""
    problems: List[Dict[str, str]] = []
    ldc = spec.get("ldc")
    lut = spec.get("lut")
    if not ldc and not lut:
        problems.append({"key": "targets", "message": "choose TI LDC, Isaac Sim LUT, or both"})
    out_ldc = out_lut = None
    if ldc:
        size = list(ldc.get("output_size") or [])
        factor, balance = ldc.get("downsample_factor", 4), ldc.get("balance", 0.5)
        try:
            factor, balance = int(factor), float(balance)
            size = [int(v) for v in size]
        except (TypeError, ValueError):
            problems.append({"key": "ldc", "message": "the mesh settings must be numbers"})
        else:
            if size and (len(size) != 2 or min(size) <= 0):
                problems.append({"key": "ldc.output_size",
                                 "message": "give a width and a height above 0, or leave it empty"})
            if not 0 <= factor <= 7:
                problems.append({"key": "ldc.downsample_factor",
                                 "message": "node spacing must be 1 to 128 pixels"})
            if not 0.0 <= balance <= 1.0:
                problems.append({"key": "ldc.balance", "message": "must be between 0 and 1"})
            out_ldc = {"output_size": size, "downsample_factor": factor, "balance": balance}
    if lut:
        size = list(lut.get("texture_size") or [])
        try:
            size = [int(v) for v in size]
        except (TypeError, ValueError):
            size = [0]
        if size and (len(size) != 2 or min(size) <= 0):
            problems.append({"key": "lut.texture_size",
                             "message": "give a width and a height above 0, or leave it empty"})
        out_lut = {"texture_size": size}
    if problems:
        raise ExportProblem("The export settings have problems.", problems)
    return out_ldc, out_lut


def chosen(spec: Mapping[str, Any], listed: Mapping[str, Any]) -> List[str]:
    """The camera names to export: those asked for, each one readable."""
    names = [str(c) for c in spec.get("cameras") or []]
    rows = {r["camera"]: r for r in listed["cameras"]}
    if not names:
        raise ExportProblem("Choose at least one camera.", [{"key": "cameras",
                                                            "message": "no camera chosen"}])
    problems = []
    for name in names:
        row = rows.get(name)
        if row is None:
            problems.append({"key": "cameras", "message": f"{name} is not in this source"})
        elif row["problem"]:
            problems.append({"key": "cameras", "message": f"{name}: {row['problem']}"})
    if problems:
        raise ExportProblem("Some chosen cameras cannot be exported.", problems)
    return names


def check(spec: Mapping[str, Any], inside: Callable[[str], Path],
          result_folder: Callable[[str], Path], *, stride: int = 4) -> Dict[str, Any]:
    """What each chosen camera's LDC mesh would be like, without writing anything.

    The error is measured on every ``stride``-th output pixel, so it is a close preview of the
    number the export writes into its record, not the same number.
    """
    from ..io.camera_file import load_calibration
    from ..ldc import TI_LDC_MeshGenerator, mesh_accuracy

    listed = list_source(spec.get("source") or {}, inside, result_folder)
    names = chosen(spec, listed)
    ldc, lut = _settings(spec)
    rows = []
    if spec["source"].get("kind") == "typed":
        typed = {n: (m, s) for n, m, s in _typed_models(spec["source"]["cameras"])}
    for name in names:
        if spec["source"].get("kind") == "typed":
            model, size = typed[name]
        else:
            loaded = load_calibration(listed["file"], camera=name)
            model, size = loaded.model, loaded.resolution
        row: Dict[str, Any] = {"camera": name, "model": model.name,
                               "width": size[0], "height": size[1]}
        if ldc:
            out_w, out_h = ldc["output_size"] or size
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = TI_LDC_MeshGenerator(model).generate_mesh_and_intrinsics(
                    int(out_w), int(out_h), ldc["downsample_factor"], ldc["balance"])
            acc = mesh_accuracy(model, result, input_size=size, stride=stride)
            mesh_h, mesh_w = result["config"]["mesh_size"][:2]
            row["ldc"] = {
                "output_size": [int(out_w), int(out_h)], "mesh_size": [int(mesh_w), int(mesh_h)],
                "q3_overflow": bool(result["config"]["q3_overflow"]),
                "n_invalid_nodes": int(result["config"]["n_invalid_nodes"]),
                "median_px": acc["median_px"], "p99_px": acc["p99_px"], "max_px": acc["max_px"],
                "in_view_fraction": acc["in_view_fraction"],
            }
        rows.append(row)
    return {"cameras": rows, "ldc": ldc, "lut": lut}


def check_spec(spec: Mapping[str, Any], inside: Callable[[str], Path],
               result_folder: Callable[[str], Path]) -> None:
    """Refuse a spec the export cannot run, before any folder is made."""
    source = spec.get("source") or {}
    listed = list_source(source, inside, result_folder)
    chosen(spec, listed)
    _settings(spec)


def command_args(spec: Mapping[str, Any], folder: Path, inside: Callable[[str], Path],
                 result_folder: Callable[[str], Path]) -> List[str]:
    """Check a spec and return the ``ds-msp export`` arguments that run it into ``folder``.

    Typed values are written next to the export as :data:`TYPED_FILE` first, so the export
    stands on its own and can be repeated from a terminal.
    """
    source = spec.get("source") or {}
    listed = list_source(source, inside, result_folder)
    names = chosen(spec, listed)
    ldc, lut = _settings(spec)
    if source.get("kind") == "typed":
        from .lineup import write_lens_values

        folder.mkdir(parents=True, exist_ok=True)
        path = write_lens_values(list(source["cameras"]), folder, dest=folder / TYPED_FILE)
    else:
        path = Path(listed["file"])
    args = ["--input", str(path)]
    if len(names) != len(listed["cameras"]):
        for name in names:
            args += ["--camera", name]
    if ldc:
        args.append("--ldc")
        if ldc["output_size"]:
            args += ["--ldc-output-size", *[str(v) for v in ldc["output_size"]]]
        args += ["--ldc-downsample-factor", str(ldc["downsample_factor"]),
                 "--ldc-balance", repr(float(ldc["balance"]))]
    if lut:
        args.append("--isaac-lut")
        if lut["texture_size"]:
            args += ["--lut-texture-size", *[str(v) for v in lut["texture_size"]]]
    args += ["--output-dir", str(folder)]
    return args


def summary_of(folder: Path) -> Optional[Dict[str, Any]]:
    """The ``exports.json`` of a result or export folder, if there is one."""
    import json

    path = folder / SUMMARY_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# -- pictures of a mesh or a LUT at work (ds_msp.studio.lensview) ---------------------------------
def _cam_index(name: str) -> Optional[int]:
    tail = str(name).rsplit("_", 1)[-1]
    tail = tail[3:] if tail.startswith("cam") else tail
    return int(tail) if tail.isdigit() else None


def _photo(folder: Optional[Path], cam: Optional[int], frame: Optional[int], readable
           ) -> Tuple[Optional[Any], List[int], Optional[int]]:
    """A photo of ``cam`` in the rig result ``folder``: the image, the frames that have one, and
    the frame shown. ``(None, [], None)`` when the folder has no photos Studio may read."""
    import cv2

    from .scene import frame_list, photo_index

    if folder is None or cam is None or not (folder / "calibration_report.json").is_file():
        return None, [], None
    index = photo_index(folder)["index"]                 # read once, not once per frame

    def usable(f: int) -> Optional[Path]:
        raw = index.get((cam, f))
        if not raw:
            return None
        path = Path(raw).resolve()
        if not path.is_file():
            return None
        if readable is not None and not any(r == path or r in path.parents for r in readable):
            return None
        return path

    paths = {}
    for f in (x["frame"] for x in frame_list(folder, cam)["frames"]):
        p = usable(f)
        if p is not None:
            paths[f] = p
        if len(paths) >= 60:
            break
    if not paths:
        return None, [], None
    frames = list(paths)
    shown = frame if frame in paths else frames[0]
    img = cv2.imread(str(paths[shown]), cv2.IMREAD_COLOR)
    return img, frames, (shown if img is not None else None)


def _render(model, size, kind: str, ldc: Optional[Dict[str, Any]], mesh=None, lut_texture=None,
            photo=None, overlay: bool = False, display: int = 720) -> Dict[str, Any]:
    from . import lensview

    out: Dict[str, Any] = {}
    stats: Dict[str, float] = {}
    if kind == "ldc":
        if photo is not None and tuple(photo.shape[1::-1]) != tuple(size):
            photo = None                                     # a photo of another size: use the room
        out["camera"] = lensview.camera_view(model, size, display, photo)
        out["corrected"] = lensview.ldc_view(model, size, mesh["mesh_lut"], mesh["K_new"],
                                             mesh["downsample_factor"], mesh["output_size"], display,
                                             photo, overlay, stats if overlay else None)
    else:
        out["render"] = (lensview.lut_view_from_texture(lut_texture, display) if lut_texture is not None
                         else lensview.lut_view_from_model(model, size, display))
    return {"images": lensview.encode(out), "used_photo": photo is not None,
            "overlay": stats or None}


def view(spec: Mapping[str, Any], camera: str, kind: str, inside, result_folder, *,
         use_photo: bool = False, frame: Optional[int] = None, overlay: bool = False,
         display: int = 720, readable=None) -> Dict[str, Any]:
    """Pictures of one camera's LDC mesh or LUT as the spec would write them; nothing is written."""
    from ..io.camera_file import load_calibration
    from ..ldc import TI_LDC_MeshGenerator

    if kind not in ("ldc", "lut"):
        raise ExportProblem("Show the LDC mesh or the LUT.", [{"key": "kind", "message": "unknown"}])
    listed = list_source(spec.get("source") or {}, inside, result_folder)
    chosen({**spec, "cameras": [camera]}, listed)
    ldc, _lut = _settings({**spec, "ldc": spec.get("ldc") or (None if kind == "lut" else {}),
                           "lut": spec.get("lut") or ({} if kind == "lut" else None)})
    src = spec["source"]
    if src.get("kind") == "typed":
        model, size = {n: (m, s) for n, m, s in _typed_models(src["cameras"])}[camera]
    else:
        loaded = load_calibration(listed["file"], camera=camera)
        model, size = loaded.model, loaded.resolution
    folder = result_folder(str(src.get("id"))) if src.get("kind") == "result" else None
    photo, frames, shown = _photo(folder, _cam_index(camera), frame, readable) if use_photo else (None, [], None)
    mesh = None
    if kind == "ldc":
        out_size = tuple(ldc["output_size"]) if ldc and ldc["output_size"] else tuple(size)
        factor = ldc["downsample_factor"] if ldc else 4
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = TI_LDC_MeshGenerator(model).generate_mesh_and_intrinsics(
                int(out_size[0]), int(out_size[1]), factor, ldc["balance"] if ldc else 0.5)
        mesh = {"mesh_lut": r["mesh_lut"], "K_new": r["K_new"], "downsample_factor": factor,
                "output_size": out_size}
    doc = _render(model, size, kind, ldc, mesh=mesh, photo=photo, overlay=overlay, display=display)
    doc.update(camera=camera, kind=kind, frames=frames, frame=shown, from_files=False)
    return doc


def view_files(folder: Path, camera: str, kind: str, *, use_photo: bool = False,
               frame: Optional[int] = None, overlay: bool = False, display: int = 720,
               readable=None) -> Dict[str, Any]:
    """Pictures of one camera's written LDC mesh or LUT, read from the files themselves."""
    import json

    from ..isaac_sim.lut import _read_rgb32_exr
    from ..ldc import read_ldc_mesh
    from ..models.registry import model_class

    summary = summary_of(folder)
    part = (summary or {}).get("ldc" if kind == "ldc" else "isaac_lut")
    row = next((r for r in (part or {}).get("cameras", []) if r.get("camera") == camera), None)
    if row is None or row.get("problem"):
        raise ExportProblem(f"{camera} has no {'LDC mesh' if kind == 'ldc' else 'Isaac LUT'} here.",
                            [{"key": "camera", "message": "not exported"}])
    if kind == "ldc":
        mesh_q3, record = read_ldc_mesh(folder / part["folder"] / row["record"])
        cam = record["camera_model"]
        model = model_class(cam["name"]).from_dict({"model": cam["name"], **(cam.get("params") or {})})
        src = record.get("source") or {}
        size = (int(src.get("input_width") or record["output_width"]),
                int(src.get("input_height") or record["output_height"]))
        mesh = {"mesh_lut": mesh_q3, "K_new": np.asarray(record["K_new"]),
                "downsample_factor": record["downsample_factor"],
                "output_size": (record["output_width"], record["output_height"])}
        photo, frames, shown = _photo(folder, _cam_index(camera), frame, readable) if use_photo else (None, [], None)
        doc = _render(model, size, "ldc", None, mesh=mesh, photo=photo, overlay=overlay, display=display)
    else:
        manifest = json.loads((folder / part["folder"] / row["manifest"]).read_text())
        enter = manifest["isaac_sim"]["attributes"]["omni:lensdistortion:lut:rayEnterDirectionTexture"]
        texture = _read_rgb32_exr(folder / part["folder"] / str(enter))
        frames, shown = [], None
        doc = _render(None, None, "lut", None, lut_texture=texture, display=display)
    doc.update(camera=camera, kind=kind, frames=frames, frame=shown, from_files=True)
    return doc
