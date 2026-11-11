"""Read camera intrinsics from every calibration file format DS-MSP supports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import cv2
import numpy as np
import yaml

from ..core.contracts import CameraModel
from ..models.registry import canonical_name, model_class


@dataclass(frozen=True)
class LoadedCalibration:
    """A reconstructed model plus its nominal sensor resolution."""

    model: CameraModel
    resolution: Tuple[int, int]
    source_format: str
    camera: str


def model_from_params(model_name: str, values: Sequence[str | float]) -> CameraModel:
    """Construct a model from ordered numbers or ``name=value`` strings.

    Ordered values follow the model class's public ``param_names`` tuple. Named
    values may be given in any order, making shell commands self-documenting.
    """
    cls = model_class(model_name)
    raw = list(values)
    if not raw:
        raise ValueError(
            f"--params is required for model {canonical_name(model_name)!r}; order: "
            + " ".join(cls.param_names)
        )
    named = ["=" in str(value) for value in raw]
    if any(named) and not all(named):
        raise ValueError("do not mix ordered values and name=value entries in --params")
    if all(named):
        parsed = {}
        for item in raw:
            key, separator, value = str(item).partition("=")
            key = key.strip()
            if not separator or key not in cls.param_names:
                raise ValueError(
                    f"unknown parameter {key!r} for {cls.name}; expected {cls.param_names}"
                )
            if key in parsed:
                raise ValueError(f"parameter {key!r} was provided more than once")
            parsed[key] = float(value)
        missing = [name for name in cls.param_names if name not in parsed]
        if missing:
            raise ValueError(f"missing parameters for {cls.name}: {', '.join(missing)}")
        vector = [parsed[name] for name in cls.param_names]
    else:
        if len(raw) != len(cls.param_names):
            raise ValueError(
                f"model {cls.name!r} needs {len(cls.param_names)} values in order "
                f"{cls.param_names}; received {len(raw)}"
            )
        vector = [float(value) for value in raw]
    return cls.from_params(np.asarray(vector, dtype=np.float64))


def _camera_number(camera: str | int) -> int:
    text = str(camera).strip().lower()
    for prefix in ("camera_", "camera", "cam"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"camera must look like cam0, camera_0, or 0; got {camera!r}") from exc


def _resolution_or_override(
    stored: Tuple[int, int], resolution: Tuple[int, int] | None
) -> Tuple[int, int]:
    chosen = stored if resolution is None else tuple(int(v) for v in resolution)
    if len(chosen) != 2 or min(chosen) <= 0:
        raise ValueError(
            "camera artifact has no usable resolution; provide --resolution WIDTH HEIGHT"
        )
    return int(chosen[0]), int(chosen[1])


def _load_mccalib(
    path: Path,
    camera: str | int,
    model_name: str | None,
    resolution: Tuple[int, int] | None,
) -> LoadedCalibration:
    from .mccalib import _DISTORTION_LAYOUT, _camera_model_field, load_camera

    camera_id = _camera_number(camera)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    node = fs.getNode(f"camera_{camera_id}")
    if node.empty():
        fs.release()
        raise KeyError(f"no camera_{camera_id} in {path}")
    width = int(node.getNode("img_width").real())
    height = int(node.getNode("img_height").real())
    detected_name = _camera_model_field(node)
    if model_name is None:
        fs.release()
        model = load_camera(str(path), camera_id)
    else:
        selected = canonical_name(model_name)
        if detected_name is not None and canonical_name(detected_name) != selected:
            fs.release()
            raise ValueError(
                f"artifact says model {detected_name!r}, but --model requested {selected!r}"
            )
        matrix = node.getNode("camera_matrix").mat()
        dist_node = node.getNode("distortion_vector")
        distortion = dist_node.mat() if not dist_node.empty() else np.zeros((1, 0))
        fs.release()
        distortion = np.asarray(distortion, dtype=np.float64).ravel()
        layout = _DISTORTION_LAYOUT[selected]
        if distortion.size != len(layout):
            raise ValueError(
                f"camera_{camera_id}: model {selected!r} expects distortion values "
                f"{layout}, artifact has {distortion.size}"
            )
        kwargs = dict(zip(layout, distortion.tolist()))
        kwargs.update(
            fx=float(matrix[0, 0]), fy=float(matrix[1, 1]),
            cx=float(matrix[0, 2]), cy=float(matrix[1, 2]),
        )
        model = model_class(selected)(**kwargs)
    return LoadedCalibration(
        model, _resolution_or_override((width, height), resolution),
        "mccalib", f"camera_{camera_id}",
    )


def _generic_resolution(data: Mapping) -> Tuple[int, int]:
    for key in ("resolution", "image_size", "imageSize", "nominal_resolution"):
        value = data.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return int(value[0]), int(value[1])
    if "width" in data and "height" in data:
        return int(data["width"]), int(data["height"])
    if "img_width" in data and "img_height" in data:
        return int(data["img_width"]), int(data["img_height"])
    return 0, 0


def _load_generic(
    data: Mapping,
    path: Path,
    model_name: str | None,
    resolution: Tuple[int, int] | None,
) -> LoadedCalibration:
    # Accept both CameraModel.to_dict() and a compact {model, params, resolution}
    # artifact. A generated Isaac manifest is accepted too, which makes rebaking
    # at another texture resolution straightforward.
    model_field = data.get("model")
    model_data = model_field if isinstance(model_field, Mapping) else data
    if not isinstance(model_data, Mapping):
        raise ValueError(f"camera model block in {path} must be a mapping")
    detected = model_data.get("name") or (
        model_data.get("model") if not isinstance(model_data.get("model"), Mapping) else None
    )
    selected = canonical_name(model_name or detected)
    cls = model_class(selected)
    if model_name is not None and detected is not None and canonical_name(detected) != selected:
        raise ValueError(f"artifact says model {detected!r}, but --model requested {selected!r}")

    params = model_data.get("params")
    if params is not None:
        if isinstance(params, Mapping):
            model = model_from_params(selected, [f"{name}={params[name]}" for name in cls.param_names])
        else:
            model = model_from_params(selected, params)
    else:
        missing = [name for name in cls.param_names if name not in model_data]
        if missing:
            raise ValueError(f"camera artifact is missing parameters: {', '.join(missing)}")
        model = cls.from_dict({"model": selected, **model_data})
    stored_resolution = _generic_resolution(data)
    return LoadedCalibration(
        model, _resolution_or_override(stored_resolution, resolution), "ds-msp", "cam0"
    )


def _colmap_cameras_path(source: Path) -> Path | None:
    if source.is_dir():
        candidate = source / "cameras.txt"
        return candidate if candidate.is_file() else None
    return source if source.name == "cameras.txt" else None


def _colmap_rows(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(line.split())
    return rows


def _load_colmap(
    path: Path,
    camera: str | int,
    model_name: str | None,
    resolution: Tuple[int, int] | None,
) -> LoadedCalibration:
    from .colmap import colmap_to_model

    rows = _colmap_rows(path)
    if not rows:
        raise ValueError(f"{path} lists no cameras")
    text = str(camera).strip().lower()
    if text in ("cam0", "camera_0", "0") and not any(r[0] == "0" for r in rows):
        row = rows[0]  # the default camera is the first one listed
    else:
        wanted = str(_camera_number(camera))
        matches = [r for r in rows if r[0] == wanted]
        if not matches:
            ids = ", ".join(r[0] for r in rows)
            raise KeyError(f"no camera {wanted} in {path}; it lists cameras {ids}")
        row = matches[0]
    try:
        model = colmap_to_model(row[1], [float(v) for v in row[4:]])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"camera {row[0]} in {path}: COLMAP model {row[1]!r} is not supported") from exc
    _check_requested_model(model, model_name)
    return LoadedCalibration(
        model, _resolution_or_override((int(row[2]), int(row[3])), resolution), "colmap", row[0]
    )


def _load_nerfstudio(
    data: Mapping, path: Path, model_name: str | None, resolution: Tuple[int, int] | None
) -> LoadedCalibration:
    from .nerfstudio import read_nerfstudio

    loaded = read_nerfstudio(str(path))
    model = loaded["model"]
    _check_requested_model(model, model_name)
    return LoadedCalibration(
        model,
        _resolution_or_override((loaded["width"], loaded["height"]), resolution),
        "nerfstudio",
        "cam0",
    )


def _check_requested_model(model: CameraModel, model_name: str | None) -> None:
    if model_name is not None and canonical_name(model_name) != model.name:
        raise ValueError(
            f"artifact says model {model.name!r}, but --model requested "
            f"{canonical_name(model_name)!r}"
        )


def _is_mccalib(source: Path) -> bool:
    text_prefix = source.read_bytes()[:4096].decode("utf-8", errors="ignore")
    return "%YAML:1.0" in text_prefix or "camera_0:" in text_prefix


def _read_mapping(source: Path) -> Mapping:
    if source.suffix.lower() == ".json":
        data = json.loads(source.read_text())
    else:
        data = yaml.safe_load(source.read_text())
    if not isinstance(data, Mapping):
        raise ValueError(f"camera artifact root in {source} must be a mapping")
    return data


def _is_nerfstudio(data: Mapping) -> bool:
    return "camera_model" in data and "fl_x" in data and "frames" in data


def load_calibration(
    path: str | Path,
    *,
    camera: str | int = "cam0",
    model_name: str | None = None,
    resolution: Tuple[int, int] | None = None,
) -> LoadedCalibration:
    """Load one camera from an existing calibration file.

    Supported inputs are MC-Calib OpenCV YAML (also what a DS-MSP rig run writes),
    Kalibr/DS-MSP camchain YAML, COLMAP ``cameras.txt`` (or the folder holding it),
    nerfstudio ``transforms.json``, ``CameraModel.to_dict()`` JSON/YAML, and DS-MSP
    Isaac LUT manifests. ``model_name`` can supply the missing model label in older
    MC-Calib files.
    """
    source = Path(path).expanduser().resolve()
    colmap = _colmap_cameras_path(source)
    if colmap is not None:
        return _load_colmap(colmap, camera, model_name, resolution)
    if not source.is_file():
        raise FileNotFoundError(source)
    if _is_mccalib(source):
        return _load_mccalib(source, camera, model_name, resolution)

    data = _read_mapping(source)
    if _is_nerfstudio(data):
        return _load_nerfstudio(data, source, model_name, resolution)

    cam_key = str(camera)
    if cam_key not in data and cam_key.startswith("camera_"):
        cam_key = "cam" + cam_key.removeprefix("camera_")
    if cam_key not in data and str(camera).isdigit():
        cam_key = "cam" + str(camera)
    block = data.get(cam_key)
    if isinstance(block, Mapping) and "intrinsics" in block:
        from .kalibr import from_kalibr_cam

        model = from_kalibr_cam(dict(block))
        _check_requested_model(model, model_name)
        stored = block.get("resolution", [0, 0])
        return LoadedCalibration(
            model,
            _resolution_or_override((int(stored[0]), int(stored[1])), resolution),
            "kalibr",
            cam_key,
        )
    return _load_generic(data, source, model_name, resolution)


@dataclass(frozen=True)
class CameraInFile:
    """One camera found in a calibration file: loaded, or the reason it could not be."""

    camera: str
    loaded: LoadedCalibration | None
    problem: str | None


def _camera_keys(source: Path) -> list[str]:
    colmap = _colmap_cameras_path(source)
    if colmap is not None:
        return [row[0] for row in _colmap_rows(colmap)]
    if not source.is_file():
        raise FileNotFoundError(source)
    if _is_mccalib(source):
        fs = cv2.FileStorage(str(source), cv2.FILE_STORAGE_READ)
        try:
            root = fs.root()
            keys = [k for k in root.keys() if k.startswith("camera_")]
        finally:
            fs.release()
        return sorted(keys, key=_camera_number)
    data = _read_mapping(source)
    kalibr = [k for k, v in data.items() if isinstance(v, Mapping) and "intrinsics" in v]
    return [str(k) for k in kalibr] or ["cam0"]


def list_cameras(path: str | Path, *, model_name: str | None = None) -> list[CameraInFile]:
    """Every camera in a calibration file, in file order.

    A camera that cannot be read (for example an older MC-Calib entry without a
    model label) is listed with its problem, so one bad entry hides no others.
    """
    source = Path(path).expanduser().resolve()
    found = []
    for key in _camera_keys(source):
        try:
            loaded = load_calibration(source, camera=key, model_name=model_name)
            found.append(CameraInFile(key, loaded, None))
        except (KeyError, TypeError, ValueError) as exc:
            found.append(CameraInFile(key, None, str(exc).strip("'\"")))
    return found


__all__ = [
    "CameraInFile", "LoadedCalibration", "list_cameras", "load_calibration", "model_from_params",
]
