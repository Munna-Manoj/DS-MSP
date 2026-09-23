"""Truth-referenced SE(3) and task-space metrics for calibrated camera rigs.

The calibration estimator stores world-to-camera transforms ``T_c_g``.  MC-Calib and its
Blender ``GroundTruth.yml`` store camera-to-world poses, so public entry points make that
conversion explicit before forming gauge-invariant, all-pair relative transforms.

Baseline length is retained as one diagnostic.  It is never treated as full pose accuracy:
the same length can hide a wrong translation direction or rotation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ACCURACY_SCHEMA = "ds-msp-rig-accuracy"
ACCURACY_SCHEMA_VERSION = 1
PROFILE_SCHEMA = "ds-msp-rig-accuracy-profile"
PROFILE_SCHEMA_VERSION = 1

_EPS = 1e-12
_LIMIT_METRICS = {
    "rotation_error_deg_max": ("rotation_error_deg_max",),
    "camera_center_error_max": ("camera_center_error_max",),
    "relative_translation_error_max": ("relative_translation_error_max",),
    "relative_translation_error_pct_max": ("relative_translation_error_pct_max",),
    "translation_direction_error_deg_max": ("translation_direction_error_deg_max",),
    "baseline_length_error_pct_max": ("baseline_length_error_pct_max",),
    "task_space_p95_max": ("task_space", "worst_pair_p95"),
    "task_space_max": ("task_space", "max"),
}


def _strict_json(text: str) -> dict:
    def reject(value: str):
        raise ValueError(f"non-finite JSON number is not allowed: {value}")

    value = json.loads(text, parse_constant=reject)
    if not isinstance(value, dict):
        raise TypeError("accuracy profile must be a JSON object")
    return value


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _vec3(value: object, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float)
    if out.shape != (3,) or not np.isfinite(out).all():
        raise ValueError(f"{name} must contain exactly three finite numbers")
    return out


def _roi_points(roi: object) -> tuple[np.ndarray, dict]:
    if not isinstance(roi, dict):
        raise TypeError("accuracy profile 'roi' must be an object")
    frame = roi.get("frame", "pair_source_camera")
    unknown = sorted(set(roi) - {"frame", "points", "min", "max", "grid_shape"})
    if unknown:
        raise ValueError(f"unknown accuracy profile roi field(s): {', '.join(unknown)}")
    if frame != "pair_source_camera":
        raise ValueError("accuracy profile roi.frame must be 'pair_source_camera'")
    has_points = "points" in roi
    has_box = any(k in roi for k in ("min", "max", "grid_shape"))
    if has_points == has_box:
        raise ValueError("roi must define either 'points' or all of 'min'/'max'/'grid_shape'")
    if has_points:
        points = np.asarray(roi["points"], dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
            raise ValueError("roi.points must be a non-empty N x 3 array")
        if not np.isfinite(points).all():
            raise ValueError("roi.points must contain only finite values")
        canonical = {"frame": frame, "points": points.tolist()}
        return points, canonical

    lo = _vec3(roi.get("min"), "roi.min")
    hi = _vec3(roi.get("max"), "roi.max")
    shape_raw = np.asarray(roi.get("grid_shape"))
    if shape_raw.shape != (3,) or not np.issubdtype(shape_raw.dtype, np.number):
        raise ValueError("roi.grid_shape must contain exactly three positive integers")
    shape = shape_raw.astype(int)
    if not np.array_equal(shape_raw, shape) or np.any(shape <= 0):
        raise ValueError("roi.grid_shape must contain exactly three positive integers")
    if np.any(hi < lo):
        raise ValueError("roi.max must be greater than or equal to roi.min on every axis")
    axes = [np.linspace(lo[i], hi[i], int(shape[i])) for i in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([axis.ravel() for axis in mesh])
    canonical = {"frame": frame, "min": lo.tolist(), "max": hi.tolist(),
                 "grid_shape": shape.tolist()}
    return points, canonical


@dataclass(frozen=True)
class AccuracyProfile:
    """Declared task-space region, required camera pairs, units, and optional limits."""

    name: str
    length_unit: str
    roi_points: np.ndarray
    roi: dict
    required_pairs: tuple[tuple[int, int], ...]
    limits: dict[str, float | None]
    sha256: str

    def to_dict(self) -> dict:
        return {
            "schema": PROFILE_SCHEMA,
            "schema_version": PROFILE_SCHEMA_VERSION,
            "name": self.name,
            "length_unit": self.length_unit,
            "roi": {**self.roi, "n_points": len(self.roi_points)},
            "required_pairs": [list(pair) for pair in self.required_pairs],
            "limits": dict(self.limits),
            "sha256": self.sha256,
        }


def accuracy_profile_from_dict(value: Mapping[str, object]) -> AccuracyProfile:
    """Validate and normalize a schema-versioned accuracy-profile mapping."""
    raw = dict(value)
    unknown = sorted(set(raw) - {
        "schema", "schema_version", "name", "length_unit", "roi", "required_pairs", "limits",
    })
    if unknown:
        raise ValueError(f"unknown accuracy profile field(s): {', '.join(unknown)}")
    if raw.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"accuracy profile schema must be {PROFILE_SCHEMA!r}")
    if raw.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported accuracy profile schema_version {raw.get('schema_version')!r}; "
            f"expected {PROFILE_SCHEMA_VERSION}")
    name = str(raw.get("name", "")).strip()
    unit = str(raw.get("length_unit", "")).strip()
    if not name:
        raise ValueError("accuracy profile name must be non-empty")
    if not unit:
        raise ValueError("accuracy profile length_unit must be non-empty")
    points, roi = _roi_points(raw.get("roi"))

    pairs_raw = raw.get("required_pairs", [])
    if not isinstance(pairs_raw, list):
        raise TypeError("accuracy profile required_pairs must be a list")
    pairs = []
    for index, pair in enumerate(pairs_raw):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"required_pairs[{index}] must be [source_camera, target_camera]")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in pair):
            raise ValueError(f"required_pairs[{index}] camera ids must be integers")
        a, b = pair
        if a == b:
            raise ValueError(f"required_pairs[{index}] names the same camera twice")
        directed = (a, b)
        if directed in pairs:
            raise ValueError(f"duplicate required camera pair: {directed}")
        pairs.append(directed)

    limits_raw = raw.get("limits", {})
    if not isinstance(limits_raw, dict):
        raise TypeError("accuracy profile limits must be an object")
    unknown = sorted(set(limits_raw) - set(_LIMIT_METRICS))
    if unknown:
        raise ValueError(f"unknown accuracy profile limit(s): {', '.join(unknown)}")
    limits: dict[str, float | None] = {}
    for key in _LIMIT_METRICS:
        limit = limits_raw.get(key)
        if limit is None:
            limits[key] = None
            continue
        number = float(limit)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"accuracy profile limit {key!r} must be finite and non-negative")
        limits[key] = number

    canonical = {
        "schema": PROFILE_SCHEMA,
        "schema_version": PROFILE_SCHEMA_VERSION,
        "name": name,
        "length_unit": unit,
        "roi": roi,
        "required_pairs": [list(pair) for pair in pairs],
        "limits": limits,
    }
    return AccuracyProfile(name=name, length_unit=unit, roi_points=points, roi=roi,
                           required_pairs=tuple(pairs), limits=limits,
                           sha256=_sha256_json(canonical))


def load_accuracy_profile(path: str | Path) -> AccuracyProfile:
    """Load and validate a strict-JSON accuracy profile."""
    profile_path = Path(path).expanduser()
    try:
        raw = _strict_json(profile_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read accuracy profile {profile_path}: {exc}") from exc
    try:
        return accuracy_profile_from_dict(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid accuracy profile {profile_path}: {exc}") from exc


def camera_to_world_as_world_to_camera(poses: Mapping[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Normalize camera-to-world pose mappings to DS-MSP's world-to-camera convention."""
    return {int(cam_id): np.linalg.inv(_transform(T, f"camera {cam_id} pose"))
            for cam_id, T in poses.items()}


def _transform(value: object, name: str) -> np.ndarray:
    T = np.asarray(value, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"{name} must be a finite 4 x 4 transform")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-7) or np.linalg.det(R) <= 0:
        raise ValueError(f"{name} rotation must be a proper orthonormal matrix")
    return T


def relative_transform(world_to_camera: Mapping[int, np.ndarray], source: int,
                       target: int) -> np.ndarray:
    """Return ``T_target_source`` from two world-to-camera transforms."""
    source_T = _transform(world_to_camera[source], f"camera {source} world-to-camera")
    target_T = _transform(world_to_camera[target], f"camera {target} world-to-camera")
    return target_T @ np.linalg.inv(source_T)


def rotation_error_deg(R_est: np.ndarray, R_ref: np.ndarray) -> float:
    """SO(3) geodesic angle between two rotations, in degrees."""
    value = (np.trace(np.asarray(R_ref).T @ np.asarray(R_est)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def _direction_error_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= _EPS or nb <= _EPS:
        return None
    value = float(np.dot(a, b) / (na * nb))
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def _percent(error: float, reference: float) -> float | None:
    return 100.0 * error / reference if reference > _EPS else None


def _task_space(T_est: np.ndarray, T_ref: np.ndarray, points: np.ndarray | None) -> dict:
    if points is None:
        return {"status": "NOT_EVALUATED", "reason": "no accuracy-profile ROI supplied"}
    est = points @ T_est[:3, :3].T + T_est[:3, 3]
    ref = points @ T_ref[:3, :3].T + T_ref[:3, 3]
    errors = np.linalg.norm(est - ref, axis=1)
    worst = int(np.argmax(errors))
    return {
        "status": "EVALUATED",
        "n_points": len(errors),
        "p50": float(np.percentile(errors, 50)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(errors[worst]),
        "worst_point_source_camera": points[worst].tolist(),
        "errors": errors,
    }


def _pair_metrics(source: int, target: int, estimated: Mapping[int, np.ndarray],
                  reference: Mapping[int, np.ndarray], roi_points: np.ndarray | None) -> dict:
    T_est = relative_transform(estimated, source, target)
    T_ref = relative_transform(reference, source, target)
    R_est, t_est = T_est[:3, :3], T_est[:3, 3]
    R_ref, t_ref = T_ref[:3, :3], T_ref[:3, 3]
    center_est = -R_est.T @ t_est
    center_ref = -R_ref.T @ t_ref
    center_error = float(np.linalg.norm(center_est - center_ref))
    translation_error = float(np.linalg.norm(t_est - t_ref))
    translation_reference = float(np.linalg.norm(t_ref))
    baseline_est = float(np.linalg.norm(center_est))
    baseline_ref = float(np.linalg.norm(center_ref))
    baseline_error = abs(baseline_est - baseline_ref)
    task = _task_space(T_est, T_ref, roi_points)
    task_errors = task.pop("errors", None)
    return {
        "pair_id": f"{source}->{target}",
        "source_camera": source,
        "target_camera": target,
        "rotation_error_deg": rotation_error_deg(R_est, R_ref),
        "camera_center_estimate": center_est.tolist(),
        "camera_center_reference": center_ref.tolist(),
        "camera_center_error": center_error,
        "relative_translation_error": translation_error,
        "relative_translation_error_pct": _percent(translation_error, translation_reference),
        "translation_direction_error_deg": _direction_error_deg(t_est, t_ref),
        "baseline_length_estimate": baseline_est,
        "baseline_length_reference": baseline_ref,
        "baseline_length_error": baseline_error,
        "baseline_length_error_pct": _percent(baseline_error, baseline_ref),
        "task_space": task,
        "_task_errors": task_errors,
    }


def _finite_max(pairs: Sequence[dict], key: str) -> float | None:
    values = [pair.get(key) for pair in pairs if pair.get(key) is not None]
    return float(max(values)) if values else None


def _summary(pairs: Sequence[dict], roi_points: np.ndarray | None) -> dict:
    out = {
        "n_pairs": len(pairs),
        "rotation_error_deg_max": _finite_max(pairs, "rotation_error_deg"),
        "camera_center_error_max": _finite_max(pairs, "camera_center_error"),
        "relative_translation_error_max": _finite_max(pairs, "relative_translation_error"),
        "relative_translation_error_pct_max": _finite_max(
            pairs, "relative_translation_error_pct"),
        "translation_direction_error_deg_max": _finite_max(
            pairs, "translation_direction_error_deg"),
        "baseline_length_error_max": _finite_max(pairs, "baseline_length_error"),
        "baseline_length_error_pct_max": _finite_max(pairs, "baseline_length_error_pct"),
    }
    if roi_points is None:
        out["task_space"] = {
            "status": "NOT_EVALUATED", "reason": "no accuracy-profile ROI supplied"}
        return out
    chunks = [pair["_task_errors"] for pair in pairs if pair["_task_errors"] is not None]
    errors = np.concatenate(chunks) if chunks else np.zeros(0)
    worst_pair = max(pairs, key=lambda pair: pair["task_space"]["p95"]) if pairs else None
    out["task_space"] = {
        "status": "EVALUATED",
        "n_pair_points": len(errors),
        "p50": float(np.percentile(errors, 50)) if len(errors) else None,
        "p95": float(np.percentile(errors, 95)) if len(errors) else None,
        "max": float(np.max(errors)) if len(errors) else None,
        "worst_pair": worst_pair["pair_id"] if worst_pair else None,
        "worst_pair_p95": worst_pair["task_space"]["p95"] if worst_pair else None,
    }
    return out


def _lookup(summary: Mapping[str, object], path: Sequence[str]) -> float | None:
    value: object = summary
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return float(value) if value is not None else None


def _limit_checks(summary: Mapping[str, object], profile: AccuracyProfile | None) -> dict:
    checks = {}
    if profile is None:
        return {"status": "NOT_CONFIGURED", "all_pass": None, "checks": checks}
    for name, path in _LIMIT_METRICS.items():
        limit = profile.limits.get(name)
        if limit is None:
            continue
        value = _lookup(summary, path)
        checks[name] = {"value": value, "limit": limit,
                        "passed": value is not None and value <= limit}
    return {"status": "EVALUATED" if checks else "NOT_CONFIGURED",
            "all_pass": all(item["passed"] for item in checks.values()) if checks else None,
            "checks": checks}


def compare_world_to_camera_poses(
        estimated: Mapping[int, np.ndarray], reference: Mapping[int, np.ndarray], *,
        profile: AccuracyProfile | None = None, reference_name: str = "reference") -> dict:
    """Compare all required camera pairs after both pose sets use world-to-camera convention.

    Relative transforms remove the arbitrary rig/world gauge.  With no profile, all common
    unordered camera pairs are evaluated for SE(3), while task-space error is explicitly
    ``NOT_EVALUATED``.  An empty profile ``required_pairs`` also means all common pairs.
    """
    estimated_norm = {int(c): _transform(T, f"estimated camera {c}")
                      for c, T in estimated.items()}
    reference_norm = {int(c): _transform(T, f"reference camera {c}")
                      for c, T in reference.items()}
    common = sorted(set(estimated_norm) & set(reference_norm))
    available = list(itertools.combinations(common, 2))
    requested = list(profile.required_pairs) if profile and profile.required_pairs else available
    missing = [pair for pair in requested if pair[0] not in common or pair[1] not in common]
    used = [pair for pair in requested if pair not in missing]
    roi_points = profile.roi_points if profile is not None else None
    pairs = [_pair_metrics(a, b, estimated_norm, reference_norm, roi_points) for a, b in used]
    summary = _summary(pairs, roi_points)
    for pair in pairs:
        pair.pop("_task_errors", None)
    if missing:
        status, reason = "INCOMPLETE", "one or more required camera pairs are unavailable"
    elif not pairs:
        status, reason = "NOT_EVALUATED", "fewer than two comparable cameras"
    else:
        status, reason = "EVALUATED", None
    return {
        "status": status,
        "reason": reason,
        "reference": reference_name,
        "estimated_pose_convention": "world_to_camera",
        "reference_pose_convention": "world_to_camera",
        "pair_convention": "T_target_source; camera center of target expressed in source",
        "length_unit": profile.length_unit if profile is not None else "scene_unit",
        "profile": profile.to_dict() if profile is not None else None,
        "common_cameras": common,
        "required_pairs": [list(pair) for pair in requested],
        "missing_pairs": [list(pair) for pair in missing],
        "pairs": pairs,
        "summary": summary,
        "limit_checks": _limit_checks(summary, profile),
    }


def compare_camera_to_world_reference(
        estimated_world_to_camera: Mapping[int, np.ndarray],
        reference_camera_to_world: Mapping[int, np.ndarray], *,
        profile: AccuracyProfile | None = None, reference_name: str = "reference") -> dict:
    """Compare DS-MSP extrinsics with a camera-to-world GT/MC-Calib pose mapping."""
    reference = camera_to_world_as_world_to_camera(reference_camera_to_world)
    return compare_world_to_camera_poses(
        estimated_world_to_camera, reference, profile=profile, reference_name=reference_name)


def accuracy_metrics(estimated_world_to_camera: Mapping[int, np.ndarray], *,
                     ground_truth_world_to_camera: Mapping[int, np.ndarray] | None = None,
                     mccalib_camera_to_world: Mapping[int, np.ndarray] | None = None,
                     profile: AccuracyProfile | None = None) -> dict:
    """Build the schema-versioned accuracy section stored in rig reports.

    Blender ``GroundTruth.yml`` ``P_i`` matrices are world-to-camera. MC-Calib's
    ``calibrated_cameras_data.yml`` ``camera_pose_matrix`` is camera-to-world. The two
    deliberately separate keyword names prevent an invisible convention swap.
    """
    def compare(reference: Mapping[int, np.ndarray] | None, name: str,
                *, camera_to_world: bool) -> dict:
        if not reference:
            return {"status": "NOT_EVALUATED", "reason": f"no {name} poses supplied",
                    "reference": name}
        if camera_to_world:
            result = compare_camera_to_world_reference(
                estimated_world_to_camera, reference, profile=profile, reference_name=name)
            result["reference_input_pose_convention"] = "camera_to_world"
            return result
        result = compare_world_to_camera_poses(
            estimated_world_to_camera, reference, profile=profile, reference_name=name)
        result["reference_input_pose_convention"] = "world_to_camera"
        return result

    return {
        "schema": ACCURACY_SCHEMA,
        "schema_version": ACCURACY_SCHEMA_VERSION,
        "profile": profile.to_dict() if profile is not None else None,
        "ground_truth": compare(ground_truth_world_to_camera, "ground_truth",
                                camera_to_world=False),
        "mccalib_reference": compare(mccalib_camera_to_world, "mccalib_reference",
                                     camera_to_world=True),
    }


__all__ = [
    "ACCURACY_SCHEMA", "ACCURACY_SCHEMA_VERSION", "PROFILE_SCHEMA",
    "PROFILE_SCHEMA_VERSION", "AccuracyProfile", "accuracy_metrics",
    "accuracy_profile_from_dict", "camera_to_world_as_world_to_camera",
    "compare_camera_to_world_reference", "compare_world_to_camera_poses",
    "load_accuracy_profile", "relative_transform", "rotation_error_deg",
]
