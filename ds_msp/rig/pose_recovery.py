"""Recover under-supported frame poses without discarding their observations.

Robust bundle adjustment can suppress correct pixels when a frame starts in the
wrong pose basin. Re-estimate that pose from individual cameras, then judge each
candidate against the entire frame while keeping the calibrated rig fixed.
"""
from __future__ import annotations

import copy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..core.numerics import NumericalError
from ..ops.pose import solve_pnp, solve_pnp_robust
from . import bundle
from .types import ObjectObs, RigState


def _frame_errors(rig: RigState, observations: List[ObjectObs], pose: Optional[np.ndarray]):
    """One entry per observed corner, including invalid projections as infinity."""
    errors, valid = [], []
    for obs in observations:
        if (pose is None or obs.object_id not in rig.objects
                or obs.cam_id not in rig.cameras or obs.cam_id not in rig.T_c_g):
            errors.append(np.full(len(obs.point_rows), np.inf))
            valid.append(np.zeros(len(obs.point_rows), dtype=bool))
            continue
        points = rig.objects[obs.object_id].pts_3d[obs.point_rows]
        transform = rig.T_c_g[obs.cam_id] @ pose
        pixels, mask = rig.cameras[obs.cam_id].project(
            points @ transform[:3, :3].T + transform[:3, 3])
        pixels = np.asarray(pixels, float)
        mask = np.asarray(mask, bool).reshape(-1) & np.isfinite(pixels).all(axis=1)
        err = np.full(len(points), np.inf)
        err[mask] = np.linalg.norm(pixels[mask] - obs.pts_2d[mask], axis=1)
        errors.append(err)
        valid.append(mask)
    return np.concatenate(errors), np.concatenate(valid)


def _summary(errors: np.ndarray, valid: np.ndarray, threshold: float) -> dict:
    return {"n_observed": len(errors), "n_valid": int(valid.sum()),
            "n_supported": int(np.sum(errors <= threshold)),
            "support_fraction": float(np.mean(errors <= threshold)),
            "truncated_cost": float((np.minimum(errors, threshold) ** 2).sum())}


def frame_support(rig: RigState, object_obs: List[ObjectObs], *,
                  threshold_px: float = 3.0, min_support: float = 0.75) -> List[dict]:
    """Audit the returned rig without changing it or selecting new poses.

    Report every nonempty observed object-frame, including already supported ones.
    Missing poses/models and invalid projections count against support. Fractions
    pool corners across cameras; they are training support, not an uncertainty
    estimate or a guarantee that each camera independently supports the pose.
    """
    if not np.isfinite(threshold_px) or threshold_px <= 0:
        raise ValueError("threshold_px must be finite and positive")
    if not np.isfinite(min_support) or not 0 < min_support <= 1:
        raise ValueError("min_support must be in (0, 1]")
    by_frame: Dict[Tuple[int, int], List[ObjectObs]] = defaultdict(list)
    for obs in object_obs:
        if len(obs.point_rows):
            by_frame[(obs.object_id, obs.frame_id)].append(obs)
    records = []
    for key in sorted(by_frame):
        errors, valid = _frame_errors(rig, by_frame[key], rig.object_poses.get(key))
        summary = _summary(errors, valid, threshold_px)
        records.append({"object_id": key[0], "frame_id": key[1],
                        "threshold_px": threshold_px, "min_support": min_support,
                        "supported": summary["support_fraction"] >= min_support,
                        **summary})
    return records


def recover_frame_poses(rig: RigState, object_obs: List[ObjectObs], *,
                        threshold_px: float = 3.0, min_support: float = 0.75,
                        max_iter: int = 60) -> Tuple[RigState, List[dict]]:
    """Return a copied rig with guarded pose replacements and per-frame diagnostics.

    Only missing poses or frames below ``min_support`` within ``threshold_px`` are
    considered. Fresh
    deterministic plain and robust PnP candidates from each observing camera are
    scored over *all* frame observations. The best candidate receives a local
    Cauchy pose polish with camera intrinsics, camera extrinsics and target points
    fixed. Both the raw and polished candidate must compete with the saved pose.

    A replacement must explain at least ``min_support`` of the corners, strictly
    increase supported-corner count, strictly reduce full-frame truncated cost,
    and retain every previously valid projection. No points are removed. Frames
    with fewer than four corners or without a supported candidate stay unchanged
    and are reported as unresolved. There is no silent cap on the number of frames.

    ``threshold_px`` is an explicit pixel support band, not estimated from a bad
    frame's residuals; callers with expected noise sigma should use their existing
    band policy (for example ``3.03 * noise_bound``). The method does not establish
    intrinsic accuracy or validate observations outside the recorded pixels.
    """
    if not np.isfinite(threshold_px) or threshold_px <= 0:
        raise ValueError("threshold_px must be finite and positive")
    if not np.isfinite(min_support) or not 0 < min_support <= 1:
        raise ValueError("min_support must be in (0, 1]")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")

    out = copy.copy(rig)
    out.object_poses = dict(rig.object_poses)
    by_frame: Dict[Tuple[int, int], List[ObjectObs]] = defaultdict(list)
    for obs in object_obs:
        key = (obs.object_id, obs.frame_id)
        if obs.object_id in rig.objects and len(obs.point_rows):
            by_frame[key].append(obs)

    diagnostics = []
    expected_failures = (cv2.error, np.linalg.LinAlgError, NumericalError, ValueError)
    for key in sorted(by_frame):
        observations = sorted(by_frame[key], key=lambda obs: obs.cam_id)
        original = rig.object_poses.get(key)
        old_errors, old_valid = _frame_errors(rig, observations, original)
        before = _summary(old_errors, old_valid, threshold_px)
        if before["support_fraction"] >= min_support:
            continue
        record = {"object_id": key[0], "frame_id": key[1], "threshold_px": threshold_px,
                  "before": before, "after": before.copy(), "accepted": False,
                  "reason": "no_supported_candidate", "candidates": 0,
                  "failed_candidates": 0}
        diagnostics.append(record)
        if len(old_errors) < 4:
            record["reason"] = "too_few_corners"
            continue

        candidates = []
        for obs in observations:
            if (len(obs.point_rows) < 4 or obs.cam_id not in rig.cameras
                    or obs.cam_id not in rig.T_c_g):
                continue
            points = rig.objects[obs.object_id].pts_3d[obs.point_rows]
            for method in ("plain", "robust"):
                try:
                    solution = (solve_pnp(rig.cameras[obs.cam_id], points, obs.pts_2d)
                                if method == "plain" else solve_pnp_robust(
                                    rig.cameras[obs.cam_id], points, obs.pts_2d,
                                    noise_bound_px=threshold_px))
                    if not solution[0]:
                        record["failed_candidates"] += 1
                        continue
                    camera_pose = np.eye(4)
                    camera_pose[:3, :3] = cv2.Rodrigues(np.asarray(solution[1], float))[0]
                    camera_pose[:3, 3] = np.asarray(solution[2], float).reshape(3)
                    pose = np.linalg.solve(rig.T_c_g[obs.cam_id], camera_pose)
                    if not np.isfinite(pose).all():
                        record["failed_candidates"] += 1
                        continue
                    errors, valid = _frame_errors(rig, observations, pose)
                    summary = _summary(errors, valid, threshold_px)
                    if np.all(valid[old_valid]):
                        candidates.append((summary["truncated_cost"], pose, summary,
                                           f"camera_{obs.cam_id}_{method}"))
                except expected_failures:
                    record["failed_candidates"] += 1
        record["candidates"] = len(candidates)
        if not candidates:
            continue
        best = min(candidates, key=lambda item: item[0])
        # A rejected seed must not become a fresh route for hiding invalid corners.
        if best[0] >= before["truncated_cost"]:
            continue
        local = copy.copy(rig)
        local.object_poses = {key: best[1]}
        try:
            usable = [obs for obs in observations if obs.cam_id in rig.cameras
                      and obs.cam_id in rig.T_c_g]
            polished = bundle.refine(
                local, usable, fix_intrinsics=True, fix_extrinsics=True,
                robust_kernel="cauchy", robust_scale=threshold_px / 3.03,
                max_iter=max_iter)
            pose = polished.object_poses[key]
            errors, valid = _frame_errors(rig, observations, pose)
            summary = _summary(errors, valid, threshold_px)
            if np.all(valid[old_valid]):
                candidates.append((summary["truncated_cost"], pose, summary,
                                   best[3] + "_polished"))
        except expected_failures:
            record["failed_candidates"] += 1
        admissible = [candidate for candidate in candidates
                      if candidate[2]["support_fraction"] >= min_support
                      and candidate[2]["n_supported"] > before["n_supported"]
                      and candidate[0] < before["truncated_cost"]]
        if admissible:
            best = min(admissible, key=lambda item: item[0])
            out.object_poses[key] = best[1].copy()
            record.update(after=best[2], accepted=True, reason="recovered", source=best[3])
    return out, diagnostics
