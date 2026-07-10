"""Outlier handling by hard-dropping (not reweighting) — FR-RIG-018 / ADR-0013.

Complements ``test_outlier_robustness.py`` (reweighting, not rejection): this covers the
*opt-in* ``reproj_gate_px`` gate that hard-drops a board observation that is a genuine
blunder (e.g. a ChArUco board a different OpenCV build mis-decodes — wrong corner ids, a
plausible-but-wrong pose) rather than ordinary noise. Real-data acceptance (Seltos rig,
2026-07-09): 294.6px -> 2.61px max camera error, 30.6 -> 0.73px BA rms, extrinsic unchanged
(1193mm / 178.6deg); on a clean build the gate drops 0 observations (verified again here in
a portable, deterministic synthetic form — the real bug needs a specific OpenCV build that
can't be pinned in CI).
"""
import numpy as np
import pytest

from ds_msp.rig.calibrate import (
    _observation_reproj_rms,
    _reject_outlier_observations,
    calibrate_rig,
)

from ._synth import make_rig

pytestmark = pytest.mark.req("FR-RIG-018")


def _corrupt_copy(o, shift_px=90.0):
    """A copy of ``o`` with every point shifted by a large, uniform pixel offset — mimics a
    whole mis-decoded board landing at a plausible-but-wrong pose (coherent, not per-point
    noise)."""
    o2 = type(o)(**{**o.__dict__, "pts_2d": o.pts_2d + shift_px})
    return o2


def test_observation_reproj_rms_separates_gross_outlier_from_clean():
    obj, obs, img, gt_ext, _ = make_rig(n_cam=2, n_frame=15, noise_px=0.1, seed=0)
    rig = calibrate_rig(obj, obs, img, fix_intrinsics=False)

    clean = obs[0]
    dirty = _corrupt_copy(obs[1])
    assert _observation_reproj_rms(rig, clean) < 2.0
    assert _observation_reproj_rms(rig, dirty) > 50.0


def test_reject_outlier_observations_is_a_noop_on_clean_data():
    obj, obs, img, gt_ext, _ = make_rig(n_cam=2, n_frame=15, noise_px=0.1, seed=0)
    rig = calibrate_rig(obj, obs, img, fix_intrinsics=False)

    kept, ndrop = _reject_outlier_observations(rig, obs, gate_px=10.0)
    assert ndrop == 0
    assert len(kept) == len(obs)


def test_reject_outlier_observations_drops_only_the_corrupted_board():
    obj, obs, img, gt_ext, _ = make_rig(n_cam=2, n_frame=15, noise_px=0.1, seed=0)
    rig = calibrate_rig(obj, obs, img, fix_intrinsics=False)

    mixed = list(obs)
    mixed[3] = _corrupt_copy(obs[3])
    kept, ndrop = _reject_outlier_observations(rig, mixed, gate_px=10.0)

    assert ndrop == 1
    assert len(kept) == len(mixed) - 1
    kept_ids = {(o.cam_id, o.frame_id) for o in kept}
    assert (mixed[3].cam_id, mixed[3].frame_id) not in kept_ids


def test_calibrate_rig_reproj_gate_px_removes_gross_outlier_and_keeps_extrinsic_correct():
    """A 2-camera rig shares one pose per frame (``object_poses[(obj, frame)]``), so a gross
    corruption on cam 1's view of a frame can drag that frame's *shared* pose enough that
    cam 0's otherwise-clean view of the very same frame also reprojects past the gate -- a
    real cascade through the shared estimate, not a gate bug (verified directly: both
    ``(0, f)`` and ``(1, f)`` land far above 10px for the corrupted frame ``f``, every other
    frame stays under 1px). So this asserts the corrupted key is gone and at most its
    same-frame sibling goes with it, not an exact drop count."""
    obj, obs, img, gt_ext, _ = make_rig(n_cam=2, n_frame=20, noise_px=0.15, seed=1)
    corrupted_obs = list(obs)
    corrupted_obs[5] = _corrupt_copy(obs[5])
    corrupted_key = (corrupted_obs[5].cam_id, corrupted_obs[5].frame_id)
    n_before = len(corrupted_obs)

    rig = calibrate_rig(obj, corrupted_obs, img, fix_intrinsics=False, reproj_gate_px=10.0)

    n_dropped = n_before - len(corrupted_obs)
    assert 1 <= n_dropped <= 2, f"expected 1 (or its same-frame sibling too), got {n_dropped}"
    remaining_keys = {(o.cam_id, o.frame_id) for o in corrupted_obs}
    assert corrupted_key not in remaining_keys

    T_rel_mine = rig.T_c_g[1] @ np.linalg.inv(rig.T_c_g[0])
    T_rel_gt = gt_ext[1] @ np.linalg.inv(gt_ext[0])
    baseline_err_pct = 100 * abs(
        np.linalg.norm(T_rel_mine[:3, 3]) - np.linalg.norm(T_rel_gt[:3, 3])
    ) / np.linalg.norm(T_rel_gt[:3, 3])
    ang_err_deg = np.degrees(np.arccos(np.clip(
        (np.trace(T_rel_mine[:3, :3].T @ T_rel_gt[:3, :3]) - 1) / 2, -1, 1)))
    assert baseline_err_pct < 2.0, f"baseline off by {baseline_err_pct:.2f}%"
    assert ang_err_deg < 1.0, f"rotation off by {ang_err_deg:.2f}deg"


def test_calibrate_rig_reproj_gate_px_default_off_keeps_all_observations():
    obj, obs, img, gt_ext, _ = make_rig(n_cam=2, n_frame=15, noise_px=0.1, seed=0)
    corrupted_obs = list(obs)
    corrupted_obs[2] = _corrupt_copy(obs[2])
    n_before = len(corrupted_obs)

    calibrate_rig(obj, corrupted_obs, img, fix_intrinsics=False)   # reproj_gate_px unset

    assert len(corrupted_obs) == n_before, \
        "off by default -- down-weighting handles the estimate, nothing is dropped"
