"""Tier the rig tests: fast smoke coverage on every PR, heavy validation nightly.

The rig suite spans two very different costs:

* **Fast** (default) — graph / averaging / hand-eye / Jacobian gradient-checks /
  multi-board reconstruction / config parsing + intrinsics policy, plus small-fixture
  end-to-end smoke calibrations. Each is sub-second-to-a-few-seconds; they run on every
  PR (``-m "not slow"``) so a change to the rig pipeline gets immediate coverage.
* **Slow** — full-size statistical validation: multi-model sweeps (5 models × many seeds),
  multi-combo robustness sweeps, and the real MC-Calib **Blender** parity runs. Each is
  tens-to-hundreds of seconds; they run in the **nightly** workflow, not the per-PR gate,
  so the PR CI stays under budget.

Rather than blanket-mark the whole package ``slow`` (which hid ~34 genuinely-fast tests and
left the rig with no PR-time coverage), we mark **only** the heavy tests ``slow`` — listed
explicitly below by node-id substring — and everything else is fast by default. A per-test
timeout on the fast tier (see the CI invocation) then structurally prevents a heavy test from
silently rejoining the PR gate: it would exceed the budget and fail until marked ``slow`` here
or shrunk.
"""
from pathlib import Path

import pytest

_RIG_DIR = Path(__file__).parent

# Node-id substrings for the heavy tests that belong in the nightly slow tier: full-size
# statistical sweeps (many models/seeds/combos) and the real-Blender parity/config runs.
_SLOW = (
    "test_blender_parity",                                   # real Blender parity, 5 scenarios
    "test_model_agnostic.py::test_model_agnostic_within_1pct",   # 5 models × 12 seeds
    "test_param_pose.py::test_kb_original_robust_under_outliers",  # 3 models × 4 seeds
    "test_param_pose.py::test_param_pose_robust",           # multi-combo × 4 seeds
    "test_param_pose.py::test_model_of_choice_clean",       # 6 combos × 4 seeds
    "test_pipeline.py::",                                    # full calibrate_scenario (n_frame=45)
    "test_rig_end2end.py::",                                 # full calibrate_rig (n_frame 40/60)
    "test_robust.py::",                                      # full calibrate_rig + robust sweep
    "test_gnc_tls_ba.py::test_rig_ba_gnc_tls",             # full rig GNC-TLS recovery
    "test_gnc_tls_ba.py::test_calib_bundle_adjust_gnc_tls",  # single-cam GNC-TLS recovery
    "test_outlier_robustness.py::test_calibrate_rig_default_front_end",  # (4,40) × 3 seeds
    "test_calib_param.py::test_calibrate_from_config_raw_images",        # real Blender images
)


def pytest_collection_modifyitems(config, items):
    for item in items:
        try:
            item.path.relative_to(_RIG_DIR)
        except ValueError:
            continue  # not a rig test
        if any(s in item.nodeid for s in _SLOW):
            item.add_marker(pytest.mark.slow)
