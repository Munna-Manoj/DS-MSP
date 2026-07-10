# ADR-0011 — Multi-object board fusion + merge for non-overlapping rigs

- **Status:** Accepted (recorded 2026-07-08)
- **Deciders:** maintainer
- **Relates to:** ARC-RIG, FR-RIG-001, FR-RIG-002, FR-RIG-017
- **Supersedes:** —

## Context

MC-Calib (Rameau et al., CVIU 2022) calibrates an arbitrary camera mesh via two independent
covisibility graphs, each kept as *every* connected component (not just the largest), then
iteratively merged: `computeBoardsPairPose → init3DObjects` builds one `Object3D` per
board-covisibility component; `computeCamerasPairPose → initCameraGroup` builds one
`CameraGroup` per camera-covisibility component; `merge3DObjects` fuses objects a camera group
jointly observes across frames (even without simultaneous co-observation) via inter-object
rigid-motion consistency; `mergeCameraGroup` and `handeyeBootstraptTranslationCalibration`
(AX=XB hand-eye) bridge groups/objects with no direct shared observation at all; the loop
iterates until one connected component remains, then a staged Ceres BA refines everything.

`ds_msp/rig/reconstruct.py::reconstruct_object` (singular) ported the board-graph stage
faithfully (`build_objects`, `object3d.py:49-88`, mirrors `init3DObjects`) but then discarded
every component except the largest:

```python
objects = build_objects(valid, board_points)         # correctly builds ALL components
obj = max(objects, key=lambda o: len(o.board_ids))    # keeps ONE, drops the rest
```

`ds_msp/rig/handeye.py::link_groups` (the `mergeCameraGroup`/hand-eye equivalent) is correctly
implemented and correctly gated on `len(groups) > 1` — but that gate is never reached for a rig
whose cameras see **entirely disjoint boards** (MC-Calib topology 5/6), because the dropped
board erases 100% of one camera's observations *before* camera-grouping ever runs: 1 camera's
`ObjectObs` rows vanish, `init_camera_groups` sees only 1 camera, `len(groups) > 1` is false,
`link_groups` never fires. The rig silently calibrates as a smaller, single-camera "rig" and
reports a deceptively clean RMS.

**Confirmed on real data**, not assumed: `seltos_cameras_rig/seltos_cams/` (2 RealSense cameras,
2 ChArUco boards, `distortion_model: 0`, `fix_intrinsic: 1`, 35 frames/camera) — each camera
sees a different board, never co-observed in any frame (a double-sided ChArUco target, cameras
facing each other). Pre-fix: `[groups] 1 group(s): [[1]]` (camera 0 dropped entirely). This is
exactly MC-Calib's topology 5 ("2 cameras, each sees a different, never-co-observed board").

Merely reaching `link_groups` without fusing the objects would not be sufficient even if the
gate were reached: in a non-overlapping rig the inter-camera extrinsic `T_c1_g` is gauge-coupled
with the un-merged object's per-frame poses `T_g_objA(f)` — the joint BA can shift the extrinsic
and absorb the change entirely into the object's pose sequence with zero reprojection change
(unidentifiable, Hartley–Zisserman MVG gauge-freedom argument). Only fusing the two objects into
one rigid `Object3D` (baking in the constant relative transform) couples both cameras' residuals
through one shared object pose per frame, making the extrinsic jointly identifiable — MC-Calib's
`merge3DObjects`.

## Decision

1. **`reconstruct.py` keeps every board-covisibility component** (`reconstruct_objects`,
   plural) instead of dropping all but the largest. `build_objects`/`object3d.py` are unchanged
   — this was already a faithful, correct port; only the caller's post-hoc drop was the bug.
2. **New `ds_msp/rig/merge.py`** ports `merge3DObjects`: given per-camera extrinsics (already
   recovered via covisibility grouping + hand-eye), compute each pair of rigidly-linked
   objects' inter-object transform from paired per-frame poses, fuse into one `Object3D` with a
   unified point cloud, and relabel every `ObjectObs.object_id`/`point_rows` onto the fused
   object (re-seeding poses via gated PnP against the new geometry).
3. **`calibrate.py::_merge_and_relink`** iterates {camera-group init → hand-eye link → object
   merge} until one round changes nothing (a chain of pairwise-linked objects still converges
   to one object), then re-derives groups — reducing the non-overlapping-rig case to the
   ordinary single-object case, so every later stage (staged BA, structure refine, MC-Calib
   output) is unchanged. This is a **transient state that always collapses back to the existing
   single/multi-object invariant** `bundle.py` already supports (`RigState.objects: Dict[int,
   Object3D]`, `RigState.object_poses` keyed `(object_id, frame_id)` — verified this needed zero
   changes; only the orchestration layer above it was broken).
4. **This ADR's own review found and fixed one further bug**, not present in the branch as
   originally authored: `WebLive3DAnimator.bind_scene` (`web3d.py`) is called once, before
   `_merge_and_relink` runs, caching the pre-merge object's (smaller) `pts_3d` array. After a
   merge, `object_obs[i].point_rows` are relabeled onto the larger fused object, so every
   `on_iter` callback during the BA stages indexed past the end of the stale cached array
   (`IndexError`) — crashing the *default* config path (`webviewer: true`). Fixed by re-binding
   `on_iter` (duck-typed check for `bind_scene`) to the fused object immediately after the merge
   completes, in `calibrate.py`, right before the BA stages begin. Regression test
   `test_merge_rebinds_live_view_scene_to_fused_object` (`tests/rig/test_rig_multiobject.py`)
   verified to fail without the fix and pass with it.

## Verification (real numbers, not assumed)

- **Real data — `seltos_cameras_rig/seltos_cams/`** (2-camera, 2-board, non-overlapping,
  `fix_intrinsic=true`): `[front-end] calibrated 2 cameras` → `[merge] fused to 1 object(s):
  [[0, 1]]` → `[groups] 1 group(s): [[0, 1]]`. Per-camera RMS **0.797px (cam 0) / 0.572px (cam
  1)**. Recovered extrinsic **baseline 1.192 m, rotation 178.6°** (cameras facing each other
  across a double-sided target) — reproduces this feature's own prior documented acceptance run
  (internal planning notes: 0.73px RMS, 1.19 m / 178.6°) to 3 significant
  figures, independently re-run. Confirmed with the live web view both disabled and enabled
  (default config) — no crash either way after the `web3d.py` fix.
- **Full suite**: 576/576 passed (12 skipped for absent optional real datasets in this
  worktree), `pytest -m jac` 9/9, `pytest tests/contract` 136/136.
- **Real-data release gate** (`tests/realdata/`, pointed at this repo's MC-Calib + Blender
  datasets): 9/9 passed — no regression on the existing single-object rigs
  (`test_rig_blender.py::test_rig_extrinsics_within_1pct_of_gt`,
  `test_mccalib_calibration.py`).
- **Governance gates**: `ruff check .`, `lint-imports` (6/6 contracts kept), `mypy ds_msp/core
  --follow-imports=silent --ignore-missing-imports` (clean), `check_traceability.py --check`,
  `check_tree_hygiene.py` all pass.

## Consequences

**Positive**
- Closes a confirmed, real-data-reproduced silent-failure bug: a rig whose cameras share no
  direct board co-observation previously calibrated as a *smaller, wrong* rig with a clean
  (misleadingly low) RMS, with no error or warning distinguishing it from a correct 1-camera
  run.
- DS-MSP now mirrors MC-Calib's full topology-1-through-6 handling (previously only 1-3), per
  the capability matrix in `docs/RIG_MULTIOBJECT_DIAGNOSIS.md`.
- No changes to the optimizer/BA core, camera models, or the existing single-object code path —
  `RigState`/`bundle.py` already supported multi-object; only the reconstruction/orchestration
  layer needed the fix.

**Negative / costs**
- `_merge_and_relink` re-seeds poses via gated PnP after every merge round — an added
  per-round cost proportional to the number of frames, only paid when `len(objects) > 1` at
  reconstruction time (zero cost for the common single-object rig).
- The merge assumes the fused objects are genuinely one rigid body (a real physical constraint
  of the target, e.g. two ChArUco faces glued to one rigid mount) — if two "objects" the
  covisibility graph considers linkable are not actually rigidly attached in reality, the fuse
  would silently produce a wrong rigid geometry. Not a new risk this ADR introduces (MC-Calib's
  `merge3DObjects` carries the identical assumption) and not exercisable by the existing
  covisibility-linkage precondition (objects only merge when hand-eye-consistent motion links
  them), but not independently guarded against a mislabeled/non-rigid target either.

## Scope explicitly deferred (not accidental omissions)

- **Multi-group/multi-object MC-Calib output** (`io/mccalib.py` writing more than one fused
  object/group) and **`refine_object_structure` for objects beyond the first** — deferred scope
  from this feature's internal planning notes, not required for the merge to
  correctly collapse to, and refine, a single fused object (today's only steady state after
  `_merge_and_relink`).
- **A rigidity/spread gate on inter-object transforms** (detecting a mislabeled non-rigid
  "object" before fusing it) — flagged as a real gap above, not built here; would need its own
  design (e.g. a residual-spread or reprojection-consistency threshold on the recovered
  inter-object transform before accepting the fuse).

## Alternatives considered

- *Stop at `link_groups` (hand-eye-linked groups, no object merge)* — rejected: as argued above,
  this leaves the inter-camera extrinsic gauge-coupled with the un-merged object's pose sequence
  and unidentifiable by the joint BA; measured, not assumed, via the gauge-freedom argument this
  ADR's Context section restates from Hartley–Zisserman MVG.
- *Silently keep dropping minority objects but emit a loud warning* — rejected: still calibrates
  the wrong (smaller) rig; a louder warning does not recover the dropped camera's extrinsic,
  which is the actual capability gap.
