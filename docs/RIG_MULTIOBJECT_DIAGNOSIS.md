# DS-MSP[rig] vs MC-Calib — Non-Co-Observed Board Diagnosis

**Status:** Diagnosis (root-cause + architecture comparison)
**Scope:** `ds_msp/rig/*`, compared against MC-Calib (`rameau-fr/MC-Calib`)
**Trigger:** `ds-msp-calibrate-rig --config calib_param_gaze.yml` calibrates **1 of 2 cameras**; MC-Calib calibrates both via hand-eye.
**Dataset:** `data/` — 2 cameras (`Cam_001`, `Cam_002`), 2 ChArUco boards, 35 frames each.

> **Framing.** This is purely a **camera-calibration topology** problem in DS-MSP[rig]: a
> multi-camera rig whose cameras observe *different, never-co-observed* calibration boards. The
> `calib_param_gaze.yml` / `in_cabin` names are just where this particular dataset came from —
> nothing here is specific to gaze or in-cabin use. Throughout this document "the rig" means this
> **non-overlapping two-object rig**, not an application.

---

## 1. Executive summary

`ds-msp-calibrate-rig` fails on this non-overlapping rig because the two cameras never look at the
same board, and DS-MSP[rig]'s reconstruction stage **collapses all boards into a single rigid
object and discards every board that is not co-observed with the largest one**. Discarding
board 1 deletes the only target camera 0 can see, so camera 0 produces zero observations, only
one camera group forms, and the hand-eye linker — which *is* implemented — is never reached.

The hand-eye capability exists (`ds_msp/rig/handeye.py`) but is wired for a **different
topology**: non-overlapping camera *groups that share one common fused object*. This rig is
the other non-overlapping topology: cameras that each see a **different, never-co-observed
object**. DS-MSP has no path that keeps the second object alive long enough for hand-eye to run.

**Verdict.** DS-MSP[rig] is a faithful port of MC-Calib's *single-rigid-object* pipeline
(topologies that reduce to one calibration object), but it does **not** implement MC-Calib's
*multi-object merge* hierarchy. It is not a strict subset — DS-MSP is *more* capable on
per-camera model diversity and robustness — but it is *less* capable on rig topology.

---

## 2. What the data actually is (empirically verified)

Running DS-MSP's own detector (`detect_board_obs_images`) + robust PnP over `data/`:

| Camera | Directory | Boards seen | Detections |
|-------:|-----------|:-----------:|:----------:|
| **0** | `Cam_001` | **board 1 only** | 32 |
| **1** | `Cam_002` | **board 0 only** | 34 |

- **Images with both boards co-visible: 0.** Boards 0 and 1 are never seen together.
- `build_objects` → **two independent rigid objects**: `object 0 = [board 0]`, `object 1 = [board 1]`.
- **32 frames** have a valid object pose in *both* cameras, with board-0-in-cam-1 rotational
  motion spanning **6.8°–40.9°** (mean 23°).

The last row is the key point: the data is **fully sufficient** for hand-eye. What MC-Calib
exploits (paired object motion across the two cameras) is present in abundance. DS-MSP throws it
away before it can be used.

> Reproduce: `python` over `detect_board_obs_images("data", [0,1], boards)` then
> `build_objects` — the covisibility matrix and object count fall straight out.

---

## 3. Root-cause: the exact failure chain

Every step below is confirmed against the code and the runtime log.

1. **`calib_param.calibrate_from_config`** (`calib_param.py:573-581`) — `number_board == 2`,
   no pre-built object → `_reconstruct`.

2. **`reconstruct.reconstruct_object`** (`reconstruct.py:266-276`) — the fault line:
   ```python
   objects = build_objects(valid, board_points)        # ← correctly builds TWO objects
   obj = max(objects, key=lambda o: len(o.board_ids))  # ← keeps ONE (board 0)
   if len(objects) > 1:
       warnings.warn(… "dropping them" …)               # ← board 1 discarded
   ```
   `build_objects` (`object3d.py:49-88`) is a faithful port of MC-Calib `init3DObjects` and
   returns *all* components. The next line collapses them. **MC-Calib has no such line.**

3. **`reconstruct.object_obs_from_board_obs`** (`reconstruct.py:280-305`) — maps detections onto
   the single surviving object. Camera 0 only ever saw board 1 (now gone) ⇒ **camera 0 yields
   zero `ObjectObs`.** Camera 0 has been erased from the problem.

4. **`calibrate_from_config`** (`calib_param.py:584`) —
   `cam_ids = sorted({o.cam_id for o in object_obs})` ⇒ `{1}`.
   → log: `[front-end] calibrated 1 cameras`.

5. **`calibrate.init_camera_groups`** (`calibrate.py:506`) — one camera ⇒ **one group `[[1]]`**.
   → log: `[groups] 1 group(s): [[1]]`.

6. **`calibrate_rig`** (`calibrate.py:509`) —
   ```python
   if len(groups) > 1:
       from .handeye import link_groups
       extr = link_groups(groups, extr, object_obs, he_approach=he_approach)
   ```
   `len(groups) > 1` is **False** ⇒ **hand-eye never runs.** Result: one camera, no inter-camera
   extrinsic — exactly the observed output.

---

## 4. Why the hand-eye code cannot save this case

`handeye.link_groups` (`handeye.py:123-165`) links non-overlapping camera groups, but its driver
pairs poses of **one shared object**: `ref_poses` (`handeye.py:143-150`) collects `o.T_c_o` for a
group's reference camera. The unit test `test_handeye.py:66` makes the assumption explicit — **all
four cameras carry `object_id=0`**. The Tsai core (`_tsai_solve`, `handeye_bootstrap`) is
topology-agnostic; the *driver* is not, and — more decisively — it is **gated behind
`len(groups) > 1`**, which the drop in §3 guarantees is false.

So the hand-eye branch is reachable **only** when 2+ cameras observe the *same retained object*
but split into separate covisibility groups. In this rig the split is at the **object** level,
not the group level, so the branch is dead on arrival.

---

## 5. Architecture: MC-Calib's hierarchy vs DS-MSP's

MC-Calib is a **two-graph, iteratively-merged** pipeline. DS-MSP reproduces the left half and stops.

```
MC-Calib runCalibrationWorkflow                 DS-MSP calibrate_from_config → calibrate_rig
─────────────────────────────────────────────  ─────────────────────────────────────────────
detectBoards                                    detect_board_obs_images
initializeCalibrationAllCam                     make_bundle_front_end (per-camera intrinsics)
── BOARD graph ─────────────────────────        ── BOARD graph ─────────────────────────
computeBoardsPairPose→…→init3DObjects           build_objects
   ⇒ N Object3D  [KEEPS ALL]                        ⇒ N objects, then max(…)  [DROPS ALL BUT 1]  ✗
── CAMERA-GROUP graph ──────────────────        ── CAMERA-GROUP graph ──────────────────
computeCamerasPairPose→…→initCameraGroup        init_camera_groups
   ⇒ M CameraGroup [KEEPS ALL]                      ⇒ M groups (over the 1 surviving object)
── MERGE / BRIDGE loop ─────────────────        ── (absent) ────────────────────────────
merge3DObjects   (fuse objects a group co-sees)  —                                          ✗
mergeCameraGroup (fuse groups sharing an object) link_groups (only groups sharing object 0) ~
handeyeBootstraptTranslationCalibration          handeye_bootstrap (present, but unreached) ~
  (bridges disconnected OBJECTS *and* GROUPS)
  … iterate until one component …                (single linear pass, no iteration)         ✗
refineAllCameraGroupAndObjectsAndIntrinsic      bundle.refine ×(warm-up, group, joint, struct) ✓
```

Legend: ✓ faithful · ~ present but narrower/unreached · ✗ missing.

**Crucial nuance — the back-end is already multi-object-ready.** `RigState.object_poses` is keyed
`(object_id, frame_id)` (`observations.py:131`), `RigState.objects` is a `Dict[int, Object3D]`, and
the BA loops (`bundle.py:116, 133, 301`) already iterate arbitrary `object_id`. **Nothing in the
front-end/orchestration ever creates `object_id != 0`.** The gap is confined to `reconstruct.py`
and the `calibrate_rig` orchestration — not the optimizer, not the data model.

---

## 6. Capability matrix — topology by topology

Let **B-covis** = are all boards co-observed (→ one fused object)? **C-covis** = do all cameras
co-observe a common object (→ one group)?

| # | Topology | MC-Calib mechanism | DS-MSP[rig] | Status |
|---|----------|--------------------|-------------|:------:|
| 1 | 1 board, overlapping cameras | 1 object, 1 group | single-board object, 1 group | ✅ |
| 2 | Multi-board **all co-observed** (rigid target), overlapping cameras | fuse → 1 object, 1 group | `reconstruct_object` fuses; `refine_object_structure` polishes | ✅ |
| 3 | Multi-board all co-observed, **non-overlapping cameras** | 1 object, N groups → hand-eye | `link_groups` bridges groups over object 0 | ✅ |
| 4 | Cameras overlap, boards **not all co-observed** → one group sees ≥2 objects | `merge3DObjects` | drops all but largest object | ❌ |
| 5 | **2 cameras, each sees a different, never-co-observed board** (this dataset) | 2 objects, 2 groups → object/group hand-eye | drop board → drop camera → 1 group → no link | ❌ |
| 6 | Chained/mixed multi-object multi-group needing several merge rounds | iterative merge + hand-eye | single linear pass | ❌ |

**The line is sharp.** DS-MSP handles everything that reduces to a **single rigid calibration
object**, however the cameras are grouped (1–3). It fails the instant the scene contains **more
than one non-co-observed object/board** (4–6) — the exact class MC-Calib's merge machinery exists
to solve.

---

## 7. Verdict — is DS-MSP[rig] limited for complex rigs?

**One structural limitation, broad in reach.** DS-MSP collapses MC-Calib's two-graph iterative
merge into a single-rigid-object linear pass, so it cannot calibrate rigs whose targets are not
all mutually co-observable (topologies 4–6). Within the single-object world it is a correct,
convention-faithful port (topologies 1–3, MC-Calib `.cpp` line numbers cited throughout).

**But it is not a subset.** DS-MSP[rig] exceeds MC-Calib on two axes MC-Calib lacks:

- **Heterogeneous per-camera models** — RadTan / UCM / EUCM / Double-Sphere / Kannala-Brandt /
  OCam, mixed in one rig (MC-Calib: Brown + Kannala only).
- **High-breakdown robustness** — GNC-TLS BA past 50 % outliers, RANSAC-DLT robust seeding,
  model-aware fisheye resection, analytic-Jacobian Schur BA.

The honest characterization is a **topology ↔ model tradeoff**: MC-Calib is more general on rig
*topology*; DS-MSP is more general on intrinsic *model diversity and robustness*. Closing the
topology gap does not require abandoning either strength — see the companion implementation plan,
`RIG_MULTIOBJECT_IMPLEMENTATION_PLAN.md`.

---

## Appendix A — file/line index

| Concern | Location |
|---------|----------|
| Board→object fusion (faithful) | `ds_msp/rig/object3d.py:49` `build_objects` |
| **Object drop (fault line)** | `ds_msp/rig/reconstruct.py:269-275` |
| Obs→object mapping (single object) | `ds_msp/rig/reconstruct.py:280` |
| cam_ids from surviving obs | `ds_msp/rig/calib_param.py:584` |
| Camera grouping | `ds_msp/rig/extrinsics.py:44` / `calibrate.py:506` |
| Hand-eye gate (dead branch) | `ds_msp/rig/calibrate.py:509` |
| Hand-eye driver (single-object) | `ds_msp/rig/handeye.py:123`, test `tests/rig/test_handeye.py:66` |
| Tsai core (topology-agnostic) | `ds_msp/rig/handeye.py:26,60` |
| Multi-object-ready back-end | `ds_msp/data/observations.py:131`, `ds_msp/rig/bundle.py:116,301` |
| Single-object structure refine | `ds_msp/rig/bundle.py:565` |
| Single-group/object output | `ds_msp/io/mccalib.py:348,391,634` |
