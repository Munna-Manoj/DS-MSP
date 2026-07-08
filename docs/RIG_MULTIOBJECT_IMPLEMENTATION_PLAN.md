# DS-MSP[rig] — Multi-Object Merge: Implementation Plan

**Goal:** Give DS-MSP[rig] MC-Calib's ability to calibrate rigs whose calibration targets are
**not all mutually co-observed** — the multi-object / non-overlapping-per-board topologies
(capability-matrix rows 4–6 in `RIG_MULTIOBJECT_DIAGNOSIS.md`), with the non-overlapping
two-object rig in `data/` as the acceptance case.

> **Framing.** This is a pure DS-MSP[rig] camera-calibration-topology problem. The dataset's
> `calib_param_gaze.yml` / `in_cabin` names are incidental (just where the data came from);
> "the acceptance rig" below means a rig whose cameras observe *different, never-co-observed*
> boards — nothing about it is gaze- or application-specific.

**Companion:** read `RIG_MULTIOBJECT_DIAGNOSIS.md` first for the root cause.

---

## Status: IMPLEMENTED (branch `feat/rig-multiobject-merge`)

Phases 1–4 and the acceptance tests are done and green. What shipped:

- `reconstruct.py` — `reconstruct_objects` keeps **all** covisibility components (no drop);
  `object_obs_from_board_obs_multi`; `reconstruct_objects_from_images/keypoints`. Singular
  `reconstruct_object` retained as a backward-compatible wrapper.
- `merge.py` (**new**) — `inter_object_transforms`, `merge_objects`, `remap_object_obs`
  (object-level clone of `object3d.build_objects`, same cpp:929 convention).
- `calibrate.py` — the front-end is now **object-aware** (`_obs_points` resolves points per
  `object_id`; the process pool stashes a `{object_id: Object3D}` map; PnP obs-light tuples carry
  `object_id`). New `_merge_and_relink` runs after hand-eye: fuse objects → re-seed poses (gated
  PnP) → re-derive groups by direct covisibility of the fused object → iterate. `calibrate_rig`
  gained an `objects=[obj]` kwarg and builds a multi-object `RigState`.
- `pipeline.py` / `calib_param.py` — thread the object list through; `_reconstruct` returns all
  objects.
- Tests — `test_merge.py`, `test_rig_multiobject.py` (end-to-end extrinsic recovery),
  `_synth.make_non_overlapping_rig`, extended `test_reconstruct.py`. Targeted + regression suites
  pass (pre-existing py3.8 `test_calib_param.py` collection error is unrelated).

**Real-data acceptance (`data/`):** was **1 camera** → now **2 cameras**, boards fused
`{0: [0,1]}`, reprojection RMS 0.84/0.60 px. The recovered rig: baseline **1.19 m**, rotation
**178.6°** (the two cameras face each other) with a fused board-0↔board-1 separation of **10.3 mm**
— i.e. a **double-sided ChArUco target**, each camera viewing one face. This is exactly MC-Calib's
result shape for an opposed-camera rig.

> **Units gotcha.** `detect.charuco.board_object_points` scales corners by `spec.square_size`
> (config `square_size: 55`), so **all rig geometry is in those units** (here 55 = mm, since
> `length_square` is 0.055 m). Baselines/distances therefore come out in **mm, not m** — multiply
> by `length_square / square_size` (= 1e-3 here) for metres. This matches MC-Calib (its objects use
> `square_size` too). A single-camera calibration passes regardless of this scale (monocular is
> scale-free); only the inter-camera baseline exposes the unit.

**Design pivot vs. the original plan:** the "Phase-2-only, reach hand-eye" increment is **not**
sufficient on its own — the extrinsic is unidentifiable from reprojection until the objects are
merged (see §1). `_merge_and_relink` therefore always completes the merge and re-groups; the
Phase-2 state is transient, never a shippable stopping point for a non-overlapping rig.

**Not yet done (Phase 5 + hardening):** multi-group/object **output** in `io/mccalib.py` (still
single-group), `refine_object_structure` covers only the first object, and a spread/covariance
**gate** on the inter-object transform to refuse merging genuinely non-rigid targets.

---

## 0. Design principles (preserved, non-negotiable)

This plan is deliberately *additive*. Every existing single-object path stays byte-for-byte on its
current behaviour; the new capability is a transient stage that **collapses back to the existing
single-fused-object invariant before the BA runs**, so the optimizer, structure refinement, and
MC-Calib I/O are untouched.

1. **Faithful MC-Calib port.** New functions map 1:1 to MC-Calib stages and cite the `.cpp`
   analogue in their docstring, exactly like `object3d.py` / `extrinsics.py` / `handeye.py`.
2. **Reuse the graph/averaging primitives.** `geometry/graph.py`
   (`connected_components`, `covis_weights`, `shortest_path`) and `geometry/averaging.py`
   (`robust_average_transform`) are already applied at board- and camera-level; the object-merge
   applies the *same* skeleton at object-level. No new graph code.
3. **Robust, down-weight-don't-drop.** Inter-object transforms use `robust_average_transform`;
   hand-eye keeps its RANSAC/gate. No new hard rejection.
4. **Single rigid object downstream.** The multi-object state is **transient** (reconstruction →
   link → merge). After merge, `RigState.objects == {0: fused}` and `object_id == 0` everywhere,
   so `bundle.py`, `refine_object_structure`, and `io/mccalib.py` need **no change** in Phases 1–3.
5. **Neutral data layer untouched.** `data/observations.py` already supports multiple objects; we
   populate it, we do not change it.
6. **Config-compatible.** No new required config keys. `he_approach` already drives the linker.
   One optional key (`allow_partial_rig`) is added with a safe default.
7. **Additive, backward-compatible API.** `reconstruct_object` (singular) stays as a thin wrapper
   so existing callers/tests (`tests/rig/test_reconstruct.py`) are unaffected.

---

## 1. Why "merge to one object" is the correct design (the identifiability argument)

A tempting minimal fix — *keep two objects, reach the existing `link_groups`* — produces a linked
extrinsic but is **not** jointly refinable, and here is the proof it must be completed by a merge:

In this non-overlapping rig, camera 0 sees only object B (board 1), camera 1 sees only object A (board 0), and
`ref_cam = cam0 ⇒ T_c0_g = I` (fixed). In the joint BA:

- cam 0 residuals constrain `{T_c0_g=I (fixed), T_g_objB(f)}` — fully explained by objB poses.
- cam 1 residuals constrain `{T_c1_g, T_g_objA(f)}` — **no term couples cam 0 and cam 1.**

So `T_c1_g` (the inter-camera extrinsic) is **gauge-coupled with objA's per-frame poses**: the BA
can shift `T_c1_g` and absorb it into `T_g_objA(f)` with *zero* change in reprojection. The
extrinsic is **unidentifiable from reprojection alone** — only the hand-eye prior pins it.

The fix is to **enforce the physical rigidity** that objA and objB move together, i.e.
`T_g_objA(f) = T_g_objB(f) · T_objB_objA` with `T_objB_objA` *constant*. Baking that constant into
one **fused object** (objA ⊕ objB) makes every frame contribute *one* shared object pose that
*both* cameras reproject against — now the extrinsic is coupled through the residuals and the BA
refines it jointly. This is precisely MC-Calib `merge3DObjects`, and it lets the existing
single-object BA + `refine_object_structure` do the rest **unchanged**.

Hand-eye recovers the extrinsic; the merge recovers the inter-object geometry; together they
reduce topologies 4–6 to the already-solved topology 2.

---

## 2. Architecture of the change

New transient stage inserted between reconstruction and the BA, entirely inside the existing
`calibrate_rig` skeleton:

```
detect ─► reconstruct_objects()          [Phase 1]  ⇒ List[Object3D] + per-object ObjectObs
       ─► front-end intrinsics (unchanged; now sees all cameras)
       ─► init_camera_groups (unchanged; naturally yields ≥2 groups) [Phase 2]
       ─► link_groups / hand-eye (unchanged core; now REACHED)       [Phase 2]
       ─► merge_objects()                 [Phase 3]  ⇐ NEW: fuse objects into ONE rigid Object3D
       ─► (state is now topology-2: 1 object, linked extrinsics)
       ─► bundle.refine ×(warm-up, group, joint, structure)  (unchanged)
       ─► iterate link+merge until 1 component               [Phase 4]
       ─► save (unchanged for 1 object; extended only if partial) [Phase 5]
```

The only files touched for the core capability: **`reconstruct.py`** (return all objects),
**new `merge.py`** (the merge stage), **`calibrate.py`** (wire the stage), **`calib_param.py`**
(call plural reconstruct). Everything else is tests, output polish, and docs.

---

## 3. Phase-by-phase plan

### Phase 1 — Multi-object reconstruction (no drop)

**File:** `ds_msp/rig/reconstruct.py`

Add `reconstruct_objects` (plural) returning **all** covisibility components with per-object
observations; keep `reconstruct_object` (singular) as a wrapper.

```python
def reconstruct_objects(board_obs, specs, img_size, *, init_models=None
                        ) -> Tuple[List[Object3D], List[ObjectObs]]:
    """Resect every board and fuse into ONE Object3D per covisibility component — MC-Calib
    calibrate3DObjects, WITHOUT discarding non-co-observed components. Each returned object
    carries a distinct object_id (0,1,…); every board detection is mapped onto whichever
    object contains its board, so a camera that only sees a 'secondary' board keeps its
    observations (the drop that erased such cameras is gone)."""
    # … resection identical to reconstruct_object …
    objects = build_objects(valid, board_points)          # already returns all components
    objects.sort(key=lambda o: (-len(o.board_ids), o.board_ids[0]))  # obj 0 = largest, stable
    for oid, o in enumerate(objects):
        o.object_id = oid
    board_to_obj = {b: o.object_id for o in objects for b in o.board_ids}
    obs = object_obs_from_board_obs_multi(board_obs, objects, board_to_obj)
    return objects, obs

def reconstruct_object(board_obs, specs, img_size, *, object_id=0, init_models=None):
    """Backward-compatible single-object wrapper: largest component, warns + drops the rest
    (unchanged behaviour for existing callers/tests)."""
    objects, _ = reconstruct_objects(board_obs, specs, img_size, init_models=init_models)
    if len(objects) > 1:
        warnings.warn(…"dropping"…)     # message preserved
    obj = objects[0]; obj.object_id = object_id
    return obj
```

`object_obs_from_board_obs_multi` = the current `object_obs_from_board_obs`, but keyed
per-object: for each `(cam, frame)` accumulate rows **into the object that owns each board**, and
emit one `ObjectObs(object_id=obj.object_id, …)` per (cam, frame, object). This is a small
generalization of the existing `pts_board_2_obj` lookup — a board's corner only maps into *its*
object.

Add `reconstruct_objects_from_images` / `…_from_keypoints` mirroring the existing
`reconstruct_from_images` / `…_from_keypoints`, returning `(objects, obs, img_size)`.

**Preserves:** resection, `build_objects`, `init_models` fisheye path — all unchanged. `object3d.py`
untouched (it already returns all components).

---

### Phase 2 — Reach hand-eye (orchestration)

**File:** `ds_msp/rig/calib_param.py` — `_reconstruct` returns all objects:

```python
def _reconstruct(cfg, *, animator=None):
    …
    objects, obs, img_size = reconstruct_objects_from_images(…)   # was reconstruct_from_images
    return objects, obs, img_size            # caller passes the *list* through
```

`calibrate_from_config` (`calib_param.py:576-582`): when `number_board > 1` and no pre-built
object, take the object **list**; build the `Scenario` with `object=objects[0]` but pass the full
list into `calibrate_scenario`/`calibrate_rig` via a new optional `objects=…` argument (default
`[scn.object]`, so single-object callers are unchanged).

**File:** `ds_msp/rig/calibrate.py` — `calibrate_rig` accepts `objects: List[Object3D]`
(default `[obj]`). Steps 1–2 already work once obs carry all cameras:

- Front-end calibrates **all** cameras (each camera now has obs for the object it sees).
- `init_camera_groups(object_obs, cam_ids)` (`extrinsics.py`) keys pair transforms by
  `(frame, object_id)` (`extrinsics.py:25`), so cameras that never co-observe the *same object*
  land in **separate groups** — correct, no change.
- `link_groups` is now reached (`len(groups) > 1`). Its `ref_poses` already pairs by frame
  ignoring `object_id` (`handeye.py:147`), so it pairs objB-in-cam0 with objA-in-cam1 across the
  32 common frames and returns `T_c1_c0` — **no change needed to the hand-eye core**.

At the end of Phase 2 the extrinsic is linked (hand-eye quality) and both cameras survive. This
alone flips the acceptance rig from "1 camera" to "2 cameras linked"; Phase 3 makes it jointly optimal.

**Preserves:** `extrinsics.py`, `handeye.py`, front-end — all unchanged. `objects` defaulting to
`[obj]` keeps every existing `calibrate_rig` caller identical.

---

### Phase 3 — Object merge (the new stage) ← core deliverable

**New file:** `ds_msp/rig/merge.py` — MC-Calib `merge3DObjects` (McCalib.cpp).

```python
def inter_object_transforms(object_obs, objects, extr):
    """Object-level analogue of extrinsics._camera_pair_transforms (McCalib merge3DObjects).
    With cameras placed in one frame by extr (T_c_g after hand-eye/grouping), lift each
    observed object to the group frame T_g_o = inv(extr[cam]) @ o.T_c_o, and for every frame
    where two DIFFERENT objects are posed, accumulate T_oj_oi = inv(T_g_oj) @ T_g_oi.
    Returns (samples[(oi,oj)] -> [T…], counts) — same shape object3d._pair_transforms uses."""

def merge_objects(objects, object_obs, extr, *, min_covis=3):
    """Fuse objects into ONE Object3D per connected inter-object-covisibility component, using
    the SAME build path as object3d.build_objects but treating each object as a super-board:
    shortest-path compose the averaged T_o_ref transforms, transform each object's pts_3d into
    the merged frame, concatenate, and rebuild pts_board_2_obj (board ids stay globally unique).
    Returns (merged_objects, remap) where remap maps old (object_id, row) -> new (0, row)."""
```

Then a driver in `calibrate.py` (or `merge.py`) inserted **after** `link_groups`, **before** the
per-object pose warm-up (`calibrate.py:516`):

```python
if len(groups) > 1 or len(objects) > 1:
    from .merge import merge_objects, remap_object_obs
    merged = merge_objects(objects, object_obs, extr)          # inter-object geometry
    if len(merged) == 1:
        objects = merged
        object_obs = remap_object_obs(object_obs, remap)       # all obs -> object_id 0
        # groups fold to one: all cameras now see the single fused object
        groups, extr = _refold_single_group(groups, extr, ref_cam)
    else:
        # genuinely unlinkable pieces (e.g. independently-moving targets)
        _warn_or_raise(cfg.allow_partial_rig, merged)
```

**Result:** `objects == [fused]`, `object_id == 0`, one group, extrinsics initialized. The state is
now *identical in shape* to topology 2, so the existing staged BA (`calibrate.py:533-580`) and
`refine_object_structure` run **unchanged** and jointly refine the extrinsic + fused geometry.

**Gauge/rigidity note:** the merged object's `ref_board_id` anchors the gauge exactly as today;
`refine_object_structure` (`bundle.py:565`, already single-object) then frees all corners of *both*
boards against 3 anchor corners — the inter-board pose recovered by the merge is polished by the
same code that polishes any multi-board target.

**Preserves:** reuses `graph.py` + `averaging.py`; mirrors `object3d.build_objects` line-for-line at
object granularity; downstream BA/IO untouched.

---

### Phase 4 — Iterative merge (topology 6, general rigs)

Wrap Phases 2–3 in MC-Calib's alternation until a fixed point:

```python
while True:
    groups, extr = init_camera_groups(object_obs, cam_ids)
    if len(groups) > 1:
        extr = link_groups(groups, extr, object_obs, he_approach)   # bridge groups
    objects, object_obs, changed = merge_objects(objects, object_obs, extr)  # bridge objects
    if not changed:
        break
```

Each pass merges any objects a (now-linked) group co-sees and any groups a merged object now
shares; iterate until one component (or stable). Bounded by `#objects + #groups` iterations.
Handles chained topologies (A–B–C where A,C never co-seen) that a single pass cannot.

**Preserves:** pure orchestration over Phase-2/3 primitives; no new math.

---

### Phase 5 — Output & partial-rig handling

- **Multi-group / partial output** (`io/mccalib.py`): when the rig genuinely cannot be reduced to
  one component (independently-moving targets), keep MC-Calib's per-object /
  per-`camera_group` output. `save_mccalib_object_poses` (`:348`) and
  `save_mccalib_reprojection_error` (`:391`, currently `nb_camera_group=1`) gain a loop over
  object ids / groups. **For the acceptance rig this path is not hit** (it merges to one object), so this
  is only needed for rows-4/6 partial cases.
- **Config:** add optional `allow_partial_rig` (default `false` = raise a clear, actionable error
  naming the unlinkable objects/cameras and why — mirrors the current warning ethos;
  `true` = emit per-component output).

---

### Phase 6 — Tests & validation

**Unit / synthetic** (`tests/rig/`):
- Extend `_synth.make_rig` with `non_overlapping=True`: two rigidly-linked boards at an angle,
  camera 0 sees board 1 only, camera 1 sees board 0 only, shared random rig motion — the synthetic
  non-overlapping rig. Assert `reconstruct_objects` → 2 objects; after link+merge → 1 object and recovered
  `T_c1_c0` within tolerance of GT.
- `test_merge.py` (new): `inter_object_transforms` + `merge_objects` recover a known inter-object
  transform (mirror `test_handeye.py`'s style).
- Topology-4 test: one 2-camera group, two non-co-observed objects, verify covis merge (no
  hand-eye) fuses them.
- Topology-6 test: 3 objects chained across 3 groups, verify iterative merge → 1 object.
- Regression: `test_reconstruct.py`, `test_handeye.py`, `test_pipeline.py`, `test_rig_end2end.py`
  must pass unchanged (guaranteed by the singular wrapper + `objects` default).

**Real-data acceptance** (`data/`):
- `ds-msp-calibrate-rig --config data/calib_param_gaze.yml` (paths retargeted to `data/`) →
  `=== 2 cameras … 2 calibrated ===`, verdict PASS, extrinsic comparable to MC-Calib's
  `Results/calibrated_cameras_data.yml` baseline (if present) or within a plausibility band.
- Add a scripted check under `scripts/validate_rig.py` / `tests/realdata/`.

---

## 4. File-change summary

| File | Change | Risk |
|------|--------|:----:|
| `ds_msp/rig/reconstruct.py` | + `reconstruct_objects`, `object_obs_from_board_obs_multi`, `*_from_images/keypoints`; keep singular wrappers | Low |
| `ds_msp/rig/merge.py` | **new** — `inter_object_transforms`, `merge_objects`, `remap_object_obs` | Med |
| `ds_msp/rig/calibrate.py` | `calibrate_rig(objects=[obj], …)`; insert merge stage after `link_groups`; optional iterative loop | Med |
| `ds_msp/rig/pipeline.py` | thread `objects` list through `calibrate_scenario` | Low |
| `ds_msp/rig/calib_param.py` | `_reconstruct` → plural; pass list; `allow_partial_rig` key | Low |
| `ds_msp/io/mccalib.py` | per-object/group output *only* for partial rigs (Phase 5) | Low |
| `tests/rig/_synth.py`, `test_merge.py`, `test_reconstruct.py`, realdata | new + extended coverage | Low |
| `docs/RIG_CALIBRATION_GUIDE.md`, `mkdocs.yml` | document non-overlapping / multi-object rigs | Low |

**Untouched (by design):** `object3d.py`, `extrinsics.py`, `handeye.py` core, `bundle.py`,
`data/observations.py`, `geometry/graph.py`, `geometry/averaging.py`.

---

## 5. Correctness, risks, mitigations

- **Extrinsic identifiability** — solved by the merge (§1); do **not** ship Phase 2 without Phase 3
  for non-overlapping rigs (extrinsic would sit at hand-eye value, unrefined). Guard: assert the
  fused object couples both cameras before the joint BA.
- **Hand-eye degeneracy** — needs rotational diversity; the gate (`handeye.py:97`, 15°) already
  rejects pure-translation motion. Surface a clear error when the gate never passes ("insufficient
  rotational motion to link groups"), not a silent identity.
- **Non-rigid / independently-moving targets** — inter-object transform has high variance;
  `robust_average_transform` + a covariance/spread check refuses the merge and routes to the
  partial-rig path. Matches MC-Calib refusing to merge unrelatable objects.
- **Board-id uniqueness across objects** — board ids are globally unique in `boards_index`, so
  merged `pts_board_2_obj` keys never collide; assert on merge.
- **Determinism** — object ordering sorted by `(-#boards, first_board_id)`; hand-eye seeded;
  merge averaging deterministic. Preserves DS-MSP's reproducibility contract.

---

## 6. Delivery order (smallest shippable increments)

1. **Phase 1 + 2** → the acceptance rig calibrates both cameras (hand-eye-quality extrinsic).
   *Visible win, low risk, no downstream change.*
2. **Phase 3** → jointly-refined extrinsic + fused geometry (MC-Calib parity on this rig).
   *The core.*
3. **Phase 6 (acceptance subset)** → lock it with synthetic + real-data tests.
4. **Phase 4** → general multi-object/multi-group rigs (topology 6).
5. **Phase 5** → partial-rig output + docs.

Phases 1–3 + acceptance tests close the reported issue; 4–5 complete general MC-Calib topology parity.

---

## 7. Outcome

After Phases 1–3, `ds-msp-calibrate-rig` on the non-overlapping two-object rig produces **2 calibrated cameras**
with a jointly-optimized inter-camera extrinsic — matching MC-Calib — while every existing
single-object rig behaves identically (the multi-object state is transient and collapses to the
current single-fused-object invariant before the BA). DS-MSP[rig] retains its differentiators
(per-camera model diversity, GNC-TLS robustness) and gains MC-Calib's multi-object topology reach.
