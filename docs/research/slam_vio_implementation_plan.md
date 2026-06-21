# DS-MSP SLAM / VIO implementation plan (book-grounded)

> The authoritative, end-to-end build plan for DS-MSP's SLAM tiers (monocular SLAM →
> VIO). Every component is anchored to a primary reference so we know the *process,
> algorithms, and implementation details* and can always go back to the official source.
>
> **Primary references** (local Claude skills, from the two standard Gao texts):
> - **`v-slam`** — Gao & Zhang, *Introduction to Visual SLAM: From Theory to Practice* (2e).
>   Cited as **VS ch.N**.
> - **`lio-slam`** — Gao, *SLAM in Robotics and Autonomous Driving*. Cited as **AD ch.N**.
>
> These are theory/framework references. **Every DS-MSP implementation is still validated
> against real-data ground truth** (TUM-VI / EuRoC), per the project rule — the book gives
> the method, the dataset gives the verdict.

---

## 0. Design principles (the contract this plan honors)

1. **Ray-native is the spine.** All geometry runs on **bearing vectors** (`project` /
   `unproject`), never undistort-to-pinhole. The calibrated epipolar constraint
   `f₂ᵀ E f₁ = 0`, PnP, triangulation, and the reprojection residual all have ray-native
   forms (VS ch.6 gives the pinhole forms; we carry the ray generalization).
2. **Ship the pinhole baseline too.** For *contrast/teaching* we implement the conventional
   undistort→pinhole path beside the ray-native one, run them **head-to-head on the same
   sequence**, and measure where pinhole breaks (periphery, >180°). This is both pedagogy
   and a quantitative ablation that *proves* the differentiator.
3. **Proper SLAM, not toy odometry.** Frontend ↔ Map ↔ Backend, keyframes, local BA, loop
   closure, relocalization (VS ch.12 architecture).
4. **Validate vs GT; target the OKVIS/BASALT band.** Full-sequence, SE(3), metric ATE on
   TUM-VI room. ORB-SLAM3 ≈ 0.009 m (top), OKVIS ≈ 0.063, BASALT ≈ 0.082 — **target 3–13 cm**;
   beating ORB-SLAM3 is an explicit non-goal.
5. **Each unit lands with a `docs/learn/` chapter** in the form *textbook pinhole way → why
   it breaks on fisheye → ray-native way → measure the gap*, referencing the book chapter.
6. **Reuse what we already own** (see §2); innovate where defensible (see §6).

---

## 1. Reference architecture (VS ch.12 — the `myslam` template)

The de-facto modern architecture (ORB-SLAM lineage), three threads sharing one locked Map:

```
Frontend (real-time)  ──new keyframes──►  Map (shared, mutex)  ──updates──►  Backend (local BA)
          ▲                                                                        │
          └──────────────── optimized poses + landmarks ◄──────────────────────────┘
                              Loop-closing (rare, blocking): detect → verify → pose-graph
```

**Core entities** (VS ch.12): `Frame` (id, pose `T_cw`, features), `Feature` (2D obs + *weak*
refs to host frame & landmark — avoids `shared_ptr` cycles), `MapPoint` (3D pos + observation
list), `Map` (all + active sets, mutex-guarded). **Frontend state machine**:
`INITING → TRACKING_GOOD ↔ TRACKING_BAD → LOST → reset`.

**DS-MSP target layout** (new):
```
ds_msp/slam/
  types.py        # Frame, Feature, MapPoint, Keyframe (dataclasses; numpy-backed)
  map.py          # Map: thread-safe store, active/sliding-window sets
  frontend.py     # tracking, keyframe decision; pluggable front-end (pinhole | ray)
  backend.py      # local sliding-window BA over the active set
  loop.py         # BoW detection + geometric verify + pose-graph (Phase B)
```
**Projection-agnostic** (identical for both methods): the architecture, Map, keyframe logic,
threading. **Projection-specific**: only the *observation residual* and the front-end's
pixel↔ray step (see §2). Python note: we implement the threads, but accuracy/correctness is
the target first; a C++ core for real-time is a later roadmap item.

---

## 2. The ray-native vs pinhole seam — what changes, what doesn't

| Component | Pinhole (book default) | Ray-native (DS-MSP) | DS-MSP status | Ref |
|---|---|---|---|---|
| Pixel ↔ geometry | `K⁻¹[u,v,1]` + undistort/rectify | `unproject → unit bearing` | **have** (all models) | VS ch.4 |
| Epipolar init | 8-pt on normalized pixels | 8-pt on bearings + cheirality | **have** `mvg/two_view` | VS ch.6 |
| Triangulation | DLT with `K·T` | midpoint/DLT on rays | **have** `triangulate_rays` | VS ch.6 |
| PnP (3D-2D) | EPnP/P3P on normalized plane | **ray PnP** (min angular err) | **build** | VS ch.6 |
| Reproj. residual | pixel `2×6` Jacobian | **angular / tangent-plane** | **have** `mvg/bundle` | VS ch.6, C5 |
| Robust kernel | Huber δ=√5.991 (px) | Huber in **radians** + GNC | **have** `core/robust` | VS ch.8 |
| Pose update | `T ← exp(δξ) T` (SE3) | same | **have** `core/lie` | VS ch.3 |
| BA / Schur | block-arrow + Schur | same (residual differs only) | **build** (Schur in optimizer) | VS ch.8 |
| Sliding window | cap active set, marginalize | same | **build** | VS ch.9 |
| Pose graph | `e=Log(T_ij⁻¹Tᵢ⁻¹Tⱼ)` | same | **build** | VS ch.9 |
| Loop closure | ORB BoW (DBoW) | + **spherical/ray BoW** | **build** | VS ch.10 |
| Sim(3) (mono scale) | for monocular loop | same | **build** | VS ch.3/13 |
| Distortion handling | **rectify at ingestion (crops wide FOV)** | none — rays carry full FOV | n/a | VS ch.4 |

**Punchline:** the *back-end machinery* (Lie, BA, Schur, sliding window, pose graph, BoW DB) is
**projection-agnostic** — write it once, both front-ends feed it. Only the **observation residual
+ Jacobian** and the **front-end lift** differ. The pinhole path additionally pays the
"rectify-at-ingestion" tax (VS ch.4) that **crops away the wide FOV** — exactly the failure the
ray-native path avoids and we measure.

**Already owned (reuse, don't rebuild):** `core/lie.py` (left-perturbation SE3/SO3),
`core/optimize.py` (manifold LM — extend with Schur), `mvg/{two_view,ransac,bundle}` (rays,
RANSAC, angular BA), `ops/reproject.py` (charts + undistort for the pinhole baseline),
`core/robust.py` (Huber/GNC), `io/` (Kalibr, COLMAP).

---

## 3. Tier 2 — Monocular SLAM (grow VO → proper SLAM)

### Phase A — shared back-end + two front-ends (head-to-head)
- **Frontend** (VS ch.6/7/12): feature (ORB via OpenCV) or KLT track; constant-velocity prior
  `T_init = ΔT · T_last` (VS pattern); **PnP** (motion-only) per frame; keyframe when inliers/
  parallax drop; triangulate new landmarks at keyframes.
  - *pinhole front-end*: undistort→pinhole→standard pixel pipeline.
  - *ray-native front-end*: `unproject`→bearings (reuse `mvg`).
- **Back-end** (VS ch.8/12): local **sliding-window BA** over ~7 active keyframes.
  - Vertices: `VertexPose` (SE3, our Lie) + `VertexPoint` (XYZ).
  - Edges: projection (pixel **or** angular), **Huber** δ=√5.991.
  - **Schur**: marginalize landmarks first (block-arrow `H`); add Schur-complement reduction to
    `core/optimize.py` (VS ch.8 pattern). Sparsity of the reduced system = covisibility graph.
- **Map** (VS ch.12): `types.py` + `map.py` with active set.
- **Build new:** `ds_msp/slam/{types,map,frontend,backend}.py`; generic **ray PnP**; Schur in the
  optimizer.

### Phase B — loop closure + relocalization
- **BoW** (VS ch.10): DBoW/ORB pinhole baseline **vs** a **spherical/ray BoW** (innovation, §6);
  precision-over-recall; **geometric verify (PnP RANSAC)** before accepting an edge.
- **Sim(3)** loop (VS ch.3/13): monocular loop closure corrects scale drift.
- **Pose-graph optimization** (VS ch.9): `e_ij = Log(T_ij⁻¹ Tᵢ⁻¹ Tⱼ)`, anchor one vertex
  (gauge), solve with our manifold LM.

### Validation (gate)
Full-sequence on TUM-VI room1 (mono → **Sim(3)-aligned** ATE since scale is gauge-free) + EuRoC
V1_01. **Head-to-head table**: ATE, FOV/rays actually used, mean feature track length, failure
modes (where pinhole undistort crops / blows up at the rim / can't represent >180°). This is the
first contrast chapter.

---

## 4. Tier 3 — VIO (the inertial arc)

We choose the **pre-integration + graph** recipe (VINS-Mono / LIO-SAM family; AD ch.4, ch.8)
over the IEKF filter recipe, because it fits our existing **graph + manifold-LM** stack and makes
loop closure natural (AD ch.4 table). The ESKF (AD ch.3) is implemented too, as the faster
filter contrast / teaching path.

### 3a — Camera–IMU calibration  `ds_msp/calib/cam_imu.py`
Estimate `T_cam_imu` + time offset `t_d`. **Static initialization** (AD ch.3 §3.5.5): average
gyro → `b_g`, `−Rᵀg` from accel → `b_a` + gravity direction. Reuse the Schur BA + SE3 Lie.
*Sensor-agnostic* (the IMU side is identical for camera or LiDAR).
- **Gate:** recovered `T_cam_imu` vs the **published Kalibr camchain** (`dso/camchain.yaml`) to
  < ~0.5° / few-mm; `t_d` within a frame period.

### 3b — IMU pre-integration  `ds_msp/inertial/preintegration.py`
On-manifold pre-integration (AD ch.4, **eq. 4.7**), the graph-optimization counterpart of the
ESKF predict. *Entirely sensor-agnostic.*
- Accumulate `(ΔR_ij, Δv_ij, Δp_ij)` (Alg. 4.1), propagate **covariance** `Σ_ij` (the factor's
  information matrix), accumulate the **five bias Jacobians** for first-order bias correction
  (no re-integration on bias shift).
- **15-D residual** (eq. 4.8): `r_ΔR, r_Δv, r_Δp, r_bg, r_ba`; gravity terms `gΔt`, `½gΔt²` must
  appear. Windows ≤ ~1 s for MEMS.
- **Gate:** preintegrated `ΔR/Δv/Δp` vs brute-force numerical integration of a synthetic IMU
  stream to < 1e-6; **gradient-check** the bias Jacobians; covariance PSD.

### 3c — Tightly-coupled VIO  `ds_msp/vio/`
Sliding-window factor graph fusing visual + inertial, solved on the manifold by `core/optimize.py`.
- **Vertices** (AD ch.4): `Pose` (SE3, 6), `Velocity` (ℝ³), `Bias` (ℝ⁶) per keyframe.
- **Edges:** pre-integration (15-D, 3b) **+** visual **angular** reprojection (ours, ray-native) —
  *and* a pinhole-VIO baseline edge for the contrast.
- **Sliding window + marginalization** (VS ch.9; AD ch.8 SMW view): cap the window, marginalize
  old keyframes with care (fill-in). Bootstraps from Tier-2 VO + static init (gravity/bias/scale).
- **Metric scale** falls out of the IMU — no Sim(3) needed; align with SE(3) only.
- **Gate (the money number):** **full-sequence SE(3) metric ATE on TUM-VI room → OKVIS/BASALT band
  (3–13 cm)**; show VIO beats Tier-2 VO and recovers absolute scale.

> ESKF alternative (AD ch.3, Alg. 3.1): 18-D error state `(p,v,R,b_g,b_a,g)`, predict/update/inject.
> Implement as a teaching contrast and a fast filter path; the graph recipe is the primary.

---

## 5. Module / build inventory

**Reuse (already shipped):** `core/lie.py`, `core/optimize.py` (+Schur), `core/robust.py`,
`mvg/{two_view,ransac,bundle}`, `ops/reproject.py` (charts + undistort), `io/{kalibr,colmap}`,
`models/*`.

**Build new:**
- `ds_msp/slam/{types,map,frontend,backend,loop}.py` — the SLAM system (both front-ends).
- generic **ray PnP** (min angular reprojection; seed + manifold refine).
- **Schur complement** in `core/optimize.py`.
- **BoW** (pinhole DBoW baseline + spherical/ray BoW); **pose-graph**; **Sim(3)**.
- `ds_msp/inertial/{preintegration.py, eskf.py}`.
- `ds_msp/vio/` (tightly-coupled sliding-window estimator).

**Import-linter note:** `slam`, `vio`, `inertial` are *composition* layers above `mvg`/`core`
(not the mutually-independent service-layer set); `core` stays dependency-free.

---

## 6. Where we innovate (defensible + teachable)

- **Fully-angular back-end** — landmarks as **bearing + inverse-depth on the sphere**;
  FOV-invariant **radian** thresholds; keyframe selection by **angular parallax**. (Pinhole
  systems can't do this cleanly.)
- **>180° FOV exploitation** — retain features **behind the 90° plane** (`z ≤ 0`) that pinhole
  methods discard → wider covisibility, longer tracks, near-omni loop closure. Enabled by the
  Double-Sphere validity cone. **The headline differentiator** — measured vs the pinhole baseline.
- **Spherical data structures** — angular covisibility graph; HEALPix/icosahedral feature
  bucketing; **spherical BoW** for ray-based loop closure; tangent-image (C3) patches for
  descriptors on curved regions.
- **Owned optimizer** — teach **sliding-window marginalization / Schur** from scratch on our
  manifold LM + GNC, rather than calling g2o/Ceres.

Guardrail: innovate only where it's defensible *and* teachable; never dress a standard method up
as novel; always show the measured win vs the baseline.

---

## 7. Learning course mapping (each chapter ⇒ official book reference)

Format per chapter: **textbook pinhole way → why it breaks on fisheye → ray-native way → measure.**

| Part | Chapter | Ref |
|---|---|---|
| II Geometry & 3D (Ch 8–12, planned) | two-view on rays, charts, sphere-sweep, angular BA, rectification | VS ch.6, ch.11 |
| III Monocular SLAM | VO frontend (PnP/triangulation/keyframes) | VS ch.6/7/12 |
| | Back-end: BA + Schur + Huber | VS ch.8 |
| | Sliding window + pose graph | VS ch.9 |
| | Loop closure + BoW + Sim(3) | VS ch.10, ch.3/13 |
| | The 3-thread architecture | VS ch.12 |
| IV VIO | IMU model + ESKF | AD ch.3 |
| | Pre-integration (15-D factor) | AD ch.4 |
| | Cam–IMU calibration + static init | AD ch.3 |
| | Tightly-coupled VIO (sliding window) | AD ch.4/8 |

---

## 8. Validation matrix & milestones

| Unit | Dataset | Metric / GT | Target | Ref |
|---|---|---|---|---|
| Ray PnP | synthetic + TUM-VI | pose vs known | <1e-6 (synthetic) | VS ch.6 |
| Mono SLAM (Phase A) | TUM-VI room, EuRoC V1_01 | Sim(3) ATE vs GT | competitive open-loop; pinhole-vs-ray table | VS ch.8/12 |
| Mono SLAM (Phase B) | TUM-VI room | Sim(3) ATE w/ loop | drift reduced vs Phase A | VS ch.9/10 |
| Cam–IMU calib (3a) | TUM-VI calib-imu1 | vs published `T_cam_imu` | <0.5° / few-mm | AD ch.3 |
| Pre-integration (3b) | synthetic IMU | vs numerical integ | <1e-6; Jac grad-check | AD ch.4 |
| **VIO (3c)** | **TUM-VI room, EuRoC V1_01** | **full-seq SE(3) ATE** | **3–13 cm (OKVIS/BASALT band)** | AD ch.4/8 |

---

## 9. Honest risks

- **Full-sequence SE(3) ATE in the 3–13 cm band is hard** — it needs solid keyframing, robust
  initialization, and correct marginalization. Early systems will land *above* the band before
  tuning; we report the real number at each step and only call a unit done when it's in-band.
- **Ray PnP, Schur, marginalization, BoW, Sim(3)** are each real engineering, not glue.
- **Python is not real-time** like the books' C++ — we target *correctness and accuracy first*;
  a C++/pybind core for real-time is a later roadmap item, not a Tier-2/3 gate.
- The book references are **L4-AD (LiDAR)** for the inertial text — we use only the
  **sensor-agnostic** IMU/ESKF/pre-integration parts (AD ch.3/4); LiDAR registration (NDT/ICP,
  AD ch.5–8) is **not** in scope for the visual VIO.
</content>
