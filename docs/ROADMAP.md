# Roadmap

DS-MSP has two intertwined goals, and the roadmap serves both:

1. **A clean, correct, well-tested wide-FOV camera library** — Double Sphere plus a
   uniform multi-model layer (UCM, EUCM, Kannala-Brandt, RadTan, OCamCalib), model
   conversion, analytic Jacobians, and Kalibr I/O.
2. **A learning resource** — a [guided, runnable curriculum](learn/README.md) that teaches
   the geometry behind SLAM / AR / robot perception on real public data.

The principle that keeps these from fighting each other: **the library stays terse and
production-grade; the teaching lives in `docs/learn/` and `examples/`.** New features land
with both a clean implementation *and* a chapter that explains them.

## Now (foundation — done)
- Double Sphere model with the correct >180° projection-validity condition.
- Multi-model library + parameter conversion between models.
- Generic Levenberg–Marquardt calibration with analytic Jacobians.
- Kalibr camchain I/O; OpenCV-compatible wrappers; TI LDC hardware export.
- Reproducible setup: `pip install -e .`, 237 passing tests, dataset fetcher.
- Learning track: **Chapters 1–2** — camera models on real TUM-VI data, and the
  Double Sphere model reproducing TUM-VI's published calibration to 0.025 px.
- **Capstone**: calibrate a real fisheye end-to-end from AprilGrid footage
  (`detect → correspond → bundle-adjust`) and match the published intrinsics to
  0.003% focal / 0.08 px median (multi-scale, periphery-robust detection — see the
  [deep-dive](learn/robust_aprilgrid_detection.md)). AprilGrid detection adds `ds_msp/calib/{targets,detect}.py`
  and the `[calib]` extra.

## Next (learning curriculum)
Build out [`docs/learn/`](learn/README.md) in order, each chapter anchored to existing
code and a runnable real-data script:
- **Ch.2** Double Sphere from first principles → `ds_msp/models/ds_math.py` ✅
- **Ch.3** Projection validity & the >180° cone (why `z>0` is the classic bug) ✅ —
  measures the 227° valid cone + the balance/coverage trade (`examples/07_fov_and_validity.py`)
- **Ch.4** Analytic Jacobians vs autodiff — derive, then gradient-check
- **Ch.5** Calibration by Levenberg–Marquardt from corner detections — the theory behind
  the **[capstone](learn/capstone_calibrating_a_real_camera.md)** (already runnable)
- **Ch.6** Model conversion without re-shooting images
- **Ch.7** Reproducing a *published* calibration — the capstone already does this for
  TUM-VI; chapter writes up the method and extends to EuRoC

## Later (capability — geometry → systems)
Extensions that turn "camera models" into "perception systems", each laptop-runnable on
the public datasets already wired up:
- **Stereo extrinsic calibration** ✅ — recovers `T_cam1_cam0` from TUM-VI's synced stereo
  AprilGrid footage, matching the published `T_cn_cnm1` to **0.062° / 0.25 mm** (~0.2%
  baseline; `--stride 8`, 55 frames). See the
  [chapter](learn/stereo_extrinsics_calibration.md) and
  `examples/06_stereo_extrinsics_tumvi.py`. Next: **camera–IMU calibration** (`T_cam_imu`).
- **Monocular visual odometry** on EuRoC / TUM-VI, reusing the bundle adjuster; reported
  with standard ATE/RPE against ground truth.
- **A C++ core** (pybind11) for the hot kernels, plus one Ceres/Eigen calibration.
- **Inference-only modern-3D demos**: learned features (SuperPoint/LightGlue) in the VO
  loop; monocular metric depth metrically corrected by the fisheye calibration.

## Tier 1 — representation-aware 3D: stereo · SfM · reconstruction
**Research-driven.** This tier is scoped from a verified deep-research study of how image-domain
charts (pinhole / equirectangular / cubemap / tangent) are used across 3D tasks — see the
[findings record](research/representations_for_3d_tasks_findings.md) (24/25 claims passed 3-vote
adversarial verification) and the [implementation spec](research/tier1_implementation_spec.md)
(`C1`–`C9`: each with math, algorithm, verification number, and target module). The thread tying
it together: **a fisheye measures rays, so the wide-FOV 3D stack is built on `project` /
`unproject`, not on a pinhole detour** — epipolar lines become curves, disparity becomes angular,
and the essential matrix still holds on **unit bearing vectors**.

Builds on the verified pixel↔ray reprojection already shipped
([deep-dive](learn/spherical_and_cylindrical_reprojection.md), `examples/08`).

**Core library capability — shipped & tested** ✅ (the wide-FOV 3D stack now exists in code):
- **`C1`+`C2` — two-view geometry on rays** (`ds_msp/mvg/two_view.py`, `ransac.py`): essential
  matrix via 8-point on bearing vectors (+ spherical 360-8PA normalization), pose recovery with
  **ray cheirality**, ray triangulation, on-sphere (angular) RANSAC, and an end-to-end
  `estimate_relative_pose`. *(5-point minimal solver deferred.)*
- **`C3` — chart library** (`ds_msp/ops/reproject.py`): sphere / cylinder / pinhole / **cubemap** /
  **tangent-image (gnomonic)** charts with valid masks and FOV-aware auto-intrinsics.
- **`C4` — sphere-sweep stereo** (`ds_msp/stereo/sphere_sweep.py`): depth directly on calibrated
  fisheye, **no rectification**.
- **`C5` — angular BA residual** (`ds_msp/mvg/bundle.py`) + **Schur-complement sparse BA** for
  calibration (`ds_msp/calib/bundle.py`).
- **`C6` — spherical epipolar rectification** (`ds_msp/stereo/rectify.py`): the clean teaching
  complement to `C4`; depth agrees to `<1%`.

**Manifold-optimization foundation — shipped & tested** ✅ (underpins all of the above):
- **SO(3)/SE(3) Lie primitives** (`ds_msp/core/lie.py`) — manifold-correct pose updates; flat
  parameterization is a *correctness* bug, not just slower.
- **In-house Levenberg–Marquardt solver** (`ds_msp/core/optimize.py`) — re-basing manifold LM that
  replaced scipy; made the Lie path both fast and robust. Plus robust M-estimation kernels with
  graduated non-convexity (`ds_msp/core/robust.py`).

**Finishing Tier 1 — active** 🟩:
- **`C9` — ecosystem interop** (`ds_msp/io/`): export calibrated intrinsics + camera poses + sparse
  points to **COLMAP** (`OPENCV_FISHEYE`≈KB, `FOV`), **nerfstudio**, and **openMVG/OpenMVS**. Leans
  on the existing model-conversion + Kalibr I/O layers — no new heavy deps. **This is the bridge to
  Gaussian Splatting (Tier 4) and to every external SfM/MVS tool**, and the exact export the
  DS-MSP→LichtFeld plugin needs. *In progress — the first remaining unit.*

**Deferred to a research extra (`[recon]`) — not started** 🟦:
- **`C7`** multi-chart MVS depth fusion · **`C8`** optical-flow ERP dense reconstruction. Both need a
  heavy external ML estimator (monocular depth / dense optical flow), which cuts against the "free,
  small, laptop-runnable, no special hardware" design rule — so they live behind an optional
  `[recon]` extra and land *after* the inertial arc. See the
  [spec](research/tier1_implementation_spec.md#c7--multi-chart-mvs-depth-fusion-).

**⚠️ Active focus — the teaching layer lags the code.** The Tier-1 library shipped *ahead* of
its curriculum: C1–C6 and the manifold foundation have passing tests but **no `docs/learn/`
chapters and no runnable `examples/`** yet. This violates the design rule below ("each unit lands
with a chapter"). Closing that gap is the next priority — see the
[learning-docs audit](research/learning_docs_audit.md) for the chapter plan (new **Part II —
Geometry & 3D**, Ch.8–12).

**Killed assumption to honor:** there is *no* canonical spherical chart — keep everything
chart-agnostic, keyed off `project` / `unproject`.

## Tier 2 — Monocular visual odometry (VO)
**Goal.** Track the camera's trajectory from a single fisheye stream and report it with the
*standard* metrics — **ATE / RPE against ground truth** — on data we already host (TUM-VI room1
`mocap0`, EuRoC `V1_01` GT). This is the first "system": it consumes Tier-1 geometry end to end
and produces a trajectory a roboticist recognizes.

**The through-line.** A fisheye measures rays, so VO is built on `unproject`/`project`, not a
pinhole detour — exactly the Tier-1 stack. **No new math is invented here; VO is the integration
test for C1–C5 + the manifold LM.**

**Module:** `ds_msp/vo/` (new pure-numpy service layer; independent in the import-linter contract).

**Pipeline.**
1. **Track** features frame-to-frame (KLT / FAST+descriptor) → pixel correspondences.
2. **Lift** to bearing vectors via `cam.unproject` → ray correspondences (chart-agnostic).
3. **Relative pose** with `C1`/`C2` (`essential_from_rays` + on-sphere RANSAC, `recover_pose`).
4. **Local map** by `C1.5` ray triangulation; grow with new keyframes.
5. **Windowed refinement**: sliding-window / local **BA on the angular residual** (`C5`,
   `ds_msp/mvg/bundle.py`) over the manifold LM (`ds_msp/core/optimize.py`, SE(3) Lie).

**Verification (prove a number).** Monocular VO is up-to-scale → align the estimated trajectory
to GT with a **Sim(3) Umeyama** fit, then assert:
- **ATE RMSE** on a TUM-VI room1 / EuRoC V1_01 segment **below a published-baseline threshold**
  (record the exact number once measured; target single-digit-cm on the easy segment).
- **RPE** drift per metre within tolerance; loop-free segment monotonic.
- Synthetic noise-free sequence → ATE `< 1e-6` (the estimator is exact when the data is).

**References (study, don't vendor):** Dong-Won Shin (ex-StradVision, GitHub
[JustWon](https://github.com/JustWon)) — `dvo_slam`, `visual_slam_lecture`, and `my_evo` /
`SLAM_eval` for ATE/RPE tooling; classic feature-based VO. Ships with a `docs/learn/` chapter +
runnable `examples/`.

## Tier 3 — Inertial: cam–IMU calibration → preintegration → VIO
**Goal.** The flagship arc. Fuse the **200 Hz IMU** our datasets already carry into the estimator
to get a **metric, drift-resistant** trajectory — full **visual-inertial odometry**. This is the
capability the TUM-VI / EuRoC datasets exist for, and the headline portfolio artifact. Built as
three dependent units; each ships with its own verification number and chapter.

**`3a` · Camera–IMU calibration** 🟩 — `ds_msp/calib/cam_imu.py`
Estimate **`T_cam_imu`** (the rigid camera↔IMU transform) **and the camera–IMU time offset**
`t_d` from synchronized motion (TUM-VI `calib-imu1`). Reuses the Schur-complement BA and the
SE(3) Lie layer; the IMU enters as a rotation/gravity-alignment constraint.
- *Verify:* recovered `T_cam_imu` vs the published Kalibr `camchain` value (`dso/camchain.yaml`)
  to **< ~0.5° rotation / < few-mm translation**; estimated `t_d` within a frame period.
- *Reference:* Kalibr cam-IMU (canonical); Dong-Won Shin `LIO-SAM` (IMU handling in practice).

**`3b` · IMU preintegration** 🟩 — `ds_msp/inertial/preintegration.py`
On-manifold **preintegrated IMU factors** (Forster et al.): integrate gyro+accel between keyframes
into a relative motion constraint with **analytic bias Jacobians** and noise propagation, so the
back-end never re-integrates raw IMU.
- *Verify:* preintegrated `ΔR, Δv, Δp` vs brute-force numerical integration of a synthetic IMU
  stream to **< 1e-6**; **gradient-check** the first-order bias-update Jacobians; covariance PSD.
- *Reference:* Forster RSS'15 / GTSAM (canonical preintegration); Dong-Won Shin `LIO-SAM`
  (preintegration in practice); 93won/`lightweight_vio` (Ceres + preintegration + sliding window).

**`3c` · Visual-inertial odometry (VIO)** 🟩 — `ds_msp/vio/`
**Tightly-coupled sliding-window / fixed-lag smoother** fusing visual factors (angular
reprojection, `C5`) with IMU preintegration factors (`3b`), solved on the SE(3)+velocity+bias
manifold by the in-house LM (`ds_msp/core/optimize.py`). Bootstraps from Tier-2 VO; calibrated by
`3a`.
- *Verify (the money number):* **ATE / RPE on TUM-VI room1 / EuRoC V1_01 vs GT, in metric scale**
  (IMU resolves scale — no Sim(3) alignment needed, only SE(3)). Assert **VIO beats Tier-2 VO** on
  the same sequence and **recovers absolute scale to within a few %**.
- *Reference:* VINS-Mono, OKVIS, Kimera-VIO, HybVIO (canonical tightly-coupled VIO);
  93won/`lightweight_vio` (Ceres sliding-window VIO, readable).

## Tier 4 — On-device fisheye → 3D Gaussian Splatting (vision / capstone)
**Vision.** DS-MSP becomes the **geometric front-end** for radiance-field reconstruction:
*calibration → VO/VIO poses → SfM sparse init → 3D Gaussian Splatting*, the **whole pipeline
running on this Apple-Silicon laptop with no NVIDIA/CUDA**. This is the through-line that ties every
tier together and closes the loop on the LichtFeld analysis — but via a path that actually runs
locally.

**Why it's possible *now* (the answer to "can't I already train GS with my data?").** Gaussian
Splatting needs three inputs, and after Tiers 1–3 DS-MSP produces all three:
1. **Intrinsics** — calibrated (the capstone), in any of our models.
2. **Posed images** — from ground truth *or, better, from our own VO/VIO* (Tiers 2–3), giving
   **metric** poses.
3. **A sparse point cloud to initialize** — from `C1`–`C5` SfM (two-view → triangulation → BA).
   This matters: **[OpenSplat](https://github.com/pierotofy/OpenSplat) requires sparse points —
   random init is not supported** — so DS-MSP's SfM output is not optional polish, it's the entry
   ticket.

**Backend (external, not reimplemented):** **OpenSplat** — production C++ 3DGS that runs on
**Apple Metal (MPS)** on the M-series GPU (CPU fallback ~100× slower). It ingests **COLMAP /
nerfstudio / openMVG** projects — exactly what **`C9`** exports. (Second target: the **LichtFeld
Studio** CUDA plugin on a rented GPU; `C9` feeds both. See the local-only assessment note.)

**Concrete pipeline.**
```
fisheye imgs (TUM-VI/EuRoC)
  → DS-MSP calibrate            (intrinsics; Tier-1 capstone)
  → DS-MSP VIO                  (metric poses; Tier 3)
  → DS-MSP SfM triangulation    (sparse points; C1–C5)
  → C9 export                   (COLMAP / nerfstudio project)
  → C3 reproject fisheye→pinhole/tangent + valid masks   (OpenSplat is pinhole-only)
  → OpenSplat (Metal)           (train) → .ply / .splat
```

**Guidelines / guardrails.**
- **DS-MSP owns geometry only.** Never vendor a GS trainer into the library — the splatting backend
  stays an external tool, exercised from `examples/` and documented, with **no heavy dep in core**.
- **The fisheye bridge is `C3`.** OpenSplat is pinhole-only → feed it `C3` pinhole/tangent
  reprojections (or undistorted crops) **with valid masks** so rim garbage never enters the loss.
- **Prove a number, as always:** sparse-init **reprojection consistency** after `C9` export
  (round-trips into OpenSplat without coordinate-frame drift), and **PSNR on a held-out view** of a
  short reconstructed segment.
- **Metric scale is the differentiator.** Because Tier-3 VIO poses are metric, the resulting splat
  is metric — a property pure-SfM / COLMAP GS pipelines don't get for free.

**References:** [OpenSplat](https://github.com/pierotofy/OpenSplat) (CPU/Metal 3DGS backend);
dense-reconstruction lineage from Dong-Won Shin ([JustWon](https://github.com/JustWon)) —
`BundleFusion`, `Kintinuous`, `MyElasticFusion` (posed-frames → dense model, the classical analogue
of this front-end→reconstruction pipeline). *(If a CPU/Ceres-based 3DGS repo by Dong-Won Shin
surfaces, prefer it as the reference backend — link TBD.)*

## Design rules (so the two goals stay aligned)
- Every new capability ships with a test that **proves a number**, not a screenshot.
- Teaching verbosity goes in `docs/`/`examples/`, never in `ds_msp/`.
- Demos run on **free, small (<10 GB) public data** on a laptop — no special hardware.
- Prefer **validating against a published reference** over self-asserted correctness.
