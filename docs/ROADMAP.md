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
This tier covers how image-domain charts (pinhole / equirectangular / cubemap / tangent) are
used across 3D tasks, building each capability with its math, algorithm, verification number,
and target module. The thread tying
it together: **a fisheye measures rays, so the wide-FOV 3D stack is built on `project` /
`unproject`, not on a pinhole detour** — epipolar lines become curves, disparity becomes angular,
and the essential matrix still holds on **unit bearing vectors**.

Builds on the verified pixel↔ray reprojection already shipped
([deep-dive](learn/spherical_and_cylindrical_reprojection.md), `examples/08`).

**Core library capability — shipped & tested** ✅ (the wide-FOV 3D stack now exists in code):
- **Two-view geometry on rays** (`ds_msp/mvg/two_view.py`, `ransac.py`): essential
  matrix via 8-point on bearing vectors (+ spherical 360-8PA normalization), pose recovery with
  **ray cheirality**, ray triangulation, on-sphere (angular) RANSAC, and an end-to-end
  `estimate_relative_pose`. *(5-point minimal solver deferred.)*
- **Chart library** (`ds_msp/ops/reproject.py`): sphere / cylinder / pinhole / **cubemap** /
  **tangent-image (gnomonic)** charts with valid masks and FOV-aware auto-intrinsics.
- **Sphere-sweep stereo** (`ds_msp/stereo/sphere_sweep.py`): depth directly on calibrated
  fisheye, **no rectification**.
- **Angular BA residual** (`ds_msp/mvg/bundle.py`) + **Schur-complement sparse BA** for
  calibration (`ds_msp/calib/bundle.py`).
- **Spherical epipolar rectification** (`ds_msp/stereo/rectify.py`): the clean teaching
  complement to sphere-sweep; depth agrees to `<1%`.

**Manifold-optimization foundation — shipped & tested** ✅ (underpins all of the above):
- **SO(3)/SE(3) Lie primitives** (`ds_msp/core/lie.py`) — manifold-correct pose updates; flat
  parameterization is a *correctness* bug, not just slower.
- **In-house Levenberg–Marquardt solver** (`ds_msp/core/optimize.py`) — re-basing manifold LM that
  replaced scipy; made the Lie path both fast and robust. Plus robust M-estimation kernels with
  graduated non-convexity (`ds_msp/core/robust.py`).

**Finishing Tier 1 — active** 🟩:
- **Ecosystem interop** (`ds_msp/io/`): export calibrated intrinsics + camera poses + sparse
  points to **COLMAP** (`OPENCV_FISHEYE`≈KB, `FOV`), **nerfstudio**, and **openMVG/OpenMVS**. Leans
  on the existing model-conversion + Kalibr I/O layers — no new heavy deps. The bridge to external
  SfM / MVS / Gaussian-Splatting tools (Tier 4). *Shipped & tested* ✅.

**Deferred to a research extra (`[recon]`) — not started** 🟦:
- **Multi-chart MVS depth fusion** · **optical-flow ERP dense reconstruction**. Both need a
  heavy external ML estimator (monocular depth / dense optical flow), which cuts against the "free,
  small, laptop-runnable, no special hardware" design rule — so they live behind an optional
  `[recon]` extra and land *after* the inertial arc.

**⚠️ Active focus — the teaching layer lags the code.** The Tier-1 library shipped *ahead* of
its curriculum: the geometry stack and manifold foundation have passing tests but **no `docs/learn/`
chapters and no runnable `examples/`** yet. This violates the design rule below ("each unit lands
with a chapter"). Closing that gap is the next priority — a new **Part II — Geometry & 3D**
(Ch.8–12).

**Killed assumption to honor:** there is *no* canonical spherical chart — keep everything
chart-agnostic, keyed off `project` / `unproject`.

## Tier 2 — Monocular visual odometry (VO)
Track the camera trajectory from a single fisheye stream directly on bearing vectors
(`unproject` / `project`, no pinhole detour), reusing the Tier-1 two-view geometry, ray
triangulation, angular BA residual, and the manifold LM. Reported with standard **ATE / RPE**
against ground truth on the public datasets already wired up (TUM-VI, EuRoC).

**Shipped & tested** ✅ — `ds_msp/vo/`:
- `estimate_trajectory` — two-view relative pose chained with landmark scale-propagation; a
  monocular trajectory recovered up to one global similarity.
- `metrics` — `align_sim3` (Umeyama), `ate_rmse`, `rpe_rmse`: the up-to-scale evaluation toolkit.
- Verified on synthetic trajectories to ATE `< 1e-6`; a runnable example evaluates against TUM-VI
  room1 ground truth (`examples/09_monocular_vo_tumvi.py`).

**Next:** keyframing + local sliding-window BA + loop closure for full-sequence robustness, and a
`docs/learn/` chapter.

## Tier 3 — Visual-inertial odometry (VIO)
Fuse the IMU the datasets already carry to recover a **metric, drift-resistant** trajectory. Three
dependent units, each validated against ground truth:
- **Camera–IMU calibration** (`ds_msp/calib/cam_imu.py`) — estimate `T_cam_imu` + time offset;
  validated against the published Kalibr camchain.
- **IMU preintegration** (`ds_msp/inertial/preintegration.py`) — on-manifold preintegrated factors
  (Forster et al.) with bias Jacobians and covariance; verified against numerical integration.
- **Tightly-coupled VIO** (`ds_msp/vio/`) — sliding-window optimization fusing the angular
  reprojection residual with IMU factors on the SE(3)+velocity+bias manifold; evaluated by
  full-sequence SE(3) metric ATE on the public benchmarks, aiming for parity with established
  open-source VIO.

## Tier 4 — Integration with external 3D reconstruction
Use the ecosystem-interop exporters (COLMAP / nerfstudio) so a DS-MSP calibration + poses + sparse points feed
external Structure-from-Motion, MVS, and Gaussian-Splatting tools. DS-MSP provides the wide-FOV
geometry front-end; the reconstruction / splatting backend stays an external tool, exercised from
`examples/` with no heavy dependency in the core library.

## Design rules (so the two goals stay aligned)
- Every new capability ships with a test that **proves a number**, not a screenshot.
- Teaching verbosity goes in `docs/`/`examples/`, never in `ds_msp/`.
- Demos run on **free, small (<10 GB) public data** on a laptop — no special hardware.
- Prefer **validating against a published reference** over self-asserted correctness.
