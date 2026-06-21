# Tier-1 implementation spec — representations for stereo / SfM / reconstruction

> Turns the verified [findings](representations_for_3d_tasks_findings.md) into **buildable
> units**: each capability carries its **math**, a **core algorithm**, the **verification
> number** to assert (DS-MSP's "prove a number, not a screenshot" rule), the **target module**,
> dependencies, and tier. `[F#]` links a unit to the finding that justifies it.
>
> **Conventions** (match the library). Ray frame: `x` right, `y` down, `z` forward.
> `unproject(u,v) → f` returns a **unit bearing vector**; `project(f) → (u,v)`. A ray's angles:
> azimuth `λ = atan2(x, z)`, elevation `ψ = atan2(-y, hypot(x,z))`. Everything below is
> **chart-agnostic** and built on `project`/`unproject` — per killed claim, no chart is canonical.

Legend — **Tier**: 🟩 core library capability · 🟦 research / nice-to-have.

---

## C1 · Bearing-vector two-view geometry  🟩  `[F5][F6]` — ✅ **implemented**
**Module:** `ds_msp/mvg/two_view.py` (pure-numpy service layer, mutually independent in the
import-linter contract). Shipped: `essential_from_rays` (eight-point on rays + manifold
projection), `decompose_essential`, `triangulate_rays` (midpoint), `recover_pose` (ray
cheirality), `relative_pose`, `epipolar_residual`. Verified on synthetic scenes *and*
end-to-end through `DoubleSphereModel` (project → unproject → recover) to <1e-3° pose error
(`tests/mvg/test_two_view.py`, 8 tests). **Still C2/next:** spherical (360-8PA) normalization,
5-point minimal solver (or OpenGV/PoseLib wrap), and the RANSAC layer below.
The highest-leverage unit: pure math on the bearing vectors `unproject` already returns,
needs **no chart**, and unlocks SfM.

### C1.1 Calibrated epipolar constraint
For correspondences with unit rays `f₁` (cam1) and `f₂` (cam2) and relative pose `(R, t)`
mapping cam1→cam2:
```
f₂ᵀ E f₁ = 0 ,   E = [t]_× R   (rank 2, two equal singular values)
```
Identical to the pinhole calibrated case, but on rays instead of `K⁻¹` pixels — so it works for
**any** central model (DS/UCM/EUCM/KB/…).

### C1.2 Eight-point essential matrix on rays
**Algorithm.**
1. (Recommended) **Spherical preconditioning** `[F6]`: scale each ray so the constraint is well
   conditioned — Robust 360-8PA (arXiv:2104.10900). v1 may skip it; add as `normalize="sphere"`.
2. For each correspondence build row `aᵢ = vec(f₂ ⊗ f₁) ∈ ℝ⁹` (Kronecker of the two rays).
3. Stack `A ∈ ℝ^{N×9}`; `e = ` smallest right-singular vector of `A` (SVD); reshape to `Ê (3×3)`.
4. **Project to the essential manifold:** `Ê = U Σ Vᵀ → E = U·diag(1,1,0)·Vᵀ`.

### C1.3 Five-point relative pose
Nistér's 5-point for calibrated cameras → up to 10 real `E`. Works unchanged on bearing vectors.
v1: **wrap OpenGV / PoseLib**; v2: native implementation as a teaching artifact.

### C1.4 Pose recovery from E (ray cheirality)
`E = U·diag(1,1,0)·Vᵀ`, `W = [[0,-1,0],[1,0,0],[0,0,1]]`. Candidates
`R ∈ {U W Vᵀ, U Wᵀ Vᵀ}` (fix `det R = +1`), `t = ±u₃`. Disambiguate by **ray cheirality**: the
triangulated point must lie in the *positive* direction of both rays — for wide-FOV use
`depth = s` from C1.5 with `s > 0`, **not** `z > 0` (a ≥180° ray can have `z ≤ 0`).

### C1.5 Ray triangulation
Point `X` from rays `f₁,f₂` with centers `c₁,c₂` (world frame).
- **Midpoint (closed form):** solve `min_{s₁,s₂} ‖(c₁+s₁ f₁) − (c₂+s₂ f₂)‖²`; 2×2 linear
  system in `(s₁,s₂)`; `X = ½[(c₁+s₁f₁)+(c₂+s₂f₂)]`.
- **DLT:** rows `f × (R X + t) = 0` per view, SVD.
- **Refine:** minimize **angular reprojection error** `Σ ∠(fᵢ, normalize(Rᵢ X + tᵢ))` (see C5).

### Verification (noise-free synthetic — assert these)
- Random `(R,t)` + 50 points → project through DS → recover: `max |f₂ᵀ E f₁| < 1e-10`;
  rotation error `< 1e-6°`; translation-direction error `< 1e-6°`.
- Triangulation recovers known points to `< 1e-10` (world units).
- **Dependencies:** SVD (numpy), RANSAC layer (C2), optional OpenGV/PoseLib for C1.3.

---

## C2 · Robust estimation on the sphere  🟩  `[F6]` — ✅ **implemented (RANSAC + whitening; 5-pt deferred)**
**Module:** `ds_msp/mvg/ransac.py` — `ransac_relative_pose` (adaptive RANSAC over the eight-point,
angular Sampson scoring) + `sampson_residual`, plus `essential_from_rays(normalize=True)` spherical
whitening in `two_view.py`. Verified: exact noise-free, lower median error on clustered rays,
exact pose under 30 % outliers vs >13° naïve (`tests/mvg/test_ransac.py`, 7 tests). **Deferred:**
the 5-point minimal solver (no OpenGV/PoseLib installed) — RANSAC currently samples 8; wrapping a
5-point solver would cut iterations on low-inlier data. Original plan below.

**Module:** `ds_msp/mvg/ransac.py`. Minimal-sample RANSAC wrapping C1.
- **Sample** 5 (C1.3) or 8 (C1.2) correspondences.
- **Score** with an **angular / on-sphere Sampson** residual, *not* pixel Sampson:
  `rᵢ = ∠(fᵢ_pred, fᵢ_obs)` of the triangulated point, or the first-order Sampson distance of
  `f₂ᵀ E f₁` using ray Jacobians. Threshold in **radians**, FOV-independent.
- **Refine** on the inlier set (C1.2 + nonlinear angular refinement).

### Verification
- 30% gross outliers → recovered pose within tolerance of the inlier-only fit; inlier set
  precision/recall `> 0.95`.

---

## C3 · Chart library (reprojection front-ends)  🟩  `[F1][F2][F3][F4]` — ✅ **implemented**
**Module:** `ds_msp/ops/reproject.py` — `Chart` base + `Equirectangular`, `Cylindrical`,
`Pinhole`, `TangentImage` (gnomonic patch), `cubemap_charts` (6 guard-banded faces), and
`reproject_maps` / `reproject_image` (chart `pixel_to_ray` → `cam.project` → `cv2.remap`,
returning a `valid` mask). Round-trips `pixel→ray→pixel` to <1e-9 px, cubemap tiles the full
sphere, verified end-to-end on `DoubleSphereModel` (`tests/ops/test_reproject.py`, 9 tests).
Charts are pure functions of `cam` (no stored intrinsics) — chart-agnostic, honouring the killed
"no canonical chart" assumption. *Follow-up:* point `examples/08` at this module; tangent-patch
icosahedron layout + fusion is C7. Original notes below.

### C3.1 Existing (already verified to 1e-13 px round-trip) — lift to library
Equirectangular `[F2]`, cylindrical, rectilinear/pinhole `[F1]`. API mirrors `compute_K_new`:
panorama intrinsics `(f px/rad, size, λ/ψ extent)` as parameters with an **FOV-aware default**
derived from the camera's valid cone.

### C3.2 Cubemap `[F2][F3]`
Six faces, each a **90°-FOV pinhole**: `f_face = W_face/2` (since `tan45°=1`). Face `k` has
rotation `R_k ∈ {±X,±Y,±Z}`; face pixel → local pinhole ray → `R_k·ray` → `project` → sample.
Per-face `valid_mask`.

### C3.3 Tangent images (gnomonic patches) `[F4]`
Tangent plane at center direction `n₀(λ₀,φ₀)`; standard **gnomonic** forward/inverse:
```
inverse (patch (x,y) → sphere (λ,φ)):
  ρ = hypot(x,y);  c = atan(ρ)
  φ = asin( cos c · sin φ₀ + (y · sin c · cos φ₀)/ρ )
  λ = λ₀ + atan2( x · sin c ,  ρ·cos φ₀·cos c − y·sin φ₀·sin c )
```
Place patches at **cubemap (6)** or **icosahedron (20)** centers. Convert `(λ,φ)→ray→project`.
**Fusion** of per-patch results (🟦, C6).

### C3.4 Robustness (the "we may have missed" list) `[F2]`
- `valid_mask` everywhere (rays outside the model's valid cone → masked, not garbage).
- **Seam/pole handling:** wrap longitude mod 2π at ±180°; guard `hypot(x,z)→0` at poles.
- **Antialiasing:** area/mip sampling when a chart *downsamples* the source (else moiré).
- **Calibration consistency:** charts are pure functions of `project`/`unproject` — never store
  a second copy of intrinsics.

### Verification
- Round-trip `chart pixel → ray → project → ray → chart pixel < 1e-12 px` per chart (extends the
  existing example-08 corner table to cubemap/tangent).
- Cubemap face seams agree across the shared edge to `< 1 px` after resampling.

---

## C4 · Sphere-sweep stereo (depth)  🟩  `[F7][F8][F9]` — ✅ **implemented**
**Module:** `ds_msp/stereo/sphere_sweep.py` — `sphere_sweep` (per-pixel depth by sweeping
inverse-depth candidates: `X = d·f` → `cam_j.project(R_j X + t_j)` → photo-cost volume → argmin),
`inverse_depth_samples`, `sweep_to_points`. Runs **directly on calibrated fisheye, no
rectification** `[F9]`, using only `unproject`/`project`. Verified on a synthetic textured-plane
DS pair — recovers per-pixel depth to <5% median (`tests/stereo/test_sphere_sweep.py`, 3 tests),
registered as a 5th independent service layer. *Follow-up:* sub-pixel/soft-argmin depth, feature
(census/NCC) cost, and C6 spherical rectification. Preferred modern wide-FOV stereo — runs
**directly on calibrated fisheye, no rectification** `[F9]`, dodging ERP's position-dependent
disparity `[F7]`.

### Math / algorithm
For a reference view with per-pixel ray `f_ref` and candidate **depths** `{dₖ}` (sample **inverse
depth** uniformly — depth candidates sidestep the arc-length nonlinearity `[F7]`):
```
for each pixel p (ray f_ref(p)), each dₖ:
    X = c_ref + dₖ · f_ref(p)               # 3D hypothesis
    for each source view j:
        u,v = camⱼ.project(R_j X + t_j)     # reuse exact project()
        cost[p,k] += matching_cost(ref(p), srcⱼ(u,v))   # SAD/SSD/feature
depth(p) = d_{argmin_k cost[p,k]}           # or soft-argmin
```
No rectification, no homography — just `project` per hypothesis. Cost volume shape
`(H, W, K_depths)`.

### Verification
- Synthetic scene (known depth) → mean depth error `< 1%`; or photometric reprojection
  consistency on a real fisheye pair.
- **Dependencies:** C3 (reference spherical/ERP view), pose from C1, a cost backend
  (numpy → optional torch for speed).

---

## C5 · Angular reprojection residual for BA  🟩  `[F7]` — ✅ implemented (mvg/bundle.py)
**Module:** extend `ds_msp/calib/`. Pixel reprojection error is **anisotropic** under heavy
distortion; the principled wide-FOV error is **angular**.
```
pixel residual (current):  r = (u,v)_obs − project(R X + t)
angular residual (new):    r = ∠( unproject(u,v)_obs ,  normalize(R X + t) )
tangent-plane residual:    project both rays onto the tangent plane at f_obs → 2-vector
```
Jacobian via chain rule over the existing analytic `project` Jacobian + the `unproject`
Jacobian. Expose as `residual="pixel" | "angular" | "tangent"` in `calibrate`.

### Verification
- Low distortion: angular-BA ≈ pixel-BA (same optimum to `< 1e-3 px`).
- High distortion (periphery): angular-BA achieves lower **angular** RMS than pixel-BA.

---

## C6 · Spherical epipolar rectification  🟦→🟩  `[F5][F10]` — ✅ implemented (stereo/rectify.py)
**Module:** `ds_msp/stereo/rectify_spherical.py`. Complements C4; cleaner pedagogically.
- **Top-bottom rig** `[F10]`: rotate both cameras so the baseline aligns with the polar axis;
  resample both into ERP with pole-on-baseline → **epipolar great circles become vertical
  meridians** (constant longitude) → 1D *vertical* search.
- **Angular disparity** `d = θ_b − θ_t`; depth follows from **C1.5 ray triangulation**
  specialized to the meridian plane (two coplanar rays + known baseline). Per `[F7]`, depth is
  nonlinear in `d` — derive it from triangulation, don't assume linearity.

### Verification
- After rectification, a known correspondence's two pixels share longitude to `< 0.5 px`.
- Recovered depth matches C4 on the same synthetic scene to `< 1%`.

---

## C7 · Multi-chart MVS depth fusion  🟦  `[F3][F4]`
**Module:** `ds_msp/stereo/fuse.py`. Recombine per-tangent/per-cube-face depth into one
consistent map: **deformable multi-scale alignment + gradient-domain (Poisson) blending**
(360MonoDepth). Research-grade; depends on C3.3 + C4/external monocular depth.

---

## C8 · Optical-flow ERP rectification & dense recon  🟦  `[F11]`
**Module:** `ds_msp/stereo/flow_recon.py`. Pathak et al.: vertically displaced spherical pair →
dense ERP optical flow → single non-linear minimization jointly refines 5-DOF epipolar geometry
and reads structure from converged flow magnitude. Research-grade; depends on an external dense-
flow estimator.

---

## C9 · Ecosystem interop  🟩  `[F2]` — ✅ **implemented (COLMAP + nerfstudio; openMVG/OpenMVS deferred)**
**Module:** `ds_msp/io/colmap.py` + `ds_msp/io/nerfstudio.py` (extend the existing Kalibr I/O +
model-conversion layers). Shipped:
- **COLMAP text sparse model** read/write (`export_colmap` / `read_colmap`) — `cameras.txt` /
  `images.txt` / `points3D.txt`, with the DS-MSP↔COLMAP camera-model map (KB→`OPENCV_FISHEYE`,
  RadTan→`OPENCV`, pinhole→`PINHOLE`) and the **world-to-camera, (w,x,y,z)-quaternion** pose
  convention handled internally (public API uses 4×4 `T_cam_world`). DS/EUCM/UCM are *refused* with
  a convert-to-KB pointer rather than silently approximated.
- **nerfstudio `transforms.json`** export/read (`export_nerfstudio` / `read_nerfstudio`) — global
  intrinsics + per-frame `transform_matrix` in the OpenGL camera-to-world convention (the
  `inv(T_cam_world)·diag(1,−1,−1,1)` flip is applied/inverted internally).

This is the bridge that feeds **OpenSplat / LichtFeld / nerfstudio** for Gaussian Splatting (Tier 4):
DS-MSP's SfM sparse points + VO/VIO poses + calibrated intrinsics export straight into a
trainer-ready project. *Verified:* export→read round-trip recovers intrinsics, poses (R,t), and 3D
points/colours to `<1e-9` (`tests/io/test_colmap.py`, `tests/io/test_nerfstudio.py`, 17 tests).
**Deferred:** `OPENCV`/`FOV`-only variants, **openMVG** spherical SfM and **OpenMVS** export — add
when a concrete downstream need lands.

---

## Build order & dependency graph

```
C3 (charts)  ─┬─────────────► C4 (sphere-sweep) ──► C7 (fusion)
              └─► C6 (spherical rectification)
C1 (two-view on rays) ─► C2 (RANSAC) ─► [SfM front-end]
C1 ───────────────────────────────────► C5 (angular BA residual)
                                         C8 (flow recon)   C9 (interop)
```

**Recommended first PRs (core, highest leverage):**
1. **C1 + C2** — `ds_msp/mvg/`: essential matrix on rays, pose recovery, ray triangulation,
   RANSAC. Pure math on existing `unproject`; biggest unlock per line of code.
2. **C3** — lift example-08 maps into `ds_msp/ops/reproject.py`, add cubemap + tangent + masks.
3. **C4** — `ds_msp/stereo/sphere_sweep.py` on top of C3.

Each ships with its verification number and a `docs/learn/` chapter (per the
[ROADMAP](../ROADMAP.md) design rules).
