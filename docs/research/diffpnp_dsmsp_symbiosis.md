# DS-MSP ↔ diffpnp — survey, gap analysis, and a symbiosis path

> Survey + gap analysis of how DS-MSP and the companion **diffpnp** repository complement each
> other, and in what order to integrate them. Companion to the
> [Tier-1 spec](tier1_implementation_spec.md) and the
> [two-view geometry proofs](../explain/two_view_geometry.md). diffpnp is a separate repository.

## Thesis — is the plan proper? **Yes, and it's the natural architecture.**

Strip both repos to one sentence and they are the **two halves of the same problem**:

> *optimize a pose (and structure / intrinsics) given a **camera projection model**.*

- **DS-MSP** owns the *projection model* half: verified wide-FOV models (Double Sphere, UCM,
  EUCM, Kannala-Brandt, RadTan, OCam), analytic Jacobians, ray-based geometry, calibration that
  matches published references — **but** its pose optimizer is flat-Euclidean axis-angle in SciPy
  (not manifold-correct) and its robustness is a single Cauchy loss.
- **diffpnp** owns the *optimizer* half: manifold-correct SE(3)/SO(3) Lie machinery, a
  differentiable robust LM with IFT backward, GNC, studentized IRLS, learnable kernels,
  Cramér–Rao covariance — **but** it is **pinhole-only**.

Each has exactly what the other lacks. And the *shared abstraction* that unifies them already
exists on both sides: **a camera model is `(project, unproject)` on bearing vectors plus an
intrinsics vector; pose optimization is manifold least-squares on a residual.** DS-MSP's whole
ray-based Tier-1 (bearing-vector two-view, the **angular residual** C5, ray-based PnP) is
*already* the camera-agnostic shape diffpnp needs to leave pinhole behind.

So the integration plan — *validate wide-angle correctness in DS-MSP, reuse the Lie +
optimization-robustness layer from diffpnp, and once DS-MSP is ready port the camera models into
diffpnp* — is not just feasible; it is the architecture the two codebases are already converging
toward.

## Survey — what each side exposes at the seam

**DS-MSP `CameraModel` contract** (`ds_msp/core/contracts.py`) — already a clean model plugin:
`project(P)→(uv,valid)`, `unproject(uv)→(unit ray,valid)`, `project_jacobian(P)→(uv, J_point,
J_param, valid)`, `from_params`, `param_bounds`, `initialize_from_correspondences(ray↔pixel)`,
`to_dict/from_dict`. Six models implement it, each cross-checked to OpenCV / published calibration.

**diffpnp pinhole coupling** — the differentiable core is camera-*agnostic*; only three seams are
baked to pinhole (confirmed by reading the source):

1. **Projection** (`projection/pinhole.py`): `residual_single/residual/project/cost` hard-code the
   gnomonic `xy = X/Z` and a `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]`.
2. **Intrinsics**: `K (3×3)` is threaded through `forward`, `backward`, and the IFT VJPs
   (`functional.py`); gradients w.r.t. it (`grad_K`) come from autograd.
3. **Init** (`init.py`): `dlt_init` / `ippe_init` / `ransac_init` all start from `K⁻¹[u,v,1]`
   normalized coordinates and a *linear* DLT — pinhole-specific.

Everything else is general: the **autodiff Jacobian backends** (`jacrev`/`jacfwd` over
`residual_single`, `jacobians.py`) differentiate *whatever* projection they're handed; the **IFT
backward, LM solver, GNC, robust kernels, and `lie.py`** operate on `(residual, Jacobian)`, not on
pinhole specifics. diffpnp even already has a **learnable-intrinsics calibration example**
(`examples/04_camera_calibration.py`) — it just optimizes `K` today.

## Gap analysis A — diffpnp → *any* camera space

| Seam | Pinhole today | Generalization | DS-MSP asset to reuse |
|---|---|---|---|
| **Projection fn** | `residual_single` with `X/Z`, `K` | a `projection/<model>.py` registry; `PnPLayer(projection='double_sphere')` | DS-MSP's per-model project math (port to torch) |
| **Intrinsics** | `K (3×3)` everywhere | generic `intr` tensor `(B, P_model)`; grads via autograd | DS-MSP param vectors (`ξ,α`, `k1..k4`, …) + `param_bounds` |
| **Jacobian** | analytic pinhole + autodiff | autodiff backends already cover **any** torch projection; analytic optional | DS-MSP analytic Jacobians → fast per-model backend (later) |
| **Init** | DLT/IPPE via `K⁻¹` linear | **ray-based**: `unproject → bearings → P3P/EPnP on rays` | DS-MSP `solve_pnp` (ray-based) is the template |
| **Cheirality** | `z.clamp(min=1e-6)` (z>0) | **depth-along-ray > 0** (valid past 90°) | DS-MSP C1.4 ray cheirality |
| **Validity** | none | model validity mask (e.g. DS half-space) | DS-MSP `project` valid mask |

**MVP to make diffpnp multi-camera:** a small projection *protocol* + **one** model (Double
Sphere) implemented as a torch `residual_single` + a ray-based init + ray cheirality. Because the
autodiff Jacobian path is already general, **no hand-derived Jacobian is needed to start** — the
analytic backend can come later for speed. This is a genuinely novel artifact: *robust,
differentiable, wide-FOV PnP* exists in neither library today.

## Gap analysis B — DS-MSP ← diffpnp's optimization

| Capability | DS-MSP today | Borrow from diffpnp |
|---|---|---|
| **Rotation parametrization** | axis-angle as flat ℝ³ in SciPy (`bundle.py`, C5 `refine_two_view`) — biased >30°, can drift near `‖r‖=π` | **SO(3)/SE(3) `exp`/`log` retraction + right-perturbation LM** (`lie.py`) — *port the math to NumPy* |
| **Pose loss** | pixel residual | **geodesic SE(3) loss** + DS-MSP's own **angular residual** (already shipped, C5) |
| **Robustness** | single Cauchy loss | **GNC** annealing, **learnable Barron α**, **studentized IRLS** (bounded-influence) |
| **Uncertainty** | none | **Cramér–Rao covariance** output for fusion / weighting |
| **Differentiability** | none | (later) optional torch bridge for learned front-ends |

**MVP to make DS-MSP manifold-correct:** a pure-NumPy `ds_msp/mvg/lie.py` (`so3_exp/log`, the
SO(3) right-Jacobian, SE(3) retraction) and switch C5's `refine_two_view` + the calibration bundle
to step *on the manifold*. **No torch** — keeps DS-MSP lean. This is the direct fix for the
"why are we optimizing pose in flat axis-angle?" observation, and a prerequisite for trustworthy
cross-validation.

## The cross-validation harness — the glue that makes this safe

The symbiosis only works if the *same* camera model means the same thing in both repos. Two checks:

1. **Geometry oracle (DS-MSP validates diffpnp's port).** A shared parametric test: same
   `(model, intrinsics, pose, points)` → `DS-MSP.project` vs `diffpnp.<model>.project` agree to
   fp64 `~1e-12`; DS-MSP's **analytic Jacobian** vs diffpnp's **autodiff Jacobian** agree to
   `~1e-8`. DS-MSP answers *"is the projection geometry right?"*
2. **Optimizer oracle (diffpnp validates DS-MSP's manifold port).** Same residual + Jacobian →
   diffpnp's manifold LM and DS-MSP's new manifold LM reach the same optimum; diffpnp answers
   *"is the descent right at large rotations?"* And: DS-MSP's classical LM calibration vs
   diffpnp's **differentiable** fisheye calibration on the same TUM-VI data should land on the
   same intrinsics.

This is exactly *"confirm wide-angle correctness on DS-MSP while utilizing Lie + robustness from
diffpnp."*

## Phased plan (interleaves with Tier-1)

- **Phase 0 — shared spec (cheap).** A one-page *camera-model contract* both implement (the
  intersection of DS-MSP's `CameraModel` and diffpnp's projection seam) + this analysis. Done here.
- **Phase 1 — DS-MSP goes manifold-correct (NumPy, no torch). ✅ DONE.** `ds_msp/core/lie.py`
  (so3/se3 `exp`/`log`, right Jacobian — verified vs cv2 + the right-Jacobian identity by finite
  difference) is now used by both `mvg.refine_two_view` (C5) and `calib/bundle.py`: each optimizes
  a **local perturbation** `R ← R_base·exp([δω]_×)` from `δω=0` instead of a flat absolute
  axis-angle, with the analytic right-perturbation Jacobian `∂Xc/∂δω = -R[Xw]_× J_r(δω)` in the
  calibrator. **No regression** — the capstone still matches TUM-VI to 0.0 % focal / 0.04 px
  principal / 0.081 px median — and `refine_two_view` now stays well-conditioned at large rotations
  (a ~165° scene where flat axis-angle nears the `‖r‖=π` singularity). *Answers the Lie gap.*
- **Phase 2 — diffpnp goes multi-camera (torch).** Projection protocol + **Double Sphere** backend
  (port DS-MSP's verified math) + ray-based init + ray cheirality. `PnPLayer(projection=…)`.
  Cross-test against DS-MSP (harness check #1).
- **Phase 3 — differentiable fisheye calibration.** diffpnp learns DS/KB intrinsics by backprop;
  compare to DS-MSP's classical calibration on TUM-VI (harness check #2). Optionally backport
  **GNC** into DS-MSP's calibration.
- **Phase 4 — learned wide-FOV front-end (Tier-4 "modern-3D").** Train a fisheye
  keypoint/confidence head end-to-end through diffpnp pose error with the DS backend — a learned
  wide-FOV differentiable-3D capability.

## Risks & guardrails

- **Keep DS-MSP lean.** The Lie port (Phase 1) is NumPy. Any torch lives in **diffpnp** or in a
  DS-MSP *optional* `[torch]` bridge — **never** DS-MSP core (numpy/scipy/opencv stays the floor).
- **One source of truth per model.** Derive each projection once, port, and pin it with the
  cross-validation harness so the two implementations can't drift.
- **Scope boundary.** diffpnp is **PnP** (known 3D) — it complements DS-MSP's BA / calibration /
  pose refinement, but does **not** replace C1/C2 (relative pose from *unknown* structure) or
  C3/C4/C6 (charts, sphere-sweep, rectification). The overlap is precisely "optimize a pose."
- **Dtype discipline.** diffpnp defaults to fp32 (GPU); DS-MSP is fp64. Cross-tests pin dtype.

## Recommendation

Start with **Phase 1** — it's pure-NumPy, directly fixes the flat-axis-angle optimization the
calibration uses today, improves the C5 BA already shipped, and is the prerequisite that lets
diffpnp's optimizer and DS-MSP's geometry validate each other. Phases 2–4 then port DS-MSP's
camera models into diffpnp and close the loop into learned wide-FOV pose.
