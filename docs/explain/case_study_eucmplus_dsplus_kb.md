# A fair fight: EUCM⁺ vs DS⁺ vs Kannala-Brandt

/// note | Historical record — EUCM⁺ has since been removed
This page's own finding — **EUCM⁺ is strictly Pareto-dominated** (Question 1 below) — is the
measured reason EUCM⁺ was later removed from the shipped library entirely (see
[ADR-0010](../process/architecture/decisions/ADR-0010-mit-relicense-and-eucmplus-removal.md)).
The numbers and analysis below are kept as the real, reproducible-at-the-time record that led
to that decision. `examples/10_evaluating_camera_models.py` now compares **EUCM vs DS⁺** only
(EUCM⁺ is no longer importable), so the EUCM⁺ rows here can no longer be reproduced verbatim —
they were captured from the version of the script that still had it.
///

> **Run alongside this:** `python examples/10_evaluating_camera_models.py` (core install
> only). This page applies the [model-evaluation framework](choosing_a_camera_model.md) to a
> real question that was live at the time: should we ship EUCM⁺ as a default, and does DS⁺ beat
> the standard model? Read this, then read the printed numbers.

The library ships two **extended** wide-<abbr title="Field Of View — the angular extent of the
scene a lens captures.">FOV</abbr> models — **DS⁺** and **EUCM⁺**.

Each adds a [division-distortion](https://en.wikipedia.org/wiki/Distortion_(optics)) term and
a tilt (decentering) term on top of a classic core — UCM for DS⁺, EUCM for EUCM⁺ — for 9
parameters each.

The obvious questions a user, or a reviewer, will ask:

1. Is **EUCM⁺** a sound design, or a parameter-padded one that should be retired?
2. Does **DS⁺** have any real edge over **Kannala-Brandt** (KB), the de-facto standard?

This is a worked example of answering both *with measurements*. It also demonstrates
something harder: **reporting a result that contradicts what you hoped for.**

We set out half-expecting to "prove EUCM⁺ is wrong." The data forced a more precise — and more
useful — conclusion.

That correction is the most important thing on this page, so it's told in full, not hidden.

We evaluate on three real lenses (plus the two reproducible ground-truth lenses in the
[framework script](choosing_a_camera_model.md)):

| lens | FOV | character |
|---|---|---|
| **<abbr title="Technical University of Munich Visual-Inertial">TUM-VI</abbr> cam0** (public) | ~169° | benign, smooth profile |
| **Industrial fisheye, full-FOV capture** | ~158° | strong θ⁵ bend, periphery observed |
| **Same lens class, inner-field only** | ~158° | as seen by a multi-camera rig (limited FOV) |

All numbers below are **median reprojection error in pixels** from a from-scratch,
multi-start fit — multi-start matters, see the [false start](#a-false-start-we-corrected)
below.

The **TUM-VI** row is fully reproducible from public data.

The two **industrial 158°** rows come from private lens captures *not* shipped with this repo
— only the lens character is given, no proprietary data.

They corroborate, but the reproducible anchor for everything on this page is the
[framework script](choosing_a_camera_model.md)'s two ground-truth lenses.

| lens | KB (8p) | EUCM (6p) | EUCM⁺ (9p) | DS⁺ (9p) | note |
|---|---|---|---|---|---|
| TUM-VI 169° (benign) | 0.082 | 0.083 | 0.082 | 0.082 | all tie |
| Industrial 158° full-FOV | 0.285 | 2.096 | 0.311 | 0.224 | DS⁺ best, EUCM fails |
| Industrial 158° inner-field | 0.366 | 0.659 | 0.607 | 0.365 | DS⁺ ties KB |

---

## Question 1: is EUCM⁺ a wrong choice?

### A false start we corrected

Our first reading of one lens — the inner-field 158° fit — was this claim:

*"EUCM⁺'s extra parameter λ₁ is collinear with EUCM's β, the shape Jacobian is numerically
rank-deficient (condition number ~10¹⁸), so λ₁ is redundant — EUCM⁺ is mathematically a wrong
model."*

It's a tidy story. It is also **over-claimed**, and we retracted it. Two checks broke it:

- **Capacity check across lenses (diagnostic A).** On the full-FOV lens, EUCM alone fails badly
  (2.10 px) and adding λ₁ rescues it to 0.31 px — a 7× improvement. A *redundant* parameter
  can't do that.
- **The constraint check (diagnostic D), initially skipped.** The rank-deficiency was at a fit
  where EUCM's α had hit its `≤1` bound and β had collapsed to 0.001 — a *boundary* artifact.
  Raising the α ceiling and refitting, EUCM still floors at ~1.2 px, nowhere near EUCM⁺'s 0.31.
  So λ₁ adds **real radial capacity EUCM lacks at any bound**, the opposite of redundant.

/// tip | Lesson — the kind this curriculum is built on
A single condition number at a single fit is not a proof. Identifiability (B) is only
meaningful *together* with capacity (A) and constraint (D) checks.

The honest verdict is not "EUCM⁺ is wrong" — that's false — but the more careful one below.
///

### What is actually true: EUCM⁺ is *Pareto-dominated*

λ₁ helps, but EUCM⁺ is still the model you shouldn't pick — because **at every parameter budget
something beats it**:

- **Redundancy (C).** λ₁'s first-order radial effect is ~collinear with β: residual outside
  span{α,β} is **0.45% (benign) and 1.19% (demanding)**, `|cos|` up to 0.9995. One of EUCM⁺'s
  three shape <abbr title="Degrees Of Freedom — independent numbers that can vary to reshape
  the fit; more DOF means more ways the curve family can bend.">DOF</abbr> mostly re-describes
  another, paying the conditioning cost (diagnostic B: cond 5.9e6 on the benign lens) for
  little new capacity.
- **Dominated at equal cost.** At the *same* 9 parameters, **DS⁺ beats it on every demanding
  lens** (0.224 vs 0.311 full-FOV; 0.365 vs 0.607 inner-field) — because DS⁺ spends its third
  shape DOF on something independent (next section) instead of a near-duplicate of β.
- **Never beats KB**, and has the **worst valid coverage** (92% of pixels unproject on the
  inner-field lens, vs 100% for KB/EUCM).
- On the **benign** lens its λ₁ is *fully* idle (C: 0.45% residual) — pure parameter padding.

**Verdict on Q1.** Deprecate EUCM⁺ — but for the correct, defensible reason: it is **strictly
Pareto-dominated** (EUCM is cheaper when 2 DOF suffice; DS⁺ is more accurate at equal 9 params
when they don't), *not* because "λ₁ does nothing." Use EUCM for benign lenses, DS⁺ for
demanding ones, and skip the dominated middle.

---

## Question 2: does DS⁺ have any edge over Kannala-Brandt?

Here the honest answer is split — a real edge in two places, a clear loss in another. We report
all three rather than cherry-picking the win.

### Accuracy: a small but *consistent* edge ✅

DS⁺ matches-or-beats KB on all three lenses — it ties on benign (0.082) and inner-field
(0.365 ≈ 0.366), and posts a genuine **~21% win on the demanding full-FOV lens** (0.224 vs
0.285). It never loses to KB.

Against the *closed-form* family the gap is decisive: on the demanding lens, DS⁺'s 0.224 beats
EUCM's 2.096 and EUCM⁺'s 0.311. DS⁺ is the only closed-form-invertible model that reaches
KB-class accuracy.

That holds because (diagnostic C) its second division term λ₂ stays genuinely independent
(residual **4–13%**, vs EUCM⁺'s ~1%), and (diagnostic B) all three shape DOF stay identifiable
on a lens that needs them.

### Compute: KB wins the unproject-bound operations ❌

This is where the "more modern model = better" intuition fails.

DS⁺'s 2-term division needs a **Ferrari quartic** (complex arithmetic) whenever λ₂≠0, and λ₂
is *essential* on both 158° lenses. So its closed form is **~2× slower than KB's Newton
iteration**, not faster:

| operation (per the questions a 3D pipeline asks) | KB | DS⁺ | winner |
|---|---|---|---|
| 3D → image (project) | 10.6 ms | 9.8 ms | ~tie |
| 2D → bearing (unproject) | 10–16 ms / 160k | 22–32 ms | **KB ~2×** |
| 2D → 3D (= unproject × depth) | = unproject | = unproject | **KB** |
| full-frame undistortion map (1.05M px) | 105 ms | 194 ms | **KB** |
| round-trip exactness | **7e-15** (machine) | 5e-13 | **KB** |
| valid pixel coverage (demanding) | **100%** | 98% | **KB** |

/// note | A correction we owe the reader
We earlier described KB's iterative unproject as "approximate ~1e-6." That's wrong — KB's
Newton converges to **machine precision (7e-15)**.

"Iterative" means *variable-latency and occasionally non-convergent at the rim*, not
inaccurate.
///

### DS⁺'s genuine edge over KB: closed-form, non-iterative invertibility

A finite algebraic unproject (no Newton loop) buys things throughput numbers don't show,
*plus* the small accuracy win:

- exact analytic differentiability — Jacobians for bundle adjustment / autodiff without
  unrolling an iteration;
- deterministic, data-independent latency (matters on GPU / SIMD, where divergent per-pixel
  iteration counts stall a warp);
- no convergence-failure path (KB left 0.4% of extreme-periphery pixels invalid on TUM-VI; DS⁺
  is algebraic).

**Verdict on Q2.** DS⁺ does **not** dominate KB. It wins on accuracy (small, consistent) and on
analytic invertibility (differentiable / deterministic pipelines). KB wins on raw
unproject/undistort throughput (~2×), and ties on project.

Pick DS⁺ when you need a differentiable or closed-form unproject at the accuracy edge measured above.
Pick KB — or EUCM on a benign lens — when unproject throughput dominates your cost.

---

## The bottom line — and the meta-lesson

- **EUCM⁺:** don't ship it as a default — strictly Pareto-dominated (a near-redundant λ₁). But
  "mathematically wrong / useless" is false; say *dominated*, which is what the data supports.
- **DS⁺:** keep it — best accuracy of any model tested and the only closed-form one that matches
  KB on hard lenses; just don't claim a CPU win it doesn't have.
- **KB:** still the right default when unproject throughput rules and you can tolerate an
  iterative inverse.

The meta-lesson is the reason this is in the *learning* track at all: **we reported the
finding that contradicted our hypothesis.**

The tidy "EUCM⁺ is provably wrong" story was more satisfying than "EUCM⁺ is dominated but its
λ₁ does real work" — and it was false. Diagnostics A and D caught it; honesty kept it caught.

A camera-model evaluation is only worth as much as its willingness to publish the result it
didn't want.

## Try it yourself
1. In the current framework script, diagnostic D shows the same shape of proof for DS⁺ that
   originally cleared EUCM⁺: unbounding EUCM's α helps a little but plateaus far short of the
   richer model's RMS — that gap is what "real radial capacity, not a redundant parameter"
   looks like as a number.
2. Force DS⁺'s λ₂ to 0 and re-time its unproject. It drops to the cheap quadratic branch — now
   compare against KB. (You've just traded the demanding-lens accuracy for KB-class speed; that
   trade *is* the design space.)
3. Add your own lens and run diagnostics A–F on EUCM vs DS⁺ — the same procedure that led to
   this page's EUCM⁺ verdict works for deciding between any two candidates on your camera.

**Built on:** [the model-evaluation framework](choosing_a_camera_model.md). **Related:**
[Are two models the same camera?](are_two_models_the_same_camera.md) ·
[The Double Sphere model](../learn/02_double_sphere_model.md).
