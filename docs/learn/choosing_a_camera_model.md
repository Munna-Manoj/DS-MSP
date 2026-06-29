# Is this camera model right for *my* lens and task?

> **Run alongside this:** `python examples/10_evaluating_camera_models.py`
> (core install only — no dataset needed; it builds two ground-truth lenses for you).
> Read this, then read the printed numbers.

Almost everyone picks a camera model the wrong way: by **reputation** ("everyone uses
Kannala-Brandt") or by **parameter count** ("more parameters must be more accurate"). Both
heuristics fail often enough to quietly ruin a calibration — a model that's perfect for a
GoPro is over-parameterized junk on a webcam, and a model that nails a 160° drone lens
can't bend enough for a 195° security dome.

This page gives you the alternative: **derive the choice from measurements.** Six
diagnostics, each a number you can compute and verify, that together answer the only
questions that matter — *can this model represent my lens, are its parameters actually
pinned down by my data, what does it cost, and where will it lie to me?* None of them
require you to trust a paper's marketing.

We'll demonstrate the framework on two deliberately different lenses (both Kannala-Brandt
ground truth, so the "right answer" is known exactly):

| lens | what it is | the catch |
|---|---|---|
| **BENIGN** | mild ~170° lens, smooth radial profile | 2 shape DOF are plenty |
| **DEMANDING** | ~158° industrial lens, strong θ⁵ bend | needs ≥3 radial DOF |

and score three candidates against each — **EUCM** (6 params), **EUCM⁺** (9), **DS⁺** (9) —
with **KB** itself as the reference. The headline result, which no rule-of-thumb predicts:
*the same model is a great choice on one lens and a poor one on the other.* Only the
diagnostics tell you which.

---

## The mental model: a camera is a 1-D curve, a fit is a projection

Before the diagnostics, two facts make all of them simple. For a central, radially-symmetric
camera (`fx≈fy`, principal point stable — almost every real lens), *all* the optics live in
one curve, the **radial profile** `r(θ)`: a ray at angle `θ` off-axis lands `r` pixels from
the principal point ([why](are_two_models_the_same_camera.md)). A "camera model" is just a
*parametric family* of such curves; **calibration / conversion is choosing the curve in that
family closest to your lens's curve**, in a least-squares sense, over the angles your data
covers. Every diagnostic below is a property of that family of curves and that projection.

---

## A. CAPACITY — can the model even represent my lens?

**Question.** Is my lens's `r(θ)` *inside* the candidate's family at all? If not, no amount of
optimization will help — you're projecting onto a surface that doesn't contain the target.

**How.** Fit the candidate to a known-good reference of the same lens over your FOV and read
the residual. In the library this is exactly what `adapt.convert` does (reference model →
candidate, least squares on a pixel grid). The example converts the KB ground truth into each
candidate, bounded to the lens FOV, and prints in-FOV RMS:

```
DEMANDING lens (~158°):    eucm 6.62 px    eucm+ 1.14 px    ds+ 0.91 px
BENIGN   lens (~170°):     eucm 0.24 px    eucm+ 0.16 px    ds+ 0.019 px
```

**Read it.** RMS ≫ 0.1 px means the family *cannot bend like this lens* — a capacity wall, not
a tuning problem. EUCM hits that wall on the demanding lens (6.6 px); DS⁺ tracks KB on both.

**The subtlety that bites everyone — measure capacity over *your* FOV.** On real calibration
corners, which mostly land in the inner field (TUM-VI corners: median 35°, 88% within 55°),
all of these models tie at ~0.08 px. The differences above only appear over the *full* FOV.
So diagnostic A must sample the field **your task actually uses** (`convert`'s `max_fov_deg`).
A model can be "enough" for calibration and still wrong for full-frame undistortion.

---

## B. IDENTIFIABILITY — are the parameters pinned by the data?

**Question.** Even if a curve fits, are its parameters *determined*? An over-parameterized
model has directions in parameter space the data can't see — it'll fit, but the parameters
are noise, won't transfer, and make the optimizer wander.

**How.** Build the **shape Jacobian** `J` — columns are `∂r(θ)/∂pᵢ` sampled across the FOV for
each shape parameter — and look at the spectrum of the Gram matrix `JᵀJ` (its eigenvalues are
the squared sensitivities). The **condition number** `σ_max/σ_min` is the headline:

```
DEMANDING:  eucm cond 7e2   eucm+ cond 1.8e5   ds+ cond 2.9e5
BENIGN:     eucm cond 1.2e3 eucm+ cond 5.9e6   ds+ cond 1.7e5
```

**Read it.** cond ~10²–10³ = healthy (every parameter independently visible); ~10⁶⁺ = a
near-null direction (one parameter barely affects the curve — practically unidentifiable);
~10¹⁵⁺ = numerically rank-deficient (a true redundancy). EUCM is beautifully conditioned but
(diagnostic A) lacks capacity; EUCM⁺ on the benign lens has a near-null third direction
(cond 5.9e6, smallest singular value ~20 against ~10⁸) — a parameter that isn't earning its
place. **The smallest singular value, not the count, tells you the effective DOF.**

---

## C. REDUNDANCY — does an *added* parameter add a new direction?

**Question.** When you bolt an extra term onto a model, does it open a genuinely new way for
the curve to bend, or just re-describe a wiggle the existing parameters already had?

**How.** Take the new parameter's Jacobian column and project it onto the span of the existing
shape columns. The **residual fraction** left outside that span is how much *new* DOF it adds;
`|cos|` against the nearest existing column shows the overlap directly:

```
DEMANDING:  eucm+ lam1: |cos|=0.9995  residual 1.19%      ds+ lam2: |cos|=0.985  residual 4.17%
BENIGN:     eucm+ lam1: |cos|=1.0000  residual 0.45%      ds+ lam2: |cos|=0.958  residual 12.8%
```

**Read it.** Residual < 1% ⇒ the parameter is *nearly redundant* (collinear with one it
already has) — it adds conditioning trouble (diagnostic B) for almost no capacity. EUCM⁺'s
division term λ₁ is ~collinear with EUCM's β on both lenses (residual ≤1.2%): a structural
inefficiency in *that* model's design. DS⁺'s second division term λ₂ keeps 4–13% genuinely
its own — a parameter that pays for itself. **This is how you tell a well-designed extension
from a padded one, before you ever fit real data.**

---

## D. CONSTRAINTS — is a bound hiding the real story?

**Question.** When a model fits poorly, is it a true capacity wall, or just a *bound* you
imposed (e.g. EUCM's conventional `α ≤ 1`) clipping a parameter that wants to run free?

**How.** Relax the bound and refit. If the residual drops, the bound was the problem; if it
barely moves, the limit is real capacity. The example raises EUCM's α ceiling:

```
DEMANDING:  EUCM α≤1 → 6.62 px    α≤3 → 4.70 px    α≤8 → 4.70 px    (EUCM⁺ with λ₁ → 1.14)
```

**Read it.** Unbounding α helps a little (6.6→4.7) but gets nowhere near EUCM⁺'s 1.14 — so on
this lens λ₁ supplies **genuine radial capacity EUCM lacks at any bound**, not a workaround for
the α≤1 clip. This is the diagnostic that keeps you *honest*: it's tempting to dismiss an
extra parameter as "just compensating for a bound," and sometimes that's true — this test is
how you find out instead of assuming.

---

## E. COST — what does project / unproject actually cost?

**Question.** Per frame you may unproject a million pixels (undistortion), project for every
bundle-adjustment residual, and round-trip for triangulation. The model's *operation class*,
not its parameter count, sets the bill.

**How.** Time `project` / `unproject` on large batches and check the round-trip error. But the
*why* matters more than the milliseconds, because it generalizes across hardware:

| model | unproject is… | consequence |
|---|---|---|
| UCM / EUCM / EUCM⁺ | one square root (closed form) | fastest; branchless; trivially differentiable |
| DS⁺ | a Ferrari **quartic** when λ₂≠0 | ~2× slower than KB; closed-form but heavy |
| KB | **Newton iteration** (no closed form) | exact to machine precision, data-dependent latency, can fail to converge at the rim |

Measured (demanding lens, 160k-px unproject): KB 10 ms, EUCM 3 ms, EUCM⁺ 5 ms, DS⁺ 22 ms;
round-trip all ≈ 5e-13 px or better (machine-precision class). **Correction worth internalizing:** KB's Newton converges to
*machine precision* (≈6e-14), so "iterative" does **not** mean "approximate" — it means
*variable-latency and occasionally non-convergent*, which is the real reason to prefer a
closed form on a GPU or in an autodiff graph, not accuracy.

---

## F. TASK WEIGHT — where does my sampling put the error?

**Question.** A least-squares fit minimizes *average* error under whatever sampling you chose.
Uniform pixels? Uniform angles? They are **not** the same, and the difference decides which
part of the lens your model gets right.

**How.** For pixel-uniform sampling the implicit angular weight is `w(θ) = r(θ)·r'(θ)` (the
image area per unit field angle — derivation in
[the conversion deep-dive](are_two_models_the_same_camera.md)). For a fisheye that's
periphery-heavy:

```
inner half-FOV (θ < 40°) carries 24% of the fit weight;  outer half carries 76%.
```

**Read it.** Sample where your *task* lives. Calibrating for SLAM that tracks features near
the center? An unweighted full-image fit will sacrifice the center to chase the rim. Match the
sampling FOV to the working FOV (`convert`'s `max_fov_deg`) — this single knob is the
difference between a model that's "1–2 px off centrally" and one that's 0.005 px on the data
you care about.

---

## Putting it together: a decision procedure

Run the six in order; each one's answer tells you whether the next even matters:

1. **A (capacity, over *your* FOV).** RMS ≫ 0.1 px → reject; the family can't bend like your
   lens. (EUCM on the demanding lens.)
2. **D (constraints).** Borderline capacity? Relax bounds and re-check before blaming the
   family.
3. **B (identifiability) + C (redundancy).** Among models that fit, prefer the one whose
   parameters are *all* pinned (low cond, no near-null singular value) and whose every term
   adds real DOF (high residual-outside-span). A model that ties on capacity but has a near-
   null parameter is **over-parameterized** — it won't transfer and it slows the solver.
4. **E (cost) + F (task weight).** Tie-break on the operations your pipeline runs most, and
   make sure your fit's sampling matches your working FOV.

The output you should be able to defend is not "I used model X because it's standard," but
"on *my* lens over *my* FOV, X has the capacity (A: 0.09 px), all parameters identifiable
(B: cond 7e3), no redundant term (C: 5% min residual), and the cheapest unproject my task can
use (E)." That sentence is a *measurement*, and it's the whole point of this curriculum.

## Try it yourself
1. Add **your** lens: build a KB (or any) model from your calibration and pass it as the
   ground truth. Which candidates clear diagnostic A over your real working FOV?
2. In diagnostic A, sweep `max_fov_deg` from your inner field out to the full FOV and watch
   EUCM's RMS climb. At what angle does it cross your error budget? That angle is the honest
   edge of "EUCM is enough."
3. Add **UCM** (1 shape param) to the candidate list. Diagnostic B should show it perfectly
   conditioned but A should show it failing both lenses — the cleanest "well-conditioned but
   no capacity" example.

**See it applied, with an honest good-and-bad verdict:**
[A fair fight — EUCM⁺ vs DS⁺ vs Kannala-Brandt](case_study_eucmplus_dsplus_kb.md).
**Related:** [Are two models the same camera?](are_two_models_the_same_camera.md) ·
[Robust losses & honest evaluation](robust_losses_and_evaluation.md).
