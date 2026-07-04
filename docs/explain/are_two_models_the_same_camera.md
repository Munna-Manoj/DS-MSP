# Are two different camera models the *same camera*?

> **Run alongside this:** `python examples/05_model_equivalence.py`
> (after `pip install -e .[calib]` and the TUM-VI download). Read this, then read the
> printed numbers.

In the [capstone](../learn/capstone_calibrating_a_real_camera.md) we calibrated the *same* TUM-VI
lens two ways and got parameters that look nothing alike:

```
KB:  fx=190.990  fy=190.974  cx=254.955  cy=256.841  k=[0.0067, -0.0052, 0.0019, -0.0006]
DS:  fx=248.513  fy=248.492  cx=254.950  cy=256.843  xi=0.3008  alpha=0.7191
```

The focal lengths differ by **58 pixels (30%)**. Either one calibration is wrong, or
something subtler is going on. This page settles it — with a derivation and measured
numbers, not hand-waving. The punchline: **they are the same camera where it was measured,
and provably so, once you compare the right things.**

## 1. A camera is a radial profile, not a parameter vector

Both lenses here are central and (very nearly) radially symmetric: `fx≈fy`, and the
principal points agree to a fraction of a pixel. For such a camera *all* the physics lives
in one 1-D curve — the **radial profile** `r(θ)`: a ray arriving at angle `θ` off the
optical axis lands at distance `r` from the principal point. Project is just "wrap `r(θ)`
around the optical center"; unproject is its inverse.

So the real question isn't "do the parameter vectors match" (they're just two coordinate
systems). It's **do the two `r(θ)` curves coincide?** Parameters are coordinates; the camera
is the curve.

## 2. The focal mystery, solved: `fx` is model-relative

Near the optical axis every reasonable model is locally linear: `r(θ) ≈ f_eff · θ` for small
`θ`. That slope `f_eff = dr/dθ|₀` is the **paraxial focal length** — the honest,
model-independent focal. Let's compute it for each model.

**Kannala-Brandt** defines the profile directly as a polynomial in the angle:
```
r(θ) = fx_KB · (θ + k₁θ³ + k₂θ⁵ + k₃θ⁷ + k₄θ⁹)
⇒   dr/dθ|₀ = fx_KB
```
Here `fx` *is* the paraxial focal. Easy.

**Double Sphere** builds the profile geometrically (see [Chapter 2](../learn/02_double_sphere_model.md)).
For a unit ray `(sinθ, 0, cosθ)` the projection in [`ds_math.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/models/ds_math.py#L20-L51)
gives, with `d1 = 1`:
```
z₁  = cosθ + ξ
d₂  = √(sin²θ + (cosθ+ξ)²) = √(1 + 2ξcosθ + ξ²)
den = α·d₂ + (1−α)·z₁
r(θ) = fx_DS · sinθ / den
```
Now take θ → 0 (`sinθ → θ`, `cosθ → 1`):
```
d₂  → √(1 + 2ξ + ξ²) = √((1+ξ)²) = 1 + ξ
den → α(1+ξ) + (1−α)(1+ξ) = 1 + ξ
⇒   r(θ) → fx_DS · θ / (1+ξ)
⇒   dr/dθ|₀ = fx_DS / (1 + ξ)
```

**So in Double Sphere, `fx` is *not* the focal length — `fx/(1+ξ)` is.** Plug in the
calibrated numbers:

| | paraxial focal `dr/dθ|₀` |
|---|---|
| KB | `fx_KB` = **190.990** |
| DS | `fx_DS/(1+ξ)` = 248.513 / 1.3008 = **191.045** |

**0.056 px apart — 0.03%.** The 30% gap in the raw `fx` was an illusion of reading a
model-relative number literally. The example confirms the formula by finite-differencing
each model's radius at the axis: KB 190.990, DS 191.045 — same to the digit.

## 3. Do the full maps agree? (measured, across the field)

The paraxial match only covers `θ→0`. To compare the *whole* lens, push identical rays and
pixels through both calibrated models:

```
PROJECT — pixel distance between the two images of the same ray
   θ(deg)     mean Δpx     max Δpx
       0        0.006        0.006
      15        0.011        0.015
      30        0.008        0.012
      45        0.011        0.016      <- still sub-0.02 px out here
      60        0.019        0.024
      75        0.006        0.007
      90        0.499        0.504      <- still where they part ways, just less dramatically

UNPROJECT — angle between the KB-ray and DS-ray over 1020 pixels
   median = 0.0038°   mean = 0.0655°   max = 3.405°
```

Out to ~75° the two models now agree to **better than 0.03 px** — an order of magnitude
tighter than the ~0.08 px calibration residual itself (today's multi-scale AprilGrid
detection recovers far more of the periphery than when this comparison was first measured).
In that region they are, for any practical purpose, the identical map. Only right at the
90° rim — a ray parallel to the image plane, which a lens like this barely if ever
observes directly — do the two models still part ways.

And each model is internally exact — `project(unproject(·))` round-trips to **1e-13 px**
(machine precision) for both. So neither is "broken"; they're each self-consistent maps
that happen to disagree at the edges.

## 4. Why the rim diverges — and why it's not a contradiction

Look at where the calibration board actually was:

```
field angle of detected corners:  median 42°,  p95 70°,  max 86°
73% of corners are within 55° — today's multi-scale detector reaches well past that.
```

The two models now agree to sub-0.03 px all the way out to 75° — matching how far the
detected corners actually reach (p95 70°). Only right at the 90° rim, past essentially
every corner this lens ever measured, do the models still part ways: both **extrapolate
with zero constraints** there, and they extrapolate differently by construction — KB's
`k₄θ⁹` term in particular grows explosively, DS's geometric profile cannot follow it. The
0.5-px gap at 90° isn't two models disagreeing about a measured fact; it's two models
*guessing* about the sliver of the field neither one ever saw. (This is the capstone's
recurring lesson, made quantitative: a calibration is trustworthy only inside its data —
and better peripheral detection directly shrinks how much of the field is left to guess
about.)

## Verdict

- **DISPROVEN — they are not bit-exact identical maps.** Double Sphere and Kannala-Brandt
  are different function families. There is no exact reparametrization from one to the
  other, and they still differ right at the 90° rim, past any ray this lens's calibration
  data ever reached. "Matches exactly everywhere" is false.
- **PROVEN — they represent the same camera over the field that was calibrated.** The
  paraxial focal agrees to 0.03%; projection agrees to < 0.03 px out to 75° and
  unprojection to a 0.004° median — all well *below* the calibration's own ~0.08 px
  residual. The differing parameter vectors are just two coordinate systems for one set
  of optics.

**The takeaway that generalizes:** never compare cameras by their parameters — `fx`, `ξ`,
the `k`'s mean different things in different models. Compare them by **behavior**: the
`r(θ)` curve, or directly the reprojection error of one model's rays through the other.
Two calibrations are "the same camera" exactly as far as their data reached, and no
farther.

## Try it yourself
1. In the example, also print `fy_DS/(1+ξ)` vs `fy_KB`. Does the vertical paraxial focal
   match too? (It should — same derivation, `y` instead of `x`.)
2. Restrict the project-agreement loop to `θ ≤ 55°` (the data boundary) and report a single
   max Δpx. That one number is the honest "are they the same camera" answer.
3. Re-run the capstone with `--stride 2` so more wide-angle corners are included, then redo
   this comparison. Does the agreement extend to larger θ as the data reaches further out?

**Next:** if two models can describe the same camera, *which one should you actually pick?*
That's a measurable question — see
[Is this model right for my lens?](choosing_a_camera_model.md) and the worked
[EUCM⁺ vs DS⁺ vs KB case study](case_study_eucmplus_dsplus_kb.md).
**Back to:** the [capstone](../learn/capstone_calibrating_a_real_camera.md), or the
[robust-loss deep-dive](../learn/robust_losses_and_evaluation.md).
