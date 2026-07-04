# Are two different camera models the *same camera*?

> **Run alongside this:** `python examples/05_model_equivalence.py`
> (after `pip install -e .[calib]` and the TUM-VI download). Read this, then read the
> printed numbers.

In the [capstone](../learn/capstone_calibrating_a_real_camera.md) we calibrated the *same* TUM-VI
lens two ways and got parameters that look nothing alike:

```
KB:  fx=192.74  fy=192.71  cx=254.96  cy=256.63   k=[0.0045, -0.0027, -0.0027, 0.0019]
DS:  fx=152.68  fy=152.66  cx=254.99  cy=256.62   xi=-0.2087  alpha=0.5830
```

The focal lengths differ by **40 pixels (26%)**. Either one calibration is wrong, or
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
For a unit ray `(sinθ, 0, cosθ)` the projection in [`ds_math.py`](../../ds_msp/models/ds_math.py)
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
| KB | `fx_KB` = **192.74** |
| DS | `fx_DS/(1+ξ)` = 152.68 / 0.7913 = **192.95** |

**0.21 px apart — 0.11%.** The 26% gap in the raw `fx` was an illusion of reading a
model-relative number literally. The example confirms the formula by finite-differencing
each model's radius at the axis: KB 192.738, DS 192.949 — same to the digit.

## 3. Do the full maps agree? (measured, across the field)

The paraxial match only covers `θ→0`. To compare the *whole* lens, push identical rays and
pixels through both calibrated models:

```
PROJECT — pixel distance between the two images of the same ray
   θ(deg)     mean Δpx     max Δpx
       0        0.034        0.034
      15        0.048        0.074
      30        0.036        0.051
      45        0.034        0.039      <- still sub-0.05 px out here
      60        0.121        0.151
      75        0.769        0.803
      90       10.026       10.062      <- they fly apart at the rim

UNPROJECT — angle between the KB-ray and DS-ray over 1024 pixels
   median = 0.0255°   mean = 0.473°   max = 10.74°
```

Out to ~45° the two models agree to **better than 0.05 px** — *below* the 0.12 px
calibration residual itself. In that region they are, for any practical purpose, the
identical map. Then past ~60° they diverge, hard.

And each model is internally exact — `project(unproject(·))` round-trips to **1e-13 px**
(machine precision) for both. So neither is "broken"; they're each self-consistent maps
that happen to disagree at the edges.

## 4. Why the rim diverges — and why it's not a contradiction

Look at where the calibration board actually was:

```
field angle of detected corners:  median 35°,  p95 62°,  max 84°
88% of corners are within 55° — the periphery was never observed.
```

The divergence in §3 sets in right where the data runs out (~60°). Beyond it both models
**extrapolate with zero constraints**, and they extrapolate differently by construction —
KB's `k₄θ⁹` term in particular grows explosively, DS's geometric profile cannot follow it.
The 10-px gap at 90° isn't two models disagreeing about a measured fact; it's two models
*guessing* about a region neither one ever saw. (This is the capstone's recurring lesson,
made quantitative: a calibration is trustworthy only inside its data.)

## Verdict

- **DISPROVEN — they are not bit-exact identical maps.** Double Sphere and Kannala-Brandt
  are different function families. There is no exact reparametrization from one to the
  other, and they differ by up to 10 px at the extreme periphery. "Matches exactly
  everywhere" is false.
- **PROVEN — they represent the same camera over the field that was calibrated.** The
  paraxial focal agrees to 0.11%; projection agrees to < 0.05 px out to 45° and
  unprojection to a 0.025° median — all *below* the calibration's own residual. The
  differing parameter vectors are just two coordinate systems for one set of optics.

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
