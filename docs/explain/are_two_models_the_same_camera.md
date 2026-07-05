# Are two different camera models the *same camera*?

/// tip | Run alongside this
`pip install -e .[calib]` then `bash scripts/download_datasets.sh tumvi`, then
`python examples/05_model_equivalence.py`. Read this, then read the printed numbers.
///

In the [capstone](../learn/capstone_calibrating_a_real_camera.md) we calibrated the *same*
<abbr title="Technical University of Munich Visual-Inertial">TUM-VI</abbr> lens two ways and
got parameters that look nothing alike:

| model | fx | fy | cx | cy | shape parameters |
|---|---|---|---|---|---|
| Kannala-Brandt | 190.990 | 190.974 | 254.955 | 256.841 | `k=[0.0067, -0.0052, 0.0019, -0.0006]` |
| Double Sphere | 248.513 | 248.492 | 254.950 | 256.843 | `xi=0.3008  alpha=0.7191` |

The focal lengths differ by **58 pixels (30%)**. Either one calibration is wrong, or
something subtler is going on.

This page settles it — with a derivation and measured numbers, not hand-waving. The
punchline: **they are the same camera where it was measured**, and provably so, once you
compare the right things.

## 1. A camera is a radial profile, not a parameter vector

Both lenses here are central and (very nearly) radially symmetric: `fx≈fy`, and the
principal points agree to a fraction of a pixel.

For such a camera *all* the physics lives in one 1-D curve — the **radial profile**
`r(θ)`: a ray arriving at angle `θ` off the optical axis lands at distance `r` from the
principal point.

Project is just "wrap `r(θ)` around the optical center"; unproject is its inverse.

So the real question isn't "do the parameter vectors match" — they're just two coordinate
systems. It's **do the two `r(θ)` curves coincide?** Parameters are coordinates; the camera
is the curve.

## 2. The focal mystery, solved: `fx` is model-relative

Near the optical axis every reasonable model is locally linear: `r(θ) ≈ f_eff · θ` for
small `θ`. That slope `f_eff = dr/dθ|₀` is the **paraxial focal length** — the honest,
model-independent focal. Let's compute it for each model.

#### Kannala-Brandt: `fx` already *is* the paraxial focal

Kannala-Brandt defines the profile directly as a polynomial in the angle:

$$r(\theta) = f_{KB} \left(\theta + k_1\theta^3 + k_2\theta^5 + k_3\theta^7 + k_4\theta^9\right)$$

Differentiating at `θ = 0` kills every higher-order term, since each one carries a positive
power of `θ`:

$$\left.\frac{dr}{d\theta}\right|_0 = f_{KB}$$

#### Double Sphere: `fx` is *not* the paraxial focal

Double Sphere builds the profile geometrically (see
[Chapter 2](../learn/02_double_sphere_model.md)). For a unit ray `(sinθ, 0, cosθ)` the
projection in [`ds_math.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/models/ds_math.py#L20-L51)
gives, with `d1 = 1`:

$$z_1 = \cos\theta + \xi$$

$$d_2 = \sqrt{\sin^2\theta + (\cos\theta+\xi)^2} = \sqrt{1 + 2\xi\cos\theta + \xi^2}$$

$$\mathrm{den} = \alpha\, d_2 + (1-\alpha)\, z_1$$

$$r(\theta) = f_{DS} \cdot \frac{\sin\theta}{\mathrm{den}}$$

Now take `θ → 0` (`sinθ → θ`, `cosθ → 1`):

$$d_2 \to \sqrt{1 + 2\xi + \xi^2} = \sqrt{(1+\xi)^2} = 1 + \xi$$

$$\mathrm{den} \to \alpha(1+\xi) + (1-\alpha)(1+\xi) = 1 + \xi$$

$$r(\theta) \to f_{DS}\, \frac{\theta}{1+\xi}$$

$$\left.\frac{dr}{d\theta}\right|_0 = \frac{f_{DS}}{1+\xi}$$

**So in Double Sphere, `fx` is *not* the focal length — `fx/(1+ξ)` is.**

The bundled fixture checks this general identity directly: finite-differencing the real
`DoubleSphereModel.project` at a tiny angle off-axis reproduces `fx/(1+ξ)` to
**1.56e-09 px** — confirming the derivation, not the specific TUM-VI numbers below (those
need the external dataset; see the box above).

{* docs_src/explain/are_two_models_the_same_camera/paraxial_focal_check.py *}

<!-- termynal -->
```
$ python -m docs_src.explain.are_two_models_the_same_camera.paraxial_focal_check
fx=711.5745  xi=0.1832
closed form   fx/(1+xi)        = 601.392276
finite diff   radius(h)/h      = 601.392276   (h=1e-05 rad)
difference                     = 1.56e-09 px
```

Now plug in the calibrated TUM-VI numbers from the top of the page:

| | paraxial focal `dr/dθ|₀` |
|---|---|
| KB | `fx_KB` = **190.990** |
| DS | `fx_DS/(1+ξ)` = 248.513 / 1.3008 = **191.045** |

**0.056 px apart — 0.03%.** The 30% gap in the raw `fx` was an illusion of reading a
model-relative number literally.

## 3. Do the full maps agree? (measured, across the field)

The paraxial match only covers `θ→0`. To compare the *whole* lens, push identical rays and
pixels through both calibrated models. That comparison needs the two full TUM-VI fits, so
it runs in
[`examples/05_model_equivalence.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/examples/05_model_equivalence.py),
trimmed here to its two measurement loops:

```python
import numpy as np

# kb, ds: KannalaBrandtModel / DoubleSphereModel, both calibrated on the same corners
# W, H: the frame resolution

# PROJECT — pixel distance between the two images of the same ray, at each field angle
for deg in [0, 15, 30, 45, 60, 75, 90]:
    th = np.deg2rad(deg)
    phis = np.linspace(0, 2 * np.pi, 60, endpoint=False)          # sweep azimuth at this angle
    dirs = np.stack([np.sin(th) * np.cos(phis), np.sin(th) * np.sin(phis),
                     np.cos(th) * np.ones_like(phis)], axis=1)    # (60, 3) unit rays
    ukb, vk = kb.project(dirs)
    uds, vd = ds.project(dirs)
    ok = vk & vd
    d = np.linalg.norm(ukb[ok] - uds[ok], axis=1)                 # per-ray pixel gap

# UNPROJECT — angle between the KB-ray and DS-ray, over a grid of pixels
ys, xs = np.mgrid[8:H:16, 8:W:16]                                  # every 16th pixel
px = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)     # (N, 2) pixel grid
rk, ok1 = kb.unproject(px)
rd, ok2 = ds.unproject(px)
ok = ok1 & ok2
ang = np.rad2deg(np.arccos(np.clip(np.sum(rk[ok] * rd[ok], axis=1), -1, 1)))
```

<!-- termynal -->
```
$ python examples/05_model_equivalence.py --stride 6
PROJECT — pixel distance between the two images of the same ray
   θ(deg)     mean Δpx     max Δpx
       0        0.006        0.006
      15        0.011        0.015
      30        0.008        0.012
      45        0.011        0.016
      60        0.019        0.024
      75        0.006        0.007
      90        0.499        0.504

UNPROJECT — angle between the KB-ray and DS-ray over 1020 pixels
   median = 0.0038°   mean = 0.0655°   max = 3.405°
```

At 45°, the models are still sub-0.02 px apart. Out to ~75° they agree to
**better than 0.03 px** — an order of magnitude tighter than the ~0.08 px calibration
residual itself.

/// note
Today's multi-scale AprilGrid detection recovers far more of the periphery than when this
comparison was first measured, which is why the agreement now extends further out than it
once did.
///

In that region the two models are, for any practical purpose, the identical map. Only at
the 90° rim do they still part ways — a ray parallel to the image plane, which a lens like
this barely if ever observes directly.

And each model is internally exact — `project(unproject(·))` round-trips to **1e-13 px**
(machine precision) for both. Neither is "broken": they're each self-consistent maps that
happen to disagree at the edges.

## 4. Why the rim diverges — and why it's not a contradiction

Look at where the calibration board actually was:

<!-- termynal -->
```
$ python examples/05_model_equivalence.py --stride 6
# (excerpt -- part 3 of the full run)
field angle of detected corners: median=42°  p95=70°  max=86°
73% of corners are within 55° — today's multi-scale detector reaches well past that.
```

The two models agree to sub-0.03 px all the way out to 75° — matching how far the detected
corners actually reach (p95 70°). Only at the 90° rim, past essentially every corner this
lens ever measured, do they still part ways.

Both **extrapolate with zero constraints** there, and they extrapolate differently by
construction:

- KB's `k₄θ⁹` term grows explosively past the fitted range.
- DS's geometric profile cannot follow that growth at all.

The 0.5 px gap at 90° isn't two models disagreeing about a measured fact — it's two models
*guessing* about the sliver of the field neither one ever saw.

/// tip | The capstone's recurring lesson, made quantitative
A calibration is trustworthy only inside its data. Better peripheral detection directly
shrinks how much of the field is left to guess about.
///

## Verdict

- **DISPROVEN — they are not bit-exact identical maps.** Double Sphere and Kannala-Brandt
  are different function families. There is no exact reparametrization from one to the
  other, and they still differ right at the 90° rim, past any ray this lens's calibration
  data ever reached. "Matches exactly everywhere" is false.
- **PROVEN — they represent the same camera over the field that was calibrated.** The
  paraxial focal agrees to 0.03%; projection agrees to under 0.03 px out to 75° and
  unprojection to a 0.004° median — all well *below* the calibration's own ~0.08 px
  residual. The differing parameter vectors are just two coordinate systems for one set of
  optics.

**The takeaway that generalizes:** never compare cameras by their parameters — `fx`, `ξ`,
the `k`'s mean different things in different models.

Compare them by **behavior** instead: the `r(θ)` curve, or directly the reprojection error
of one model's rays through the other. Two calibrations are "the same camera" exactly as far
as their data reached, and no farther.

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
