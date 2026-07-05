# Chapter 2 — The Double Sphere model, from first principles

> **Run alongside this:** `python examples/02_double_sphere_tumvi.py`
> (after the [setup](README.md#setup-once)). Read this, then read the printed numbers.

In [Chapter 1](01_fisheye_and_camera_models.md) a camera model was a black box: a
`project`/`unproject` pair that happened to be inverses. This chapter opens one specific
box — **Double Sphere** (Usenko, Demmel & Cremers, 3DV 2018).

Its math is short, geometric, and exactly invertible. By the end you'll read
[`ds_math.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/models/ds_math.py) and
recognize every line.

![A 3D point projected through both spheres to a fisheye pixel](https://raw.githubusercontent.com/Munna-Manoj/DS-MSP/main/assets/learn/double_sphere_pipeline.gif)

*Follow the bright point as it travels **3D point → sphere 1 → sphere 2 → α-centre**, while a
colourful world of directions fills in the image.*

*The same projection ray meets **two equivalent image planes** — the model's normalized
**z = 1 plane** (virtual, upright) and the **physical sensor** behind both spheres (real,
inverted). Every coloured pixel is the exact `ds_project` of its 3D direction — even the ones
past 90°, which a normal camera cannot capture.*

The model is radially symmetric, so a 2-D cross-section is the complete picture — the same
construction with both image planes labelled:

![Double Sphere 2D cross-section — ray to sphere 1, shift to sphere 2, projection onto the z=1 plane and the inverted physical sensor](https://raw.githubusercontent.com/Munna-Manoj/DS-MSP/main/assets/learn/double_sphere_2d.gif)

*The two spheres sit between the 3-D world (right) and the sensor (left, behind the α-centre),
matching the paper's figure; the z = 1 plane in front carries the equivalent upright image.*

*The sections below dissect each step.*

**You'll learn**
- Derive Double Sphere's projection as two sequential unit-sphere projections — shifted by
  `ξ` and blended by `α` — and read it directly in
  [`ds_math.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/models/ds_math.py#L20-L51).
- Why Double Sphere unprojects in **closed form** (one square root, no iteration), unlike
  Kannala-Brandt's polynomial, which needs Newton's method.
- Verify projection and its closed-form inverse are exact to machine precision
  (round-trip mean 2.17e-14 px, max 1.17e-13 px over 1600 real pixels).
- Use [`convert()`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/adapt/convert.py#L96-L182) to re-express TUM-VI's published
  Kannala-Brandt calibration as Double Sphere, matching it to **0.025 px** max reprojection
  error over 179.8° of field of view.

**Prerequisites**
- Finish [Chapter 1](01_fisheye_and_camera_models.md) — this chapter assumes `project`/
  `unproject` and the `CameraModel` contract are already familiar.
- Same [setup](README.md#setup-once) as Chapter 1; no new installs.

## 1. Why another model after Kannala-Brandt?

Kannala-Brandt (Chapter 1's model) describes the lens by a polynomial in the incidence
angle θ: `r(θ) = θ + k1·θ³ + k2·θ⁵ + …`. It fits well, but it has a practical wart:
**unprojection has no closed form.**

To go from a pixel back to a ray you must invert that polynomial numerically (Newton
iterations) for *every pixel, every frame*. In a
<abbr title="Visual Odometry / Simultaneous Localization and Mapping — real-time pose and map estimation from a camera feed.">VO/SLAM</abbr>
front-end unprojecting thousands of features per image, that adds up.

Double Sphere was designed to fix exactly this: it matches fisheye lenses as well as KB
while keeping **both** projection and unprojection in closed form — no iteration, no
root-finding.

That single property is why it shows up in modern visual-inertial systems (it's the model
behind Basalt). The whole point of this chapter is to see *why* it inverts cleanly.

## 2. The geometric picture: two spheres

Pinhole projection divides by `Z`. That explodes as a ray approaches 90° (`Z → 0`).

The fix every wide-<abbr title="Field Of View — the angular extent of the scene a lens captures.">FOV</abbr>
model uses is the same idea: **first map the ray onto a unit sphere (where nothing explodes),
then do a perspective division from a shifted center.** Models differ only in *where* that
second center sits.

Double Sphere uses *two* unit spheres in sequence, governed by two new numbers:

#### The three-step construction

1. **Project the 3D point onto a first unit sphere** — just normalize it. Now every
   direction, even one 100° off-axis, is a finite point on a sphere.
2. **Shift by `ξ` (xi) and project onto a second unit sphere.** `ξ` is the gap between the
   two sphere centers. This second bending is what lets the model curve enough for real
   fisheye glass.
3. **Pinhole-project from a center blended by `α` (alpha).** `α` slides the projection
   center between "the second sphere's center" (`α=1`) and "one sphere-radius behind it"
   (`α=0`). It controls how much perspective foreshortening remains.

So Double Sphere = pinhole + two shape knobs: **`ξ` (sphere spacing)** and **`α` (which
center you project from)**. Everything else (`fx, fy, cx, cy`) is the ordinary intrinsic
matrix you already know.

![The Double Sphere two-sphere projection](https://raw.githubusercontent.com/Munna-Manoj/DS-MSP/main/assets/learn/double_sphere_projection.gif)

*The construction in cross-section (the model is radially symmetric, so this slice is the
whole story): an incoming ray (green) lands on the **first** unit sphere, is **shifted by
`ξ`** onto the **second** sphere (orange), then **projected from the `α`-blended centre**
onto the image plane — a pixel (pink).*

*The shaded wedge is the valid field of view; watch `θ` climb **past 90°** and still land
inside it — the >180° reach a pinhole can never have.*

*(Rendered from the exact `ds_project` geometry — every point matches the library to ~1e-16.)*

## 3. Read the projection in code

Here is the entire forward map from
[`ds_math.py`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/models/ds_math.py#L26-L49) — six lines of real
arithmetic, run below on a real detected corner (`test_config.json`) pushed out to an
arbitrary depth along its own ray:

{* docs_src/learn/double_sphere_model/forward_projection.py hl[29:34] *}

<!-- termynal -->
```
$ python -m docs_src.learn.double_sphere_model.forward_projection
3D point (camera frame, metres): x=-1.2837 y=-0.5700 z=1.8213
d1=2.3000  z1=2.2427  d2=2.6462  den=2.5690
u=593.6100  v=361.0100  (matches the original detected corner 593.61, 361.01)
ds_project() agrees:  u=593.6100  v=361.0100  valid=True
```

#### Match the code to the geometry

- `d1` normalizes onto **sphere 1**.
- `z1 = z + ξ·d1` is the **ξ shift** — it pushes the point's `z` toward the second sphere's
  center. With `ξ = 0` this line vanishes and the two spheres collapse into one (Double
  Sphere degenerates to the **Unified Camera Model**, UCM).
- `den = α·d2 + (1−α)·z1` is the **α blend** of the two possible denominators. With
  `α = 0` you divide by `z1` (pure UCM-style); with `α = 1` you divide by `d2`. Real
  fisheyes land in between — TUM-VI's is `α ≈ 0.71` (the example prints it).
- The last two lines are the pinhole division you've seen a hundred times, just with `den`
  in place of `Z`.

That's the whole model — two extra scalars on top of a pinhole. The point above lands on the
*same* pixel whether it sits 1 m or 100 m along that ray: only direction matters, the
hallmark of a central camera.

## 4. Why it inverts in closed form

The reason Double Sphere unprojects without iteration: the forward map is a composition of
a normalization and a *quadratic* perspective step. A quadratic can be solved with a square
root rather than Newton's method.

You can see the solved result directly in
[`ds_unproject`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/models/ds_math.py#L139-L155), walked below
for the same real corner and then cross-checked as a full round trip over all 30 bundled
corners:

{* docs_src/learn/double_sphere_model/closed_form_unprojection.py hl[24:28] *}

<!-- termynal -->
```
$ python -m docs_src.learn.double_sphere_model.closed_form_unprojection
mx=-0.4997  my=-0.2219  r2=0.2989  mz=0.8730
pixels tested: 30 / 30
project(unproject(u)) round-trip: mean=1.89e-14px  max=1.14e-13px
```

No loop. That `np.sqrt` is the analytic inverse of the quadratic in §3. `1e-14 px` is
float64's last bit — this isn't "close enough", it's *the model is its own exact inverse*.

/// tip
The same precision holds at full scale. `examples/02_double_sphere_tumvi.py` runs the
identical round trip over 1600 real TUM-VI pixels:

<!-- termynal -->
```
$ python examples/02_double_sphere_tumvi.py
# (excerpt -- part 2 of the full run)
pixels tested: 1600 / 1600 (rest fall outside the lens circle)
project(unproject(u)) round-trip: mean=2.17e-14px  max=1.17e-13px
```
///

Contrast Chapter 1's verify-don't-trust habit: this is proof, not a plausibility argument.

/// note
The `s ≥ 0` and `sqrt_arg ≥ 0` checks in the code are where rays that the lens physically
can't see get flagged invalid — that's [Chapter 3](03_projection_validity.md).
///

## 5. Is Double Sphere expressive enough for a *real* lens?

A model that inverts cleanly is useless if it can't actually fit real glass. TUM-VI's
authors calibrated their fisheye and published it as a Kannala-Brandt model.

Can a Double Sphere model describe the **same** camera? The example re-expresses it with the
library's own [`convert()`](https://github.com/Munna-Manoj/DS-MSP/blob/main/ds_msp/adapt/convert.py#L96-L182) —
sample pixels, unproject through the reference, seed, then Levenberg-Marquardt refine with
the model's *analytic* Jacobian.

#### Measuring agreement across the frame

<!-- termynal -->
```
$ python examples/02_double_sphere_tumvi.py
Recovered Double Sphere: fx=240.178 fy=240.172 cx=254.932 cy=256.897 xi=0.2584 alpha=0.7110

Evaluated over 1879 rays spanning 179.8 deg of field of view:
    RMS  reprojection error : 0.0106 px
    max  reprojection error : 0.0249 px
```

**0.025 px maximum disagreement across a ~180° field** — Double Sphere has the expressive
power to capture this lens to a fortieth of a pixel.

/// note
Why does `fx` change from 191 to 240? Focal length is *model-relative* — the same lens has a
different `fx` under KB vs DS because the denominators differ. The true *paraxial* (near-axis)
focal is `fx_DS/(1+ξ)`. A whole deep-dive proves this:
**[are two models the same camera?](../explain/are_two_models_the_same_camera.md)**. What's
invariant is where rays land, not the raw numbers.
///

/// tip
**This is model *conversion*, not calibration.** We re-expressed one set of published numbers
as another model's numbers — no images, no board, no detected corners.

Proving the model on *real measurements* is the
**[capstone](capstone_calibrating_a_real_camera.md)**: detect AprilGrid corners in TUM-VI's
raw calibration footage and bundle-adjust intrinsics from scratch that land on the published
reference. Do Chapter 2, then jump to it — it's the artifact everything here builds toward.
///

## Try it yourself
1. In the example, after `convert()`, print `ds.xi` while forcing `xi = 0`
   (`DoubleSphereModel(ds.fx, ds.fy, ds.cx, ds.cy, 0.0, ds.alpha)`) and re-measure the
   reprojection error. How much worse does the single-sphere (UCM) fit get? That gap is
   what the second sphere buys you.
2. Convert to `EUCMModel` and `UCMModel` instead and compare their max reprojection error
   to Double Sphere's. Which models reproduce this lens best?
3. Run the round-trip grid (§4) out to the image corners (`np.linspace(0, W, …)`) and watch
   how many pixels drop out of the valid mask near the edge — a preview of Chapter 3.

**Next:** the **[capstone](capstone_calibrating_a_real_camera.md)** — calibrate this camera
for real from AprilGrid footage and match the published numbers.

Or continue the theory thread with **[Chapter 3](03_projection_validity.md)** — projection
validity and the >180° cone (why `z > 0` is the classic fisheye bug).
