# Robust losses vs hard rejection — and why naive RMS lies

> **Run alongside this:** `python examples/04_robust_vs_rejection.py`
> (after `pip install -e .[calib]` and the TUM-VI download). Read this, then watch the
> three rows of the table move.

Real calibration data has a few bad corners — a peripheral AprilGrid tag on the curved
part of the lens where `cornerSubPix` latches onto the wrong edge, a near-miss decode. The
question this page answers: **what do you do with them, and how do you then measure how
well you did?** Both halves trip people up.

**You'll learn**
- Why plain least-squares (L2) lets a single bad corner drag the whole calibration's focal
  length off by ~5 px — the square in "sum of squared residuals" is the culprit.
- Fight back with robust M-estimation / IRLS instead of two-pass hard rejection, and read
  the weight function `w(r) = ρ'(r)/r` for L2, Huber, and Cauchy losses.
- Run the Cauchy loss (`calibrate(..., loss="cauchy", f_scale=0.5)`) and see it beat both
  L2 and hard rejection on focal accuracy (Δfx 1.29 vs. 4.94 vs. 2.12 px) while keeping
  all 5,180 corners.
- Avoid the evaluation trap: why all-corner RMS makes a *better* robust fit look worse
  (naiveRMS 3.608 px for Cauchy), and score with median (0.115 px) or inlier RMS (0.247 px)
  instead.

**Prerequisites**
- Finish the [capstone](capstone_calibrating_a_real_camera.md) — the [learn README](README.md)
  lists this as a companion to read after it; it puts the capstone's Cauchy-loss step under
  a microscope.
- Same `[calib]` install and TUM-VI dataset as the capstone.

## 1. Why one bad corner wrecks a least-squares fit

Calibration minimizes reprojection error. Plain least squares (L2) minimizes the **sum of
squared** residuals:

```
minimize  Σ_i  r_i²        where  r_i = ‖ project(X_i) − u_i ‖   (pixels)
```

The square is the problem. A corner at `r = 6 px` contributes `36`; a good corner at
`0.2 px` contributes `0.04` — **900× less**. So a handful of outliers shout down thousands
of good corners and pull the parameters (here: focal length) toward themselves. In the
example, L2's focal is off by ~5 px for exactly this reason.

## 2. Two ways to fight back

**Hard rejection** (a two-pass loop): fit once, *delete* every corner with `r > τ`, refit.
It works, but it has two faults — it **throws away data** (a 1.01 px corner that was mostly
fine is gone), and it's **brittle at the threshold** (0.99 px stays, 1.01 px vanishes; move
`τ` slightly and the answer jumps). It's a binary in/out decision applied to a continuous
problem.

**Robust M-estimation** keeps every corner but replaces `r²` with a function `ρ(r)` that
grows more slowly for large `r`:

```
minimize  Σ_i  ρ(r_i)
```

The magic is in the derivative. The optimizer effectively solves a **weighted** least
squares where each corner's weight is

```
w(r) = ρ'(r) / r          (this is Iteratively Reweighted Least Squares, IRLS)
```

Each iteration re-weights every corner by its current residual, so an outlier's pull shrinks
as the fit reveals it to be an outlier:

```mermaid
graph LR
    A["compute residuals<br/>rᵢ = ‖project − observed‖"] --> B["weight each corner<br/>wᵢ = ρ'(rᵢ)/rᵢ"]
    B --> C["weighted least-squares step<br/>(analytic Jacobian)"]
    C --> A
```

`w(r)` is how much that corner is allowed to influence the answer. Compare three choices:

| loss | ρ(r) (small → large r) | weight `w(r)` | what large outliers do |
|---|---|---|---|
| **L2** | `½ r²` | `1` (constant) | full influence — they dominate |
| **Huber** | quadratic, then linear past `δ` | `min(1, δ/|r|)` | **bounded**: capped, but never zero |
| **Cauchy** | `½c²·log(1+(r/c)²)` | `1 / (1 + (r/c)²)` | **redescending**: influence → 0 |

Read the weights as "how loud each corner is allowed to shout":

- **L2**: everyone shouts in proportion to their error — outliers loudest.
- **Huber**: past the scale `δ` a corner's influence stops growing. A 6 px outlier pulls no
  harder than a corner at `δ`. Bounded, never silenced.
- **Cauchy**: influence *rises then falls*. A 6 px outlier with `c = 0.5` gets weight
  `1/(1+144) ≈ 0.007` — effectively muted, **decided by the data, not a hand-set threshold**.
  This is the soft, continuous version of rejection: garbage fades out smoothly instead of
  being clipped at a cliff.

The scale parameter (`f_scale` in SciPy / `ds_msp.calib.calibrate`) is `δ` / `c` — the
residual size where down-weighting begins. Set it near your *inlier* noise: subpixel corner
detection is good to ~0.1–0.3 px, so `f_scale ≈ 0.5 px` treats anything past half a pixel as
increasingly suspect. (Mechanically, `calibrate` just forwards `loss=` / `f_scale=` to
`scipy.optimize.least_squares`; the analytic Jacobian is unchanged — SciPy applies the
reweighting internally.)

## 3. The result, and the evaluation trap

```
method                   corners     Δfx   median  inlierRMS  naiveRMS
L2 (no robustness)     5180/5180    4.94    0.213      0.323     0.827
hard reject >1px       4625/5180    2.12    0.126      0.252     0.883
Cauchy f_scale=0.5     5180/5180    1.29    0.115      0.247     3.608
```

The headline: **Cauchy gets the best focal (Δfx 1.29) and the best median (0.115 px) while
keeping all 5180 corners** — it beats hard rejection *and* discards nothing.

Now look at Cauchy's **`naiveRMS = 3.608`**. By that number alone Cauchy looks like the
*worst* fit — 4× worse than L2. That is the trap, and here's why it's wrong:

**RMS is itself an L2 statistic.** `RMS = √(mean(r²))` squares every residual, so it is
dominated by exactly the large outliers Cauchy *deliberately ignored*. Computing all-corner
RMS on a robust fit doesn't measure "how good is the fit on real data" — it measures **"how
big are the outliers the model chose not to explain."** The better the robust loss is at
spotting garbage and leaving it with a big residual, the *worse* its naive RMS looks. Using
naive RMS to score a robust fit penalizes it for doing its job.

**The honest reads** describe the fit where the data is trustworthy:

- **Median** reprojection error — the 50th percentile literally cannot be moved by a
  minority of outliers (it has a 50% breakdown point). Cauchy's `0.115 px` says half the
  corners are better than an eighth of a pixel.
- **Inlier RMS** — RMS over the corners the model actually explains (here `< 1 px`). Same L2
  units everyone expects, but computed on the set the fit is responsible for: `0.247 px`.
- **Inlier fraction** — what share made the cut (`~89%`). If this were 50% your model or your
  detector is broken, not your loss.

A robust *fit* and a robust *metric* go together: down-weight outliers in the optimization,
then evaluate with a statistic outliers can't hijack. Mixing a robust fit with an L2 metric
is how you talk yourself out of the better calibration.

## Try it yourself
1. Sweep `f_scale` (0.3, 0.5, 1.0, 2.0) in the Cauchy run. As it grows, Cauchy → L2 (nothing
   gets down-weighted) and `Δfx` drifts back up. As it shrinks, more corners are treated as
   suspect. Find where median stops improving — that's your noise scale.
2. Swap `loss="cauchy"` for `"huber"`. Huber bounds but never mutes outliers, so its `Δfx`
   sits between L2 and Cauchy. Confirm it on the numbers.
3. Print `median` *and* `naiveRMS` for the L2 row too. Why are they much closer for L2 than
   for Cauchy? (Hint: L2 didn't leave any residuals deliberately large.)

**Back to:** the **[capstone](capstone_calibrating_a_real_camera.md)**, which uses the Cauchy
fit and median/inlier reporting throughout.
