# Two-view geometry on bearing vectors — derivations, proofs, and numerical notes

Formal companion to [`ds_msp/mvg/two_view.py`](../../ds_msp/mvg/two_view.py). Every
claim here is checked by a named test in [`tests/mvg/test_two_view.py`](../../tests/mvg/test_two_view.py),
so the math and the code can't drift.

## Setup and conventions

Two central cameras observe a 3D point. The relative pose $(R, t)$, $R \in SO(3)$, maps a point
from camera 1 to camera 2:
$$X_2 = R\,X_1 + t .$$
Each camera measures a **unit bearing vector** (a ray), not a pixel: $f_1 = X_1/\lVert X_1\rVert$
and $f_2 = X_2/\lVert X_2\rVert$. This is the only thing a calibrated camera reports, and it is
why everything below is **model-agnostic** — Double Sphere, UCM, KB, … all produce bearing
vectors and share this geometry. Write $\lambda_i = \lVert X_i\rVert > 0$ for the **depth along
the ray**, so $X_1 = \lambda_1 f_1$, $X_2 = \lambda_2 f_2$.

## 1. The epipolar constraint holds on rays

**Claim.** $f_2^\top E\, f_1 = 0$ with $E = [t]_\times R$, where $[t]_\times$ is the skew matrix
with $[t]_\times v = t \times v$.

**Proof.** From $X_2 = R X_1 + t$, substitute $X_i = \lambda_i f_i$:
$$\lambda_2 f_2 = \lambda_1 R f_1 + t .$$
Left-multiply by $f_2^\top [t]_\times$. Since $[t]_\times$ is skew, $f_2^\top [t]_\times f_2 = 0$,
killing the left side:
$$0 = \lambda_1\, f_2^\top [t]_\times R\, f_1 + f_2^\top [t]_\times t .$$
And $[t]_\times t = t \times t = 0$, so the last term vanishes. With $\lambda_1 \neq 0$,
$$f_2^\top \underbrace{[t]_\times R}_{E}\, f_1 = 0. \qquad\blacksquare$$
Geometrically: $f_2$, $t$, and $R f_1$ are coplanar (all lie in the epipolar plane), and the
scalar triple product $f_2 \cdot (t \times R f_1) = 0$ states exactly that.

*Tests:* `test_epipolar_constraint_holds_on_rays`, `test_essential_matches_skew_t_times_R_up_to_sign`.

## 2. Properties of the essential matrix

**Rank and singular values.** $[t]_\times$ has singular values $\{\lVert t\rVert, \lVert t\rVert, 0\}$
(it is skew, rank 2). Right-multiplying by the orthogonal $R$ preserves singular values, so
$E = [t]_\times R$ has singular values $\{\lVert t\rVert, \lVert t\rVert, 0\}$ — **two equal,
one zero**, hence $\operatorname{rank} E = 2$ and $\det E = 0$. The code normalizes to
$(1, 1, 0)$ since scale is unrecoverable.

*Tests:* `test_essential_singular_values_are_one_one_zero`, `test_essential_determinant_is_zero`.

**Algebraic characterization (Huang–Faugeras).** A real $3\times3$ matrix is essential **iff**
$$2\,E E^\top E - \operatorname{tr}(E E^\top)\,E = 0 .$$
This is the polynomial form of "two equal singular values and one zero": with the SVD
$E = U\,\mathrm{diag}(a,b,c)\,V^\top$ the equation reduces to $a=b,\ c=0$ (up to permutation). It
is the constraint a minimal 5-point solver enforces directly.

*Test:* `test_essential_satisfies_characterization_equation`.

**Scale invariance.** $E$ depends only on ray *directions*; scaling any $f_i$ leaves it
unchanged, and $E$ itself is defined only up to scale (and sign).

*Test:* `test_essential_is_scale_invariant_in_the_rays`.

## 3. The eight-point algorithm and the manifold projection

Each correspondence gives one linear equation in the 9 entries of $E$. Vectorize row-major,
$e = \operatorname{vec}(E)$; then $f_2^\top E f_1 = (f_2 \otimes f_1)^\top e$, so stacking $N\ge 8$
rows gives $A e = 0$, $A_i = f_2^{(i)} \otimes f_1^{(i)} \in \mathbb{R}^9$. The least-squares
solution under $\lVert e\rVert = 1$ is the **right-singular vector of $A$ with the smallest
singular value** (SVD), reshaped to $3\times3$.

That raw $\hat E$ does not generally satisfy the rank/equal-singular-value structure. We project
onto the essential manifold: with $\hat E = U\,\mathrm{diag}(\sigma_1,\sigma_2,\sigma_3)\,V^\top$,
replace the singular values by $(1,1,0)$:
$$E = U\,\mathrm{diag}(1,1,0)\,V^\top .$$
**Optimality (Eckart–Young).** Among all matrices with singular values $(\bar\sigma,\bar\sigma,0)$,
the Frobenius-nearest to $\hat E$ takes $\bar\sigma = (\sigma_1+\sigma_2)/2$ and the same singular
vectors; the third singular value drops to 0 because that is the nearest rank-2 matrix. Fixing
$\bar\sigma = 1$ only rescales (scale is free). So the projection is the optimal essential-matrix
correction of the noisy linear estimate.

*Tests:* `test_recover_pose_matches_ground_truth`, and `test_too_few_correspondences_raises`
(the $N\ge8$ guard).

## 4. Decomposition into four $(R, t)$ candidates

Let $E = U\,\mathrm{diag}(1,1,0)\,V^\top$ with $\det U = \det V = +1$ (flip signs otherwise so the
results are rotations, not reflections). With
$W = \begin{pmatrix}0&-1&0\\1&0&0\\0&0&1\end{pmatrix}$,
$$R \in \{\,U W V^\top,\; U W^\top V^\top\,\}, \qquad t = \pm\,U_{:,3}\ (\text{unit}).$$
This yields **four** combinations. They are the genuine reconstructions plus their *twisted
pairs* (a 180° rotation about the baseline and a reflected reconstruction); exactly one places
the scene in front of both cameras.

*Test:* `test_decompose_gives_four_proper_rotations` (each $R$ orthogonal, $\det = +1$, $t$ unit).

## 5. Cheirality — and why `z > 0` is the wrong test for wide-FOV

To pick the physical solution, triangulate each correspondence and keep the candidate whose
points lie **in front of both cameras**. For a pinhole one writes $z > 0$. For a fisheye that is
**wrong**: a ray past $90°$ has $z \le 0$ yet is a perfectly valid observation (this is the same
>180° point made in [Chapter 3](../learn/03_projection_validity.md)). The correct, model-free
test is **positive depth *along the bearing vector***:
$$\lambda_1 > 0 \quad\text{and}\quad \lambda_2 > 0,$$
where $\lambda_i$ are the triangulated depths of §6. `recover_pose` counts correspondences
satisfying this and takes the maximum — robust to a few mistriangulated points.

*Tests:* `test_recover_pose_matches_ground_truth` (selects the correct sign: $\hat t \cdot t > 0$),
`test_all_triangulated_points_are_in_front`.

## 6. Midpoint triangulation

Given $(R, t)$, express both rays in **camera-1** coordinates. Ray 1 is the line
$P(\lambda_1) = \lambda_1 f_1$ (through the origin). Camera 2's centre in camera-1 frame is
$c_2 = -R^\top t$ (from $X_2 = 0$), and ray 2's direction is $d_2 = R^\top f_2$, giving the line
$Q(\lambda_2) = c_2 + \lambda_2 d_2$. The point closest to both lines minimizes
$\lVert P(\lambda_1) - Q(\lambda_2)\rVert^2$. Setting the gradient to zero gives the standard
$2\times2$ system (with $f_1, d_2$ unit, so $f_1\!\cdot\!f_1 = d_2\!\cdot\!d_2 = 1$):
$$b = f_1\!\cdot d_2,\quad d = f_1\!\cdot w_0,\quad e = d_2\!\cdot w_0,\quad w_0 = -c_2,$$
$$\lambda_1 = \frac{b e - d}{1 - b^2}, \qquad \lambda_2 = \frac{e - b d}{1 - b^2},$$
and $X = \tfrac12\big(\lambda_1 f_1 + c_2 + \lambda_2 d_2\big)$. The denominator $1 - b^2 = \sin^2\theta$
where $\theta$ is the angle between the rays — it vanishes only when the rays are **parallel**
(zero parallax / infinite depth), the sole degeneracy. The recovered point reprojects onto both
rays to $\sim0°$.

*Tests:* `test_triangulation_recovers_points_at_true_scale`,
`test_triangulated_points_reproject_onto_both_rays`.

## 7. Numerical stability and degeneracies

- **Conditioning & spherical whitening.** Unit bearing vectors are already $O(1)$ and
  well-scaled, so $A$ is far better conditioned than the pixel design matrix the classic
  eight-point needs Hartley-normalized. Pixel-domain Hartley normalization does **not** transfer
  to the sphere; the analogue (`normalize=True`, the 360-8PA idea) whitens the ray covariance,
  $T = (\mathrm{Cov} + \varepsilon I)^{-1/2}$, solves in the whitened frame, and maps back
  $E = T_2^\top E' T_1$. It is **exact in the noise-free limit** (changes nothing) and lowers the
  median pose error on clustered rays (e.g. a forward cone at $\sigma = 3\,$mrad: $10.5° \to 4.7°$).
  The $\varepsilon I$ (with $\varepsilon = 10^{-2}\lambda_{\max}$) is **load-bearing**: for a
  *very* narrow cone $\mathrm{Cov}$ is near-singular and an unregularized $\mathrm{Cov}^{-1/2}$
  amplifies the degenerate axis and makes the estimate *worse* — the regularization caps that, so
  whitening helps for moderate clustering and is safe otherwise. *Tests:*
  `test_spherical_normalization_is_exact_in_the_noise_free_limit`,
  `test_spherical_normalization_improves_conditioning_on_clustered_rays`.
- **Robust estimation.** A few mismatched rays break the least-squares eight-point, so
  `ransac_relative_pose` wraps it in RANSAC scored by the **angular Sampson residual** (radians,
  tangent-plane gradients), with an adaptive iteration count. On 30 % outliers it recovers the
  pose exactly where the naïve eight-point lands $>13°$ off. *Tests:*
  `test_ransac_recovers_pose_under_30pct_outliers`, `test_ransac_beats_naive_eight_point_with_outliers`.
- **Backward stability.** Both the null-space solve and the decomposition are pure SVD, which is
  backward stable; the noise-free pipeline recovers pose to $\sim10^{-6}$°.
- **Graceful degradation.** Pose error grows roughly *linearly* with ray noise and does not blow
  up at small $\sigma$ (e.g. $10^{-4}$ rad → $\sim0.01°$). *Test:*
  `test_noise_degrades_gracefully_and_monotonically`.
- **Degeneracy — pure rotation ($t \to 0$).** $E = [t]_\times R \to 0$; the translation direction
  is undefined (it's the null singular vector of an essentially arbitrary matrix). Detect via a
  small ratio of the two non-zero singular values of the *raw* $\hat E$, or a tiny median
  parallax, and fall back to a rotation-only (homography) estimate.
- **Degeneracy — planar / collinear scene.** When all points are coplanar, the calibrated
  two-view problem has a **two-fold ambiguity** and the eight-point becomes unreliable (a planar
  test scene here lands several degrees off). Use a plane-aware (homography-decomposition) path,
  or ensure non-degenerate 3D coverage — the same "the data must exercise the geometry" lesson as
  the [calibration FOV-coverage](../learn/are_two_models_the_same_camera.md) point.

## Manifold-correct refinement

The nonlinear refinement (`mvg.refine_two_view`) and the calibration bundle do **not** optimize an
absolute axis-angle vector (biased >30°, singular at `‖r‖=π`). They optimize a **local
perturbation** retracted through the exponential map, `R ← R₀·Exp([δω]_×)` with `δω` starting at
`0`, using the SO(3) `exp`/`log` and **right Jacobian** in [`ds_msp/core/lie.py`]
(`∂(Exp(w)v)/∂w = -Exp(w)[v]_× J_r(w)`, verified by finite difference). The calibrator's analytic
extrinsic Jacobian is then `∂Xc/∂δω = -R[Xw]_× J_r(δω)` — same numbers in benign
regimes, stable at large rotation.

## What this unlocks

Relative pose + triangulation on rays is the front end of **Structure-from-Motion**: chain
two-view poses, triangulate a point cloud, and refine by manifold bundle adjustment with the
**angular reprojection residual** — all without ever flattening the fisheye to a
pinhole, with a robust wrapper (RANSAC + spherical normalization) on top.
