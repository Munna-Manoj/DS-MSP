# Research notes

Captured, verified research that feeds the [roadmap](../ROADMAP.md). Each report is a
**fact-checked record** plus an **implementation spec** that maps findings to concrete
`ds_msp` modules with math, algorithms, and the verification number to assert.

| Report | What it is |
|---|---|
| [Findings — representations for 3D tasks](representations_for_3d_tasks_findings.md) | Faithful record of a multi-source deep-research run on image-domain charts (ERP / cubemap / tangent / pinhole) for stereo, SfM, and reconstruction. 24/25 claims passed 3-vote adversarial verification; 1 killed. Votes, evidence, sources, caveats, open questions. |
| [Tier-1 implementation spec](tier1_implementation_spec.md) | Each finding turned into a buildable unit (C1–C9): math, core algorithm, verification test, target module, dependencies, tier (🟩 core / 🟦 research). The direct map from research → Python. |
| [Two-view geometry — derivations & proofs](../explain/two_view_geometry.md) | Formal companion to the shipped `ds_msp/mvg/` (C1): epipolar-constraint-on-rays proof, essential-matrix properties, eight-point + manifold projection optimality, ray cheirality, midpoint triangulation, and numerical stability / degeneracies. Each claim ↔ a named test. Now lives at `docs/explain/two_view_geometry.md`. |
| [DS-MSP ↔ diffpnp symbiosis](diffpnp_dsmsp_symbiosis.md) | Survey + gap analysis of the (same-author) **diffpnp** differentiable robust PnP library: where the two repos complement each other (DS-MSP = verified wide-FOV models; diffpnp = manifold-correct robust optimizer), the three pinhole seams to generalize, the manifold-optimization gap to borrow, a cross-validation harness, and a 5-phase plan. |

**How this connects:** the spec's units (C1 two-view-on-rays, C3 charts, C4 sphere-sweep, …) are
scheduled in [`../ROADMAP.md`](../ROADMAP.md) under **Tier 1**. The geometry these reports cover
extends the existing reprojection deep-dive,
[`../learn/spherical_and_cylindrical_reprojection.md`](../learn/spherical_and_cylindrical_reprojection.md).

> Research records are point-in-time (2026-06). Method rankings in fast-moving deep-learning
> areas may age; the **geometric facts** (epipolar curves, arc-length disparity, essential matrix
> on rays) are stable. Re-verify before treating any single method as state-of-the-art.
