# Explanation: the math and the "why" behind the library

This section is understanding-oriented, not task-oriented. Nothing here has a "run this
script" checklist to complete — for that, go to [**Learn**](../learn/README.md), the guided,
runnable tutorial track. These pages exist instead to settle a conceptual question in depth,
with a derivation, a proof, or a measured comparison — the kind of material that would slow a
tutorial down if it were inlined there.

Read a page here when a tutorial chapter links to it for "the theory," or when you're chasing
a specific question about *why* the library behaves the way it does.

## Pages

- **[Two-view geometry on bearing vectors](two_view_geometry.md)** — the epipolar-constraint
  proof, essential-matrix properties, the four-fold pose decomposition, and numerical-stability
  notes behind [Learn Chapter 8](../learn/08_two_view_geometry_on_rays.md).
- **[Are two different camera models the same camera?](are_two_models_the_same_camera.md)** —
  why a Double Sphere and a Kannala-Brandt calibration of the *same* lens report focal lengths
  that differ by 26%, and the paraxial-focal derivation that proves they're the same optics
  where the data reached.
- **[Is this camera model right for my lens and task?](choosing_a_camera_model.md)** — a
  measurable framework for picking a model: six diagnostics (capacity, identifiability,
  parameter redundancy, bound sensitivity, compute cost, FOV sampling weight), each backed by
  a number you can verify on real lenses.
- **[A fair fight: EUCM⁺ vs DS⁺ vs Kannala-Brandt](case_study_eucmplus_dsplus_kb.md)** — the
  framework above applied to a real ship/retire decision on three lenses, including a
  hypothesis retracted when the data contradicted it.

## What doesn't live here

Anything with a runnable script and a "you'll be able to…" outcome belongs in
[Learn](../learn/README.md) instead, not here — that's the Diátaxis line this project holds.
If a page here starts accumulating "Try it yourself" steps that teach a new skill rather than
deepen a conceptual answer, it should move.
