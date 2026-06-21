# Explanation

The math and the reasoning behind the library: derivations, design choices, and the geometry
that the tutorials use but don't stop to prove. Each page links to the tutorial that exercises it
and the tests that pin its claims.

Two pages are available now:

- [Projection validity, FOV & undistortion](projection_validity_and_fov.md) — the tilted half-space test, maximum incidence angle, and why a wide fisheye can't fully un-distort into a pinhole image.
- [Two-view geometry (derivations)](two_view_geometry.md) — the algebra behind essential matrices, ray-based epipolar constraints, and the fisheye-compatible formulation.

Coming soon:

- Manifold optimization (Lie / LM / robust)
- Model equivalence (paraxial focal)
- Charts (sphere / cylinder / pinhole / cubemap / tangent)
