# ADR-0032 — Run the observability audit on the block-arrowhead information matrix

- **Status:** Accepted (recorded 2026-11-07)
- **Relates to:** ARC-RIG, NFR-PERF-006, NFR-PERF-007, NFR-PERF-008

## Context

After a rig fit, the audit takes the eigen-decomposition of the equilibrated information matrix
`H = JᵀWJ` to find weak directions. `H` has one column per camera parameter and per board pose, so K
grows with the number of poses: 6,000 to 10,000 on the rendered 25-camera rigs. The audit formed the
dense Jacobian (rows x K), then up to three K x K arrays at once, and set the peak memory of every rig
run measured: 18 GB on a 25-camera rig at 6aed1ae, and an estimated 110 GB at 100 cameras.

The audit does not need the full spectrum. It reads the smallest and largest eigenvalues, the median,
every eigenvalue below `tau · median` with its eigenvector, the next eigenvalue up, and the entries of
the equilibrated matrix whose magnitude exceeds the pair threshold.

## Decision

1. **No dense Jacobian.** `H` is summed from each observation's Jacobian block over the columns it
   touches.
2. **One K x K array when `H` is formed.** `H` is symmetrised in place. Each spectrum comes from LAPACK
   `dsyevd` with `jobz='N'` on the transposed view, which is the same symmetric matrix in column-major
   order, so no copy is made. LAPACK overwrites only the triangle it reads; the other triangle is
   checksummed and `H` is restored from it.
3. **Block-arrowhead route.** Each board pose only meets its own observations, so `H` is block-arrowhead:
   a camera block `A`, the coupling `B`, and 6 x 6 pose blocks. With each block rotated to its eigenbasis
   (an orthogonal similarity), the number of eigenvalues below `σ` is the number of block eigenvalues
   below `σ` plus the number of negative eigenvalues of the camera-block Schur complement (Sylvester's
   law of inertia, Haynsworth's inertia additivity). Bisection on that count isolates each eigenvalue
   the audit needs, and safeguarded Newton on the Schur complement finishes it; eigenvectors are the
   Schur complement's null vectors lifted back to the poses. A pose coordinate whose elimination would
   grow the Schur complement by more than 100 times its diagonal scale is kept beside the camera block
   instead, so a near-singular board pose cannot corrupt a count. Only `A`, `B` and the blocks are stored.
4. **Route choice.** The arrowhead route is used when its estimated cost is at most twice the dense
   cost: rigs with many board poses per camera. Camera-heavy rigs, where K x K is small, keep route 2.
5. **Freed heap.** After image detection and after the audit, freed heap is returned to the operating
   system (`malloc_trim` on glibc; a no-op elsewhere).

## Consequences

- Both routes report the same findings. On the audit matrices of three rendered rigs (K 6,128 to 10,266)
  the weak directions, pairs, scales and participating parameters are identical, and eigenvalues agree
  within 6e-16 of the largest. Ratios near 1e-8 agree to about 2e-7 relative, the rounding level of any
  backward-stable solver.
- The arrowhead route is 5 to 11 times faster than the dense spectra on those rigs, and needs about
  `C² + 2C·poses` numbers instead of `K²`.
- Calibration results do not depend on the audit, and are byte-identical to the previous code.
- The arrowhead route is plain numpy. Route 2 uses SciPy's LAPACK wrappers, which SciPy ships on every
  platform it supports.
- The full spectrum is no longer computed on the arrowhead route, so `eigvals` in its result holds only
  the eigenvalues the audit reads.

The evaluation is in the pull request and in `tests/core/test_arrowhead.py`,
`tests/rig/test_normal_matrix.py` and `tests/test_heap.py`. The growth bound of 100 and the factor of
two in the route choice are numerical-analysis choices, not fitted to a dataset.
