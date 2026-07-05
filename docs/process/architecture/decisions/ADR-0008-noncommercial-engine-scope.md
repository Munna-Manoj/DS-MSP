# ADR-0008 — Noncommercial license covers the robust calibration/conversion engine, not just the Plus models

- **Status:** Superseded by [ADR-0010](ADR-0010-mit-relicense-and-eucmplus-removal.md)
- **Deciders:** maintainer
- **Relates to:** ARC-GEOMETRY, ARC-CALIB, ARC-ADAPT, ARC-MODELS
- **Supersedes:** —

## Context

DS-MSP is dual-licensed: the generic library is MIT, and the project's differentiated work is
PolyForm Noncommercial 1.0.0 with attribution to the author. The initial noncommercial scope was
just the four Plus-model files (`models/{dsplus,eucmplus}{,_math}.py`).

That boundary is a shallow moat:

1. **The math is not the moat.** The Double Sphere and Extended UCM models are published academic
   work; copyright protects this project's *specific code expression*, not the algorithm. Anyone may
   reimplement the model math independently regardless of license.
2. **The value sits in the surrounding engineering, which was MIT.** The genuinely hard, R&D-derived
   contribution is the *robust from-scratch* path: a pure-NumPy intrinsic seed + RANSAC resection,
   an auto-initialized bundle-adjustment `calibrate()` that converges sub-pixel from a generic init,
   and an image-free `convert()` with a deterministic shape sweep (ADR-0007). With only the model
   files restricted, a commercial user could drive (re-implemented or old-MIT) models with all of
   that MIT tooling — taking the real IP for free.
3. **Relicensing is forward-only.** Already-distributed copies under MIT keep their MIT grant
   permanently (MIT has no revocation clause); this ADR governs the current and future versions
   only. No prior PyPI release contains the robust engine, so the first published artifact carrying
   it is also the first under this scope.

## Decision

Extend the PolyForm Noncommercial 1.0.0 scope to the **robust calibration/conversion engine** in
addition to the Plus models. The noncommercial set is now:

    ds_msp/models/dsplus.py, dsplus_math.py, eucmplus.py, eucmplus_math.py   (Plus models)
    ds_msp/geometry/resection.py     (robust intrinsic seed + RANSAC resection)
    ds_msp/calib/bundle.py           (robust auto-initialized calibrate())
    ds_msp/adapt/convert.py          (robust image-free model conversion)

Everything else stays MIT (the `CameraModel` protocol and standard KB/DS/UCM/EUCM/RadTan models,
geometry helpers, sampling/evaluation, detection, I/O). Each noncommercial file carries an SPDX
`LicenseRef-PolyForm-Noncommercial-1.0.0` header; `LICENSE-NONCOMMERCIAL.txt`, `LICENSING.md` and
`README.md` enumerate the set; the distribution SPDX expression remains
`MIT AND LicenseRef-PolyForm-Noncommercial-1.0.0`.

## Consequences

**Positive**
- The defensible engineering (robust seed → calibrate → convert), not just published math, is the
  licensed asset; commercial use requires a license from the author.
- Honest, enumerated per-file scope; no claim over the underlying algorithms.

**Negative / costs**
- `geometry/resection.py` is a low layer that `calib`/`rig` build on, so the *maintained library as a
  whole* becomes noncommercial-to-use even though most files stay MIT-labeled. This is the intended
  effect, accepted with eyes open.
- Reduced permissive adoption of the robust path. Mitigated: standard models + helpers remain MIT and
  independently usable; a user may build their own initializer/optimizer on the MIT models.
- Forward-only: the engine shipped to `main` immediately before this record stays MIT in that git
  history. Acceptable — no released artifact carried it, and relicensing cannot reach distributed
  copies regardless.

## Alternatives considered

- *Keep noncommercial scope at the four model files (status quo).* Rejected — protects published math
  while leaving the real IP MIT; trivially bypassed.
- *Relicense the whole project noncommercial / copyleft (AGPL).* Rejected — kills permissive adoption
  of the generic, genuinely reusable library.
- *Patent the algorithms.* Rejected — costly, likely unpatentable given prior publication, and out of
  proportion to the project.
- *Rewrite history / delete old releases to revoke MIT.* Rejected — ineffective (distributed copies
  keep their grant) and destructive.
