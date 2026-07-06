# ADR-0010 — Remove EUCM⁺; relicense the whole project to plain MIT

- **Status:** Accepted (recorded 2026-07-05)
- **Deciders:** maintainer
- **Relates to:** ARC-MODELS, ARC-GEOMETRY, ARC-CALIB, ARC-ADAPT
- **Supersedes:** [ADR-0008](ADR-0008-noncommercial-engine-scope.md) (in full); partially
  supersedes [ADR-0005](ADR-0005-dsplus-eucmplus.md) (drops the EUCM⁺ half of that decision;
  the DS⁺ half stands)

## Context

Since [ADR-0008](ADR-0008-noncommercial-engine-scope.md), DS-MSP shipped dual-licensed: the
generic library MIT, and the Plus models (DS⁺, EUCM⁺) plus the robust
calibrate/convert engine (`geometry/resection.py`, `calib/bundle.py`, `adapt/convert.py`)
under PolyForm Noncommercial 1.0.0 with attribution.

The maintainer decided to reverse this. Two changes, made together:

1. **Drop EUCM⁺ entirely.** DS⁺ already covers the same "closed-form-invertible extension of
   a sphere model" niche EUCM⁺ occupied ([ADR-0005](ADR-0005-dsplus-eucmplus.md)), and
   real-data measurement during this same change (`tests/realdata/test_mccalib_calibration.py`)
   showed plain EUCM (EUCM⁺'s non-extended base) cannot reach sub-pixel on this project's own
   reference wide-FOV lens (median 1.5–2.1px vs. DS⁺'s <0.4px from the same generic init) —
   maintaining a second Plus model family added surface (two more files under the full contract
   + gradient-check suites, a second noncommercial file pair, a second web-studio port) without
   a distinct use case DS⁺ didn't already serve.
2. **Relicense everything to plain MIT.** No noncommercial tier remains. Every file DS-MSP has
   built so far — including DS⁺ and the robust calibrate/convert engine that ADR-0008 scoped as
   noncommercial — is MIT. This is an explicit reversal of ADR-0008's core argument (that the
   robust engineering, not the published math, was the defensible moat): the maintainer chose
   permissive adoption over that protection.

## Decision

- Delete `ds_msp/models/eucmplus.py` and `eucmplus_math.py`, their registry/IO/rig
  wiring, their contract/gradcheck/robustness/realdata test coverage, their web-studio
  (`web/`) TypeScript port, and their `docs/reference/models.md` entry. `ds_msp.models` now
  ships seven models: DoubleSphere, UCM, EUCM, KannalaBrandt, RadTan, OCam, DS⁺.
- Delete `LICENSE-NONCOMMERCIAL.txt`. Strip the `SPDX-License-Identifier:
  LicenseRef-PolyForm-Noncommercial-1.0.0` header from every file that carried it
  (`models/dsplus.py`, `models/dsplus_math.py`, `geometry/resection.py`, `calib/bundle.py`,
  `adapt/convert.py`) — they now carry no per-file header, matching every other MIT file in
  this repo. `pyproject.toml`'s `license`/`license-files`, `CITATION.cff`'s `license` field, and
  `LICENSING.md` are updated to plain MIT. The distribution SPDX expression becomes simply
  `MIT`.
- Prior work this project cites (Fisheye-Calib-Adapter, MC-Calib, the camera-model papers) was
  never vendored code and was never covered by the noncommercial scope — those citations in
  the README's Credits section are unaffected by this ADR; they document design lineage and
  academic attribution, not a license grant over those external repositories.
- Real, published academic math (the DS⁺/EUCM⁺ formulations themselves) was already
  unprotectable by copyright per ADR-0008 §Context point 1 — that reasoning is unchanged by
  this ADR; it was never the basis for either license scope.

## Consequences

**Positive**
- One license for the whole project — no per-file SPDX bookkeeping, no "is the *maintained
  library as a whole* noncommercial-to-use" caveat (ADR-0008's own acknowledged cost).
- Simpler model surface: one closed-form-invertible Plus model (DS⁺) instead of two, with a
  measured (not assumed) real-data justification for why the second didn't earn its keep.
- Removes the case where a trivial, mechanical helper (ADR-0009 §Decision point 5,
  `seed_from_K`) could be silently relicensed to noncommercial purely by which file it lived
  in — that constraint no longer exists, though the placement itself (in `models/registry.py`)
  remains correct on independent layering grounds and is not reverted by this ADR.

**Negative / costs**
- Reverses ADR-0008's stated goal (commercial use of the robust engine required a separate
  license) — a business decision, not a technical one; recorded here for the architectural
  record it touches (five files' licensing, one model's existence), not re-litigated.
- Breaking API change: any caller constructing `EUCMPlusModel` or naming `"eucmplus"`/`"eucm+"`
  in a config, a Kalibr `camera_model: eucm_plus` field, or an MC-Calib `distortion_type`
  entry now gets `KeyError`/`ImportError` instead of a model instance. No deprecation window —
  the model is deleted, not soft-removed, per the maintainer's decision.
- `tests/realdata/test_mccalib_calibration.py` lost its one cross-family "DS⁺ is a faithful
  conversion target for an independently-fit real model" check: EUCM⁺ was that check's partner
  specifically because it is architecturally close to DS⁺ (both sphere/division-model
  extensions), so a from-scratch EUCM⁺ fit converted into DS⁺ sub-pixel. Substituting KB (a
  different, polynomial family) for real-data measurement in the same test found the converted
  fit diverges past sub-pixel at the periphery (up to ~1.5px RMS at 50–70° ray angle on this
  dataset) — a genuine, measured cross-family gap, not a same-family one. That specific
  real-data cross-family assertion was removed rather than loosened to a number that would
  silently under-test the claim; the broader "DS⁺ is a faithful universal conversion target"
  claim remains covered *synthetically*
  (`tests/adapt/test_convert_robustness.py::test_dsplus_is_faithful_universal_target`, sub-pixel
  across UCM/EUCM/DS/KB ground truth). A same-family real-data partner for DS⁺ is a documented
  gap, not a silently dropped one.

## Verification

- Full suite green post-removal under this project's pinned environment (`uv run pytest`):
  578 passed / 1 failed on the fast tier (`-m "not slow"`) and 30 passed / 5 skipped on the slow
  tier (`-m "slow" --timeout=1200`), 608 passing total. The one failure,
  `tests/calib/test_charuco.py::test_parity_vs_mccalib_keypoints`, is pre-existing and unrelated
  — confirmed present identically on the pre-change branch before any of this ADR's changes.
- `pytest -m jac` and `pytest tests/contract` green (no analytic-Jacobian or contract
  regression from either the model removal or the license-header strips).
- `tests/realdata/test_mccalib_calibration.py` green (6/6) after substituting KB for the
  self-conversion checks; the removed cross-family assertion is documented above, not silently
  dropped.
- `mkdocs build --strict`, `ruff check .`, `lint-imports`, `python tools/check_traceability.py
  --check`, `python tools/check_tree_hygiene.py`, `python tools/check_docs_src_coverage.py`,
  `python tools/check_packaging.py`, and `mypy ds_msp/core` all green.

## Alternatives considered

- *Keep EUCM⁺, only relicense.* Rejected by the maintainer — EUCM⁺ was explicitly named for
  removal alongside the relicensing, not kept as a side effect of it.
- *Scope the relicense to "DS⁺ and its use case stay noncommercial; `convert()`/`calibrate()`/
  `rig_calibrate()` become MIT."* Rejected — a partial relicense still needs per-file SPDX
  bookkeeping and leaves the same "is the *maintained library as a whole* noncommercial-to-use"
  caveat ADR-0008 already accepted as a cost; a full relicense is simpler and maximizes
  permissive adoption, which is the actual goal here.
- *Loosen the real-data cross-family conversion threshold to accommodate KB instead of removing
  the assertion.* Rejected — the measured KB→DS⁺ divergence (up to ~1.5px) is a real, different
  claim from the original EUCM⁺→DS⁺ sub-pixel one; silently loosening the bound to make an
  unrelated claim pass would misrepresent what was actually verified.
