# Quality Assurance & Verification/Validation Plan `[QVP]`

> Standards-informed after ISO/IEC/IEEE 29119-2 (test process) and 12207 (V&V activities). Defines
> *how* DS-MSP is verified (does it meet the spec?) and validated (does it work on real data?), the
> entry/exit criteria for each stage, and the release policy plus its current automated controls.

## 1. Verification vs validation

- **Verification** — the software meets its specification. Done with the deterministic, dataset-free
  test levels ([test-levels.md](test-levels.md)): unit, contract, gradient-check, integration, and
  statistical. The fast levels run on every PR; statistical tests run nightly and locally before
  merge when relevant to a change.
- **Validation** — the software produces correct results on **real** data (real lenses, real noise,
  real board detection and pose distributions). Done with `realdata` tests/scripts against real
  datasets. Required before a release for release-gated requirements (ADR-0006).

## 2. Quality gates (per PR, all must pass)

The CI jobs ([CICD_PIPELINE.md](../management/CICD_PIPELINE.md)) apply the following gates. Code
changes run items 1–4, governance runs on every pull request, and the strict docs build runs when
code or documentation changes:

1. **Lint** — `ruff check .` clean.
2. **Layering** — `lint-imports` (import-linter contracts) clean; mirrored by
   `tests/contract/test_independence.py`.
3. **Types** — `mypy` clean on the typed core surface.
4. **Tests** — `pytest -m "not slow"` green on the Python 3.10 / 3.11 / 3.12 matrix, with coverage
   reported. Relevant slow synthetic evidence comes from a local or nightly run before merge.
5. **Governance** — traceability, tree hygiene, docs-zone, docs-source-coverage, and packaging checks
   clean (no orphan requirements, dangling links, unregistered docs, copied/unverified examples,
   tracked local-only content, or advertised-but-unshipped package surface).
6. **Documentation** — the MkDocs site builds in strict mode through `tests/docs/`.

## 3. Entry / exit criteria

**Verification (synthetic) — entry:** a change with the relevant tests added/updated and its
requirement marker(s) in place. **Exit:** all §2 gates green.

**Validation (real data) — entry:** verification passed; the change touches a requirement whose
canonical row has `release_gated=yes`. **Exit:** every linked `realdata` test actually executed and
passed on the real dataset, within the tolerance stated in the requirement.

**Release — entry:** `check_traceability.py --release` confirms structural synthetic and real-data
coverage, and execution records confirm the linked real-data tests did not skip and passed.
**Exit:** release-please cuts the tag and the PyPI OIDC publish succeeds (ADR-0006, CON-07).

## 4. Coverage expectations

- New public behaviour ships with tests at the appropriate level (unit + contract for a model;
  integration for a pipeline; `realdata` for release-gated accuracy claims).
- Coverage is reported per PR (`pytest --cov`); the bar is *meaningful* coverage of new code paths,
  not a single global percentage. Numerical claims (accuracy, tolerances) are backed by an asserting
  test, never by prose alone.

## 5. The release gate

The policy: no release-gated requirement may ship without green synthetic **and** real-data evidence.
Current controls are:

- **Active structural control:** `tools/check_traceability.py --release` fails if a release-gated
  requirement lacks linked synthetic or `realdata` coverage.
- **Active execution path:** `nightly.yml` runs the real-data suite on schedule or on demand when
  datasets are provisioned.
- **Remaining RSK-07 gap:** `release.yml` does not depend on a non-skipping result, and the
  dataset-gated job can be green after all tests skip. Until an automated release dependency proves
  execution, the maintainer must verify and record that the linked tests ran and passed before
  merging the release PR.

See ADR-0006 and [CHANGE_RELEASE_MGMT.md](../management/CHANGE_RELEASE_MGMT.md).

## 6. Internal verification protocol (high-risk changes)

Beyond the public gates above, high-risk changes (new camera model, solver/optimizer changes,
anything affecting calibration accuracy) additionally go through a **local deep-verification
protocol** before entering the public PR flow: extended adversarial review of derivations, additional
synthetic stress scenarios, and broader real-data validation than the release gate requires. This
protocol is a local development practice; its artifacts are kept out of the tracked tree (CON-06,
NFR-PRIV-001) and are not a prerequisite for external contributors — the public gates in §2–§5 are the
contract every change is held to.

## 7. Defect handling

Failures and regressions are tracked per [ISSUE_DEFECT_PROCESS.md](../management/ISSUE_DEFECT_PROCESS.md)
(IEEE-1044-style taxonomy). A fixed defect ships with a regression test that fails before the fix and
passes after.
