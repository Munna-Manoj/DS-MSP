# CI/CD Pipeline `[CICD]`

> The automated pipeline that enforces the quality gates and ships releases (ISO/IEC/IEEE 12207
> §6.3). Maps each GitHub Actions workflow to what it guards, and records how to keep this document in
> sync with the workflows.

## Test tiers (the design that keeps the PR gate fast)

Tests are split into three tiers by cost and purpose, so the per-PR gate stays **under 15 minutes**
while the expensive validation still runs — just not on the PR path:

| Tier | Marker | What | Where it runs | Budget |
|------|--------|------|---------------|--------|
| **fast** | *(default, `not slow`)* | unit + contract + Jacobian gradient-checks + **rig smoke** (small-fixture end-to-end) | **every PR / push** (`ci.yml`) | seconds-to-minutes; per-test `--timeout` |
| **slow** | `@pytest.mark.slow` | heavy synthetic statistical validation (multi-model × multi-seed sweeps, robustness sweeps, full-size bundle adjustment) | **nightly** (`nightly.yml`) | ≤ 60 min |
| **realdata** | `@pytest.mark.realdata` | validation against real datasets (TUM-VI, MC-Calib Blender); dataset-gated, self-skips without data | **nightly** when provisioned; manually dispatched and evidenced before a release-gated release | ≤ 30 min |

The rig tests are tiered in `tests/rig/conftest.py`: only the genuinely-heavy tests are marked `slow`
(listed explicitly there), everything else is a fast smoke test. This was a deliberate correction — the
suite used to blanket-mark **all** rig tests `slow`, which hid ~34 fast tests and left the rig pipeline
with no PR-time coverage while the slow job grew past the CI timeout.

**Structural guard against regressing this.** The fast tier runs under a per-test `--timeout`
(`ci.yml`): a `not slow` test that exceeds it *fails*, so a heavy full-BA test can never silently rejoin
the <15-min PR gate — it must be marked `slow` (→ nightly) or shrunk. A `--cov-fail-under` gate on the
same run keeps coverage from regressing.

## Workflows

### `ci.yml` — the PR / push gate (fast, < 15 min)

| Job | Steps | Guards |
|-----|-------|--------|
| `detect changes` | `dorny/paths-filter` → `code` output | routes docs-only changes around the test work (see below) |
| `lint + types + layering` | `ruff check .`; `lint-imports`; `mypy ds_msp/core` | code style; the layered architecture (NFR-ARCH-001/002); typed core |
| **`governance`** | `check_traceability.py --check`; `check_tree_hygiene.py`; `check_docs_zone.py`; `check_docs_src_coverage.py`; `check_packaging.py` | requirement↔test↔ADR traceability; no tracked local-only/leak content; top-level docs allowlist; source-backed tested examples; **docs never advertise a subpackage the wheel excludes** |
| `tests (py3.10/3.11/3.12)` | `pytest -m "not slow" --timeout=120 --cov=ds_msp --cov-fail-under=80` on the version matrix | the fast tier, parallelized; portability (NFR-PORT-001); the timeout + coverage budget guards |

The **slow** and **realdata** tiers are **not** in this workflow — they run in `nightly.yml`. This is the
change that keeps the PR gate fast: the heavy synthetic suite (30+ min) no longer gates a merge.

The traceability, hygiene, and docs-structure checks use only the standard library. The packaging
check installs the project and imports each advertised entry point, because static inspection alone
cannot prove that the wheel's public commands resolve. The governance job always runs; a docs-only PR
is exactly when these contracts still matter.

**Concurrency.** `concurrency: { group: ci-<workflow>-<ref>, cancel-in-progress: true }`, so a second
push to a PR cancels the superseded run instead of paying for both.

**Path filtering (docs-only changes).** The `detect changes` job decides whether the diff touches code
(`ds_msp/`, `tests/`, `scripts/`, `tools/`, `configs/`, `examples/`, `benchmarks/`, build files, or any
workflow). On a **docs-only** change the `lint`/`tests` job *steps* are gated off (the required jobs still
report green in seconds), while `governance` always runs in full. If the diff cannot be determined, the
filter defaults to running everything.

### `nightly.yml` — the heavy validation (scheduled + on-demand, not on PRs)

Runs daily (`cron: 0 6 * * *`) and via `workflow_dispatch`:

| Job | Runs | Purpose |
|-----|------|---------|
| `slow synthetic suite` | `pytest -m "slow" --timeout=1200` | the full multi-model × multi-seed rig statistical validation; ≤ 60 min |
| `fast tier + coverage (matrix)` | `pytest -m "not slow" --cov` on 3.10/3.11/3.12 | daily portability + coverage snapshot |
| `real-data validation (dataset-gated)` | `pytest -m "realdata"` with `DSMSP_*_DIR` secrets | real-data evidence for release-gated requirements when data is provisioned; self-skips otherwise, and an all-skipped run is not evidence |

The slow synthetic job is serial at the pytest level but internally parallel: rig calibration tests
exercise the production `ProcessPoolExecutor` path. Running those cases through an additional xdist
pool would multiply the per-calibration worker pools and starve them. A POSIX signal timeout
preserves the 1,200-second per-test cap without starting a watchdog thread before the calibration
workers fork.

### `release.yml` — on push to `main`

`release-please` maintains a release PR (version bump + `CHANGELOG.md` from Conventional Commits).
Merging it cuts the tag + GitHub Release and triggers the PyPI publish via **Trusted Publishing
(OIDC)** — no stored token (CON-07, ADR-0006). See [`RELEASING.md`](../../../RELEASING.md).

### `deploy-pages.yml`

Builds and publishes the documentation site.

## The release gate

The policy (ADR-0006): a release involving any requirement whose `requirements.csv` row has
`release_gated=yes` requires both `tools/check_traceability.py --release` to pass **and** its linked
`realdata` tests to execute and pass against real datasets. `realdata` tests are dataset-gated and
skipped in ordinary PR CI to keep PRs fast.

Status: the structural `--release` check and the scheduled/on-demand `nightly.yml` real-data runner
are wired. `release.yml` does not yet depend on a non-skipping real-data result, and a dataset-gated
job can report green when every test skipped. Until that remaining RSK-07 control is wired, the
maintainer must record evidence that every linked real-data test actually executed and passed before
merging a release PR that contains release-gated work.

## Lifecycle mapping (ISO/IEC/IEEE 12207)

```
requirement (§6.4.1)  →  design / ADR (§6.4.3, 42010)  →  branch (§6.3.2)
   →  implement (§6.4.4: ruff / mypy / import-linter + independence)
   →  verify-synthetic (§6.4.6: pytest, -m jac, contract, integration)
   →  validate-real-data (§6.4.9: realdata tests/scripts)
   →  review (§6.3.7: CODEOWNERS)  →  release (§6.4.10: release-please + OIDC)
```

## Keeping this document current

This file is **descriptive of the workflows**, which are authoritative. When a workflow changes:

1. Update the matching row/section here in the same PR.
2. If a new gate is added, add it to the [Definition of Done](../quality/DEFINITION_OF_DONE.md) and the
   PR template.
3. If a gate enforces a requirement, ensure that requirement's `verify_method` points at the workflow
   or test, so traceability stays complete.
