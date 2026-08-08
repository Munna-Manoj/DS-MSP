# Contributing to DS-MSP

Thanks for contributing! DS-MSP is governed by a lightweight, standards-informed engineering process
so every change is controlled, tested on synthetic data, and (where it makes an accuracy claim)
validated on real data before it ships. This file is the short version; the full system lives in
[`docs/process/`](docs/process/) and is indexed by the
[**Engineering Handbook**](docs/process/HANDBOOK.md).

## TL;DR

1. **Branch off `main`** — `feat/…`, `fix/…`, `docs/…`, `chore/…`. No direct commits to `main`.
2. **Link a requirement** — reference the `FR-…`/`NFR-…` ID(s) your change implements, or add a row to
   [`docs/process/srs/requirements.csv`](docs/process/srs/requirements.csv).
3. **Write tests** at the right [level](docs/process/quality/test-levels.md); tag new ones with
   `@pytest.mark.req("FR-…")`.
4. **Run the gates locally:**
   ```bash
   ruff check .
   mypy ds_msp/core --follow-imports=silent --ignore-missing-imports
   lint-imports
   pytest -q -m "not slow"
   pytest -q -m "slow"  # required locally for relevant solver/optimizer changes
   python tools/check_traceability.py --check
   python tools/check_tree_hygiene.py
   python tools/check_docs_zone.py
   python tools/check_docs_src_coverage.py
   python tools/check_packaging.py
   ```
5. **Conventional Commits** — `feat:` / `fix:` / `feat!:` (breaking) drive versioning automatically.
6. **Open a PR** — the template is the [Definition of Done](docs/process/quality/DEFINITION_OF_DONE.md).

## Adding something common?

Follow the matching playbook — they walk REQ → ADR → code → tests → docs → release:

- [Add a camera model](docs/process/playbooks/add-a-camera-model.md)
- [Add a robust kernel](docs/process/playbooks/add-a-robust-kernel.md)
- [Add an IO format](docs/process/playbooks/add-an-io-format.md)
- [Add a capability or pipeline](docs/process/playbooks/add-a-pipeline-capability.md)

## Ground rules

- Keep the **math foundation** (`core`/`data`/`geometry`/`models`) NumPy-only — no `cv2`/`scipy`
  there ([ADR-0004](docs/process/architecture/decisions/ADR-0004-cv2-scipy-free-foundation.md)).
- Respect the [layering](docs/process/architecture/ARCHITECTURE.md); a new cross-layer edge needs an
  ADR and updated import-linter contracts.
- New models need analytic Jacobians that pass the `-m jac` gradient check at ≤1e-6
  ([ADR-0003](docs/process/architecture/decisions/ADR-0003-analytic-jacobians.md)).
- Back numerical claims with an asserting test, never prose. Keep large datasets and any local-only
  content out of the tree.

Security issues: see [SECURITY.md](SECURITY.md) — report privately, never in a public issue.
