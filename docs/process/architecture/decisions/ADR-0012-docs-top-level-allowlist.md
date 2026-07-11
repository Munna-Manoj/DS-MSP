# ADR-0012 — Positive allowlist for top-level docs/ entries, enforced in CI

- **Status:** Accepted (recorded 2026-07-11)
- **Deciders:** maintainer
- **Relates to:** ARC-DOCS, NFR-DOCS-003
- **Supersedes:** —

## Context

`docs/` has no structural check that every top-level file or directory belongs there. A stray
draft, a half-finished note, or a file created for local convenience can land directly under
`docs/` and ship to the public site unnoticed — there is nothing that inspects *new* top-level
entries and asks whether they were meant to be published.

Content-pattern scanning (matching known-bad phrasings) is one defense against this, but it is
bounded by what someone already thought to write a pattern for: a file with different wording,
or one added before a given pattern existed, passes clean. That approach also does not run as
part of the CI pipeline today, so it cannot be relied on as the only gate.

## Decision

Add a **positive allowlist** for anything living directly under `docs/`
(`docs/process/governance/DOCS_TOP_LEVEL_MANIFEST.txt`), enforced by a new, tracked, pure-stdlib
script (`tools/check_docs_zone.py`) wired into the CI `governance` job
(`.github/workflows/ci.yml`) as a required check on every PR, plus both local `pre-commit` and
`pre-push` hooks for earlier feedback.

This is structurally different from a content scan: any top-level `docs/` entry not on the
manifest fails **regardless of its name or content** — there is no phrasing to guess correctly
in advance, because the check is set membership, not pattern matching. Adding a genuinely public
document requires a one-line, reviewable diff to the manifest in the same PR that adds it — a
deliberate act, not a default.

Scope is deliberately narrow: only *top-level* entries directly under `docs/`. Nested pages
under the Diataxis directories (`explain/`, `how-to/`, `learn/`, `reference/`, `process/`) are
already governed by other means — the documentation-authoring workflow's own review step,
`mkdocs build --strict`, and `docs_src` coverage (NFR-DOCS-002). This check targets specifically
an ad hoc top-level file landing outside all of that.

## Verification (real numbers, not assumed)

- `python3 tools/check_docs_zone.py` against the real `docs/` tree: `DOCS ZONE: OK (21 top-level
  docs/ entries, all on the manifest)`.
- Simulated the exact failure mode this closes: dropped an unlisted file into `docs/` and
  re-ran the check — `DOCS ZONE: FAIL — new top-level docs/ entries not on the manifest: -
  docs/SOME_SNEAKY_INTERNAL_NOTE.md`, exit code 1. Removed the file; check passes again.
- `ruff check .`, `lint-imports`, `mypy ds_msp/core --follow-imports=silent
  --ignore-missing-imports`, `check_traceability.py --check`, `check_tree_hygiene.py`,
  `check_docs_src_coverage.py` all pass with the new files present.

## Consequences

**Positive**
- The check runs in CI on a clean, freshly-checked-out worktree — independent of local machine
  state (a misconfigured or skipped local hook, or a contributor who never installed one).
- Absence from the manifest is itself the failure signal, so a new top-level entry cannot be
  missed due to unfamiliar wording the way a content scan can miss one.
- Cheap: pure stdlib, one `Path.iterdir()` call, no network/build dependency; adds negligible
  time to CI or either local hook.

**Negative / costs**
- Every future *legitimately public* top-level `docs/` addition requires remembering to update
  the manifest, or CI fails — a small, deliberate friction cost, which is the point (it forces a
  reviewable decision instead of a silent default).
- Only covers the top level of `docs/`; content nested inside an already-allowed subdirectory is
  not in scope for this check — that stays the job of the authoring workflow and existing
  content checks.

## Scope explicitly deferred (not accidental omissions)

- **Frontmatter/metadata self-declaration** (e.g. requiring every page to declare
  `audience: public`) — considered and not built; the allowlist already gives an exhaustive
  guarantee for a misplaced top-level file with far less machinery.
- **Extending the allowlist to nested directories** (per-subdirectory manifests) — not needed
  today; those trees already have their own gates (authoring workflow, `docs_src` coverage,
  mkdocs strict build).

## Alternatives considered

- *Add more phrases to a content-pattern scan and call it fixed* — rejected: narrows a known gap
  without changing the fact that the next unfamiliar phrasing still gets through silently.
- *Frontmatter-tag every doc file, fail the build if a file lacks a valid tag* — considered (a
  positive-declaration scheme, structurally similar in spirit); rejected for now as more
  machinery than the current scope needs — see Scope-deferred above.
- *A file-naming convention check* (flag names matching common draft/internal patterns) —
  rejected as the sole mechanism: still a content-based guess, just over filenames instead of
  prose, with the same fundamental gap. An allowlist has no such gap by construction.
