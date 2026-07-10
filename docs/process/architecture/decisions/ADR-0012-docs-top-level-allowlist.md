# ADR-0012 — Positive allowlist for top-level docs/ entries, enforced in CI

- **Status:** Accepted (recorded 2026-07-11)
- **Deciders:** maintainer
- **Relates to:** ARC-DOCS, NFR-DOCS-003
- **Supersedes:** —

## Context

`docs/RIG_MULTIOBJECT_IMPLEMENTATION_PLAN.md` — an internal planning document written for a
single feature's own AI-assisted development process — was added directly under the public
`docs/` tree in PR #49 (2026-07-08) instead of this repository's local-only planning-notes
location (git-ignored; several sibling planning docs already live there). It sat publicly on
`main`, unreviewed as a leak, until discovered on 2026-07-11 while working an unrelated branch.

It was not caught earlier for two independent reasons, confirmed by reading the actual
mechanism rather than assumed:

1. **The only scanner for this class of content is a pre-existing, local-only content regex
   denylist**, matching phrasings like "phase N" seen in *previously* leaked content. A denylist
   is bounded by what someone already thought to list — different wording in a new file (or a
   file added before the pattern existed) passes clean. Confirmed: re-running that scanner
   against this exact file returned 23 matches once the pattern existed, meaning the *content*
   was always detectable in principle — the gap was never having run the scanner against this
   file's commit before it merged.
2. **That scanner is local-only and wired only into a local `pre-push` git hook**
   (`.git/hooks/pre-push`, itself gitignored/never shipped) — it has never run in CI. Worse, the
   hook itself had an independent, unrelated bug at the time this file was added: in a
   git-worktree checkout it validated whichever branch happened to be checked out in the *main*
   working directory, not the branch actually being pushed (fixed separately, same day, not part
   of this ADR's scope). Net effect: for months, the only thing standing between internal
   process content and the public tree was a local, occasionally-misconfigured, always
   bypassable (`--no-verify`) hook that a contributor might not even have installed.

## Decision

Add a **positive allowlist** for anything living directly under `docs/`
(`docs/process/governance/DOCS_TOP_LEVEL_MANIFEST.txt`), enforced by a new, tracked, pure-stdlib
script (`tools/check_docs_zone.py`) wired into the CI `governance` job
(`.github/workflows/ci.yml`) as a required check on every PR, plus both local hooks
(`pre-commit`, `pre-push`) for earlier feedback.

This is structurally different from a denylist, not just another rule added to one: any
top-level `docs/` entry not on the manifest fails **regardless of its name or content** — there
is no phrasing to guess correctly in advance, because the check is set membership, not pattern
matching. Adding a genuinely public document requires a one-line, reviewable diff to the
manifest in the same PR that adds it — a deliberate act, not a default. Internal content stays
in this repository's local-only planning-notes location (git-ignored), which is on the manifest
only so a contributor's local checkout (where that directory exists on disk) doesn't spuriously
fail.

Scope is deliberately narrow: only *top-level* entries directly under `docs/`. Nested pages
under the Diataxis directories (`explain/`, `how-to/`, `learn/`, `reference/`, `process/`) are
governed by other means already in place — the doc-drafter → … → doc-finalizer pipeline (with
its own `leak-guard` gate before publish), `mkdocs build --strict`, and `docs_src` coverage
(NFR-DOCS-002). The failure this closes is specifically an ad hoc top-level file dropped outside
all of that, which is exactly what happened.

The existing local content regex scanner is **not replaced** — it stays as defense-in-depth for
content leaked *within* an otherwise-legitimate page (e.g. a stray internal aside inside a real
chapter), which a top-level allowlist cannot catch. The two are complementary: the allowlist is
exhaustive-by-construction for *misplaced files*; the denylist is best-effort for *misplaced
content within an allowed file*.

## Verification (real numbers, not assumed)

- `docs/RIG_MULTIOBJECT_IMPLEMENTATION_PLAN.md` moved out of the tracked tree to a local-only
  location in this change. The pre-existing local scanner reported 23 blockers from this file
  before the move (its "phase N" pattern) plus 1 from a stale reference in
  `ADR-0011-rig-multiobject-merge.md` (line 130, reworded here as an unrelated fix); after the
  move + reword: 0 blockers.
- `python3 tools/check_docs_zone.py` against the real `docs/` tree: `DOCS ZONE: OK (21 top-level
  docs/ entries, all on the manifest)`.
- Simulated the exact failure mode this closes: dropped an unlisted file
  (`docs/SOME_SNEAKY_INTERNAL_NOTE.md`) into `docs/` and re-ran the check —
  `DOCS ZONE: FAIL — new top-level docs/ entries not on the manifest: - docs/SOME_SNEAKY_INTERNAL_NOTE.md`,
  exit code 1. Removed the file; check passes again.
- `ruff check .`, `lint-imports`, `mypy ds_msp/core --follow-imports=silent
  --ignore-missing-imports`, `check_traceability.py --check`, `check_tree_hygiene.py`,
  `check_docs_src_coverage.py` all pass with the new files present.

## Consequences

**Positive**
- The check runs in CI on a clean, freshly-checked-out worktree — immune by construction to the
  class of local-machine issue that let this leak go undetected (broken hook, `--no-verify`,
  never-installed hooks, wrong-branch validation in a worktree).
- Absence from the manifest is itself the failure signal, so it cannot miss a new leak due to
  unfamiliar wording the way a content scanner can.
- Cheap: pure stdlib, one `Path.iterdir()` call, no network/build dependency; adds negligible
  time to CI or either local hook.

**Negative / costs**
- Every future *legitimately public* top-level `docs/` addition requires remembering to update
  the manifest, or CI fails — a small, deliberate friction cost, which is the point (it forces a
  reviewable decision instead of a silent default).
- Only covers the top level of `docs/`; a leak nested inside an allowed subdirectory (e.g. an
  internal aside added to a real `docs/explain/` page) is not caught by this check — that
  remains the content scanner's and the doc pipeline's job, per Decision above.

## Scope explicitly deferred (not accidental omissions)

- **Frontmatter/metadata self-declaration** (e.g. requiring every page to declare
  `audience: public`) — considered and not built; the allowlist already gives an exhaustive
  guarantee for the demonstrated failure mode (a misplaced top-level file) with far less
  machinery. Would only add value for catching leaked content *inside* an allowed file, which is
  the content scanner's job already.
- **Extending the allowlist to nested directories** (per-subdirectory manifests) — not needed
  today; those trees already have their own gates (doc pipeline, `docs_src` coverage, mkdocs
  strict build).
- **Promoting the pre-existing local denylist scanner's full pattern list into a tracked,
  CI-run script** — not done: that scanner's specific patterns intentionally name private
  codenames/paths, so tracking it would itself be a leak. Only the zoning/allowlist mechanism —
  which names nothing private — is promoted here.

## Alternatives considered

- *Add "implementation plan"/"phase N" as more denylist patterns and call it fixed* — rejected:
  this is the exact failure mode being closed, not a fix for it. It narrows today's known gap
  without changing the fact that the *next* unfamiliar phrasing still gets through silently.
- *Frontmatter-tag every doc file, fail the build if a file lacks a valid tag* — considered (a
  positive-declaration scheme, structurally similar in spirit); rejected for now as more
  machinery than the demonstrated failure mode needs — see Scope-deferred above. Revisit if a
  future leak is found *inside* an already-allowed file, which the allowlist cannot catch.
- *A file-naming convention check* (flag `*_PLAN.md`/`*_IMPLEMENTATION*.md` outside this
  repository's local-only planning-notes location) — rejected as the sole mechanism: still a
  denylist, just over filenames instead of content, with the same fundamental gap (an internal
  doc with an innocuous name still gets through). An allowlist has no such gap by construction.
