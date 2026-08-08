# Change & Release Management `[CRM]`

> How changes are proposed, versioned, and released (ISO/IEC/IEEE 12207 §6.3.5/§6.4.10). DS-MSP uses
> **Conventional Commits → release-please → PyPI (OIDC)**; SemVer is derived from commit history, not
> set by hand.

## 1. Change requests

A change request is a **GitHub issue** (enhancement) or, for anything architecturally significant, an
**ADR** ([../architecture/decisions/INDEX.md](../architecture/decisions/INDEX.md)). The change is
realized on a branch ([BRANCHING_CONTRIBUTION.md](BRANCHING_CONTRIBUTION.md)) and merged via a PR that
meets the [Definition of Done](../quality/DEFINITION_OF_DONE.md).

## 2. Conventional Commits → versions

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/). release-please
parses them to compute the next SemVer and update `CHANGELOG.md`:

| Prefix | Effect on version | Use for |
|--------|-------------------|---------|
| `fix:` | patch (x.y.**z**) | bug fix |
| `feat:` | minor (x.**y**.0) | new backward-compatible feature |
| `feat!:` / `BREAKING CHANGE:` footer | major (**x**.0.0) | breaking change to a public interface (IFC-*) |
| `docs:` / `chore:` / `refactor:` / `test:` | no release on its own | non-shipping work |

A breaking change to any public interface in [`../srs/interfaces.md`](../srs/interfaces.md) **must** use
the `!` / `BREAKING CHANGE` form and note the migration in the PR and `CHANGELOG.md`.

Every commit ends with the project attribution trailer (`Co-Authored-By: …`); attribution is **not**
internal-process content and is kept.

## 3. Release flow

1. Merges to `main` accumulate; release-please maintains a **release PR** (version bump + changelog).
2. **Before merging the release PR for a release-gated change**, run
   `tools/check_traceability.py --release` and retain evidence that every linked `realdata` test
   actually executed (not skipped) and passed. The dataset-gated nightly job can be dispatched for
   this purpose, but an all-skipped green run is not validation (ADR-0006).
3. Merging the release PR creates the tag + GitHub Release and triggers the PyPI publish via Trusted
   Publishing (OIDC) — no stored token (CON-07). See [`RELEASING.md`](../../../RELEASING.md).
4. Docs are published by `deploy-pages.yml`.

## 4. Release IDs & traceability

Releases are `REL-vX.Y.Z`. The chain `FR/NFR ↔ ARC ↔ code ↔ test ↔ ISS ↔ REL` lets any shipped
behaviour be traced from a release tag back to the requirement and the test that validated it.

## 5. Hotfixes

A critical (S1) defect on a released version is fixed with a `fix:` commit on a branch off `main`,
fast-tracked through the same gates; release-please cuts the patch release on merge.
