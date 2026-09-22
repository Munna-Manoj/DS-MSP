# ADR-0022 — Attributable rig runs and a locked reference environment

- **Status:** Accepted (recorded 2026-09-22)
- **Deciders:** maintainer
- **Relates to:** ARC-RIG, NFR-REPRO-002, IFC-11
- **Supersedes:** —

## Context

A calibration report previously contained the estimated parameters and residual statistics but not
enough evidence to reproduce how they were obtained. It omitted the Git revision/dirty state, exact
resolved configuration and overrides, numerical-library versions, thread settings, and cryptographic
identities of the source files and effective 2D/3D estimator inputs. Two compatible Python/OpenCV
environments can legitimately produce slightly different numerical results; compatibility and
reproducibility are therefore separate contracts.

Pinning the library's public dependencies globally would make installation unnecessarily brittle.
Conversely, keeping only broad compatibility ranges cannot define a reference environment for golden
baselines and release qualification.

## Decision

1. Every persisted `ds-msp-calibrate-rig` result writes a strict-JSON
   `calibration_manifest.json` beside the calibration report.
2. The manifest records schema/run identity, UTC creation time, replay arguments, seed, Git
   commit/branch/tracked-dirty-state digest plus a separate untracked-path digest, exact
   runtime/dependency versions, numerical-backend digests,
   thread environment, resolved configuration, SHA-256 source-file records, and SHA-256 digests of
   the exact observation and target-geometry arrays consumed by calibration.
3. Output artifacts are hashed only after they are written. The report embeds a small provenance
   summary and the manifest path; the manifest hashes the report, but never itself, avoiding a
   circular digest.
4. `requirements-production.txt` defines the exact release/golden-baseline runtime. The broad
   compatibility ranges in `pyproject.toml` remain the consumer installation contract. A packaged
   `reference_environment.json` lets every run state whether it matches the reference profile, and a
   test prevents the two files from drifting. The profile rejects simultaneous
   `opencv-python`/`opencv-python-headless` installations because both provide the same `cv2` module
   and import resolution is not a trustworthy production contract.
5. `tools/replay_rig_manifest.py` verifies every recorded input hash, replays into a new non-empty-safe
   output directory, and compares effective input digests plus deterministic MC-Calib artifacts.
   Replay never overwrites the original result.
6. Source-file hashes and effective-array hashes are both required. File hashes establish data
   identity; effective hashes reveal decode/detection differences even when source files are equal.

## Consequences

**Positive**

- A calibration is attributable to exact code, inputs, configuration, and numerical environment.
- Golden results have one declared environment without narrowing the public library unnecessarily.
- Replays fail closed on missing or changed inputs and cannot overwrite the evidence being checked.
- OpenCV/detector behavior changes become visible through the effective-observation digest.

**Negative / costs**

- Full SHA-256 hashing adds I/O proportional to the input image volume.
- Manifests contain absolute local paths and environment details by design; they are calibration
  evidence and should be redacted before public sharing when paths are sensitive.
- A dirty worktree can still run, but the report marks it explicitly; production policy may later
  choose to reject it.
- The manifest proves computational identity, not physical calibration accuracy.

## Verification

- `tests/rig/test_provenance.py` changes one pixel/one target coordinate and proves the corresponding
  digest changes, checks source/output hashes, strict JSON, and lock/profile consistency.
- `tests/rig/test_report_deliverable.py` proves the CLI persists the manifest, the report points to
  it, and the manifest hashes the report.
- `python tools/check_traceability.py --check` verifies the requirement/ADR/test links.
- A real config replay is required as the Stage-0 acceptance run; its old/new effective digests and
  deterministic MC-Calib output hashes must match.

## Alternatives considered

- *Store only the command line.* Rejected: it omits dependency, code, data, and effective decode
  identity.
- *Store only source image hashes.* Rejected: identical files can produce different detected points
  under another OpenCV/detector build.
- *Pin `pyproject.toml` dependencies exactly.* Rejected: a reference deployment and a reusable
  library have different compatibility needs.
- *Embed the finalized manifest in the report.* Rejected: hashing the report from a manifest that
  itself embeds the report hash is circular.
