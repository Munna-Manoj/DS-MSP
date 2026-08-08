# DS-MSP Engineering Handbook `[HBK]`

> The master charter for how DS-MSP is engineered. It indexes the whole Software Engineering
> Management System (SEMS), maps it to the standards it draws on, and tells **any contributor — human
> or AI — exactly what to do**. If you read one file, read this one, then follow the links.

## 1. What this system is (and isn't)

A lightweight, **standards-informed** governance layer that makes every change *controlled, traceable,
verified on synthetic data, and validated on real data before it goes public* — without certification
ceremony. It is informed by, not certified against:

| Concern | Standard | Where it lands |
|---------|----------|----------------|
| Requirements | ISO/IEC/IEEE 29148 | [SRS](srs/SRS.md) + [`requirements.csv`](srs/requirements.csv) |
| Life cycle | ISO/IEC/IEEE 12207 | [CI/CD](management/CICD_PIPELINE.md), [Branching](management/BRANCHING_CONTRIBUTION.md) |
| Test process | ISO/IEC/IEEE 29119 | [QA & V&V](quality/QA_VV_PLAN.md), [Test levels](quality/test-levels.md) |
| Defects | IEEE 1044 | [Issue & defect process](management/ISSUE_DEFECT_PROCESS.md) |
| Architecture | ISO/IEC/IEEE 42010 + ADRs | [Architecture](architecture/ARCHITECTURE.md), [ADRs](architecture/decisions/INDEX.md) |
| Traceability | Automotive-SPICE-style | [`tools/check_traceability.py`](../../tools/check_traceability.py), [matrix](traceability/TRACEABILITY.md) |

## 2. Document map

```
docs/process/
├─ HANDBOOK.md                      ← you are here (index + charter)
├─ srs/
│  ├─ SRS.md                        scope, stakeholders, constraints, FR/NFR narrative
│  ├─ requirements.csv              canonical FR/NFR registry (machine-checked)
│  ├─ stakeholders.csv  constraints.csv
│  └─ interfaces.md                 public API + file-format surface (IFC-*)
├─ architecture/
│  ├─ ARCHITECTURE.md               layered stack + enforcement
│  ├─ components.csv                ARC-* components + depends_on
│  └─ decisions/INDEX.md + ADR-NNNN records
├─ quality/
│  ├─ QA_VV_PLAN.md  DEFINITION_OF_DONE.md  test-levels.md
├─ management/
│  ├─ CICD_PIPELINE.md  CHANGE_RELEASE_MGMT.md  BRANCHING_CONTRIBUTION.md
│  ├─ ISSUE_DEFECT_PROCESS.md  RISK_REGISTER.md  risks.csv
├─ playbooks/                       add-a-{camera-model,robust-kernel,io-format,pipeline-capability}.md
│                                   extend-the-rig-pipeline.md
└─ traceability/TRACEABILITY.md     generated bidirectional matrix (CI-checked in sync)
```
Root: [`CONTRIBUTING.md`](../../CONTRIBUTING.md), [`SECURITY.md`](../../SECURITY.md),
`.github/` (CODEOWNERS, PR & issue templates), [`tools/`](../../tools) (the governance checkers).

## 3. ID schemes (the traceability backbone)

`STK-NN` · `FR-<AREA>-NNN` · `NFR-<AREA>-NNN` · `CON-NN` · `ARC-<LAYER>` · `ADR-NNNN` · `RSK-NN` ·
`IFC-NN` · `REL-vX.Y.Z` · `ISS`/`CR` = GitHub issue/PR numbers.
Areas mirror packages and shared capabilities: CORE, MODEL, CALIB, RIG, MVG, STEREO, OPS, ADAPT,
IO, VO, INTEROP; NFR areas: NUM, ARCH, PORT, REPRO, PRIV, PERF, DOCS.

The chain is **`STK → FR/NFR ↔ ARC ↔ code ↔ test ↔ ISS ↔ REL`**. The REQ↔test link is a
`@pytest.mark.req("FR-…")` marker co-located with each test (it can't drift); the matrix joins it all
and CI fails on any break.

## 4. Roles

- **Contributors (human or AI)** — implement under the playbooks + Definition of Done; keep
  traceability green. Start: [CONTRIBUTING.md](../../CONTRIBUTING.md).
- **Reviewers** — enforce the DoD and the gates; CODEOWNERS gates `docs/process/` and ADRs.
- **Maintainer / release owner** — owns the process and the release gate; ensures no unverified
  release and no internal-process leakage.

## 5. How to… (quick routes)

| I want to… | Go to |
|------------|-------|
| Add a camera model / kernel / IO format / capability | the matching [playbook](playbooks/) |
| Know when my change is "done" | [Definition of Done](quality/DEFINITION_OF_DONE.md) |
| Understand the layers / add a cross-layer edge | [Architecture](architecture/ARCHITECTURE.md) + a new [ADR](architecture/decisions/INDEX.md) |
| File a bug / request a feature | `.github/ISSUE_TEMPLATE/` (see [defect process](management/ISSUE_DEFECT_PROCESS.md)) |
| Understand versioning / releases | [Change & release mgmt](management/CHANGE_RELEASE_MGMT.md) |
| Know what CI enforces | [CI/CD pipeline](management/CICD_PIPELINE.md) |
| See open risks | [Risk register](management/RISK_REGISTER.md) |
| Report a security issue | [SECURITY.md](../../SECURITY.md) |

## 6. The two non-negotiables

1. **Nothing reaches `main` unverified.** Code changes pass lint + types + layering + the fast test
   matrix; relevant slow evidence is reviewed before merge; every change passes the always-on
   `governance` gate via a reviewed PR. (ISO 12207 §6.3; enforced in CI and the DoD.)
2. **Nothing ships without the required validation.** Every requirement whose canonical
   `requirements.csv` row has `release_gated=yes` needs *both* synthetic and real-data tests linked,
   and the linked real-data tests must execute (not skip) and pass before release.
   ([ADR-0006](architecture/decisions/ADR-0006-synthetic-real-release-gate.md);
   `check_traceability.py --release` checks the structural links.)

## 7. Keeping the system honest

The governance is self-enforcing: `check_traceability.py` validates requirements, test links, ADRs,
and the generated matrix; `check_tree_hygiene.py` rejects tracked local-only content;
`check_docs_zone.py` enforces the top-level docs allowlist; `check_docs_src_coverage.py` keeps shown
examples source-backed and tested; and `check_packaging.py` checks documented shipping surfaces
against the package. All run in the `governance` CI job on every PR. When you change a workflow, a
requirement, or an interface, update the matching doc **in the same PR** — that is itself part of the
Definition of Done.
