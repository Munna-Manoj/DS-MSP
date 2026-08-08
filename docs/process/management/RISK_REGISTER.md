# Risk Register `[RSK]`

> Active engineering and process risks, their mitigations, and status. The canonical machine list is
> [`risks.csv`](risks.csv); this document adds the narrative and points each risk at the control that
> manages it. Reviewed when a release is planned or a new risk is found.

## Scoring

Likelihood × Impact, each {low, medium, high}. A risk is **mitigated** when a control is in place and
verified; **planned** when the control is agreed but not yet fully wired; **open** when active work is
still reducing it.

## Register

| ID | Risk | Cat. | L | I | Status | Mitigation / control |
|----|------|------|---|---|--------|----------------------|
| RSK-01 | Calibration converges to a wrong basin (DS fold / EUCM α→1) | technical | M | H | **open** | Model-aware multi-start + UCM bootstrap + GNC; release-gated real-data validation (NFR-NUM-004) |
| RSK-02 | Planar two-fold (mirror) pose ambiguity inflates mean/p95 error | technical | M | H | **open** | Homography-based two-fold disambiguation in `geometry.resection`; synthetic planted-flip test |
| RSK-03 | Internal R&D / process content leaks into the public repo | process | L | H | **mitigated** | Defence in depth: `.gitignore` + the CI tree-hygiene gate + a local pre-push guard |
| RSK-04 | Analytic Jacobian regression slips in undetected | technical | L | H | **mitigated** | `-m jac` gradient-check at 1e-6 on every model PR (NFR-NUM-001) |
| RSK-05 | Architecture erodes (cycles / cv2 in math path) | process | M | M | **mitigated** | import-linter contracts + `test_independence.py` (NFR-ARCH-001/002) |
| RSK-06 | Requirements and tests drift out of sync | process | M | M | **mitigated** | CI traceability gate (`check_traceability.py`) fails on orphan/dangling links and an out-of-date generated matrix |
| RSK-07 | A release ships without real-data validation | process | L | H | **planned** | Structural release check + nightly/on-demand real-data runner are active; a non-skipping result is not yet an automated dependency of `release.yml` |
| RSK-08 | Third-party dependency CVE (numpy/opencv/scipy/pyyaml) | security | L | M | **planned** | Pinned dev deps; `SECURITY.md` reporting; dependency review |

## Notes on the open risks

- **RSK-01 / RSK-02** are the focus of the in-flight robust auto-calibration work; they move to
  *mitigated* when the real-data validation (NFR-NUM-004) is green and the planted-flip / multi-start
  tests are in CI. Until then ADR-0006 requires real-data evidence before release; RSK-07 records the
  remaining automation gap in enforcing that policy.
- **RSK-06** is *mitigated*: the `governance` CI job runs `check_traceability.py --check` on every PR
  and push to `main`, rejecting orphan requirements, dangling test links, incomplete ADR indexing,
  and a stale generated matrix.
- **RSK-07** stays *planned*: `check_traceability.py --release` provides the structural check and
  `nightly.yml` provides a scheduled/on-demand real-data runner, but `release.yml` does not depend on
  a non-skipping result. A green all-skipped dataset-gated run is not evidence, so the maintainer must
  confirm actual execution manually until that dependency is wired.

## Maintenance

Add a risk when a near-miss, a recurring defect, or a design discussion surfaces one. Update status
when a control lands or is verified. A risk that materialises becomes a tracked defect
([ISSUE_DEFECT_PROCESS.md](ISSUE_DEFECT_PROCESS.md)) and keeps its RSK entry until the control proves
durable.
