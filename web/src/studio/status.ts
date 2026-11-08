// Per-camera status, from checks the library itself records. Only these set a status:
//  - the lens check (lens 1-sigma and parameters at a bound; validated on the synthetic
//    starved-camera matrix, 128 of 139 bad lenses flagged),
//  - per-camera coverage below the configured minimum,
//  - the opt-in split check, when it was run,
//  - an observability finding that names the camera.
// Anything else is shown as a number, not turned into a verdict.

import type { AuditFinding, RunDoc } from "./api";

export type CamState = "ready" | "attention" | "check";

export interface CamStatus {
  id: string;
  state: CamState;
  /** One-line reasons, in plain words. */
  reasons: string[];
}

export function camIds(doc: RunDoc): string[] {
  const n = doc.report.capture?.n_cameras ?? doc.report.intrinsics?.length ?? 0;
  return Array.from({ length: n }, (_, i) => String(i));
}

export function findings(doc: RunDoc): AuditFinding[] {
  return doc.acceptance?.observability?.findings ?? doc.report.audit?.findings ?? [];
}

export function findingsFor(doc: RunDoc, cam: string): AuditFinding[] {
  const c = Number(cam);
  return findings(doc).filter((f) => f.cam === c || (f.cams ?? []).includes(c));
}

export function cameraStatus(doc: RunDoc, cam: string): CamStatus {
  const reasons: string[] = [];
  let state: CamState = "ready";
  const lens = doc.acceptance?.lens?.cameras?.[cam];
  if (lens && !lens.ok) {
    state = "attention";
    if (lens.reasons.includes("sigma") || lens.sigma_median_widened_px == null) {
      reasons.push(
        lens.sigma_median_widened_px == null
          ? `Lens not determined from ${lens.frames} photo${lens.frames === 1 ? "" : "s"}`
          : `Lens uncertain: at least ±${lens.sigma_median_widened_px.toFixed(2)} px`,
      );
    }
    if (lens.params_at_bound.length)
      reasons.push(`Lens parameter at its limit: ${lens.params_at_bound.join(", ")}`);
  }
  const cov = doc.acceptance?.coverage?.[cam];
  const minCov = doc.acceptance?.min_camera_coverage;
  if (cov && minCov != null && cov.coverage < minCov) {
    state = "attention";
    reasons.push(`Fit explains ${(cov.coverage * 100).toFixed(0)}% of its corners`);
  }
  const cv = doc.acceptance?.crossval?.cameras?.[cam];
  if (cv && cv.applicable && cv.ok === false) {
    if (state === "ready") state = "check";
    reasons.push(`Split check: halves disagree (≈${cv.error_estimate_px?.toFixed(2)} px)`);
  }
  if (findingsFor(doc, cam).length) {
    if (state === "ready") state = "check";
    reasons.push("Weakly linked to the rest of the rig");
  }
  return { id: cam, state, reasons };
}

export const STATE_LABEL: Record<CamState, string> = {
  ready: "Ready",
  attention: "Needs photos",
  check: "Check",
};
