// Per-camera status. The server computes it (ds_msp/studio/verdict.py) from checks the library
// itself records: the lens check, corner coverage, the opt-in split check, observability
// findings, pair uncertainty, solver status and the outlier share (a provisional 3% "check").
// Results written before the verdict existed fall back to the lens/coverage/split/findings rules.

import type { AuditFinding, RunDoc, VerdictLevel } from "./api";

export type CamState = VerdictLevel;

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
  const v = doc.verdict?.cameras?.[cam];
  if (v) return { id: cam, state: v.level, reasons: v.reasons };
  const reasons: string[] = [];
  let state: CamState = "accept";
  const lens = doc.acceptance?.lens?.cameras?.[cam];
  if (lens && !lens.ok) {
    state = lens.sigma_median_widened_px == null ? "reject" : "improve";
    reasons.push(
      lens.sigma_median_widened_px == null
        ? `lens not determined from ${lens.frames} photo${lens.frames === 1 ? "" : "s"}`
        : `lens uncertain: at least ±${lens.sigma_median_widened_px.toFixed(2)} px`,
    );
  }
  const cov = doc.acceptance?.coverage?.[cam];
  const minCov = doc.acceptance?.min_camera_coverage;
  if (cov && minCov != null && cov.coverage < minCov) {
    state = "reject";
    reasons.push(`the fit explains only ${(cov.coverage * 100).toFixed(0)}% of its corners`);
  }
  if (findingsFor(doc, cam).length) {
    if (state === "accept") state = "improve";
    reasons.push("weakly tied to the rest of the rig");
  }
  return { id: cam, state, reasons };
}

export const STATE_LABEL: Record<CamState, string> = {
  accept: "Ready",
  check: "Check",
  improve: "Improve",
  reject: "Not ready",
};

export const STATE_TONE = { accept: "ok", check: "info", improve: "warn", reject: "bad" } as const;
