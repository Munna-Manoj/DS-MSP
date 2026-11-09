// Client for the `ds-msp studio` JSON API (ds_msp/studio/server.py). Types cover the fields
// the viewer reads; the documents carry more.

export type Level = "PASS" | "WARN" | "FAIL" | string;

export interface RunSummary {
  id: string;
  root: number;
  path: string;
  name: string;
  run_id?: string | null;
  created_utc?: string | null;
  n_cameras?: number | null;
  n_frames?: number | null;
  models?: string[];
  fit_level?: Level | null;
  median_px?: number | null;
  p95_px?: number | null;
  solver_status?: string | null;
  n_weak?: number | null;
  lens_failed?: string[] | null;
  accuracy_status?: string | null;
  error?: string;
}

export interface ErrorStats {
  n: number;
  mean: number;
  median: number;
  p95: number;
  max: number;
  rms: number;
  inlier_frac?: number;
  n_gross?: number;
}

export interface Intrinsics {
  cam_id: number;
  model: string;
  width: number;
  height: number;
  params: { name: string; value: number; sigma: number | null }[];
}

export interface Extrinsics {
  cam_id: number;
  is_ref: boolean;
  T_c_ref: number[][];
  center_in_ref: number[];
  baseline: number;
  rot_deg: number;
}

export interface LensCam {
  frames: number;
  sigma_median_px: number | null;
  sigma_p95_px?: number | null;
  sigma_median_widened_px: number | null;
  t_factor: number | null;
  params_at_bound: string[];
  ok: boolean;
  reasons: string[];
}

export interface CrossvalCam {
  frames_halves: number[];
  applicable: boolean;
  error_estimate_px?: number | null;
  ok?: boolean;
}

export interface AuditFinding {
  kind: string;
  cam: number | null;
  cams?: number[] | null;
  params?: string[];
  ratio?: number;
  message?: string;
  hint?: string;
}

export interface PairSigma {
  rotation_sigma_deg: number | null;
  centre_sigma: number | null;
}

export interface Report {
  fit?: { level: Level; message: string; basis?: string };
  capture?: { n_cameras: number; ref_cam_id: number; n_objects: number; n_frames: number;
    n_observations: number; n_corners: number };
  models?: Record<string, string>;
  per_camera_errors_px?: Record<string, ErrorStats>;
  overall_errors_px?: ErrorStats;
  intrinsics?: Intrinsics[];
  extrinsics?: Extrinsics[];
  audit?: { n_weak: number; n_soft: number; findings: AuditFinding[];
    coverage?: Record<string, { tilt_range_deg?: number; periphery_frac?: number }> } | null;
}

export interface Acceptance {
  accuracy_status?: string;
  profile?: string;
  min_camera_coverage?: number;
  coverage?: Record<string, { coverage: number; n_observed: number }> | null;
  rejected_observations?: Record<string, unknown>[];
  config_warnings?: string[];
  solver?: { final_stage?: string; final_status?: string;
    stages?: { stage: string; status: string; iterations: number; rms: number }[] } | null;
  extrinsic_uncertainty?: { units?: Record<string, string>; pairs?: Record<string, PairSigma>;
    within_limits?: boolean | null } | null;
  observability?: { n_weak: number; findings: AuditFinding[] } | null;
  lens_gate?: string;
  lens?: { limit_px: number; failed: string[]; cameras: Record<string, LensCam> } | null;
  crossval_gate?: string;
  crossval?: { applicable: boolean; reason?: string; limit_px?: number; failed?: string[];
    cameras?: Record<string, CrossvalCam> } | null;
  diagnostics?: { outliers?: Record<string, { outlier_frac: number | null }>;
    board_flatness?: Record<string, Record<string, { n_corners: number; sag_ptp: number | null; rms: number | null }>>;
    files?: Record<string, string> } | null;
}

export type VerdictLevel = "accept" | "check" | "improve" | "reject";

export interface VerdictAction {
  level: VerdictLevel;
  text: string;
  cams: string[];
  evidence: "lens" | "frames" | "residuals" | "rig" | "links" | "tech" | null;
  check: string;
}

export interface Verdict {
  level: VerdictLevel;
  headline: string;
  cameras: Record<string, { level: VerdictLevel; reasons: string[] }>;
  rig_reasons: { level: VerdictLevel; reason: string; check: string }[];
  actions: VerdictAction[];
  n_actions: number;
  max_shown: number;
  outlier_check_frac: number;
}

export interface CameraDiagnostics {
  cam: number;
  width: number | null;
  height: number | null;
  band_px: number | null;
  outliers: { n_observed: number; n_valid: number; n_within: number; outlier_frac: number | null } | null;
  residuals: { n_total: number; n_valid: number; n_shown: number; u: number[]; v: number[];
    dx: number[]; dy: number[]; within: boolean[]; rms_px: number | null } | null;
  worst_frames: { frame: number; n: number; rms_px: number; max_px: number; n_outside: number }[] | null;
  n_frames?: number;
  lens: { nx: number; ny: number; u: number[]; v: number[]; sigma_px: (number | null)[];
    unbounded: boolean; observed: boolean[]; limit_px: number | null;
    zones?: { answer: string; zones: ("trusted" | "extrapolated" | "uncertain" | "none")[];
      share: { trusted: number; extrapolated: number; uncertain: number }; limit_px: number;
      sigma_median_px: number | null; sigma_max_trusted_px: number | null } | null } | null;
  pattern?: { kind: "none" | "edges" | "regional"; answer: string; rms_px: number; typical_px: number;
    edge_growth: number | null;
    rings: { from: number; to: number; n: number; median_px: number | null; p90_px: number | null }[];
    grid: { nx: number; ny: number; cells: { i: number; j: number; n: number; mx: number | null; my: number | null; biased: boolean }[] };
  } | null;
}

export interface RunDoc {
  schema: string;
  schema_version: number;
  id: string;
  name: string;
  summary: RunSummary;
  report: Report;
  acceptance: Acceptance | null;
  manifest: {
    run_id?: string;
    created_utc?: string;
    software?: Record<string, unknown>;
    configuration?: { resolved?: Record<string, unknown> };
  } | null;
  files: { name: string; bytes: number }[];
  verdict?: Verdict;
  has_diagnostics?: boolean;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(path, { credentials: "same-origin" });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => getJSON<{ ok: boolean; version: string; api_version: number }>("api/health"),
  runs: () => getJSON<RunSummary[]>("api/runs"),
  run: (id: string) => getJSON<RunDoc>(`api/run?id=${encodeURIComponent(id)}`),
  diagnostics: (id: string, cam: string) =>
    getJSON<CameraDiagnostics>(`api/run/diagnostics?id=${encodeURIComponent(id)}&cam=${encodeURIComponent(cam)}`),
  fileUrl: (id: string, name: string) =>
    `api/run/file?id=${encodeURIComponent(id)}&name=${encodeURIComponent(name)}`,
};
