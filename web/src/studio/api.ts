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
  /** Views ("cam:frame" or "cam:frame:board") left out of this calibration on request. */
  excluded_views?: string[];
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
  /** The folder holds the calib_param.yml it was calibrated with. */
  has_config?: boolean;
}

// -- a result on its own capture ---------------------------------------------------------------

export interface SceneDoc {
  boards: { id: number; points: number[][]; outline: number[][] }[];
  /** T maps target coordinates into the reference camera's frame. */
  frames: { frame: number; T: number[][] }[];
  /** How photos are matched to frames: recorded in the result, assumed from file numbers, or none. */
  photos: "recorded" | "assumed" | "none";
}

export interface FrameStat { frame: number; n: number; rms_px: number; max_px: number; n_outside: number }

export interface FrameCorners {
  cam: number; frame: number; n: number;
  u: number[]; v: number[]; dx: number[]; dy: number[]; within: boolean[];
  photo: boolean;
}

export interface PairDoc {
  ok: boolean; reason?: string;
  distance: number; position: number[];
  yaw_deg: number; pitch_deg: number; roll_deg: number; angle_deg: number;
  rotation_sigma_deg: number | null; position_sigma: number | null;
}

export interface TaskDoc {
  ok: boolean; reason?: string;
  distance: number; baseline: number; pixel_noise: number;
  depth: number; sideways: number;
  from_cameras: { depth: number; sideways: number };
  from_pixels: { depth: number; sideways: number };
  curve?: { distance: number; depth: number; sideways: number }[];
}

// -- setting up and running a calibration -------------------------------------------------

export interface SchemaProp {
  type: string | string[];
  default: unknown;
  description: string;
  enum?: unknown[];
  minimum?: number;
  items?: { type: string };
  "x-group": string;
  "x-level": "basic" | "expert";
}

export interface ConfigSchema {
  properties: Record<string, SchemaProp>;
  "x-groups": { id: string; title: string }[];
  /** Per profile, the defaults it uses in place of each property's own default. */
  "x-profiles"?: Record<string, Values>;
}

export type Values = Record<string, unknown>;
export interface Problem { key: string; message: string }

export interface BrowseDoc {
  ref: string | null;
  path: string | null;
  name: string;
  parent: string | null;
  images: number;
  dirs: { name: string; ref: string; path: string; images: number; result: boolean }[];
  files: { name: string; path: string }[];
}

export interface ScanDoc {
  cam_prefix: string | null;
  number_camera: number;
  cameras: { index: number; folder: string; images: number; width: number | null;
    height: number | null; frames: number }[];
  shared_frames: number;
  problems: string[];
}

export interface BoardCheck {
  cameras: { index: number; folder: string; tried: number; found: number;
    corners_max: number; corners_of: number;
    /** The tried photo that showed the most corners, and where they were found in it. */
    sample: { photo: number; file: string; width: number; height: number; corners: [number, number][] } | null }[];
  found: number;
  tried: number;
  message: string;
}

/** The board size Studio read from the markers in a few photos. */
export interface BoardGuess {
  ok: boolean; number_x_square: number | null; number_y_square: number | null;
  /** How many different boards appeared in the photos that were read. */
  boards_seen: number; markers: number; photos: number; message: string;
  /** Marker width over square width as measured in the photos; null when they do not settle it. */
  marker_fill?: number | null;
  /** How many pairs of touching markers the measurement rests on. */
  marker_fill_pairs?: number;
}

export interface ConfigCheck {
  ok: boolean;
  problems: Problem[];
  warnings: string[];
  yaml: string;
  values: Values;
}

export type JobState = "queued" | "running" | "done" | "failed" | "cancelled" | "interrupted";

export interface JobBrief {
  id: string;
  name: string;
  state: JobState;
  created_utc: string;
  started_utc: string | null;
  finished_utc: string | null;
  exit_code: number | null;
  error: string | null;
  has_result: boolean;
  queue_position: number | null;
  /** The earlier result whose corner detections this calibration reuses. */
  reused_from?: string | null;
}

export interface JobProgress {
  phase: "starting" | "detecting" | "solving" | "saving" | "done";
  stage: string | null;
  stage_index: number;
  stages: { name: string; t: number }[];
  detect: Record<string, [number, number]> | null;
  save: Record<string, [number, number]> | null;
  iter: { it: number; max_iter: number; rms: number | null; cost: number | null } | null;
  elapsed: number | null;
  history: [number, number][];
  scene?: { cameras: number; observations: number };
  final_rms?: number | null;
}

export interface RepeatCompare { ok: boolean; reason?: string; same?: boolean; rows?: { what: string; same: boolean }[] }

export interface JobDoc extends JobBrief {
  progress: JobProgress;
  command: string;
  log: string[];
  /** Set on a job that repeats a result: its id, and once done whether it reproduced it. */
  repeat_of?: string | null;
  repeat?: RepeatCompare;
}

export interface KeypointsScan {
  path: string;
  number_camera: number;
  number_board: number;
  cameras: { index: number; width: number; height: number; frames: number; sightings: number; boards: number[] }[];
  pixel_convention: string | null;
  reused_target: string | null;
  problems: string[];
}

export interface IntrinsicsScan {
  path: string;
  cameras: { index: number; model: string | null; fx: number; fy: number; cx: number; cy: number;
    width: number | null; height: number | null }[];
}

export interface LensAdvice {
  cam: number; views: number; chosen: string;
  verdict: "ok" | "misfit" | "caution" | "unknown"; message: string; suggested: string | null;
  fov_h_deg: number | null; measured_to_deg: number | null;
  reference_px: number | null; chosen_px: number | null;
}

export interface LensHealth {
  cam: number; verdict: "ok" | "misfit" | "unknown"; message: string;
  given_px: number | null; reference_px: number | null; views: number;
}

/** How far a result's corners lie from where its calibration expects them. */
export interface ErrorSpread {
  n: number; median_px?: number; max_px?: number;
  /** above[k] = corners further than grid_px[k] pixels. */
  grid_px: number[]; above: number[];
  dx: number[]; dy: number[];
}

export interface CaptureHistory {
  root_path: string;
  runs: { id: string; name: string; created_utc: string | null; n_cameras: number | null;
    models: string[]; median_px: number | null; has_config: boolean;
    reuse?: { ok: boolean; why: string } }[];
  /** The newest earlier result whose corner detections fit the current settings. */
  reusable: string | null;
}

export interface Fingerprint {
  ok: boolean; reason?: string;
  run_id?: string; created_utc?: string; configuration_sha256?: string;
  config_path?: string; overrides?: string[];
  n_inputs?: number; input_bytes?: number;
  observations?: { n_corners?: number; n_observations?: number; sha256?: string } | null;
  software?: { version?: string | null; commit?: string | null; dirty?: boolean | null };
  python?: string | null;
  can_repeat?: boolean;
}

export interface InputCheck {
  ok: boolean; checked: number; same: boolean;
  n_missing: number; n_changed: number; n_appeared: number;
  missing: string[]; changed: string[]; appeared: string[];
}

/** An error from the API; `problems` names the settings at fault when the server listed them. */
export class ApiError extends Error {
  problems: Problem[];
  constructor(message: string, problems: Problem[] = []) {
    super(message);
    this.problems = problems;
  }
}

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    let problems: Problem[] = [];
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
      problems = body.problems ?? [];
    } catch {
      /* not JSON */
    }
    throw new ApiError(typeof detail === "string" ? detail : `${res.status}`, problems);
  }
  return (await res.json()) as T;
}

async function getJSON<T>(path: string): Promise<T> {
  return parse<T>(await fetch(path, { credentials: "same-origin" }));
}

// The header marks the request as sent by this page: the server refuses changes without it.
async function postJSON<T>(path: string, body: unknown): Promise<T> {
  return parse<T>(await fetch(path, {
    method: "POST",
    credentials: "same-origin",
    headers: { "content-type": "application/json", "x-dsmsp-studio": "1" },
    body: JSON.stringify(body),
  }));
}

export const api = {
  health: () =>
    getJSON<{ ok: boolean; version: string; api_version: number; can_run?: boolean }>("api/health"),
  schema: () => getJSON<ConfigSchema>("api/schema/rig-config"),
  browse: (ref: string | null) =>
    getJSON<BrowseDoc>(ref == null ? "api/browse" : `api/browse?ref=${encodeURIComponent(ref)}`),
  scanPhotos: (root_path: string, cam_prefix?: string) =>
    postJSON<ScanDoc>("api/photos/scan", { root_path, cam_prefix }),
  checkBoard: (values: Values) => postJSON<BoardCheck>("api/photos/check-board", { values }),
  findBoard: (values: Values) => postJSON<BoardGuess>("api/photos/find-board", { values }),
  /** A scaled-down photo of one camera in a photo folder that has no result yet. */
  thumbUrl: (root: string, cam: number, photo = 0, prefix?: string | null) =>
    `api/photos/thumb?path=${encodeURIComponent(root)}&cam=${cam}&photo=${photo}${prefix ? `&prefix=${encodeURIComponent(prefix)}` : ""}`,
  checkConfig: (values: Values) => postJSON<ConfigCheck>("api/config/check", { values }),
  importConfig: (text: string) =>
    postJSON<{ values: Values; unknown: string[]; problems: Problem[] }>("api/config/import", { text }),
  configOf: (id: string) =>
    getJSON<{ values: Values; unknown: string[]; problems: Problem[]; name: string; changed?: boolean }>(
      `api/config?id=${encodeURIComponent(id)}`),
  sourceConfig: (path: string) =>
    getJSON<{ values: Values; unknown: string[]; problems: Problem[]; path: string }>(
      `api/source/config?path=${encodeURIComponent(path)}`),
  runErrors: (id: string) => getJSON<ErrorSpread>(`api/run/errors?id=${encodeURIComponent(id)}`),
  sourceKeypoints: (path: string) => getJSON<KeypointsScan>(`api/source/keypoints?path=${encodeURIComponent(path)}`),
  sourceIntrinsics: (path: string) => getJSON<IntrinsicsScan>(`api/source/intrinsics?path=${encodeURIComponent(path)}`),
  fingerprint: (id: string) => getJSON<Fingerprint>(`api/run/fingerprint?id=${encodeURIComponent(id)}`),
  fingerprintCheck: (id: string) => getJSON<InputCheck>(`api/run/fingerprint/check?id=${encodeURIComponent(id)}`),
  repeatJob: (id: string, root: number, name: string) => postJSON<JobDoc>("api/jobs/repeat", { id, root, name }),
  jobs: () => getJSON<JobBrief[]>("api/jobs"),
  job: (id: string) => getJSON<JobDoc>(`api/job?id=${encodeURIComponent(id)}`),
  startJob: (root: number, name: string, values: Values, reuse?: string | null) =>
    postJSON<JobDoc>("api/jobs", { root, name, values, reuse: reuse ?? undefined }),
  lensAdvice: (values: Values, cam: number) => postJSON<LensAdvice>("api/photos/lens-advice", { values, cam }),
  lensHealth: (values: Values, cam: number) => postJSON<LensHealth>("api/photos/lens-health", { values, cam }),
  captureHistory: (values: Values) => postJSON<CaptureHistory>("api/capture/history", { values }),
  cancelJob: (id: string) => postJSON<{ id: string; state: JobState }>("api/job/cancel", { id }),
  runs: () => getJSON<RunSummary[]>("api/runs"),
  run: (id: string) => getJSON<RunDoc>(`api/run?id=${encodeURIComponent(id)}`),
  diagnostics: (id: string, cam: string) =>
    getJSON<CameraDiagnostics>(`api/run/diagnostics?id=${encodeURIComponent(id)}&cam=${encodeURIComponent(cam)}`),
  scene: (id: string) => getJSON<SceneDoc>(`api/run/scene?id=${encodeURIComponent(id)}`),
  frames: (id: string, cam: string) =>
    getJSON<{ cam: number; frames: FrameStat[] }>(`api/run/frames?id=${encodeURIComponent(id)}&cam=${cam}`),
  frame: (id: string, cam: string, frame: number) =>
    getJSON<FrameCorners>(`api/run/frame?id=${encodeURIComponent(id)}&cam=${cam}&frame=${frame}`),
  photoUrl: (id: string, cam: string | number, frame: number, w = 1280) =>
    `api/run/photo?id=${encodeURIComponent(id)}&cam=${cam}&frame=${frame}&w=${w}`,
  pair: (id: string, a: string | number, b: string | number) =>
    getJSON<PairDoc>(`api/run/pair?id=${encodeURIComponent(id)}&a=${a}&b=${b}`),
  task: (id: string, a: string | number, b: string | number, distance: number, pixelNoise: number) =>
    getJSON<TaskDoc>(`api/run/task?id=${encodeURIComponent(id)}&a=${a}&b=${b}&distance=${distance}&pixel_noise=${pixelNoise}`),
  fileUrl: (id: string, name: string) =>
    `api/run/file?id=${encodeURIComponent(id)}&name=${encodeURIComponent(name)}`,
};
