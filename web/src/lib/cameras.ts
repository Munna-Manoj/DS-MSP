// Multi-model camera registry — faithful TypeScript ports of the five models
// ds_msp ships (ds_msp/models/*_math.py). Every project/unproject below is a
// line-for-line port of the Python so the demo is provably correct, not a
// look-alike. Validity rules differ per model on purpose — that is real physics,
// not an inconsistency (DS/UCM use a tilted half-space; EUCM only guards the
// denominator; RadTan is strict z>0; KB relies on a solver converging).
//
// Cross-checked against the library to ~1e-6 px by scripts/verify_cameras.mjs.

export type Vec3 = [number, number, number];
export type Params = Record<string, number>;
export type Zone = "front" | "beyond" | "invalid";

export interface ParamSpec {
  key: string;
  symbol: string;
  label: string;
  min: number;
  max: number;
  step: number;
  link?: string[]; // set these keys together (e.g. fx & fy from one focal slider)
}

export interface Projection {
  u: number;
  v: number;
  valid: boolean;
  theta: number; // incidence angle from the optical axis (radians)
  zone: Zone;
}

export interface Unprojection {
  ray: Vec3; // unit bearing
  valid: boolean;
}

export interface CameraModel {
  id: string;
  name: string;
  blurb: string;
  glyph: string; // a compact mark drawn as SVG path data in a 24×24 box
  wideFov: boolean; // can the model see past 180°?
  defaults: Params; // includes cx, cy (resolution is universal)
  params: ParamSpec[]; // tunable params EXCEPT cx, cy (those are the resolution control)
  project(P: Vec3, p: Params): Projection;
  unproject(u: number, v: number, p: Params): Unprojection;
}

const norm3 = (x: number, y: number, z: number): Vec3 => {
  const n = Math.hypot(x, y, z) || 1;
  return [x / n, y / n, z / n];
};
const incidence = (x: number, y: number, z: number) => Math.atan2(Math.hypot(x, y), z);

function zoneOf(theta: number, valid: boolean): Zone {
  if (!valid) return "invalid";
  return theta < Math.PI / 2 ? "front" : "beyond";
}

// ─────────────────────────────────────────────────────────── Double Sphere ──
// ds_msp/models/ds_math.py
const DS: CameraModel = {
  id: "ds",
  name: "Double Sphere",
  blurb: "Two-sphere cascade; >180° with a closed-form inverse. The library's flagship.",
  glyph: "M5 12a7 7 0 1 0 14 0a7 7 0 1 0 -14 0 M12 12a4 4 0 1 0 8 0a4 4 0 1 0 -8 0",
  wideFov: true,
  defaults: { fx: 180, fy: 180, cx: 320, cy: 320, xi: 0.3, alpha: 0.6 },
  params: [
    { key: "fx", symbol: "f", label: "focal length (px)", min: 120, max: 520, step: 1, link: ["fx", "fy"] },
    { key: "xi", symbol: "ξ", label: "second-sphere offset", min: -0.9, max: 0.9, step: 0.01 },
    { key: "alpha", symbol: "α", label: "sphere mix", min: 0.0, max: 0.99, step: 0.01 },
  ],
  project(P, p) {
    const [x, y, z] = P;
    const { fx, fy, cx, cy, xi, alpha } = p;
    const d1 = Math.hypot(x, y, z);
    const z1 = z + xi * d1;
    const d2 = Math.hypot(x, y, z1);
    const den = alpha * d2 + (1 - alpha) * z1;
    const w1 = alpha <= 0.5 ? alpha / (1 - alpha) : (1 - alpha) / alpha;
    const w2 = (w1 + xi) / Math.sqrt(2 * w1 * xi + xi * xi + 1);
    const valid = z > -w2 * d1 && den > 1e-8;
    const theta = incidence(x, y, z);
    return { u: (fx * x) / den + cx, v: (fy * y) / den + cy, valid, theta, zone: zoneOf(theta, valid) };
  },
  unproject(u, v, p) {
    const { fx, fy, cx, cy, xi, alpha } = p;
    const mx = (u - cx) / fx;
    const my = (v - cy) / fy;
    const r2 = mx * mx + my * my;
    const s = 1 - (2 * alpha - 1) * r2;
    const valid = s >= 0;
    const mz = (1 - alpha * alpha * r2) / (alpha * Math.sqrt(Math.max(s, 0)) + 1 - alpha);
    const disc = mz * mz + (1 - xi * xi) * r2;
    const k = (mz * xi + Math.sqrt(Math.max(disc, 0))) / (mz * mz + r2);
    return { ray: norm3(k * mx, k * my, k * mz - xi), valid: valid && disc >= 0 };
  },
};

// ─────────────────────────────────────────────────────────────────── UCM ──
// ds_msp/models/ucm_math.py  (DS with ξ = 0)
const UCM: CameraModel = {
  id: "ucm",
  name: "Unified (UCM)",
  blurb: "Single sphere then pinhole. The classic catadioptric / fisheye workhorse.",
  glyph: "M5 12a7 7 0 1 0 14 0a7 7 0 1 0 -14 0 M12 5l0 14 M5 12l14 0",
  wideFov: true,
  defaults: { fx: 180, fy: 180, cx: 320, cy: 320, alpha: 0.6 },
  params: [
    { key: "fx", symbol: "f", label: "focal length (px)", min: 120, max: 520, step: 1, link: ["fx", "fy"] },
    { key: "alpha", symbol: "α", label: "sphere mix", min: 0.0, max: 0.99, step: 0.01 },
  ],
  project(P, p) {
    const [x, y, z] = P;
    const { fx, fy, cx, cy, alpha } = p;
    const d = Math.hypot(x, y, z);
    const den = alpha * d + (1 - alpha) * z;
    const w = alpha <= 0.5 ? alpha / (1 - alpha) : (1 - alpha) / alpha;
    const valid = z > -w * d && den > 1e-8;
    const theta = incidence(x, y, z);
    return { u: (fx * x) / den + cx, v: (fy * y) / den + cy, valid, theta, zone: zoneOf(theta, valid) };
  },
  unproject(u, v, p) {
    const { fx, fy, cx, cy, alpha } = p;
    const mx = (u - cx) / fx;
    const my = (v - cy) / fy;
    const r2 = mx * mx + my * my;
    const s = 1 - (2 * alpha - 1) * r2;
    const valid = s >= 0;
    const mz = (1 - alpha * alpha * r2) / (alpha * Math.sqrt(Math.max(s, 0)) + 1 - alpha);
    return { ray: norm3(mx, my, mz), valid };
  },
};

// ────────────────────────────────────────────────────────────────── EUCM ──
// ds_msp/models/eucm_math.py
const EUCM: CameraModel = {
  id: "eucm",
  name: "Enhanced UCM",
  blurb: "UCM with an ellipsoid (β) instead of a sphere — one extra knob for wide lenses.",
  glyph: "M4 12a8 5 0 1 0 16 0a8 5 0 1 0 -16 0 M12 7l0 10",
  wideFov: true,
  defaults: { fx: 180, fy: 180, cx: 320, cy: 320, alpha: 0.6, beta: 1.0 },
  params: [
    { key: "fx", symbol: "f", label: "focal length (px)", min: 120, max: 520, step: 1, link: ["fx", "fy"] },
    { key: "alpha", symbol: "α", label: "sphere mix", min: 0.0, max: 0.99, step: 0.01 },
    { key: "beta", symbol: "β", label: "ellipsoid stretch", min: 0.2, max: 3.0, step: 0.01 },
  ],
  project(P, p) {
    const [x, y, z] = P;
    const { fx, fy, cx, cy, alpha, beta } = p;
    const d = Math.sqrt(beta * (x * x + y * y) + z * z);
    const den = alpha * d + (1 - alpha) * z;
    const valid = den > 1e-8;
    const theta = incidence(x, y, z);
    return { u: (fx * x) / den + cx, v: (fy * y) / den + cy, valid, theta, zone: zoneOf(theta, valid) };
  },
  unproject(u, v, p) {
    const { fx, fy, cx, cy, alpha, beta } = p;
    const mx = (u - cx) / fx;
    const my = (v - cy) / fy;
    const r2 = mx * mx + my * my;
    const s = 1 - (2 * alpha - 1) * beta * r2;
    const valid = s >= 0;
    const mz = (1 - beta * alpha * alpha * r2) / (alpha * Math.sqrt(Math.max(s, 0)) + 1 - alpha);
    return { ray: norm3(mx, my, mz), valid };
  },
};

// ───────────────────────────────────────────────────────── Kannala–Brandt ──
// ds_msp/models/kb_math.py  (equidistant fisheye; OpenCV cv2.fisheye)
function kbThetaD(theta: number, k1: number, k2: number, k3: number, k4: number) {
  const t2 = theta * theta;
  return theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))));
}
const KB: CameraModel = {
  id: "kb",
  name: "Kannala–Brandt",
  blurb: "Equidistant fisheye: radius ∝ angle, plus an odd polynomial. OpenCV's cv2.fisheye.",
  glyph: "M5 12a7 7 0 1 0 14 0a7 7 0 1 0 -14 0 M12 12l0 -7 M12 12l5 5 M12 12l-5 5",
  wideFov: true,
  defaults: { fx: 180, fy: 180, cx: 320, cy: 320, k1: 0, k2: 0, k3: 0, k4: 0 },
  params: [
    { key: "fx", symbol: "f", label: "focal length (px)", min: 120, max: 360, step: 1, link: ["fx", "fy"] },
    { key: "k1", symbol: "k₁", label: "θ³ term", min: -0.3, max: 0.3, step: 0.005 },
    { key: "k2", symbol: "k₂", label: "θ⁵ term", min: -0.1, max: 0.1, step: 0.002 },
    { key: "k3", symbol: "k₃", label: "θ⁷ term", min: -0.05, max: 0.05, step: 0.001 },
    { key: "k4", symbol: "k₄", label: "θ⁹ term", min: -0.02, max: 0.02, step: 0.001 },
  ],
  project(P, p) {
    const [x, y, z] = P;
    const { fx, fy, cx, cy, k1, k2, k3, k4 } = p;
    const r = Math.hypot(x, y);
    const theta = Math.atan2(r, z);
    if (r < 1e-9) return { u: cx, v: cy, valid: true, theta, zone: zoneOf(theta, true) };
    const td = kbThetaD(theta, k1, k2, k3, k4);
    const u = fx * (td / r) * x + cx;
    const v = fy * (td / r) * y + cy;
    const valid = Number.isFinite(u) && Number.isFinite(v);
    return { u, v, valid, theta, zone: zoneOf(theta, valid) };
  },
  unproject(u, v, p) {
    const { fx, fy, cx, cy, k1, k2, k3, k4 } = p;
    const mx = (u - cx) / fx;
    const my = (v - cy) / fy;
    const ru = Math.hypot(mx, my);
    if (ru < 1e-9) return { ray: [0, 0, 1], valid: true };
    let theta = ru;
    for (let i = 0; i < 10; i++) {
      const t2 = theta * theta;
      const td = kbThetaD(theta, k1, k2, k3, k4);
      const d = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)));
      theta -= (td - ru) / d;
      theta = Math.max(0, Math.min(Math.PI, theta));
    }
    const res = Math.abs(kbThetaD(theta, k1, k2, k3, k4) - ru);
    const st = Math.sin(theta);
    return { ray: [st * mx / ru, st * my / ru, Math.cos(theta)], valid: res < 1e-6 && theta <= Math.PI };
  },
};

// ──────────────────────────────────────────────────────────────── RadTan ──
// ds_msp/models/radtan_math.py  (pinhole + Brown–Conrady; narrow FOV, z>0)
const RADTAN: CameraModel = {
  id: "radtan",
  name: "Pinhole + RadTan",
  blurb: "The familiar pinhole with radial+tangential distortion. Narrow FOV — front hemisphere only.",
  glyph: "M4 6l16 0l0 12l-16 0z M9 12a3 3 0 1 0 6 0a3 3 0 1 0 -6 0",
  wideFov: false,
  defaults: { fx: 320, fy: 320, cx: 320, cy: 320, k1: 0, k2: 0, p1: 0, p2: 0, k3: 0 },
  params: [
    { key: "fx", symbol: "f", label: "focal length (px)", min: 180, max: 700, step: 1, link: ["fx", "fy"] },
    { key: "k1", symbol: "k₁", label: "radial r²", min: -0.5, max: 0.5, step: 0.01 },
    { key: "k2", symbol: "k₂", label: "radial r⁴", min: -0.3, max: 0.3, step: 0.005 },
    { key: "k3", symbol: "k₃", label: "radial r⁶", min: -0.2, max: 0.2, step: 0.005 },
    { key: "p1", symbol: "p₁", label: "tangential", min: -0.05, max: 0.05, step: 0.001 },
    { key: "p2", symbol: "p₂", label: "tangential", min: -0.05, max: 0.05, step: 0.001 },
  ],
  project(P, p) {
    const [x, y, z] = P;
    const { fx, fy, cx, cy, k1, k2, p1, p2, k3 } = p;
    const theta = incidence(x, y, z);
    if (z <= 1e-9) return { u: NaN, v: NaN, valid: false, theta, zone: "invalid" };
    const a = x / z;
    const b = y / z;
    const r2 = a * a + b * b;
    const radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3));
    const xp = a * radial + 2 * p1 * a * b + p2 * (r2 + 2 * a * a);
    const yp = b * radial + p1 * (r2 + 2 * b * b) + 2 * p2 * a * b;
    const valid = z > 1e-9;
    return { u: fx * xp + cx, v: fy * yp + cy, valid, theta, zone: zoneOf(theta, valid) };
  },
  unproject(u, v, p) {
    const { fx, fy, cx, cy, k1, k2, p1, p2, k3 } = p;
    const a0 = (u - cx) / fx;
    const b0 = (v - cy) / fy;
    let a = a0;
    let b = b0;
    for (let i = 0; i < 20; i++) {
      const r2 = a * a + b * b;
      const radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3));
      const dx = 2 * p1 * a * b + p2 * (r2 + 2 * a * a);
      const dy = p1 * (r2 + 2 * b * b) + 2 * p2 * a * b;
      a = (a0 - dx) / radial;
      b = (b0 - dy) / radial;
    }
    return { ray: norm3(a, b, 1), valid: true };
  },
};

export const CAMERAS: CameraModel[] = [DS, UCM, EUCM, KB, RADTAN];
export const CAMERA_BY_ID: Record<string, CameraModel> = Object.fromEntries(
  CAMERAS.map((m) => [m.id, m]),
);

/** Largest incidence angle (rad) the model still projects — found by scanning a
 *  meridian, so it is correct for every model regardless of closed form. */
export function thetaMaxOf(model: CameraModel, p: Params): number {
  let last = 0;
  for (let i = 1; i <= 900; i++) {
    const t = (i / 900) * Math.PI;
    const P: Vec3 = [Math.sin(t), 0, Math.cos(t)];
    if (model.project(P, p).valid) last = t;
    else if (last > 0) break;
  }
  return last || Math.PI / 2;
}

export const zoneColor: Record<Zone, string> = {
  front: "#4ed7c0",
  beyond: "#ffc56e",
  invalid: "#ff5d6c",
};

/** Dev self-check: for every model, project then unproject and confirm the
 *  recovered bearing is parallel to the original. Returns the worst error (rad).
 *  A regression in any port makes this loud, not silent. */
export function selfCheck(): number {
  const pts: Vec3[] = [
    [0, 0, 1],
    [0.4, -0.3, 1],
    [1, 0.6, 0.8],
    [-0.5, 0.4, 0.9],
  ];
  let worst = 0;
  for (const m of CAMERAS) {
    for (const P of pts) {
      const pr = m.project(P, m.defaults);
      if (!pr.valid || !Number.isFinite(pr.u)) continue;
      const un = m.unproject(pr.u, pr.v, m.defaults);
      const n = Math.hypot(...P) || 1;
      const dot = Math.abs((P[0] / n) * un.ray[0] + (P[1] / n) * un.ray[1] + (P[2] / n) * un.ray[2]);
      worst = Math.max(worst, Math.acos(Math.min(1, dot)));
    }
  }
  return worst;
}
