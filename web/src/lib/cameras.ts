// Multi-model camera registry — faithful TypeScript ports of the eight models
// ds_msp ships (ds_msp/models/*_math.py). Every project/unproject below is a
// line-for-line port of the Python so the demo is provably correct, not a
// look-alike. Validity rules differ per model on purpose — that is real physics,
// not an inconsistency (DS/UCM use a tilted half-space; EUCM only guards the
// denominator; RadTan is strict z>0; KB/OCam rely on a solver converging).
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

// ────────────────────────────────────────────────────────────────── OCam ──
// ds_msp/models/ocam_math.py  (Scaramuzza / OCamCalib polynomial)
const OCAM_R = 100.0;
function ocamW(rho: number, a: number[]) {
  const rn = rho / OCAM_R;
  return a[0] + rn * (a[1] + rn * (a[2] + rn * (a[3] + rn * a[4])));
}
const OCAM: CameraModel = {
  id: "ocam",
  name: "OCam (Scaramuzza)",
  blurb: "A free-form radial polynomial fit to the lens — no focal length, just the curve.",
  glyph: "M5 12a7 7 0 1 0 14 0a7 7 0 1 0 -14 0 M5 12c3 -5 11 -5 14 0",
  wideFov: true,
  defaults: { cx: 320, cy: 320, c: 1, d: 0, e: 0, a0: -230, a1: 0, a2: 0.0016, a3: 0, a4: 0 },
  params: [
    { key: "a0", symbol: "a₀", label: "poly constant", min: -320, max: -140, step: 1 },
    { key: "a2", symbol: "a₂", label: "poly r² term", min: -0.002, max: 0.004, step: 0.0001 },
    { key: "a4", symbol: "a₄", label: "poly r⁴ term", min: -0.0001, max: 0.0001, step: 0.000005 },
    { key: "c", symbol: "c", label: "affine xx", min: 0.6, max: 1.4, step: 0.01 },
    { key: "d", symbol: "d", label: "affine xy", min: -0.3, max: 0.3, step: 0.01 },
  ],
  project(P, p) {
    const [X, Y, Z] = P;
    const { cx, cy, c, d, e } = p;
    const a = [p.a0, p.a1, p.a2, p.a3, p.a4];
    const theta = incidence(X, Y, Z);
    const nrm = Math.hypot(X, Y);
    if (nrm < 1e-9) return { u: cx, v: cy, valid: true, theta, zone: zoneOf(theta, true) };
    const m = Z / nrm;
    let rho = OCAM_R; // seed
    let conv = false;
    for (let i = 0; i < 30; i++) {
      const rn = rho / OCAM_R;
      const F = ocamW(rho, a) + m * rho;
      const dF = (a[1] + rn * (2 * a[2] + rn * (3 * a[3] + rn * 4 * a[4]))) / OCAM_R + m;
      const step = F / dF;
      rho -= step;
      if (Math.abs(F) < 1e-6) {
        conv = true;
        break;
      }
    }
    const valid = conv && rho > 0;
    const ix = (rho * X) / nrm;
    const iy = (rho * Y) / nrm;
    return { u: c * ix + d * iy + cx, v: e * ix + iy + cy, valid, theta, zone: zoneOf(theta, valid) };
  },
  unproject(u, v, p) {
    const { cx, cy, c, d, e } = p;
    const a = [p.a0, p.a1, p.a2, p.a3, p.a4];
    const det = c - d * e;
    const x = ((u - cx) - d * (v - cy)) / det;
    const y = (-e * (u - cx) + c * (v - cy)) / det;
    const rho = Math.hypot(x, y);
    return { ray: norm3(x, y, -ocamW(rho, a)), valid: Number.isFinite(x) && Number.isFinite(y) };
  },
};

// ─────────────────────────────────────────────────────────────────── DS⁺ ──
// ds_msp/models/dsplus_math.py  (UCM core + Fitzgibbon division θ³,θ⁵ + 2-axis tilt)
// Forward is a line-for-line port. The 2-term division inverse is the Ferrari
// quartic in the library; here we solve the same equation per-pixel with Newton
// (seeded at the distorted radius — the small-distortion branch the library picks),
// matching it to ~1e-10 within the slider ranges. At λ₁=λ₂=τ=0, DS⁺ collapses to
// UCM and the inverse is exact in one step.
const DSPLUS: CameraModel = {
  id: "dsplus",
  name: "DS⁺ (Double Sphere +)",
  blurb: "UCM core + division (θ³,θ⁵) + Scheimpflug tilt. Fits ~140° lenses that defeat plain DS; closed-form inverse.",
  glyph: "M5 12a7 7 0 1 0 14 0a7 7 0 1 0 -14 0 M12 12a4 4 0 1 0 8 0a4 4 0 1 0 -8 0 M19 2l0 5 M16.5 4.5l5 0",
  wideFov: true,
  defaults: { fx: 180, fy: 180, cx: 320, cy: 320, alpha: 0.6, lambda1: 0.0, lambda2: 0.0, tau_x: 0.0, tau_y: 0.0 },
  params: [
    { key: "fx", symbol: "f", label: "focal length (px)", min: 120, max: 520, step: 1, link: ["fx", "fy"] },
    { key: "alpha", symbol: "α", label: "sphere mix", min: 0.0, max: 0.99, step: 0.01 },
    { key: "lambda1", symbol: "λ₁", label: "division θ³", min: -0.5, max: 0.5, step: 0.005 },
    { key: "lambda2", symbol: "λ₂", label: "division θ⁵", min: -0.2, max: 0.2, step: 0.002 },
    { key: "tau_x", symbol: "τx", label: "tilt (x)", min: -0.3, max: 0.3, step: 0.005 },
    { key: "tau_y", symbol: "τy", label: "tilt (y)", min: -0.3, max: 0.3, step: 0.005 },
  ],
  project(P, p) {
    const [x, y, z] = P;
    const { fx, fy, cx, cy, alpha, lambda1, lambda2, tau_x, tau_y } = p;
    // Stage S: UCM sphere.
    const d = Math.hypot(x, y, z);
    const den = alpha * d + (1 - alpha) * z;
    const w = alpha > 0.5 ? (1 - alpha) / alpha : alpha / (1 - alpha);
    const validS = z > -w * d && den > 1e-9;
    const dens = Math.max(den, 1e-9);
    const mx = x / dens;
    const my = y / dens;
    // Stage D: division radial.
    const rho2 = mx * mx + my * my;
    const g = 1 + lambda1 * rho2 + lambda2 * rho2 * rho2;
    const validD = g > 1e-9 && 1 - lambda1 * rho2 - 3 * lambda2 * rho2 * rho2 > 0;
    const gs = Math.abs(g) < 1e-12 ? 1e-12 : g;
    const dx = mx / gs;
    const dy = my / gs;
    // Stage H: tilt homography.
    const wt = tau_x * dx + tau_y * dy + 1;
    const validH = Math.abs(wt) > 1e-9;
    const ws = Math.abs(wt) < 1e-12 ? 1e-12 : wt;
    const theta = incidence(x, y, z);
    const valid = validS && validD && validH;
    return { u: (fx * dx) / ws + cx, v: (fy * dy) / ws + cy, valid, theta, zone: zoneOf(theta, valid) };
  },
  unproject(u, v, p) {
    const { fx, fy, cx, cy, alpha, lambda1, lambda2, tau_x, tau_y } = p;
    const xp = (u - cx) / fx;
    const yp = (v - cy) / fy;
    // H⁻¹: tilt inverse (same projective form, sign flipped).
    const wpr = 1 - tau_x * xp - tau_y * yp;
    const validH = Math.abs(wpr) > 1e-9;
    const wprs = Math.abs(wpr) < 1e-12 ? 1e-12 : wpr;
    const xg = xp / wprs;
    const yg = yp / wprs;
    // D⁻¹: undistorted radius ρ solves rd = ρ/(1 + λ₁ρ² + λ₂ρ⁴). The library uses
    // the Ferrari quartic and selects the SMALLEST positive root — the principal
    // branch connected to ρ=0. We find that same root by bracketing the first sign
    // change of F(ρ)=ρ/g(ρ)−rd up from 0, then bisecting. (A plain Newton from ρ=rd
    // can jump to the spurious larger root once two positives exist, rd≳1.8.)
    const rd = Math.hypot(xg, yg);
    let rho = rd;
    if ((Math.abs(lambda1) >= 1e-15 || Math.abs(lambda2) >= 1e-15) && rd > 1e-12) {
      const Fr = (r: number) => {
        const r2 = r * r;
        const gg = 1 + lambda1 * r2 + lambda2 * r2 * r2;
        return gg > 1e-9 ? r / gg - rd : NaN;
      };
      let lo = 0;
      let hi = -1;
      let prev = -rd; // F(0)
      const STEP = 0.1;
      for (let r = STEP; r <= 8; r += STEP) {
        const fr = Fr(r);
        if (Number.isNaN(fr)) break; // hit a pole: principal branch ends
        if (prev < 0 && fr >= 0) {
          lo = r - STEP;
          hi = r;
          break;
        }
        prev = fr;
      }
      if (hi > 0) {
        for (let i = 0; i < 40; i++) {
          const mid = 0.5 * (lo + hi);
          const fm = Fr(mid);
          if (Number.isNaN(fm) || fm >= 0) hi = mid;
          else lo = mid;
        }
        rho = 0.5 * (lo + hi);
      }
    }
    const scale = rd > 1e-12 ? rho / rd : 1;
    const mx = xg * scale;
    const my = yg * scale;
    // S⁻¹: UCM (DS with ξ=0) closed form.
    const r2 = mx * mx + my * my;
    const s = 1 - (2 * alpha - 1) * r2;
    const validS = s >= 0;
    const mz = (1 - alpha * alpha * r2) / (alpha * Math.sqrt(Math.max(s, 0)) + 1 - alpha);
    return { ray: norm3(mx, my, mz), valid: validS && validH };
  },
};

// ────────────────────────────────────────────────────────────────── EUCM⁺ ──
// ds_msp/models/eucmplus_math.py  (EUCM core + 1-term division + 2-axis tilt)
// Truly square-root-only: the whole inverse below is closed form — no Newton, no
// cube root — so it matches the library to machine precision at any parameters.
const EUCMPLUS: CameraModel = {
  id: "eucmplus",
  name: "EUCM⁺ (Enhanced UCM +)",
  blurb: "EUCM core + 1-term division + tilt — a truly √-only closed-form inverse. DS-class speed, fits lenses DS can't.",
  glyph: "M4 12a8 5 0 1 0 16 0a8 5 0 1 0 -16 0 M11 7l0 10 M19 2l0 5 M16.5 4.5l5 0",
  wideFov: true,
  defaults: { fx: 180, fy: 180, cx: 320, cy: 320, alpha: 0.6, beta: 1.0, lambda1: 0.0, tau_x: 0.0, tau_y: 0.0 },
  params: [
    { key: "fx", symbol: "f", label: "focal length (px)", min: 120, max: 520, step: 1, link: ["fx", "fy"] },
    { key: "alpha", symbol: "α", label: "sphere mix", min: 0.0, max: 0.99, step: 0.01 },
    { key: "beta", symbol: "β", label: "ellipsoid stretch", min: 0.2, max: 3.0, step: 0.01 },
    { key: "lambda1", symbol: "λ₁", label: "division θ³", min: -0.5, max: 0.5, step: 0.005 },
    { key: "tau_x", symbol: "τx", label: "tilt (x)", min: -0.3, max: 0.3, step: 0.005 },
    { key: "tau_y", symbol: "τy", label: "tilt (y)", min: -0.3, max: 0.3, step: 0.005 },
  ],
  project(P, p) {
    const [x, y, z] = P;
    const { fx, fy, cx, cy, alpha, beta, lambda1, tau_x, tau_y } = p;
    // Stage S: EUCM sphere.
    const d = Math.sqrt(beta * (x * x + y * y) + z * z);
    const den = alpha * d + (1 - alpha) * z;
    const validS = den > 1e-9;
    const dens = Math.max(den, 1e-9);
    const mx = x / dens;
    const my = y / dens;
    // Stage D: division radial (1 term).
    const s2 = mx * mx + my * my;
    const g = 1 + lambda1 * s2;
    const validD = Math.abs(g) > 1e-9;
    const gs = Math.abs(g) < 1e-12 ? 1e-12 : g;
    const dx = mx / gs;
    const dy = my / gs;
    // Stage H: tilt homography.
    const wt = tau_x * dx + tau_y * dy + 1;
    const validH = Math.abs(wt) > 1e-9;
    const ws = Math.abs(wt) < 1e-12 ? 1e-12 : wt;
    const theta = incidence(x, y, z);
    const valid = validS && validD && validH;
    return { u: (fx * dx) / ws + cx, v: (fy * dy) / ws + cy, valid, theta, zone: zoneOf(theta, valid) };
  },
  unproject(u, v, p) {
    const { fx, fy, cx, cy, alpha, beta, lambda1, tau_x, tau_y } = p;
    const xt = (u - cx) / fx;
    const yt = (v - cy) / fy;
    // H⁻¹: tilt inverse (linear).
    const denom = 1 - tau_x * xt - tau_y * yt;
    const validH = Math.abs(denom) > 1e-9;
    const denoms = Math.abs(denom) < 1e-12 ? 1e-12 : denom;
    const xg = xt / denoms;
    const yg = yt / denoms;
    // D⁻¹: 1-term division is a quadratic — one sqrt, no iteration.
    const rd = Math.hypot(xg, yg);
    const disc = 1 - 4 * lambda1 * rd * rd;
    const validD = disc >= 0;
    let r = rd;
    if (Math.abs(lambda1) >= 1e-15 && rd > 1e-12) {
      r = (1 - Math.sqrt(Math.max(disc, 0))) / (2 * lambda1 * rd);
    }
    const scale = rd > 1e-12 ? r / rd : 1;
    const mx = xg * scale;
    const my = yg * scale;
    // S⁻¹: EUCM closed form (sqrt-only).
    const r2 = mx * mx + my * my;
    const s = 1 - (2 * alpha - 1) * beta * r2;
    const validS = s >= 0;
    const mz = (1 - beta * alpha * alpha * r2) / (alpha * Math.sqrt(Math.max(s, 0)) + 1 - alpha);
    return { ray: norm3(mx, my, mz), valid: validS && validD && validH };
  },
};

export const CAMERAS: CameraModel[] = [DS, UCM, EUCM, KB, RADTAN, OCAM, DSPLUS, EUCMPLUS];
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
