// The projection, broken into the stages a learner steps through. The first two
// and last are universal (a direction → its bearing → a pixel on a surface); the
// middle stage carries each model's own characteristic math.

import type { CameraModel } from "./cameras";

export interface Stage {
  title: string;
  eq: string;
  detail: string;
}

const MIDDLE: Record<string, Stage> = {
  ds: {
    title: "Double Sphere",
    eq: "ζ = ξ·d₁ + Z,   den = α·d₂ + (1−α)·ζ",
    detail: "Reflect through a second sphere offset by ξ, then blend by α.",
  },
  ucm: {
    title: "Unified projection",
    eq: "den = α·‖P‖ + (1−α)·Z",
    detail: "One sphere, then a pinhole shifted by α off its centre.",
  },
  eucm: {
    title: "Enhanced UCM",
    eq: "d = √(β(X²+Y²)+Z²),   den = α·d + (1−α)·Z",
    detail: "The sphere becomes an ellipsoid — β stretches it to fit the lens.",
  },
  kb: {
    title: "Equidistant + polynomial",
    eq: "θ_d = θ(1 + k₁θ² + k₂θ⁴ + …),   r = f·θ_d",
    detail: "Radius grows with the angle θ itself, not its tangent.",
  },
  radtan: {
    title: "Pinhole + distortion",
    eq: "x′ = a(1 + k₁r² + k₂r⁴ + …) + tangential",
    detail: "Perspective divide by Z, then bend the rays radially & tangentially.",
  },
  ocam: {
    title: "Scaramuzza polynomial",
    eq: "solve  w(ρ) + (Z / √(X²+Y²))·ρ = 0",
    detail: "Fit the lens's radius curve directly — no focal length at all.",
  },
  dsplus: {
    title: "DS⁺ — UCM + division + tilt",
    eq: "den = α‖P‖ + (1−α)Z,   ÷(1 + λ₁r² + λ₂r⁴),   tilt τ",
    detail: "UCM sphere, then a Fitzgibbon division (θ³,θ⁵) and a Scheimpflug tilt — each stage closed-form invertible.",
  },
  eucmplus: {
    title: "EUCM⁺ — EUCM + division + tilt",
    eq: "d = √(β(X²+Y²)+Z²),   den = αd+(1−α)Z,   ÷(1+λ₁r²),   tilt τ",
    detail: "Ellipsoid (β) sphere, one division term and a tilt — a fully √-only closed-form inverse.",
  },
};

export function stagesFor(model: CameraModel): Stage[] {
  return [
    { title: "World ray", eq: "P = (X, Y, Z)", detail: "A direction the camera is looking at." },
    { title: "Bearing", eq: "b = P / ‖P‖", detail: "Project the point onto the unit sphere." },
    MIDDLE[model.id] ?? { title: "Projection", eq: "", detail: "" },
    { title: "Image pixel", eq: "(u, v)", detail: "Land on the chosen image surface." },
  ];
}
