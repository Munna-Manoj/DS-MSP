import { useEffect, useRef } from "react";
import { thetaMaxOf, type CameraModel, type Params, type Vec3 } from "../lib/cameras";
import { flipY } from "../lib/raymap";

// Transparent overlay on the rendered fisheye image: FOV-zone rings (90° + θmax),
// a crosshair on the picked ray, and a resolution badge. Drawn in sensor pixels
// (W = 2·cx, H = 2·cy); CSS stretches it over the panel. Works for any model.

interface Props {
  model: CameraModel;
  params: Params;
  active: Vec3;
}

function radiusAt(theta: number, model: CameraModel, p: Params): number | null {
  const P: Vec3 = [Math.sin(theta), 0, Math.cos(theta)];
  const { u, valid } = model.project(P, p);
  if (!valid || !Number.isFinite(u)) return null;
  return Math.abs(u - p.cx);
}

export function FisheyeOverlay({ model, params, active }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    const W = params.cx * 2;
    const H = params.cy * 2;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = W * dpr;
    cv.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const cx = params.cx;
    const cy = params.cy;
    const tmax = thetaMaxOf(model, params);

    const r90 = radiusAt(Math.PI / 2 - 1e-3, model, params);
    const rMax = radiusAt(tmax - 2e-3, model, params);

    ctx.lineWidth = 2;
    if (r90 && tmax > Math.PI / 2) {
      ctx.strokeStyle = "rgba(255,197,110,0.5)";
      ring(ctx, cx, cy, r90);
    }
    if (rMax) {
      ctx.strokeStyle = "rgba(255,93,108,0.55)";
      ring(ctx, cx, cy, rMax);
    }

    // active is a y-up world point; the model expects the camera (y-down) frame.
    const ap = model.project(flipY(active), params);
    if (ap.valid && Number.isFinite(ap.u)) {
      const x = ap.u;
      const y = ap.v;
      ctx.strokeStyle = "#6e8bff";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(x, y, 16, 0, Math.PI * 2);
      ctx.moveTo(x - 26, y);
      ctx.lineTo(x + 26, y);
      ctx.moveTo(x, y - 26);
      ctx.lineTo(x, y + 26);
      ctx.stroke();
      ctx.fillStyle = "rgba(110,139,255,0.95)";
      ctx.font = "600 22px 'IBM Plex Mono', monospace";
      ctx.fillText(`(${ap.u.toFixed(0)}, ${ap.v.toFixed(0)})`, x + 24, y - 22);
    }

    if (r90 && tmax > Math.PI / 2) {
      ctx.font = "18px 'IBM Plex Mono', monospace";
      ctx.fillStyle = "rgba(255,197,110,0.8)";
      ctx.fillText("90°", cx + r90 - 40, cy - 8);
    }
    if (rMax) {
      ctx.font = "18px 'IBM Plex Mono', monospace";
      ctx.fillStyle = "rgba(255,93,108,0.85)";
      ctx.fillText(`${((tmax * 180) / Math.PI).toFixed(0)}°`, cx + rMax - 56, cy - 8);
    }

    ctx.fillStyle = "rgba(230,234,242,0.55)";
    ctx.font = "16px 'IBM Plex Mono', monospace";
    ctx.fillText(`${Math.round(W)} × ${Math.round(H)}px`, 12, H - 12);
  }, [model, params, active]);

  return (
    <canvas ref={ref} className="absolute inset-0 w-full h-full pointer-events-none" aria-hidden />
  );
}

function ring(ctx: CanvasRenderingContext2D, x: number, y: number, r: number) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.stroke();
}
