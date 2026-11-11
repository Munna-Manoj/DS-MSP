// The capture as a strip of bars, one per photo of the selected camera, as tall as that photo's
// error. One timeline drives every view: the photo, the board in 3D, and playback.
import { useEffect, useRef, type ReactNode } from "react";
import type { FrameStat } from "../api";
import { num } from "../format";
import { boardName } from "./boards";

/** Round tick values for an axis that reaches `top`: 3 to 5 steps of 1, 2 or 5 times a power of ten. */
export function ticks(top: number): { step: number; max: number; values: number[] } {
  const raw = Math.max(top, 1e-9) / 5;
  const pow = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * pow).find((v) => v >= raw) ?? 10 * pow;
  const n = Math.max(1, Math.ceil(Math.max(top, 1e-9) / step - 1e-9));
  return { step, max: n * step, values: Array.from({ length: n + 1 }, (_, k) => k * step) };
}

function tickLabel(v: number, step: number): string {
  const d = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  return v.toFixed(Math.min(d, 4));
}

export function Timeline({ frames, frame, onFrame, playing, onPlaying, marked }: {
  frames: FrameStat[]; frame: number | null; onFrame: (f: number) => void;
  playing: boolean; onPlaying: (p: boolean) => void;
  /** Frames of this camera marked to be left out of the next run. */
  marked?: Set<number>;
}) {
  const index = frames.findIndex((f) => f.frame === frame);
  const latest = useRef({ frames, index, onFrame });
  latest.current = { frames, index, onFrame };

  useEffect(() => {
    if (!playing) return;
    const t = window.setInterval(() => {
      const { frames: fs, index: i, onFrame: go } = latest.current;
      if (fs.length) go(fs[(i + 1) % fs.length].frame);
    }, 320);
    return () => window.clearInterval(t);
  }, [playing]);

  if (!frames.length) return null;
  const top = Math.max(...frames.map((f) => f.rms_px), 1e-9);
  const axis = ticks(top);
  const sorted = frames.map((f) => f.rms_px).sort((x, y) => x - y);
  const typical = sorted[Math.floor(sorted.length / 2)];
  const worst = frames.reduce((a, b) => (b.rms_px > a.rms_px ? b : a));
  const step = (d: number) => onFrame(frames[(index + d + frames.length) % frames.length].frame);
  const cur = index >= 0 ? frames[index] : null;
  const W = 1000, H = 100, bw = W / frames.length;
  const up = (v: number) => `${(100 * v) / axis.max}%`;

  return (
    <div className="grid gap-2">
      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-300">
        <div className="flex items-center gap-1" role="group" aria-label="Step through the photos">
          <IconButton label="Previous photo" onClick={() => step(-1)}>‹</IconButton>
          <IconButton label={playing ? "Pause" : "Play through the capture"} onClick={() => onPlaying(!playing)}>
            {playing ? "❚❚" : "▶"}
          </IconButton>
          <IconButton label="Next photo" onClick={() => step(1)}>›</IconButton>
        </div>
        <span className="tnum">
          {cur ? <>Photo <b className="text-white">{index + 1}</b> of {frames.length} · {cur.n} corners{cur.boards?.length ? ` · ${cur.boards.map(boardName).join(", ")}` : ""} · typical error <b className="text-white">{num(cur.rms_px)} px</b></> : "Choose a photo"}
        </span>
        <button type="button" onClick={() => onFrame(worst.frame)}
          className="ml-auto rounded px-2 py-1 bg-white/10 hover:bg-white/20 text-slate-100">
          Go to the worst photo ({num(worst.rms_px)} px)
        </button>
      </div>
      {/* the scale: error in pixels, with a line at each mark and one at the camera's typical photo */}
      <div className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-2 pt-1.5">
        <div className="relative h-24 w-12 text-[10px] text-slate-400 tnum" aria-hidden="true">
          {axis.values.map((v) => (
            <span key={v} className="absolute right-0 translate-y-1/2 leading-none" style={{ bottom: up(v) }}>
              {tickLabel(v, axis.step)}{v === axis.max ? " px" : ""}
            </span>
          ))}
        </div>
        <div className="relative h-24 rounded bg-black/30 overflow-hidden">
          {axis.values.slice(1).map((v) => (
            <div key={v} className="absolute inset-x-0 border-t border-white/10 pointer-events-none" style={{ bottom: up(v) }} />
          ))}
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="absolute inset-0 w-full h-full cursor-pointer"
          role="slider" tabIndex={0} aria-label="Photo in the capture" aria-valuemin={1} aria-valuemax={frames.length}
          aria-valuenow={index + 1}
          onKeyDown={(e) => { if (e.key === "ArrowRight") step(1); if (e.key === "ArrowLeft") step(-1); }}
          onPointerDown={(e) => {
            const r = e.currentTarget.getBoundingClientRect();
            const pick = (x: number) => onFrame(frames[Math.min(frames.length - 1, Math.max(0, Math.floor(((x - r.left) / r.width) * frames.length)))].frame);
            pick(e.clientX);
            const el = e.currentTarget;
            el.setPointerCapture(e.pointerId);
            const move = (ev: PointerEvent) => pick(ev.clientX);
            const up = () => { el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); };
            el.addEventListener("pointermove", move);
            el.addEventListener("pointerup", up);
          }}>
          {frames.map((f, k) => {
            const h = Math.max(1.5, (f.rms_px / axis.max) * H);
            return (
              <rect key={f.frame} x={k * bw + bw * 0.12} y={H - h} width={bw * 0.76} height={h}
                fill={marked?.has(f.frame) ? "#fbbf24" : k === index ? "#38bdf8" : f.n_outside > 0 ? "#f87171" : "#64748b"}
                fillOpacity={marked?.has(f.frame) && k !== index ? 0.5 : 1}
                stroke={marked?.has(f.frame) && k === index ? "#38bdf8" : "none"} strokeWidth={2} vectorEffect="non-scaling-stroke">
                <title>{`Photo ${k + 1} (frame ${f.frame}): ${num(f.rms_px, 3)} px${marked?.has(f.frame) ? ", marked to leave out" : ""}`}</title>
              </rect>
            );
          })}
        </svg>
          <div className="absolute inset-x-0 border-t border-dashed border-sky-300/70 pointer-events-none" style={{ bottom: up(typical) }}>
            <span className="absolute right-1 -top-3.5 text-[10px] text-sky-200 bg-[#0b1120]/80 px-1 rounded">typical {num(typical)} px</span>
          </div>
        </div>
      </div>
      <div className="flex justify-between text-[11px] text-slate-400 pl-14">
        <span>start of capture</span>
        <span>each bar is one photo; its height is that photo's error in pixels{frames.some((f) => f.n_outside > 0) ? " · red = has corners far off" : ""}{marked && marked.size > 0 ? " · amber = marked to leave out" : ""}</span>
        <span>end</span>
      </div>
    </div>
  );
}

function IconButton({ label, onClick, children }: { label: string; onClick: () => void; children: ReactNode }) {
  return (
    <button type="button" aria-label={label} title={label} onClick={onClick}
      className="size-8 grid place-items-center rounded bg-white/10 hover:bg-white/20 text-slate-100 text-sm">
      {children}
    </button>
  );
}
