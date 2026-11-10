// The capture as a strip of bars, one per photo of the selected camera, as tall as that photo's
// error. One timeline drives every view: the photo, the board in 3D, and playback.
import { useEffect, useRef, type ReactNode } from "react";
import type { FrameStat } from "../api";
import { num } from "../format";

export function Timeline({ frames, frame, onFrame, playing, onPlaying }: {
  frames: FrameStat[]; frame: number | null; onFrame: (f: number) => void;
  playing: boolean; onPlaying: (p: boolean) => void;
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
  const worst = frames.reduce((a, b) => (b.rms_px > a.rms_px ? b : a));
  const step = (d: number) => onFrame(frames[(index + d + frames.length) % frames.length].frame);
  const cur = index >= 0 ? frames[index] : null;
  const W = 1000, H = 44, bw = W / frames.length;

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
          {cur ? <>Photo <b className="text-white">{index + 1}</b> of {frames.length} · {cur.n} corners · typical error <b className="text-white">{num(cur.rms_px)} px</b></> : "Choose a photo"}
        </span>
        <button type="button" onClick={() => onFrame(worst.frame)}
          className="ml-auto rounded px-2 py-1 bg-white/10 hover:bg-white/20 text-slate-100">
          Go to the worst photo ({num(worst.rms_px)} px)
        </button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="w-full h-11 cursor-pointer rounded bg-black/30"
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
          const h = Math.max(2, (f.rms_px / top) * (H - 4));
          return (
            <rect key={f.frame} x={k * bw + bw * 0.12} y={H - h} width={bw * 0.76} height={h}
              fill={k === index ? "#38bdf8" : f.n_outside > 0 ? "#f87171" : "#64748b"} />
          );
        })}
      </svg>
      <div className="flex justify-between text-[11px] text-slate-400">
        <span>start of capture</span>
        <span>bar height = error in that photo (tallest {num(top)} px){frames.some((f) => f.n_outside > 0) ? " · red = has corners far off" : ""}</span>
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
