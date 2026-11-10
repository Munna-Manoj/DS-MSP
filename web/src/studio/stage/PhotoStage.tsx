// The calibration shown on the user's own photo: each corner where the detector found it, and
// a line to where the calibration says it should be, stretched so a sub-pixel error is visible.
import { useEffect, useState, type ReactNode } from "react";
import { api, type CameraDiagnostics, type FrameCorners } from "../api";
import { num } from "../format";

export function PhotoStage({ runId, cam, frame, size, magnify, showTrust, diagnostics, photos }: {
  runId: string; cam: string; frame: number | null; size: { width: number; height: number } | null;
  magnify: number; showTrust: boolean; diagnostics: CameraDiagnostics | null;
  photos: "recorded" | "assumed" | "none";
}) {
  const [corners, setCorners] = useState<FrameCorners | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (frame == null) return;
    let live = true;
    setFailed(false);
    api.frame(runId, cam, frame).then((c) => live && setCorners(c), () => live && setCorners(null));
    return () => { live = false; };
  }, [runId, cam, frame]);

  if (frame == null || !size)
    return <Empty>This camera has no photo with the board in it.</Empty>;
  const W = size.width, H = size.height;
  const r = W / 220;
  const hasPhoto = corners?.photo && !failed;
  const zones = showTrust ? diagnostics?.lens?.zones?.zones : undefined;
  const nx = diagnostics?.lens?.nx ?? 24, ny = diagnostics?.lens?.ny ?? 18;

  return (
    <div className="relative w-full bg-black/40 rounded-md overflow-hidden" style={{ aspectRatio: `${W} / ${H}` }}>
      {hasPhoto ? (
        <img src={api.photoUrl(runId, cam, frame)} alt={`Photo ${frame} of camera ${cam}`}
          className="absolute inset-0 w-full h-full" onError={() => setFailed(true)} draggable={false} />
      ) : (
        <div className="absolute inset-0 grid place-items-center text-xs text-slate-400 p-6 text-center">
          {corners == null ? "Loading…"
            : photos === "none" ? "The photos of this capture are not recorded with the result, so only the corners are shown."
            : "This photo is not inside the folders Studio was started with, so only the corners are shown."}
        </div>
      )}
      <svg viewBox={`-0.5 -0.5 ${W} ${H}`} className="absolute inset-0 w-full h-full" role="img"
        aria-label={corners ? `${corners.n} corners with their error, magnified ${magnify} times` : "corners"}>
        <defs>
          <pattern id="never" width={W / 60} height={W / 60} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2={W / 60} stroke="rgba(15,23,42,0.75)" strokeWidth={W / 120} />
          </pattern>
        </defs>
        {zones && zones.map((z, i) => {
          if (z === "trusted") return null;
          const cx = (i % nx) * (W / nx) - 0.5, cy = Math.floor(i / nx) * (H / ny) - 0.5;
          return (
            <rect key={i} x={cx} y={cy} width={W / nx + 0.5} height={H / ny + 0.5}
              fill={z === "uncertain" ? "rgba(245,158,11,0.45)" : "url(#never)"} />
          );
        })}
        {corners && corners.u.map((u, i) => {
          const v = corners.v[i];
          const x2 = u + corners.dx[i] * magnify, y2 = v + corners.dy[i] * magnify;
          const err = Math.hypot(corners.dx[i], corners.dy[i]);
          const color = corners.within[i] ? "#34d399" : "#f87171";
          return (
            <g key={i}>
              <title>{`error ${num(err, 3)} px`}</title>
              <line x1={u} y1={v} x2={x2} y2={y2} stroke={color} strokeWidth={r * 0.45} strokeLinecap="round" />
              <circle cx={u} cy={v} r={r * 0.55} fill="#fff" stroke="#0f172a" strokeWidth={r * 0.18} />
              <circle cx={x2} cy={y2} r={r} fill="none" stroke={color} strokeWidth={r * 0.4} />
            </g>
          );
        })}
      </svg>
      {photos === "assumed" && hasPhoto && (
        <div className="absolute left-2 bottom-2 rounded bg-black/70 text-[11px] text-amber-200 px-2 py-1 max-w-[80%]">
          Photos are matched to frames by file number. If the corners do not sit on the board, the first photos had no board in them.
        </div>
      )}
    </div>
  );
}

function Empty({ children }: { children: ReactNode }) {
  return <div className="h-64 grid place-items-center text-sm text-slate-400">{children}</div>;
}
