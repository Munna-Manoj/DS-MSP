// The calibration shown on the user's own photo: each corner where the detector found it, and
// a line to where the calibration says it should be, stretched so a sub-pixel error is visible.
import { useEffect, useState, type ReactNode } from "react";
import { api, type CameraDiagnostics, type FrameCorners } from "../api";
import { num } from "../format";
import { boardColor, boardName } from "./boards";

/** The convex hull of some points, in order. */
function hull(pts: [number, number][]): [number, number][] {
  const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o: number[], a: number[], b: number[]) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (list: [number, number][]) => {
    const out: [number, number][] = [];
    for (const q of list) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], q) <= 0) out.pop();
      out.push(q);
    }
    out.pop();
    return out;
  };
  return [...half(p), ...half([...p].reverse())];
}

/** Each board in a photo: an outline around its corners, a little larger, and where to name it. */
function boardShapes(c: FrameCorners) {
  const ids = [...new Set(c.board ?? [])].sort((a, b) => a - b);
  return ids.map((id) => {
    const pts = c.u.map((u, i) => [u, c.v[i]] as [number, number]).filter((_, i) => c.board![i] === id);
    const mx = pts.reduce((a, p) => a + p[0], 0) / pts.length, my = pts.reduce((a, p) => a + p[1], 0) / pts.length;
    const out = hull(pts).map(([x, y]) => [mx + (x - mx) * 1.16, my + (y - my) * 1.16] as [number, number]);
    const top = out.reduce((a, p) => (p[1] < a[1] ? p : a), out[0]);
    return { id, n: pts.length, out, top };
  });
}

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
        {corners && boardShapes(corners).map((b) => {
          const size = W / 62, w = size * 5.4, h = size * 1.55;
          const x = Math.min(Math.max(b.top[0] - w / 2, 2), W - w - 2), y = Math.min(Math.max(b.top[1] - h - size * 0.3, 2), H - h - 2);
          return (
            <g key={b.id}>
              <polygon points={b.out.map((p) => p.join(",")).join(" ")} fill="none" stroke={boardColor(b.id)} strokeWidth={r * 0.5} strokeLinejoin="round" />
              <rect x={x} y={y} width={w} height={h} rx={h / 4} fill={boardColor(b.id)} />
              <text x={x + w / 2} y={y + h * 0.74} textAnchor="middle" fontSize={size} fontWeight="700" fill="#0f172a">{boardName(b.id)}</text>
            </g>
          );
        })}
        {corners && corners.u.map((u, i) => {
          const v = corners.v[i];
          const x2 = u + corners.dx[i] * magnify, y2 = v + corners.dy[i] * magnify;
          const err = Math.hypot(corners.dx[i], corners.dy[i]);
          const color = corners.within[i] ? "#34d399" : "#f87171";
          return (
            <g key={i}>
              <title>{`${corners.board ? `${boardName(corners.board[i])}, corner ${corners.corner?.[i]}: ` : ""}error ${num(err, 3)} px`}</title>
              <line x1={u} y1={v} x2={x2} y2={y2} stroke={color} strokeWidth={r * 0.45} strokeLinecap="round" />
              <circle cx={u} cy={v} r={r * 0.55} fill={corners.board ? boardColor(corners.board[i]) : "#fff"} stroke="#0f172a" strokeWidth={r * 0.18} />
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
