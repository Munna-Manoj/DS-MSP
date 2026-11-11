// What the calibration's uncertainty means for a measurement: a point seen by two cameras, at a
// distance the user chooses, comes out uncertain by this much sideways and this much in depth.
import { useEffect, useState, type ReactNode } from "react";
import { api, type TaskDoc } from "../api";
import { camName } from "../format";
import { length, oneMillimetre, type Unit } from "./units";

export function MeaningStage({ runId, a, b, distance, pixelNoise, unit, onMode }: {
  runId: string; a: string; b: string; distance: number; pixelNoise: number; unit: Unit;
  onMode?: (mode: TaskDoc["mode"]) => void;
}) {
  const [doc, setDoc] = useState<TaskDoc | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    const t = window.setTimeout(() => {
      api.task(runId, a, b, distance, pixelNoise).then(
        (d) => { if (live) { setDoc(d); setError(null); if (d.ok) onMode?.(d.mode); } },
        (e: Error) => live && setError(e.message),
      );
    }, 120);
    return () => { live = false; window.clearTimeout(t); };
  }, [runId, a, b, distance, pixelNoise, onMode]);

  if (error) return <Message>{error}</Message>;
  if (!doc) return <Message>Working it out…</Message>;
  if (!doc.ok) return <Message>{doc.reason ?? "Not available for these two cameras."}</Message>;

  const mm = oneMillimetre(unit);
  const share = doc.mode === "triangulate" && doc.depth > 0 ? (doc.from_cameras.depth / doc.depth) ** 2 : 0;
  const reach = (key: "depth" | "sideways") => {
    if (mm == null || !doc.curve?.length) return null;
    const inside = doc.curve.filter((p) => p[key] <= mm);
    if (!inside.length) return 0;
    return inside.length === doc.curve.length ? Infinity : inside[inside.length - 1].distance;
  };
  const rd = reach("depth"), rs = reach("sideways");
  const say = (v: number | null) => v == null ? null : v === 0 ? "not even close up" : v === Infinity ? "over this whole range" : `out to ${length(v, unit)}`;

  if (doc.mode === "transfer") {
    const how = doc.relation === "each_other" ? "face each other" : doc.relation === "apart" ? "look in different directions" : "do not see this point together";
    return (
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] items-center p-4 sm:p-6">
        <div className="grid gap-4">
          <p className="text-[15px] text-slate-100 leading-relaxed">
            {camName(a)} and {camName(b)} {how}{doc.axes_deg != null && doc.relation !== "no_shared_view" ? ` (their views are ${Math.round(doc.axes_deg)}° apart)` : ""}, so
            they are not a stereo pair. What this calibration gives you is <b className="text-white">handing a measurement from one camera to the other</b>.
          </p>
          <p className="text-slate-300 text-sm">
            A point <b className="text-white">{length(doc.distance, unit)}</b> in front of {camName(a)} is placed in {camName(b)}'s coordinates to
          </p>
          <div className="grid grid-cols-2 gap-4">
            <Big label="sideways" value={`±${length(doc.sideways, unit)}`} />
            <Big label={`along ${camName(a)}'s line of sight`} value={`±${length(doc.depth, unit)}`} />
          </div>
          <p className="text-sm text-slate-300 leading-relaxed">
            This is only the error of where the two cameras are relative to each other. The error of your own measurement in {camName(a)} comes on top. The chart shows how it changes with the distance.
          </p>
          {mm != null && (
            <p className="text-sm text-slate-200 rounded-md bg-white/10 px-3 py-2 leading-relaxed">
              <b className="text-white">1 mm or better:</b> sideways {say(rs)}; along the line of sight {say(rd)}.
            </p>
          )}
          {mm == null && <p className="text-sm text-amber-200">Tell Studio the unit of your square size (on the right) to read this in millimetres.</p>}
          <p className="text-xs text-slate-400 leading-relaxed">
            One standard deviation, from random error only. Real errors are usually larger, because a calibration also carries errors that repeat in every photo.
          </p>
          <LensCaveat doc={doc} />
        </div>
        <div className="grid gap-3">
          <PlanView doc={doc} unit={unit} a={a} b={b} />
          {doc.curve && doc.curve.length > 2 && <Curve doc={doc} unit={unit} mm={mm} />}
        </div>
      </div>
    );
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] items-center p-4 sm:p-6">
      <div className="grid gap-4">
        <p className="text-slate-300 text-sm">
          A point <b className="text-white">{length(doc.distance, unit)}</b> away, seen by {camName(a)} and {camName(b)}, is located to
        </p>
        <div className="grid grid-cols-2 gap-4">
          <Big label="sideways" value={`±${length(doc.sideways, unit)}`} />
          <Big label="in depth" value={`±${length(doc.depth, unit)}`} />
        </div>
        <p className="text-sm text-slate-300 leading-relaxed">
          {share > 0.6 ? "Most of the depth error comes from the calibration: how well the angle between the two cameras is known."
            : share < 0.4 ? "Most of the depth error comes from finding the point in the image, not from the calibration."
            : "The calibration and finding the point in the image contribute about equally to the depth error."}
          {" "}Depth error grows with the square of the distance; the cameras are {length(doc.baseline, unit)} apart.
        </p>
        {mm != null && (
          <p className="text-sm text-slate-200 rounded-md bg-white/10 px-3 py-2 leading-relaxed">
            <b className="text-white">1 mm or better:</b> sideways {say(rs)}; in depth {say(rd)}.
          </p>
        )}
        {mm == null && (
          <p className="text-sm text-amber-200">Tell Studio the unit of your square size (on the right) to read this in millimetres.</p>
        )}
        <p className="text-xs text-slate-400 leading-relaxed">
          One standard deviation, from random error only. Real errors are usually larger, because a
          calibration also carries errors that repeat in every photo.
        </p>
        <LensCaveat doc={doc} />
      </div>
      <div className="grid gap-3">
        <TopView doc={doc} unit={unit} a={a} b={b} />
        {doc.curve && doc.curve.length > 2 && <Curve doc={doc} unit={unit} mm={mm} />}
      </div>
    </div>
  );
}

function LensCaveat({ doc }: { doc: TaskDoc }) {
  if (doc.lens_recorded !== false) return null;
  return (
    <p className="text-sm text-amber-200 leading-relaxed">
      This run has no lens uncertainty on record. If the lenses were kept as they were, any error in them is not counted here, and these numbers can be far too small.
    </p>
  );
}

function Message({ children }: { children: ReactNode }) {
  return <div className="h-64 grid place-items-center text-sm text-slate-300 p-6 text-center">{children}</div>;
}

function Big({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-white/10 px-4 py-3">
      <div className="text-3xl font-semibold text-white tnum leading-tight">{value}</div>
      <div className="text-xs text-slate-300 mt-1">{label}</div>
    </div>
  );
}

/** Seen from above: the two cameras, the point, and its uncertainty drawn large enough to see. */
function TopView({ doc, unit, a, b }: { doc: TaskDoc; unit: Unit; a: string; b: string }) {
  const W = 420, H = 290, camY = H - 46, pointY = 86;
  const half = Math.min(150, Math.max(26, (doc.baseline / doc.distance) * (camY - pointY) * 0.5));
  const x1 = W / 2 - half, x2 = W / 2 + half;
  const scale = Math.min(74 / Math.max(doc.depth, 1e-12), 0.45 * (camY - pointY) / Math.max(doc.depth, 1e-12));
  const ry = Math.max(4, doc.depth * scale), rx = Math.max(2.5, doc.sideways * scale);
  const times = (camY - pointY) / doc.distance;
  const zoom = scale / Math.max(times, 1e-12);
  return (
    <figure className="grid gap-1">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img"
        aria-label="Top view of the two cameras, the point and its uncertainty">
        <line x1={x1} y1={camY} x2={W / 2} y2={pointY} stroke="#64748b" strokeWidth="1" />
        <line x1={x2} y1={camY} x2={W / 2} y2={pointY} stroke="#64748b" strokeWidth="1" />
        <ellipse cx={W / 2} cy={pointY} rx={rx} ry={ry} fill="rgba(56,189,248,0.25)" stroke="#38bdf8" strokeWidth="1.5" />
        <circle cx={W / 2} cy={pointY} r="2.5" fill="#fff" />
        {[[x1, a], [x2, b]].map(([x, id]) => (
          <g key={String(id)}>
            <path d={`M${Number(x) - 9},${camY + 10} L${Number(x) + 9},${camY + 10} L${Number(x) + 5},${camY - 4} L${Number(x) - 5},${camY - 4} Z`} fill="#cbd5e1" />
            <text x={Number(x)} y={camY + 24} textAnchor={Number(x) < W / 2 ? "end" : "start"} dx={Number(x) < W / 2 ? 9 : -9} fontSize="11" fill="#cbd5e1">{camName(String(id))}</text>
          </g>
        ))}
        <text x={W / 2 + Math.max(rx, 8) + 8} y={pointY + 4} fontSize="11" fill="#e2e8f0">where the point could be</text>
        <text x={W / 2} y={camY + 40} textAnchor="middle" fontSize="11" fill="#94a3b8">{length(doc.baseline, unit)} apart</text>
      </svg>
      <figcaption className="text-xs text-slate-400">
        Seen from above. The blue shape is the uncertainty, drawn {zoom >= 10 ? `${Math.round(zoom)} times` : `${zoom.toFixed(1)} times`} larger than the rest so it can be seen.
      </figcaption>
    </figure>
  );
}

/** Seen from above, to scale: both cameras where they really are, and the point in front of the first. */
function PlanView({ doc, unit, a, b }: { doc: TaskDoc; unit: Unit; a: string; b: string }) {
  const W = 420, H = 230, M = 46;
  const B = doc.plan?.b ?? [0, doc.baseline], axis = doc.plan?.axis ?? [0, -1];
  const P: [number, number] = [0, doc.distance];
  const xs = [0, B[0], P[0]], zs = [0, B[1], P[1]];
  const spanZ = Math.max(...zs) - Math.min(...zs) || 1, spanX = Math.max(...xs) - Math.min(...xs);
  const k = Math.min((W - 2 * M) / spanZ, spanX > 0 ? (H - 2 * M) / spanX : Infinity);
  const midZ = (Math.max(...zs) + Math.min(...zs)) / 2, midX = (Math.max(...xs) + Math.min(...xs)) / 2;
  // the first camera's viewing axis runs left to right
  const X = (z: number) => W / 2 + (z - midZ) * k, Y = (x: number) => H / 2 + (x - midX) * k;
  const px = X(P[1]), py = Y(P[0]);
  const scale = Math.min(34 / Math.max(doc.sideways, doc.depth, 1e-12), 1e9);
  const rx = Math.max(3, doc.depth * scale), ry = Math.max(3, doc.sideways * scale);
  const zoom = scale / k;
  const camera = (cx: number, cy: number, dx: number, dy: number, id: string, under: boolean) => {
    const n = Math.hypot(dx, dy) || 1, ux = dx / n, uy = dy / n, s = 13;
    const tip = `${cx + ux * s},${cy + uy * s}`;
    const l = `${cx - uy * s * 0.7},${cy + ux * s * 0.7}`, r = `${cx + uy * s * 0.7},${cy - ux * s * 0.7}`;
    return (
      <g key={id}>
        <polygon points={`${l} ${tip} ${r}`} fill="#cbd5e1" />
        <text x={cx} y={cy + (under ? 30 : -20)} textAnchor="middle" fontSize="12" fill="#e2e8f0">{camName(id)}</text>
      </g>
    );
  };
  return (
    <figure className="grid gap-1">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label="The two cameras and the point, seen from above">
        <line x1={X(0)} y1={Y(0)} x2={px} y2={py} stroke="#64748b" strokeWidth="1" />
        <line x1={X(0)} y1={Y(0)} x2={X(B[1])} y2={Y(B[0])} stroke="#fbbf24" strokeWidth="1" strokeDasharray="4 4" />
        <text x={(X(0) + X(B[1])) / 2} y={(Y(0) + Y(B[0])) / 2 + 40} textAnchor="middle" fontSize="12" fill="#fbbf24">{length(doc.baseline, unit)} apart</text>
        {camera(X(0), Y(0), 1, 0, a, true)}
        {camera(X(B[1]), Y(B[0]), axis[1], axis[0], b, true)}
        <ellipse cx={px} cy={py} rx={rx} ry={ry} fill="rgba(56,189,248,0.25)" stroke="#38bdf8" strokeWidth="1.5" />
        <circle cx={px} cy={py} r="2.5" fill="#fff" />
        <text x={px} y={py - Math.max(ry, 8) - 6} textAnchor="middle" fontSize="12" fill="#e2e8f0">the point</text>
      </svg>
      <figcaption className="text-xs text-slate-400">
        Seen from above, to scale. The blue shape is the uncertainty, drawn {zoom >= 10 ? `${Math.round(zoom)} times` : `${zoom.toFixed(1)} times`} larger than the rest so it can be seen.
      </figcaption>
    </figure>
  );
}

function Curve({ doc, unit, mm }: { doc: TaskDoc; unit: Unit; mm: number | null }) {
  const pts = doc.curve ?? [];
  const W = 420, H = 150, L = 8, R = 8, T = 10, B = 22;
  const x0 = Math.log(pts[0].distance), x1 = Math.log(pts[pts.length - 1].distance);
  const all = pts.flatMap((p) => [p.depth, p.sideways]).filter((v) => v > 0);
  const y0 = Math.log(Math.min(...all, mm ?? Infinity)), y1 = Math.log(Math.max(...all, mm ?? 0));
  const X = (d: number) => L + ((W - L - R) * (Math.log(d) - x0)) / Math.max(x1 - x0, 1e-9);
  const Y = (v: number) => T + ((H - T - B) * (y1 - Math.log(v))) / Math.max(y1 - y0, 1e-9);
  const path = (key: "depth" | "sideways") => pts.map((p, k) => `${k ? "L" : "M"}${X(p.distance).toFixed(1)},${Y(p[key]).toFixed(1)}`).join(" ");
  const last = pts[pts.length - 1];
  return (
    <figure className="grid gap-1">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label="Uncertainty against distance">
        {mm != null && (
          <g>
            <line x1={L} x2={W - R} y1={Y(mm)} y2={Y(mm)} stroke="#34d399" strokeDasharray="4 4" strokeWidth="1" />
            <text x={L + 2} y={Y(mm) - 4} fontSize="10" fill="#34d399">1 mm</text>
          </g>
        )}
        <path d={path("depth")} fill="none" stroke="#38bdf8" strokeWidth="2" />
        <path d={path("sideways")} fill="none" stroke="#f59e0b" strokeWidth="2" />
        <text x={X(last.distance) - 4} y={Y(last.depth) + 14} textAnchor="end" fontSize="10" fill="#38bdf8">{doc.mode === "transfer" ? "line of sight" : "depth"}</text>
        <text x={X(last.distance) - 4} y={Y(last.sideways) - 6} textAnchor="end" fontSize="10" fill="#f59e0b">sideways</text>
        <line x1={X(doc.distance)} x2={X(doc.distance)} y1={T} y2={H - B} stroke="#e2e8f0" strokeWidth="1" />
        <circle cx={X(doc.distance)} cy={Y(doc.depth)} r="3.5" fill="#38bdf8" />
        <circle cx={X(doc.distance)} cy={Y(doc.sideways)} r="3.5" fill="#f59e0b" />
        <text x={L} y={H - 6} fontSize="10" fill="#94a3b8">{length(pts[0].distance, unit)}</text>
        <text x={W - R} y={H - 6} fontSize="10" fill="#94a3b8" textAnchor="end">{length(last.distance, unit)}</text>
        <text x={W / 2} y={H - 6} fontSize="10" fill="#94a3b8" textAnchor="middle">distance to the point</text>
      </svg>
      <figcaption className="text-xs text-slate-400">How the uncertainty changes with distance. Both scales are stretched so near and far fit.</figcaption>
    </figure>
  );
}
