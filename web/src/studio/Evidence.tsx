// Evidence for one camera (layer 2): where its errors are, where its lens is uncertain, and
// which photos are worst. Each view answers one question and says what to do about it.

import { useEffect, useState } from "react";
import { api, type CameraDiagnostics } from "./api";
import { int, num, px } from "./format";
import { Card, Loading, Note } from "./ui";

export function Evidence({ runId, cam, available }: { runId: string; cam: string; available: boolean }) {
  const [d, setD] = useState<CameraDiagnostics | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    setD(null);
    setError(null);
    if (available) api.diagnostics(runId, cam).then(setD, (e: Error) => setError(e.message));
  }, [runId, cam, available]);

  if (!available)
    return (
      <Card title="Evidence">
        <Note>
          This result has no diagnostic files. Recalibrate with a current ds-msp to get the
          per-corner residuals and the lens uncertainty map.
        </Note>
      </Card>
    );
  if (error) return <Card title="Evidence"><Note>Could not read the diagnostic files: {error}</Note></Card>;
  if (!d) return <Card title="Evidence"><Loading what="evidence" /></Card>;
  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <ResidualCard d={d} />
      <LensCard d={d} />
      <div className="lg:col-span-2"><FramesCard d={d} /></div>
    </div>
  );
}

function Frame({ w, h, children, label }: { w: number; h: number; children: React.ReactNode; label: string }) {
  return (
    <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label}
      className="w-full h-auto rounded border border-s-rule bg-s-sunken">
      {children}
    </svg>
  );
}

function Answer({ tone, children }: { tone: "ok" | "warn" | "bad"; children: React.ReactNode }) {
  const c = { ok: "border-s-ok/30 bg-s-ok-soft text-s-ok", warn: "border-s-warn/40 bg-s-warn-soft text-s-warn",
    bad: "border-s-bad/40 bg-s-bad-soft text-s-bad" }[tone];
  return <p className={`rounded-md border px-3 py-2 text-sm font-medium ${c}`}>{children}</p>;
}

function Toggle({ on, set, label }: { on: boolean; set: (v: boolean) => void; label: string }) {
  return (
    <button type="button" onClick={() => set(!on)} className="justify-self-start text-xs text-s-accent hover:underline">
      {on ? "Hide" : "Show"} {label}
    </button>
  );
}

/** Median error (line) and 90th percentile (band) from the image centre to its corners. */
function RingChart({ rings }: { rings: NonNullable<CameraDiagnostics["pattern"]>["rings"] }) {
  const pts = rings.filter((r) => r.median_px != null);
  if (pts.length < 2) return null;
  const W = 300, H = 120, L = 34, R = 8, T = 8, B = 22;
  const medMax = Math.max(...pts.map((r) => r.median_px ?? 0));
  const ymax = Math.min(Math.max(...pts.map((r) => r.p90_px ?? 0)), 4 * medMax) * 1.1 || 1;
  const clip = (v: number) => Math.min(v, ymax);
  const X = (r: number) => L + r * (W - L - R);
  const Y = (v: number) => T + (1 - v / ymax) * (H - T - B);
  const mid = (r: (typeof pts)[number]) => (r.from + r.to) / 2;
  const line = pts.map((r, k) => `${k ? "L" : "M"}${X(mid(r))},${Y(r.median_px!)}`).join(" ");
  const band = pts.map((r, k) => `${k ? "L" : "M"}${X(mid(r))},${Y(clip(r.p90_px!))}`).join(" ")
    + " " + [...pts].reverse().map((r) => `L${X(mid(r))},${Y(r.median_px!)}`).join(" ") + " Z";
  const ticks = [0, ymax / 2, ymax].map((t) => Number(t.toPrecision(2)));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label="error size from centre to corner">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={Y(t)} y2={Y(t)} stroke="var(--s-line)" />
          <text x={L - 4} y={Y(t) + 3} textAnchor="end" fontSize="9" fill="var(--s-faint)">{t}</text>
        </g>
      ))}
      <path d={band} fill="var(--s-accent)" fillOpacity="0.15" />
      <path d={line} fill="none" stroke="var(--s-accent)" strokeWidth="2" />
      {pts.map((r) => <circle key={r.from} cx={X(mid(r))} cy={Y(r.median_px!)} r="2.5" fill="var(--s-accent)" />)}
      <text x={L} y={H - 6} fontSize="9" fill="var(--s-faint)">centre</text>
      <text x={W - R} y={H - 6} fontSize="9" fill="var(--s-faint)" textAnchor="end">corners</text>
      <text x={L + 4} y={T + 8} fontSize="9" fill="var(--s-faint)">px</text>
    </svg>
  );
}

/** 6 x 4 regions of the image; one arrow per region for its average error. */
function RegionGrid({ grid, W, H, rms }: { grid: NonNullable<CameraDiagnostics["pattern"]>["grid"]; W: number; H: number; rms: number }) {
  const cw = W / grid.nx, ch = H / grid.ny;
  // an average error as large as the camera's RMS fills 45% of the cell; random regions stay short
  const half = 0.45 * Math.min(cw, ch);
  const k = half / Math.max(rms, 1e-9);
  const sw = W / 300;
  return (
    <Frame w={W} h={H} label="average error per image region">
      <defs>
        <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M0 0L10 5L0 10z" fill="context-stroke" />
        </marker>
      </defs>
      {grid.cells.map((c) => {
        const x = c.i * cw, y = c.j * ch, cx = x + cw / 2, cy = y + ch / 2;
        return (
          <g key={`${c.i}-${c.j}`}>
            <rect x={x} y={y} width={cw} height={ch} fill={c.biased ? "var(--s-warn)" : "transparent"} fillOpacity={c.biased ? 0.18 : 0}
              stroke="var(--s-line)" strokeWidth={sw / 2} />
            {c.mx == null ? (
              <rect x={x} y={y} width={cw} height={ch} fill="var(--s-line)" fillOpacity={0.6} />
            ) : (
              <line x1={cx} y1={cy} x2={cx + Math.max(-half, Math.min(half, c.mx * k))} y2={cy + Math.max(-half, Math.min(half, c.my! * k))} markerEnd="url(#ah)"
                stroke={c.biased ? "var(--s-warn)" : "var(--s-faint)"} strokeWidth={c.biased ? 2.2 * sw : 1.4 * sw} />
            )}
          </g>
        );
      })}
    </Frame>
  );
}

function ResidualCard({ d }: { d: CameraDiagnostics }) {
  const [raw, setRaw] = useState(false);
  const r = d.residuals;
  const W = d.width ?? 1, H = d.height ?? 1;
  const out = d.outliers;
  const p = d.pattern;
  if (!r || !r.n_shown || !p)
    return <Card title="Is there a pattern in the errors?"><Note>Too few corners of this camera project to tell.</Note></Card>;
  const lens = r.dx.map((x, i) => Math.hypot(x, r.dy[i])).sort((a, b) => a - b);
  const med = lens[Math.floor(lens.length / 2)] || 1e-6;
  const scale = Math.min(5000, Math.max(1, (0.03 * W) / med));
  const sw = W / 500;
  const cap = 0.04 * W;
  return (
    <Card title="Is there a pattern in the errors?">
      <div id="ev-residuals" className="grid gap-3">
        <Answer tone={p.kind === "none" ? "ok" : "warn"}>{p.answer}</Answer>
        <div className="grid gap-1">
          <div className="text-xs font-medium">Error size, centre to corners</div>
          <RingChart rings={p.rings} />
          <Note>Line: typical error. Band: up to 90% of corners (cut at the top when a few photos are far off). A flat line is good; one that rises toward the corners means the lens model does not fit there.</Note>
        </div>
        <div className="grid gap-1">
          <div className="text-xs font-medium">Average error per region of the image</div>
          <RegionGrid grid={p.grid} W={W} H={H} rms={p.typical_px} />
          <Note>Each arrow is the average error in that region; a full-length arrow equals the camera's typical error. Short grey arrows are normal; orange regions lean one way; shaded regions had too few corners.</Note>
        </div>
        <div className="text-xs text-s-muted tnum">
          RMS {px(r.rms_px)} over {int(r.n_valid)} corners
          {out?.outlier_frac != null ? ` · ${num(out.outlier_frac * 100, 1)}% outside the ${px(d.band_px)} band` : ""}
        </div>
        <Toggle on={raw} set={setRaw} label="every corner" />
        {raw && (
          <div className="grid gap-1">
            <Frame w={W} h={H} label="residual arrows over the image">
              {r.u.map((u, i) => {
                const k = Math.min(scale, cap / Math.max(Math.hypot(r.dx[i], r.dy[i]), 1e-12));
                return r.within[i] ? (
                  <line key={i} x1={u} y1={r.v[i]} x2={u + r.dx[i] * k} y2={r.v[i] + r.dy[i] * k}
                    stroke="var(--s-accent)" strokeWidth={sw} strokeOpacity={0.55} />
                ) : (
                  <g key={i}>
                    <line x1={u} y1={r.v[i]} x2={u + r.dx[i] * k} y2={r.v[i] + r.dy[i] * k} stroke="var(--s-bad)" strokeWidth={2 * sw} />
                    <circle cx={u} cy={r.v[i]} r={3 * sw} fill="var(--s-bad)" />
                  </g>
                );
              })}
            </Frame>
            <Note>Every corner's error, drawn {num(scale, 0)}× longer. Red dots are outside the inlier band (arrows shortened).</Note>
          </div>
        )}
      </div>
    </Card>
  );
}

const ZONE = {
  trusted: { fill: "var(--s-ok)", op: 0.45, label: "Trusted", note: "board seen here, within the limit" },
  extrapolated: { fill: "var(--s-faint)", op: 0.18, label: "Never covered", note: "no photo had the board here; the model guesses" },
  uncertain: { fill: "var(--s-warn)", op: 0.7, label: "Uncertain", note: "over the limit" },
} as const;

function LensCard({ d }: { d: CameraDiagnostics }) {
  const [raw, setRaw] = useState(false);
  const g = d.lens;
  const W = d.width ?? 1, H = d.height ?? 1;
  if (!g || !g.zones)
    return <Card title="Where can you trust this lens?"><Note>Not computed: intrinsics were fixed or the lens check was off.</Note></Card>;
  const z = g.zones;
  const cw = W / g.nx, ch = H / g.ny;
  const tone = g.unbounded || z.share.uncertain > 0.25 ? "bad" : z.share.trusted > 0.8 ? "ok" : "warn";
  const vals = g.sigma_px.filter((x): x is number => x != null);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  return (
    <Card title="Where can you trust this lens?" aside={`limit ${px(z.limit_px)}`}>
      <div id="ev-lens" className="grid gap-3">
        <Answer tone={tone}>{z.answer}</Answer>
        <Frame w={W} h={H} label="lens trust zones">
          {z.zones.map((k, i) => {
            if (k === "none") return null;
            const s = ZONE[k];
            return <rect key={i} x={g.u[i] - cw / 2} y={g.v[i] - ch / 2} width={cw} height={ch}
              fill={s.fill} fillOpacity={s.op} stroke="var(--s-surface)" strokeWidth={W / 600} />;
          })}
        </Frame>
        <ul className="grid gap-1 text-xs">
          {(Object.keys(ZONE) as (keyof typeof ZONE)[]).map((k) => (
            <li key={k} className="flex items-center gap-2">
              <span className="inline-block h-3 w-3 rounded-sm" style={{ background: ZONE[k].fill, opacity: ZONE[k].op + 0.2 }} />
              <span className="font-medium tnum w-10 text-right">{num(z.share[k] * 100, 0)}%</span>
              <span>{ZONE[k].label}</span>
              <span className="text-s-muted">· {ZONE[k].note}</span>
            </li>
          ))}
        </ul>
        <Note>
          To widen the trusted zone, take photos with the board in the grey and orange areas, close
          and tilted. Inside the green zone the lens is known to {px(z.sigma_max_trusted_px)} or better
          (photo noise only).
        </Note>
        <Toggle on={raw} set={setRaw} label="uncertainty values" />
        {raw && (
          <div className="grid gap-1">
            <Frame w={W} h={H} label="lens uncertainty values">
              {g.sigma_px.map((s, i) => s == null ? null : (
                <rect key={i} x={g.u[i] - cw / 2} y={g.v[i] - ch / 2} width={cw} height={ch}
                  fill="var(--s-accent)" fillOpacity={0.05 + 0.85 * ((s - lo) / Math.max(hi - lo, 1e-12))} />
              ))}
            </Frame>
            <div className="flex items-center gap-2 text-xs text-s-muted tnum">
              <span>{px(lo, 3)}</span>
              <span className="h-2 flex-1 rounded" style={{ background: "linear-gradient(to right, color-mix(in srgb, var(--s-accent) 5%, transparent), var(--s-accent))" }} />
              <span>{px(hi, 3)}</span>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}

function FramesCard({ d }: { d: CameraDiagnostics }) {
  const f = d.worst_frames ?? [];
  return (
    <Card title="Worst photos" aside={d.n_frames ? `of ${int(d.n_frames)}` : undefined}>
      <div id="ev-frames" className="grid gap-2">
        {f.length === 0 ? (
          <Note>No photo of this camera projects.</Note>
        ) : (
          <table className="text-sm tnum w-full">
            <thead>
              <tr className="text-left text-xs text-s-faint">
                <th className="py-1 font-medium">Frame</th>
                <th className="py-1 font-medium text-right">RMS</th>
                <th className="py-1 font-medium text-right">Max</th>
                <th className="py-1 font-medium text-right">Outside band</th>
              </tr>
            </thead>
            <tbody>
              {f.map((x) => (
                <tr key={x.frame} className="border-t border-s-rule">
                  <td className="py-1 font-studio-mono">{x.frame}</td>
                  <td className="py-1 text-right">{px(x.rms_px)}</td>
                  <td className="py-1 text-right">{px(x.max_px, 1)}</td>
                  <td className={`py-1 text-right ${x.n_outside ? "text-s-bad" : "text-s-muted"}`}>{x.n_outside} / {x.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <Note>
          A photo far worse than the rest usually has a wrong detection, blur or glare. Retake or
          remove it. If the same frame is worst in several cameras, check the camera sync.
        </Note>
      </div>
    </Card>
  );
}
