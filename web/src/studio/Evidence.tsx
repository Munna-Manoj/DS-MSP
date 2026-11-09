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
    <div className="grid gap-5 lg:grid-cols-3">
      <ResidualCard d={d} />
      <LensCard d={d} />
      <FramesCard d={d} />
    </div>
  );
}

function Frame({ w, h, children, label }: { w: number; h: number; children: React.ReactNode; label: string }) {
  return (
    <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label}
      className="w-full h-auto rounded border border-s-line bg-s-sunken">
      {children}
    </svg>
  );
}

function ResidualCard({ d }: { d: CameraDiagnostics }) {
  const r = d.residuals;
  const W = d.width ?? 1, H = d.height ?? 1;
  const out = d.outliers;
  if (!r || !r.n_shown)
    return <Card title="Where the errors are" className="" ><Note>No corner of this camera projects.</Note></Card>;
  const lens = r.dx.map((x, i) => Math.hypot(x, r.dy[i])).sort((a, b) => a - b);
  const med = lens[Math.floor(lens.length / 2)] || 1e-6;
  const scale = Math.min(5000, Math.max(1, (0.03 * W) / med));
  const sw = W / 500;
  const cap = 0.04 * W;
  return (
    <Card title="Where the errors are" aside={`arrows ${num(scale, 0)}× longer`} >
      <div id="ev-residuals" className="grid gap-2">
        <Frame w={W} h={H} label="residual arrows over the image">
          {r.u.map((u, i) => {
            // outliers would run across the image at the inlier scale: cap their drawn length
            const k = Math.min(scale, cap / Math.max(Math.hypot(r.dx[i], r.dy[i]), 1e-12));
            return r.within[i] ? (
              <line key={i} x1={u} y1={r.v[i]} x2={u + r.dx[i] * k} y2={r.v[i] + r.dy[i] * k}
                stroke="var(--s-accent)" strokeWidth={sw} strokeOpacity={0.55} />
            ) : (
              <g key={i}>
                <line x1={u} y1={r.v[i]} x2={u + r.dx[i] * k} y2={r.v[i] + r.dy[i] * k}
                  stroke="var(--s-bad)" strokeWidth={2 * sw} />
                <circle cx={u} cy={r.v[i]} r={3 * sw} fill="var(--s-bad)" />
              </g>
            );
          })}
        </Frame>
        <div className="text-xs text-s-muted tnum">
          RMS {px(r.rms_px)} over {int(r.n_valid)} corners
          {r.n_shown < r.n_valid ? ` (${int(r.n_shown)} drawn)` : ""} ·{" "}
          {out?.outlier_frac != null ? `${num(out.outlier_frac * 100, 1)}% outside the ${px(d.band_px)} band` : ""}
        </div>
        <Note>
          Random directions are normal. Arrows that grow toward the edges mean the lens model is too
          stiff; arrows pointing the same way along a line mean detector bias. Red dots are corners outside
          the inlier band; their arrows are shortened to fit.
        </Note>
      </div>
    </Card>
  );
}

function LensCard({ d }: { d: CameraDiagnostics }) {
  const g = d.lens;
  const W = d.width ?? 1, H = d.height ?? 1;
  if (!g) return <Card title="Where the lens is known"><Note>Not computed: intrinsics were fixed or the lens check was off.</Note></Card>;
  if (g.unbounded)
    return (
      <Card title="Where the lens is known">
        <div id="ev-lens"><Note>Not determined anywhere: the photos do not pin this lens down. Add photos covering the whole image.</Note></div>
      </Card>
    );
  const vals = g.sigma_px.filter((x): x is number => x != null);
  const lim = g.limit_px ?? Math.max(...vals);
  const cw = W / g.nx, ch = H / g.ny;
  const over = vals.filter((x) => x > lim).length;
  return (
    <Card title="Where the lens is known" aside={`limit ${px(lim)}`}>
      <div id="ev-lens" className="grid gap-2">
        <Frame w={W} h={H} label="lens uncertainty map">
          {g.sigma_px.map((s, i) => {
            const x = g.u[i] - cw / 2, y = g.v[i] - ch / 2;
            if (s == null) return <rect key={i} x={x} y={y} width={cw} height={ch} fill="none" />;
            const t = Math.min(1, s / lim);
            return (
              <g key={i}>
                <rect x={x} y={y} width={cw} height={ch}
                  fill={s > lim ? "var(--s-warn)" : "var(--s-accent)"} fillOpacity={s > lim ? 0.75 : 0.08 + 0.6 * t} />
                {g.observed[i] && <circle cx={g.u[i]} cy={g.v[i]} r={Math.min(cw, ch) / 9} fill="var(--s-text)" fillOpacity={0.5} />}
              </g>
            );
          })}
        </Frame>
        <div className="text-xs text-s-muted tnum">
          median {px(vals.sort((a, b) => a - b)[Math.floor(vals.length / 2)])} · {over} of {vals.length} cells over the limit
        </div>
        <Note>
          Darker means less certain; orange is over the limit. Dots mark cells where the board was
          seen. Add photos with the board in the dark and orange areas. Photo noise only: the real
          error is at least this large.
        </Note>
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
                <tr key={x.frame} className="border-t border-s-line">
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
