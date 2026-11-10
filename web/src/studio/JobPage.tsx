// One calibration started from Studio: what it is doing now, how far it is, its output, and
// where the result is once it ends. Polls GET api/job while the calibration is queued or running.
import { useEffect, useRef, useState, type ReactNode } from "react";
import { href } from "../router";
import { api, type JobDoc, type JobProgress, type JobState } from "./api";
import { num, when } from "./format";
import { Badge, Button, Callout, Card, ErrorBox, Loading, Note, Stat } from "./ui";

const STATE: Record<JobState, { label: string; tone: "ok" | "warn" | "bad" | "info" | "plain" }> = {
  queued: { label: "Waiting", tone: "plain" },
  running: { label: "Running", tone: "info" },
  done: { label: "Finished", tone: "ok" },
  failed: { label: "Failed", tone: "bad" },
  cancelled: { label: "Cancelled", tone: "plain" },
  interrupted: { label: "Interrupted", tone: "warn" },
};

/** The solver's stage labels, in plain words. */
export function stageWords(name: string | null): string {
  if (!name) return "Starting";
  if (name.startsWith("(0)")) return "Working out each lens";
  if (name.startsWith("(a)")) return "Placing the board in each photo";
  if (name.startsWith("(b)")) return "Placing the cameras";
  if (name.startsWith("(c)")) return "Refining cameras and lenses together";
  if (name.startsWith("(d)")) return "Refining the shape of the target";
  if (name.startsWith("(save)")) return "Writing the result";
  return name;
}

function clock(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const s = Math.max(0, Math.round(seconds));
  const m = Math.floor(s / 60);
  return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : m > 0 ? `${m} min ${s % 60} s` : `${s} s`;
}

export function JobPage({ id }: { id: string }) {
  const [job, setJob] = useState<JobDoc | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [cancelling, setCancelling] = useState(false);

  useEffect(() => {
    let live = true;
    let timer: number | undefined;
    const tick = () => {
      api.job(id).then(
        (doc) => {
          if (!live) return;
          setJob(doc);
          setError(null);
          if (doc.state === "queued" || doc.state === "running") timer = window.setTimeout(tick, 1000);
        },
        (e: Error) => { if (live) { setError(e.message); timer = window.setTimeout(tick, 4000); } },
      );
    };
    setJob(null);
    tick();
    return () => { live = false; window.clearTimeout(timer); };
  }, [id]);

  if (!job)
    return (
      <Page>
        {error ? <ErrorBox title="Could not open this calibration" detail={error} /> : <Loading what="calibration" />}
      </Page>
    );

  const active = job.state === "queued" || job.state === "running";
  const p = job.progress;
  const st = STATE[job.state];

  function cancel() {
    setCancelling(true);
    api.cancelJob(id).then(() => setConfirming(false), (e: Error) => setError(e.message))
      .finally(() => setCancelling(false));
  }

  return (
    <Page>
      <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
        <div className="grid gap-1 min-w-0">
          <a href={href({ page: "runs" })} className="text-xs text-s-accent hover:underline">← All results</a>
          <h1 className="text-2xl font-semibold break-all flex flex-wrap items-center gap-3">
            {job.name} <Badge tone={st.tone}>{st.label}</Badge>
          </h1>
          <p className="text-s-muted text-sm">Started {when(job.started_utc ?? job.created_utc)}</p>
        </div>
        <div className="ml-auto flex flex-wrap gap-2">
          {active && !confirming && <Button kind="danger" onClick={() => setConfirming(true)}>Cancel</Button>}
          {!active && (
            <a href={href({ page: "new", from: job.repeat_of ?? id })}
              className="inline-flex items-center rounded-md px-3.5 min-h-10 text-sm font-medium border border-s-line bg-s-surface hover:bg-s-sunken">
              New calibration with these settings
            </a>
          )}
          {job.has_result && (
            <a href={href({ page: "run", id })}
              className="inline-flex items-center rounded-md px-3.5 min-h-10 text-sm font-medium bg-s-accent text-white hover:opacity-90">
              Open the result
            </a>
          )}
        </div>
      </div>

      {error && <ErrorBox title="Lost contact with Studio" detail={`${error}. Trying again.`} />}

      {confirming && (
        <Callout tone="warn" title="Cancel this calibration?">
          <p>It stops now and no result is written. The settings stay in its folder, so it can be started again.</p>
          <div className="flex flex-wrap gap-2 mt-3">
            <Button kind="danger" onClick={cancel} busy={cancelling}>Yes, cancel it</Button>
            <Button onClick={() => setConfirming(false)}>Keep it running</Button>
          </div>
        </Callout>
      )}

      {job.state === "queued" && (
        <Callout tone="info" title={job.queue_position === 1 ? "Next in line" : `Number ${job.queue_position ?? "?"} in line`}>
          Calibrations run one at a time, because each one uses every processor core. This one
          starts when the one before it ends.
        </Callout>
      )}
      {job.state === "done" && (
        <Callout tone="ok" title="The calibration finished">
          {p.final_rms != null && <>The worst camera's typical error is {num(p.final_rms)} px. </>}
          Open the result to see whether every camera can be trusted.
          {job.exit_code !== 0 && <> The fit to the photos is poor (exit code {job.exit_code}); the result page says why.</>}
        </Callout>
      )}
      {job.reused_from && (
        <Callout tone="info" title="Corner detections reused">
          The corners found by{" "}
          <a href={href({ page: "run", id: job.reused_from })} className="text-s-accent hover:underline">{job.reused_from.split("/").pop()?.replace(/^\d+:/, "")}</a>{" "}
          are used again: same photos, same board. That result was copied from, not changed.
        </Callout>
      )}
      {job.repeat_of && (
        <Callout tone={job.repeat?.ok ? (job.repeat.same ? "ok" : "warn") : "info"}
          title={!job.repeat ? "This is a repeat of an earlier result"
            : !job.repeat.ok ? "Could not compare with the original"
            : job.repeat.same ? "Same calibration as the original" : "The repeat differs from the original"}>
          {!job.repeat && <>When it finishes, Studio compares its calibration files with the original's.</>}
          {job.repeat && !job.repeat.ok && <>{job.repeat.reason}</>}
          {job.repeat?.rows && (
            <ul className="grid gap-0.5 mt-1 text-xs">
              {job.repeat.rows.map((r) => (
                <li key={r.what} className="flex items-center gap-2">
                  <span className={r.same ? "text-s-ok" : "text-s-bad"} aria-hidden="true">{r.same ? "✓" : "×"}</span>
                  <span className="font-studio-mono">{r.what}</span>
                  <span className="text-s-muted">{r.same ? "identical" : "different"}</span>
                </li>
              ))}
            </ul>
          )}
          <a href={href({ page: "run", id: job.repeat_of })} className="inline-block mt-2 text-s-accent hover:underline text-xs">Open the original</a>
        </Callout>
      )}
      {job.state === "failed" && (
        <Callout tone="bad" title="The calibration stopped without a result">
          <p className="break-words">{job.error}</p>
          <p className="mt-2 text-s-muted">The output below shows what led to it. Change the settings and start again.</p>
        </Callout>
      )}
      {job.state === "interrupted" && <Callout tone="warn" title="Interrupted">{job.error}</Callout>}
      {job.state === "cancelled" && <Callout tone="info" title="Cancelled">No result was written.</Callout>}

      {(job.state === "running" || p.stages.length > 0) && (
        <Progress p={p} running={job.state === "running"} stopped={!active && job.state !== "done"} />
      )}

      <Card title="Output" aside="what the calibration printed">
        <LogView lines={job.log} follow={active} />
      </Card>

      <Card title="Run it from a terminal">
        <div className="grid gap-2 min-w-0">
          <Note>The same calibration, with the settings saved in this folder:</Note>
          <CopyLine text={job.command} />
        </div>
      </Card>
    </Page>
  );
}

function Page({ children }: { children: ReactNode }) {
  return <main className="mx-auto w-full max-w-5xl px-4 sm:px-6 py-6 grid gap-5">{children}</main>;
}

function Progress({ p, running, stopped }: { p: JobProgress; running: boolean; stopped: boolean }) {
  const detecting = p.phase === "detecting" && p.detect;
  const saving = p.phase === "saving" && p.save;
  const headline = p.phase === "starting" ? "Starting"
    : p.phase === "detecting" ? "Finding the board in the photos"
    : p.phase === "done" ? "Done"
    : stopped ? `Stopped while: ${stageWords(p.stage).toLowerCase()}`
    : stageWords(p.stage);
  return (
    <Card title={running ? "Now" : "What it did"} aside={p.elapsed != null ? (running ? `${clock(p.elapsed)} so far` : `took ${clock(p.elapsed)}`) : undefined}>
      <div className="grid gap-5">
        <div className="flex flex-wrap items-center gap-x-8 gap-y-3">
          <div className="grid gap-0.5 min-w-0">
            <div className="text-lg font-semibold flex items-center gap-2.5">
              {running && <span className="size-2.5 rounded-full bg-s-accent animate-pulse" aria-hidden="true" />}
              {headline}
            </div>
            {p.stage && p.phase !== "done" && <div className="font-studio-mono text-xs text-s-faint">{p.stage}</div>}
          </div>
          {p.iter && (
            <>
              <Stat k="Step" v={`${p.iter.it} of at most ${p.iter.max_iter}`} />
              <Stat k="Typical error now" v={p.iter.rms == null ? "—" : `${num(p.iter.rms)} px`} />
            </>
          )}
          {p.scene && p.scene.observations > 0 && <Stat k="Board sightings used" v={p.scene.observations.toLocaleString("en-US")} sub={`${p.scene.cameras} cameras`} />}
        </div>

        {detecting && <Bars title="Photos checked, per camera" counts={p.detect!} />}
        {saving && <Bars title="Images written, per camera" counts={p.save!} />}

        {p.history.length > 1 && <ErrorCurve history={p.history} />}

        {p.stages.length > 0 && (
          <ol className="grid gap-1.5">
            {p.stages.map((s, k) => {
              const last = k === p.stages.length - 1;
              const took = (last ? p.elapsed ?? s.t : p.stages[k + 1].t) - s.t;
              const current = last && running && p.phase !== "done";
              const broke = last && stopped && p.phase !== "done";
              return (
                <li key={`${s.name}-${k}`} className="flex items-baseline gap-3 text-sm">
                  <span className={`size-4 shrink-0 rounded-full grid place-items-center text-[10px] font-bold self-center ${broke ? "bg-s-bad-soft text-s-bad" : current ? "bg-s-accent-soft text-s-accent" : "bg-s-ok-soft text-s-ok"}`} aria-hidden="true">
                    {broke ? "×" : current ? "…" : "✓"}
                  </span>
                  <span className={current ? "font-medium" : ""}>{stageWords(s.name)}{broke && <span className="text-s-muted"> (stopped here)</span>}</span>
                  <span className="ml-auto tnum text-xs text-s-faint whitespace-nowrap">{clock(took)}</span>
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </Card>
  );
}

function Bars({ title, counts }: { title: string; counts: Record<string, [number, number]> }) {
  const cams = Object.keys(counts).sort((a, b) => Number(a) - Number(b));
  return (
    <div className="grid gap-2">
      <div className="text-xs text-s-faint">{title}</div>
      <div className="grid gap-x-6 gap-y-1.5 sm:grid-cols-2">
        {cams.map((c) => {
          const [i, n] = counts[c];
          const pct = n > 0 ? Math.round((100 * i) / n) : 0;
          return (
            <div key={c} className="flex items-center gap-3 text-xs">
              <span className="font-studio-mono w-14 shrink-0">cam{c.padStart(2, "0")}</span>
              <div className="h-2 flex-1 rounded-full bg-s-sunken overflow-hidden" role="progressbar"
                aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100} aria-label={`camera ${c}`}>
                <div className="h-full bg-s-accent rounded-full" style={{ width: `${pct}%` }} />
              </div>
              <span className="tnum w-20 text-right text-s-muted">{i} of {n}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** Typical error (px) against time, on a log scale: it falls fast, then flattens. */
function ErrorCurve({ history }: { history: [number, number][] }) {
  const pts = history.filter(([, r]) => r > 0);
  if (pts.length < 2) return null;
  const W = 640, H = 120, L = 46, B = 18, T = 6;
  const t0 = pts[0][0], t1 = pts[pts.length - 1][0];
  const lo = Math.log10(Math.min(...pts.map((q) => q[1])));
  const hi = Math.log10(Math.max(...pts.map((q) => q[1])));
  const span = Math.max(hi - lo, 0.3);
  const x = (t: number) => L + ((W - L - 4) * (t - t0)) / Math.max(t1 - t0, 1e-9);
  const y = (r: number) => T + ((H - T - B) * (hi + (span - (hi - lo)) / 2 - Math.log10(r))) / span;
  const path = pts.map(([t, r], k) => `${k ? "L" : "M"}${x(t).toFixed(1)},${y(r).toFixed(1)}`).join(" ");
  const ticks = [];
  for (let e = Math.ceil(lo - 0.001); e <= Math.floor(hi + 0.001); e++) ticks.push(e);
  return (
    <figure className="grid gap-1">
      <figcaption className="text-xs text-s-faint">Typical error while solving, in pixels (each line is ten times smaller)</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto max-h-40" role="img"
        aria-label={`Typical error fell from ${num(pts[0][1])} to ${num(pts[pts.length - 1][1])} pixels`}>
        {ticks.map((e) => (
          <g key={e}>
            <line x1={L} x2={W - 4} y1={y(10 ** e)} y2={y(10 ** e)} stroke="var(--s-line)" strokeWidth="1" />
            <text x={L - 6} y={y(10 ** e) + 3.5} textAnchor="end" fontSize="10" fill="var(--s-faint)">{10 ** e >= 1 ? 10 ** e : (10 ** e).toFixed(-e)}</text>
          </g>
        ))}
        <path d={path} fill="none" stroke="var(--s-accent)" strokeWidth="1.8" strokeLinejoin="round" />
        <text x={L} y={H - 4} fontSize="10" fill="var(--s-faint)">{clock(t0)}</text>
        <text x={W - 4} y={H - 4} fontSize="10" fill="var(--s-faint)" textAnchor="end">{clock(t1)}</text>
      </svg>
    </figure>
  );
}

function LogView({ lines, follow }: { lines: string[]; follow: boolean }) {
  const box = useRef<HTMLPreElement>(null);
  const pinned = useRef(true);
  useEffect(() => {
    const el = box.current;
    if (el && follow && pinned.current) el.scrollTop = el.scrollHeight;
  }, [lines, follow]);
  if (!lines.length) return <Note>Nothing printed yet.</Note>;
  return (
    <pre ref={box} tabIndex={0} aria-label="Calibration output"
      onScroll={(e) => { const el = e.currentTarget; pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24; }}
      className="font-studio-mono text-[11px] leading-relaxed bg-s-sunken border border-s-line rounded-lg p-3 overflow-auto max-h-96 whitespace-pre-wrap break-words">
      {lines.join("\n")}
    </pre>
  );
}

function CopyLine({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="flex gap-2 items-stretch min-w-0">
      <pre className="flex-1 min-w-0 font-studio-mono text-xs bg-s-sunken border border-s-line rounded-lg px-3 py-2.5 overflow-x-auto">{text}</pre>
      <Button onClick={() => { navigator.clipboard?.writeText(text).then(() => { setCopied(true); window.setTimeout(() => setCopied(false), 1500); }, () => undefined); }}>
        {copied ? "Copied" : "Copy"}
      </Button>
    </div>
  );
}
