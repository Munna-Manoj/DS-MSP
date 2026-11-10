import { useEffect, useState } from "react";
import { href } from "../router";
import { api, type JobBrief, type RunSummary } from "./api";
import { int, num, when } from "./format";
import { Badge, ErrorBox, Loading } from "./ui";

const JOB_LABEL: Record<JobBrief["state"], { label: string; tone: "info" | "plain" | "bad" | "warn" | "ok" }> = {
  queued: { label: "Waiting", tone: "plain" },
  running: { label: "Running", tone: "info" },
  done: { label: "Finished", tone: "ok" },
  failed: { label: "Failed", tone: "bad" },
  cancelled: { label: "Cancelled", tone: "plain" },
  interrupted: { label: "Interrupted", tone: "warn" },
};

export function RunsPage() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [jobs, setJobs] = useState<JobBrief[]>([]);
  const [canRun, setCanRun] = useState(false);

  useEffect(() => {
    api.runs().then(setRuns, (e: Error) => setError(e.message));
    api.health().then((h) => setCanRun(h.can_run === true), () => undefined);
  }, []);

  // Calibrations started here that are not a result yet: waiting, running, or ended without one.
  useEffect(() => {
    let live = true;
    let timer: number | undefined;
    let wasActive = false;
    const tick = () => {
      api.jobs().then((list) => {
        if (!live) return;
        setJobs(list);
        const active = list.some((j) => j.state === "queued" || j.state === "running");
        if (wasActive && !active) api.runs().then(setRuns, () => undefined);   // one just ended
        wasActive = active;
        if (active) timer = window.setTimeout(tick, 2500);
      }, () => undefined);
    };
    tick();
    return () => { live = false; window.clearTimeout(timer); };
  }, []);

  const pending = jobs.filter((j) => !j.has_result);

  const shown = (runs ?? []).filter((r) =>
    `${r.path} ${r.name} ${(r.models ?? []).join(" ")}`.toLowerCase().includes(query.toLowerCase()),
  );

  return (
    <main className="mx-auto w-full max-w-7xl px-4 sm:px-6 py-8 grid gap-6">
      <div className="flex flex-wrap items-end gap-4">
        <div className="grid gap-1">
          <h1 className="text-2xl font-semibold">Calibration results</h1>
          <p className="text-s-muted text-sm">
            Result folders found under the folders this Studio was started with.
          </p>
        </div>
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Filter by folder or lens model"
          aria-label="Filter results"
          className="sm:ml-auto w-full sm:w-72 rounded-md border border-s-line bg-s-surface px-3 py-2 text-sm outline-none focus:border-s-accent"
        />
        {canRun && (
          <a href={href({ page: "new" })}
            className="inline-flex items-center rounded-md px-3.5 min-h-10 text-sm font-medium whitespace-nowrap bg-s-accent text-white hover:opacity-90">
            New calibration
          </a>
        )}
      </div>

      {pending.length > 0 && (
        <section className="rounded-lg border border-s-line bg-s-surface" aria-label="Calibrations without a result yet">
          <h2 className="px-4 pt-3 text-xs font-medium uppercase tracking-wide text-s-faint">Started here, no result yet</h2>
          <ul className="p-2 grid">
            {pending.map((j) => (
              <li key={j.id}>
                <a href={href({ page: "job", id: j.id })} className="flex flex-wrap items-center gap-3 rounded-md px-2 py-2 hover:bg-s-sunken">
                  <span className="font-medium text-s-accent break-all">{j.name}</span>
                  <Badge tone={JOB_LABEL[j.state].tone}>{JOB_LABEL[j.state].label}</Badge>
                  {j.state === "failed" && j.error && <span className="text-xs text-s-muted break-words min-w-0 line-clamp-2 basis-full">{j.error}</span>}
                  <span className="ml-auto text-xs text-s-faint whitespace-nowrap">{when(j.started_utc ?? j.created_utc)}</span>
                </a>
              </li>
            ))}
          </ul>
        </section>
      )}

      {error && <ErrorBox title="Could not list results" detail={error} />}
      {!runs && !error && <Loading what="results" />}
      {runs && runs.length === 0 && (
        <div className="rounded-lg border border-s-line bg-s-surface p-6 text-s-muted">
          No results here yet.{" "}
          {canRun
            ? <>Start with <a className="text-s-accent underline" href={href({ page: "new" })}>a new calibration</a>, or start Studio with <code className="font-studio-mono">--root</code> set to a folder that holds results.</>
            : <>A result folder holds <code className="font-studio-mono">calibration_report.json</code>; start Studio with <code className="font-studio-mono">--root</code> set to a folder that contains one.</>}
        </div>
      )}

      {shown.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-s-line bg-s-surface">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-s-faint text-xs uppercase tracking-wide border-b border-s-line">
                <th className="px-4 py-2.5 font-medium">Result</th>
                <th className="px-4 py-2.5 font-medium">Created</th>
                <th className="px-4 py-2.5 font-medium text-right">Cameras</th>
                <th className="px-4 py-2.5 font-medium">Lens models</th>
                <th className="px-4 py-2.5 font-medium text-right">Typical error</th>
                <th className="px-4 py-2.5 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((r) => {
                const needs = (r.lens_failed ?? []).length;
                return (
                  <tr key={r.id} className="border-b border-s-line last:border-0 hover:bg-s-sunken">
                    <td className="px-4 py-3">
                      <a href={href({ page: "run", id: r.id })} className="font-medium text-s-accent hover:underline">
                        {r.path || r.name}
                      </a>
                    </td>
                    <td className="px-4 py-3 text-s-muted whitespace-nowrap">{when(r.created_utc)}</td>
                    <td className="px-4 py-3 text-right tnum">{int(r.n_cameras)}</td>
                    <td className="px-4 py-3 text-s-muted">{(r.models ?? []).join(", ") || "—"}</td>
                    <td className="px-4 py-3 text-right tnum whitespace-nowrap">
                      {r.median_px == null ? "—" : `${num(r.median_px)} px`}
                    </td>
                    <td className="px-4 py-3">
                      {r.error ? (
                        <Badge tone="bad">Unreadable</Badge>
                      ) : needs > 0 ? (
                        <Badge tone="warn">{needs} camera{needs === 1 ? "" : "s"} need photos</Badge>
                      ) : r.solver_status && r.solver_status !== "CONVERGED" ? (
                        <Badge tone="warn">Solver: {r.solver_status.toLowerCase().replace(/_/g, " ")}</Badge>
                      ) : (
                        <Badge tone="ok">Calibrated</Badge>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}
