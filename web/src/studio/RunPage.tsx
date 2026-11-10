import { useEffect, useMemo, useState } from "react";
import { href } from "../router";
import { api, type RunDoc } from "./api";
import { CameraDetail } from "./CameraDetail";
import { int, px, when } from "./format";
import { LinksTable } from "./LinksTable";
import { cameraStatus, camIds, STATE_TONE, type CamStatus } from "./status";
import { Actions } from "./Actions";
import { Evidence } from "./Evidence";
import { BoardFlatness } from "./BoardFlatness";
import { TechDetails } from "./TechDetails";
import { Stage } from "./stage/Stage";
import { Card, ErrorBox, Loading, Note } from "./ui";

export function RunPage({ id, cam }: { id: string; cam?: string }) {
  const [doc, setDoc] = useState<RunDoc | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDoc(null);
    setError(null);
    api.run(id).then(setDoc, (e: Error) => setError(e.message));
  }, [id]);

  const statuses = useMemo(() => (doc ? camIds(doc).map((c) => cameraStatus(doc, c)) : []), [doc]);

  if (error)
    return (
      <main className="mx-auto w-full max-w-7xl px-4 sm:px-6 py-8">
        <ErrorBox title="Could not open this result" detail={error} />
      </main>
    );
  if (!doc)
    return (
      <main className="mx-auto w-full max-w-7xl px-4 sm:px-6 py-8">
        <Loading what="result" />
      </main>
    );

  const firstAttention = statuses.find((s) => s.state !== "accept")?.id;
  const selected = cam ?? firstAttention ?? statuses[0]?.id ?? "0";
  return (
    <main className="mx-auto w-full max-w-7xl px-4 sm:px-6 py-6 grid gap-5">
      <TitleRow doc={doc} runId={id} />
      <Banner doc={doc} statuses={statuses} />
      <Actions doc={doc} runId={id} />
      <Stage doc={doc} runId={id} cam={selected} statuses={statuses} />
      <CameraDetail doc={doc} cam={selected} status={statuses.find((s) => s.id === selected)} />
      <details className="rounded-lg border border-s-line bg-s-surface group">
        <summary className="cursor-pointer px-4 py-3.5 flex flex-wrap items-baseline gap-3">
          <h2 className="text-[15px] font-semibold">All the numbers</h2>
          <span className="text-xs text-s-muted">Error charts, camera pairs, board flatness, solver stages and files</span>
        </summary>
        <div className="px-4 pb-4 grid gap-5">
          <Evidence runId={id} cam={selected} available={!!doc.has_diagnostics} />
          <BoardFlatness doc={doc} />
          <LinksTable doc={doc} />
          <SetAside doc={doc} />
          <TechDetails doc={doc} />
        </div>
      </details>
    </main>
  );
}

function TitleRow({ doc, runId }: { doc: RunDoc; runId: string }) {
  const cap = doc.report.capture;
  const models = [...new Set(Object.values(doc.report.models ?? {}))];
  return (
    <div className="flex flex-wrap items-end gap-x-6 gap-y-2">
      <div className="grid gap-1 min-w-0">
        <a href={href({ page: "runs" })} className="text-xs text-s-accent hover:underline">← All results</a>
        <h1 className="text-2xl font-semibold break-all">{doc.summary.path || doc.name}</h1>
      </div>
      <dl className="flex flex-wrap gap-x-6 gap-y-1 text-sm text-s-muted ml-auto">
        <div><dt className="inline text-s-faint">Created </dt><dd className="inline">{when(doc.summary.created_utc)}</dd></div>
        <div><dt className="inline text-s-faint">Photos </dt><dd className="inline tnum">{int(cap?.n_frames)} frames</dd></div>
        <div><dt className="inline text-s-faint">Lens </dt><dd className="inline">{models.join(", ") || "—"}</dd></div>
      </dl>
      <div className="flex flex-wrap gap-2">
        {doc.manifest && (
          <a href={href({ page: "repeat", id: runId })}
            className="inline-flex items-center rounded-md px-3.5 min-h-10 text-sm font-medium border border-s-line bg-s-surface hover:bg-s-sunken">
            Repeat exactly
          </a>
        )}
        {(doc.has_config || doc.manifest) && (
          <a href={href({ page: "new", from: runId })}
            className="inline-flex items-center rounded-md px-3.5 min-h-10 text-sm font-medium border border-s-line bg-s-surface hover:bg-s-sunken">
            Change settings and run again
          </a>
        )}
      </div>
    </div>
  );
}

function Banner({ doc, statuses }: { doc: RunDoc; statuses: CamStatus[] }) {
  const v = doc.verdict;
  const level = v?.level ?? (statuses.some((s) => s.state === "reject") ? "reject"
    : statuses.some((s) => s.state === "improve") ? "improve" : statuses.some((s) => s.state === "check") ? "check" : "accept");
  const tone = STATE_TONE[level];
  const ready = statuses.filter((s) => s.state === "accept").length;
  const fit = doc.report.fit;
  const box = { ok: "border-s-ok/30 bg-s-ok-soft", info: "border-s-accent/30 bg-s-accent-soft",
    warn: "border-s-warn/40 bg-s-warn-soft", bad: "border-s-bad/40 bg-s-bad-soft" }[tone];
  const ink = { ok: "text-s-ok", info: "text-s-accent", warn: "text-s-warn", bad: "text-s-bad" }[tone];
  return (
    <section className={`rounded-lg border p-4 grid gap-2 ${box}`} aria-live="polite">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <h2 className={`text-lg font-semibold ${ink}`}>{v?.headline ?? "Calibration result"}</h2>
        <span className="text-sm text-s-muted">
          {ready} of {statuses.length} cameras ready · typical error {px(doc.report.overall_errors_px?.median)}, 95% under{" "}
          {px(doc.report.overall_errors_px?.p95)}
        </span>
      </div>
      {(v?.rig_reasons ?? []).map((r) => (
        <p key={r.reason} className="text-sm">{r.reason}.</p>
      ))}
      <p className="text-sm text-s-muted">
        <span className="font-medium text-s-text">Not validated with a second capture.</span>{" "}
        These numbers come from the photos the calibration used; uncertainties are from photo noise
        only, so the real error is at least as large. {fit?.basis ? `Fit judged on: ${fit.basis}.` : ""}
      </p>
    </section>
  );
}

function SetAside({ doc }: { doc: RunDoc }) {
  const rej = doc.acceptance?.rejected_observations ?? [];
  const nObs = doc.report.capture?.n_observations;
  return (
    <Card title="Photos set aside" aside={nObs ? `${rej.length} of ${int(nObs)} board views` : undefined}>
      {rej.length === 0 ? (
        <Note>None. Every board view was used.</Note>
      ) : (
        <div className="grid gap-2">
          <Note>The solver left these out because they disagreed with the rest.</Note>
          <ul className="grid gap-1 text-sm font-studio-mono">
            {rej.slice(0, 50).map((r, i) => (
              <li key={i} className="break-all">{Object.entries(r).map(([k, v]) => `${k} ${String(v)}`).join(" · ")}</li>
            ))}
          </ul>
          {rej.length > 50 && <Note>And {rej.length - 50} more, listed in calibration_acceptance.json.</Note>}
        </div>
      )}
    </Card>
  );
}
