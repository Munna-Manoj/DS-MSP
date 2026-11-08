import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { href } from "../router";
import { api, type RunDoc } from "./api";
import { CameraDetail } from "./CameraDetail";
import { camName, int, num, px, when } from "./format";
import { LinksTable } from "./LinksTable";
import { cameraStatus, camIds, findings, STATE_LABEL, type CamStatus } from "./status";
import { TechDetails } from "./TechDetails";
import { Badge, Card, ErrorBox, Loading, Note } from "./ui";

const RigView = lazy(() => import("./RigView"));

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

  const firstAttention = statuses.find((s) => s.state !== "ready")?.id;
  const selected = cam ?? firstAttention ?? statuses[0]?.id ?? "0";
  return (
    <main className="mx-auto w-full max-w-7xl px-4 sm:px-6 py-6 grid gap-5">
      <TitleRow doc={doc} />
      <Banner doc={doc} statuses={statuses} />
      <div className="grid gap-5 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <Card title="Cameras" aside={`${statuses.length} in the rig`}>
          <CameraTiles runId={id} statuses={statuses} doc={doc} selected={selected} />
        </Card>
        <Card title="Rig" aside="drag to turn, scroll to zoom" className="overflow-hidden">
          <Suspense fallback={<div className="h-[340px] grid place-items-center text-s-muted text-sm">Loading 3D view…</div>}>
            <RigView doc={doc} selected={Number(selected)} />
          </Suspense>
        </Card>
      </div>
      <CameraDetail doc={doc} cam={selected} status={statuses.find((s) => s.id === selected)} />
      <LinksTable doc={doc} />
      <SetAside doc={doc} />
      <TechDetails doc={doc} />
    </main>
  );
}

function TitleRow({ doc }: { doc: RunDoc }) {
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
    </div>
  );
}

function Banner({ doc, statuses }: { doc: RunDoc; statuses: CamStatus[] }) {
  const attention = statuses.filter((s) => s.state === "attention");
  const check = statuses.filter((s) => s.state === "check");
  const ready = statuses.length - attention.length - check.length;
  const solver = doc.acceptance?.solver?.final_status;
  const solverBad = solver != null && solver !== "CONVERGED";
  const fit = doc.report.fit;
  const nWeak = findings(doc).length;
  const tone = attention.length || solverBad ? "warn" : check.length ? "info" : "ok";
  const headline = solverBad
    ? `The solver stopped with ${solver.toLowerCase().replace(/_/g, " ")}`
    : attention.length
      ? `${attention.length} camera${attention.length === 1 ? " needs" : "s need"} attention`
      : check.length
        ? `Calibrated; ${check.length} camera${check.length === 1 ? "" : "s"} to check`
        : "Your rig is calibrated";
  const lines: string[] = [];
  if (attention.length)
    lines.push(`${attention.map((s) => camName(s.id)).join(", ")}: ${attention[0].reasons[0] ?? ""}${attention.length > 1 ? ", and others" : ""}.`);
  if (nWeak)
    lines.push(`${nWeak} part${nWeak === 1 ? " of the rig is" : "s of the rig are"} weakly linked; see the cameras marked Check.`);
  return (
    <section
      className={`rounded-lg border p-4 grid gap-2 ${tone === "warn" ? "border-s-warn/40 bg-s-warn-soft" : tone === "info" ? "border-s-accent/30 bg-s-accent-soft" : "border-s-ok/30 bg-s-ok-soft"}`}
      aria-live="polite"
    >
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <h2 className={`text-lg font-semibold ${tone === "warn" ? "text-s-warn" : tone === "info" ? "text-s-accent" : "text-s-ok"}`}>{headline}</h2>
        <span className="text-sm text-s-muted">
          {ready} of {statuses.length} cameras ready · typical error {px(doc.report.overall_errors_px?.median)}, 95% under{" "}
          {px(doc.report.overall_errors_px?.p95)}
        </span>
      </div>
      {lines.map((l) => (
        <p key={l} className="text-sm">{l}</p>
      ))}
      <p className="text-sm text-s-muted">
        <span className="font-medium text-s-text">Not validated with a second capture.</span>{" "}
        These numbers come from the photos the calibration used. To validate, take a second round of
        board photos and check the calibration against it. {fit?.basis ? `Fit judged on: ${fit.basis}.` : ""}
      </p>
    </section>
  );
}

function CameraTiles({ runId, statuses, doc, selected }: {
  runId: string; statuses: CamStatus[]; doc: RunDoc; selected: string;
}) {
  const tone = { ready: "ok", attention: "warn", check: "info" } as const;
  return (
    <ul className="grid gap-2 grid-cols-[repeat(auto-fill,minmax(8.5rem,1fr))]">
      {statuses.map((s) => {
        const err = doc.report.per_camera_errors_px?.[s.id];
        const model = doc.report.models?.[s.id];
        const active = s.id === selected;
        return (
          <li key={s.id}>
            <a
              href={href({ page: "run", id: runId, cam: s.id })}
              aria-current={active ? "true" : undefined}
              className={`block rounded-md border px-3 py-2 transition ${
                active ? "border-s-accent ring-1 ring-s-accent bg-s-accent-soft/40" : "border-s-line hover:border-s-accent"
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="font-semibold text-sm">{camName(s.id)}</span>
                {String(doc.report.capture?.ref_cam_id) === s.id && <span className="text-[10px] text-s-faint">ref</span>}
              </div>
              <div className="text-xs text-s-muted">{model ?? "—"} · {num(err?.median)} px</div>
              <div className="mt-1.5"><Badge tone={tone[s.state]}>{STATE_LABEL[s.state]}</Badge></div>
            </a>
          </li>
        );
      })}
    </ul>
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
