// A result's fingerprint and the way to repeat it: what the run recorded about itself, a check
// that none of its inputs changed since, and a button that runs the same calibration again
// into a new folder. When the repeat finishes, its page says whether the files came out the same.
import { useEffect, useState, type ReactNode } from "react";
import { href } from "../router";
import { api, ApiError, type Fingerprint, type InputCheck } from "./api";
import { bytes, int, when } from "./format";
import { Button, Callout, Card, ErrorBox, Field, INPUT, Loading, Note } from "./ui";

export function RepeatPage({ id }: { id: string }) {
  const [fp, setFp] = useState<Fingerprint | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [check, setCheck] = useState<InputCheck | null>(null);
  const [checking, setChecking] = useState(false);
  const base = id.split(":").slice(1).join(":").split("/").pop() || "result";
  const [name, setName] = useState(`${base}_repeat`);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const root = Number(id.split(":", 1)[0]) || 0;

  useEffect(() => {
    setFp(null);
    setCheck(null);
    api.fingerprint(id).then(setFp, (e: Error) => setError(e.message));
  }, [id]);

  function runCheck() {
    setChecking(true);
    api.fingerprintCheck(id).then(
      (c) => { setCheck(c); setChecking(false); },
      (e: Error) => { setChecking(false); setError(e.message); },
    );
  }

  function start() {
    setStarting(true);
    setStartError(null);
    api.repeatJob(id, root, name).then(
      (job) => { window.location.hash = href({ page: "job", id: job.id }); },
      (e: Error) => { setStarting(false); setStartError(e instanceof ApiError ? e.message : String(e)); },
    );
  }

  return (
    <main className="mx-auto w-full max-w-4xl px-4 sm:px-6 py-6 grid gap-5">
      <div className="grid gap-1">
        <a href={href({ page: "run", id })} className="text-xs text-s-accent hover:underline">← Back to the result</a>
        <h1 className="text-2xl font-semibold">Repeat this calibration exactly</h1>
        <p className="text-s-muted text-sm max-w-prose">
          Every saved calibration records how it was made: its settings, a checksum of every input
          file, and the software that ran it. Studio can run the same calibration again and tell
          you whether the result is the same.
        </p>
      </div>

      {error && <ErrorBox title="Could not read this result's record" detail={error} />}
      {!fp && !error && <Loading what="the record" />}
      {fp && !fp.ok && <Callout tone="warn" title="No record to repeat from">{fp.reason}</Callout>}

      {fp?.ok && (
        <>
          <Card title="1. What this run recorded" aside={fp.run_id ? `run ${fp.run_id}` : undefined}>
            <dl className="grid gap-x-6 gap-y-2 sm:grid-cols-[auto_1fr] text-sm">
              <Row k="Made">{when(fp.created_utc)}</Row>
              <Row k="Settings file"><Mono>{fp.config_path ?? "not recorded"}</Mono></Row>
              {fp.overrides && fp.overrides.length > 0 && <Row k="Changed on the command line"><Mono>{fp.overrides.join(", ")}</Mono></Row>}
              <Row k="Input files">{int(fp.n_inputs)} files, {bytes(fp.input_bytes ?? 0)}, each with its checksum</Row>
              {fp.observations && <Row k="Detections used">{int(fp.observations.n_observations)} board views, {int(fp.observations.n_corners)} corners</Row>}
              <Row k="Software">
                DS-MSP {fp.software?.version ?? "version not recorded"}
                {fp.software?.commit && <> at <Mono>{fp.software.commit.slice(0, 10)}</Mono></>}
                {fp.software?.dirty && <span className="text-s-warn"> with uncommitted changes</span>}
                {fp.python && <>, Python {fp.python}</>}
              </Row>
              <Row k="Settings checksum"><Mono>{(fp.configuration_sha256 ?? "").slice(0, 16)}…</Mono></Row>
            </dl>
          </Card>

          <Card title="2. Has anything changed since?">
            <div className="grid gap-3">
              <Note>Studio reads every input file again and compares its checksum with the recorded one. A large capture takes a moment.</Note>
              <div><Button onClick={runCheck} busy={checking} kind={check ? "secondary" : "primary"}>{check ? "Check again" : "Check the input files"}</Button></div>
              {check && check.same && (
                <Callout tone="ok" title={`All ${int(check.checked)} input files are unchanged`}>A repeat will use exactly what the original used.</Callout>
              )}
              {check && !check.same && (
                <Callout tone="warn" title="The inputs are not what the original used">
                  <ul className="grid gap-1 mt-1 text-xs">
                    {check.n_changed > 0 && <li><b>{check.n_changed} changed:</b> <Files list={check.changed} n={check.n_changed} /></li>}
                    {check.n_missing > 0 && <li><b>{check.n_missing} missing:</b> <Files list={check.missing} n={check.n_missing} /></li>}
                    {check.n_appeared > 0 && <li><b>{check.n_appeared} new:</b> <Files list={check.appeared} n={check.n_appeared} /></li>}
                  </ul>
                  <p className="mt-2">A repeat would be a different calibration. Restore the files to repeat it exactly.</p>
                </Callout>
              )}
            </div>
          </Card>

          <Card title="3. Run it again">
            {!fp.can_repeat ? <Callout tone="info">{fp.reason}</Callout> : (
              <div className="grid gap-3 sm:max-w-md">
                <Field htmlFor="repeat-name" label="Name of the repeat" help="A new folder with this name is created next to the original. The original is not touched.">
                  <input id="repeat-name" className={INPUT} value={name} onChange={(e) => setName(e.target.value)} />
                </Field>
                {startError && <Callout tone="bad" title="Not started">{startError}</Callout>}
                <div className="flex flex-wrap items-center gap-3">
                  <Button kind="primary" onClick={start} busy={starting} disabled={!name.trim() || (check != null && !check.same)}>
                    Repeat exactly
                  </Button>
                  {!check && <span className="text-xs text-s-muted">You can start without the check; the repeat's page still compares the results.</span>}
                </div>
              </div>
            )}
          </Card>
        </>
      )}
    </main>
  );
}

function Row({ k, children }: { k: string; children: ReactNode }) {
  return <><dt className="text-s-faint">{k}</dt><dd className="min-w-0 break-words">{children}</dd></>;
}

function Mono({ children }: { children: ReactNode }) {
  return <code className="font-studio-mono text-[0.92em] break-all">{children}</code>;
}

function Files({ list, n }: { list: string[]; n: number }) {
  return (
    <span className="font-studio-mono break-all">
      {list.slice(0, 6).join(", ")}{n > 6 ? ` and ${n - 6} more` : ""}
    </span>
  );
}
