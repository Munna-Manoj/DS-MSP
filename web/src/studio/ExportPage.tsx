// Export LDC meshes and Isaac Sim LUTs for cameras the user already has. The first screen asks
// only where the camera values come from; after that, four tracked steps. Studio runs
// `ds-msp export` (ADR-0035), so the job page shows the same command for a terminal.
import { useCallback, useEffect, useMemo, useState } from "react";

import { href } from "../router";
import { api, ApiError, type ExportCheckRow, type ExportSource, type ExportSpec, type Problem, type RunSummary, type SourceCamera } from "./api";
import { ldcVerdict } from "./ExportsPanel";
import { LensViewer, type CamTone } from "./LensViewer";
import { FolderPicker } from "./FolderPicker";
import { num } from "./format";
import { LENSES, lensFields } from "./setup/CameraStep";
import { choiceCard, Explain, Mono, Part, Range, Segmented, Step, Toggle, type StepState } from "./setup/kit";
import { SizeBoxes, SPACING } from "./setup/Settings";
import { Tracker, type TrackedStep } from "./setup/Tracker";
import { Badge, Button, Callout, ErrorBox, INPUT } from "./ui";

type Start = "file" | "result" | "typed";
type Id = "source" | "cameras" | "targets" | "start";
interface TypedCam { model: string; values: Record<string, string> }

const STARTS: { id: Start; title: string; text: string; icon: string }[] = [
  { id: "file", title: "A calibration file", icon: "M6 3h8l4 4v14H6zM14 3v4h4",
    text: "From DS-MSP (rig, intrinsics or stereo), MC-Calib, Kalibr, COLMAP or nerfstudio. Studio finds every camera in it." },
  { id: "result", title: "A Studio result", icon: "M4 5h16v14H4zM4 9h16M9 9v10",
    text: "A calibration listed under Results." },
  { id: "typed", title: "I'll type the values in", icon: "M4 17h16M6 13l4-8 4 8M7.5 10h5",
    text: "For lens values from a data sheet or another program: one card per camera." },
];
const START_TITLE: Record<Start, string> = { file: "A calibration file", result: "A Studio result", typed: "Typed lens values" };

const blankCam = (): TypedCam => ({ model: "kb", values: {} });
const typedPayload = (cams: TypedCam[]) => cams.map((c) => {
  const out: Record<string, unknown> = { model: c.model };
  for (const f of [...lensFields(c.model), "width", "height"]) out[f] = c.values[f] === undefined || c.values[f].trim() === "" ? "" : Number(c.values[f]);
  return out;
});
const typedComplete = (cams: TypedCam[]) => cams.length > 0 && cams.every((c) =>
  [...lensFields(c.model), "width", "height"].every((f) => (c.values[f] ?? "").trim() !== "" && Number.isFinite(Number(c.values[f]))));

export function ExportPage({ from }: { from?: string }) {
  const [start, setStart] = useState<Start | null>(from ? "result" : null);
  const [active, setActive] = useState<Id>("source");
  const [filePath, setFilePath] = useState("");
  const [resultId, setResultId] = useState<string>(from ?? "");
  const [results, setResults] = useState<RunSummary[] | null>(null);
  const [typed, setTyped] = useState<TypedCam[]>([blankCam()]);
  const [listed, setListed] = useState<{ format: string | null; file: string | null; cameras: SourceCamera[] } | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [chosen, setChosen] = useState<string[]>([]);
  const [touchedCams, setTouchedCams] = useState(false);
  const [ldcOn, setLdcOn] = useState(false);
  const [lutOn, setLutOn] = useState(false);
  const [factor, setFactor] = useState(4);
  const [balance, setBalance] = useState(0.5);
  const [outSize, setOutSize] = useState<number[]>([]);
  const [texSize, setTexSize] = useState<number[]>([]);
  const [preview, setPreview] = useState<ExportCheckRow[] | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [name, setName] = useState("");
  const [roots, setRoots] = useState<{ path: string }[]>([]);
  const [root, setRoot] = useState(0);
  const [picker, setPicker] = useState(false);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [problems, setProblems] = useState<Problem[]>([]);
  const [viewCam, setViewCam] = useState("");

  useEffect(() => {
    api.browse(null).then((d) => setRoots(d.dirs.map((x) => ({ path: x.path }))), () => undefined);
    api.runs().then((r) => setResults(r.filter((x) => !x.error)), () => setResults([]));
  }, []);

  const source: ExportSource | null = useMemo(() => {
    if (start === "file" && filePath) return { kind: "file", path: filePath };
    if (start === "result" && resultId) return { kind: "result", id: resultId };
    if (start === "typed" && typedComplete(typed)) return { kind: "typed", cameras: typedPayload(typed) };
    return null;
  }, [start, filePath, resultId, typed]);
  const sourceKey = JSON.stringify(source);

  // read the cameras of the source each time it changes
  useEffect(() => {
    setListed(null);
    setListError(null);
    if (!source) return;
    let live = true;
    api.exportCameras(source).then(
      (doc) => {
        if (!live) return;
        setListed(doc);
        const ok = doc.cameras.filter((c) => !c.problem).map((c) => c.camera);
        setChosen((old) => (touchedCams ? old.filter((c) => ok.includes(c)) : ok));
      },
      (e: Error) => { if (live) setListError(e.message); },
    );
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceKey]);

  useEffect(() => {
    if (start === "result" && resultId && results) {
      const r = results.find((x) => x.id === resultId);
      if (r && !name) setName(`${r.name}_exports`);
      if (r) setRoot(r.root);
    }
  }, [start, resultId, results, name]);

  const spec: ExportSpec | null = source && chosen.length && (ldcOn || lutOn) ? {
    source, cameras: chosen,
    ldc: ldcOn ? { output_size: outSize, downsample_factor: factor, balance } : null,
    lut: lutOn ? { texture_size: texSize } : null,
  } : null;
  const specKey = JSON.stringify(spec);

  // preview each mesh while the settings change
  useEffect(() => {
    setPreview(null);
    setPreviewError(null);
    if (!spec || !spec.ldc) return;
    let live = true;
    setChecking(true);
    const t = window.setTimeout(() => {
      api.exportCheck(spec).then(
        (d) => { if (live) setPreview(d.cameras); },
        (e: Error) => { if (live) setPreviewError(e.message); },
      ).finally(() => { if (live) setChecking(false); });
    }, 350);
    return () => { live = false; window.clearTimeout(t); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [specKey]);

  const readable = listed?.cameras.filter((c) => !c.problem) ?? [];
  const shownCam = chosen.includes(viewCam) ? viewCam : chosen[0] ?? "";
  const camTone = (name: string): CamTone => {
    const r = preview?.find((x) => x.camera === name);
    return r?.ldc ? ldcVerdict(r.ldc).tone : "plain";
  };
  const cutOff = (preview ?? []).filter((r) => r.ldc?.q3_overflow).map((r) => r.camera);
  const sourceState: StepState = listError ? "fix" : listed ? "done" : "todo";
  const camState: StepState = !listed ? "waiting" : chosen.length === 0 ? "todo" : touchedCams ? "done" : "usual";
  const targetState: StepState = !(ldcOn || lutOn) ? "todo" : cutOff.length ? "warn" : "done";
  const nameOk = name.trim() !== "";
  const steps: TrackedStep<Id>[] = [
    { id: "source", n: 1, title: start === "typed" ? "Lens values" : start === "result" ? "Result" : "File", state: sourceState,
      note: listError ? "Cannot be read" : listed ? `${listed.cameras.length} camera${listed.cameras.length === 1 ? "" : "s"} found` : start === "typed" ? "Fill in every card" : "Not chosen yet" },
    { id: "cameras", n: 2, title: "Cameras", state: camState, word: camState === "usual" ? "All of them" : undefined,
      note: !listed ? "After step 1" : `${chosen.length} of ${listed.cameras.length} chosen` },
    { id: "targets", n: 3, title: "What to write", state: targetState,
      note: !(ldcOn || lutOn) ? "Choose LDC, LUT or both" : cutOff.length ? `${cutOff.length} mesh${cutOff.length === 1 ? "" : "es"} cut off` : [ldcOn && "TI LDC", lutOn && "Isaac LUT"].filter(Boolean).join(" and ") },
    { id: "start", n: 4, title: "Name and start", state: nameOk && spec ? "todo" : "waiting", note: nameOk ? name : "Name it" },
  ];
  const required = 3;
  const doneCount = [sourceState === "done", chosen.length > 0 && !!listed, ldcOn || lutOn].filter(Boolean).length;
  const need: { step: Id; text: string }[] = [
    ...(!listed ? [{ step: "source" as Id, text: start === "typed" ? "Fill in every value on every card" : "Choose where the values come from" }] : []),
    ...(listed && chosen.length === 0 ? [{ step: "cameras" as Id, text: "Choose at least one camera" }] : []),
    ...(!(ldcOn || lutOn) ? [{ step: "targets" as Id, text: "Choose TI LDC, Isaac Sim LUT, or both" }] : []),
    ...(!nameOk ? [{ step: "start" as Id, text: "Give the export a name" }] : []),
  ];

  const goTo = useCallback((id: Id) => { setActive(id); window.scrollTo({ top: 0, behavior: "smooth" }); }, []);
  const nav = (id: Id) => {
    const k = steps.findIndex((s) => s.id === id);
    return (
      <div className="flex flex-wrap gap-2 justify-between">
        {k > 0 ? <Button onClick={() => goTo(steps[k - 1].id)}>Back</Button> : <span />}
        {k < steps.length - 1 && <Button kind="primary" onClick={() => goTo(steps[k + 1].id)}>Next: {steps[k + 1].title}</Button>}
      </div>
    );
  };

  function begin(s: Start) { setStart(s); setActive("source"); if (s === "file") setPicker(true); }
  function startExport() {
    if (!spec) return;
    setStarting(true); setStartError(null); setProblems([]);
    api.startExport(root, name.trim(), spec).then(
      (job) => { window.location.hash = href({ page: "job", id: job.id }); },
      (e: Error) => { setStartError(e.message); if (e instanceof ApiError) setProblems(e.problems); },
    ).finally(() => setStarting(false));
  }

  if (!start)
    return (
      <Page>
        <h1 className="text-2xl font-semibold tracking-tight">Export lens files for a device or a simulator</h1>
        <Explain>Write a TI LDC mesh (lens correction on a TI Jacinto board) or an Isaac Sim lens LUT for each camera. Where do the camera values come from?</Explain>
        <div className="grid gap-3 sm:grid-cols-3">
          {STARTS.map((o) => (
            <button key={o.id} type="button" onClick={() => begin(o.id)} className={`${choiceCard(false)} text-left p-5 grid gap-2 content-start`}>
              <svg viewBox="0 0 24 24" className="size-7 text-s-accent" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={o.icon} /></svg>
              <span className="text-[17px] font-semibold">{o.title}</span>
              <span className="text-[15px] text-s-muted leading-snug">{o.text}</span>
            </button>
          ))}
        </div>
        <Explain>To write these files while calibrating instead, turn them on in step 5 of a new calibration.</Explain>
      </Page>
    );

  const icon = STARTS.find((o) => o.id === start)!.icon;
  return (
    <Page>
      <div className="grid gap-6 lg:grid-cols-[320px_minmax(0,1fr)] items-start">
        <div className="lg:sticky lg:top-4">
          <Tracker<Id> label="Export progress" from={START_TITLE[start]} icon={icon} onChangeFrom={() => setStart(null)} prelude={null}
            steps={steps} active={active} onGo={goTo} done={doneCount} total={required}
            headline={need.length ? `${need.length} thing${need.length === 1 ? "" : "s"} left before you can start` : "Ready to start"} />
        </div>
        <div className="grid gap-5 min-w-0">

          <Step n={1} of={4} title={steps[0].title} state={sourceState} active={active === "source"} footer={nav("source")}
            lead={start === "typed" ? "One card per camera. Values in pixels, as your lens data gives them." : "Studio reads every camera in it and says what it found."}>
            {start === "file" && (
              <Part>
                <div className="text-base font-semibold">Calibration file on the computer running Studio</div>
                <div className="flex gap-2 flex-wrap">
                  <input className={`${INPUT} font-studio-mono !text-sm flex-1 min-w-[16rem]`} readOnly placeholder="No file chosen" value={filePath} />
                  <Button onClick={() => setPicker(true)}>{filePath ? "Choose another" : "Choose"}</Button>
                </div>
                <Explain>A COLMAP model can be given as its <Mono>cameras.txt</Mono>.</Explain>
              </Part>
            )}
            {start === "result" && (
              <Part>
                <div className="text-base font-semibold">Which result?</div>
                {results == null ? <span className="text-s-muted">Reading the results…</span> : results.length === 0 ? <Explain>No result is listed under Results yet.</Explain> : (
                  <div className="grid gap-2 sm:grid-cols-2">
                    {results.map((r) => (
                      <button key={r.id} type="button" onClick={() => setResultId(r.id)} className={`${choiceCard(resultId === r.id)} text-left px-4 py-3 grid gap-0.5`}>
                        <span className="font-semibold break-all">{r.name}</span>
                        <span className="text-[14px] text-s-muted">{r.n_cameras ?? "?"} cameras · {(r.models ?? []).join(", ")}</span>
                      </button>
                    ))}
                  </div>
                )}
              </Part>
            )}
            {start === "typed" && (
              <>
                <div className="grid gap-4 sm:grid-cols-2">
                  {typed.map((c, k) => (
                    <article key={k} className="rounded-xl border border-s-rule p-4 grid gap-3 transition hover:border-s-accent/45 focus-within:border-s-accent">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold">camera_{k}</span>
                        {typed.length > 1 && <button type="button" className="ml-auto text-[14px] text-s-accent hover:underline cursor-pointer" onClick={() => setTyped(typed.filter((_x, j) => j !== k))}>Remove</button>}
                      </div>
                      <label className="grid gap-1 text-[15px] font-medium">Lens type
                        <select className={INPUT} value={c.model} onChange={(e) => setTyped(typed.map((x, j) => (j === k ? { ...x, model: e.target.value } : x)))}>
                          {LENSES.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
                        </select>
                      </label>
                      <div className="grid grid-cols-2 gap-2">
                        {[...lensFields(c.model), "width", "height"].map((f) => {
                          const text = c.values[f] ?? "";
                          const bad = text.trim() === "" || !Number.isFinite(Number(text));
                          return (
                            <label key={f} className="grid gap-0.5 text-[14px] font-medium text-s-muted">
                              <span className="font-studio-mono">{f}{f === "width" || f === "height" ? " (px)" : ""}</span>
                              <input inputMode="decimal" aria-label={`${f} of camera_${k}`} value={text}
                                onChange={(e) => setTyped(typed.map((x, j) => (j === k ? { ...x, values: { ...x.values, [f]: e.target.value } } : x)))}
                                className={`w-full min-w-0 rounded-md border-[1.5px] bg-s-field px-2 min-h-10 text-[14px] font-studio-mono outline-none focus:border-s-accent ${bad ? "border-s-warn" : "border-s-field-line"}`} />
                            </label>
                          );
                        })}
                      </div>
                    </article>
                  ))}
                </div>
                <Button className="justify-self-start" onClick={() => setTyped([...typed, blankCam()])}>Add a camera</Button>
              </>
            )}
            {listError && <Callout tone="bad" title="Studio could not use this">{listError}</Callout>}
            {listed && (
              <Callout tone={readable.length === listed.cameras.length ? "ok" : "warn"}
                title={`${listed.cameras.length} camera${listed.cameras.length === 1 ? "" : "s"} found${listed.format && listed.format !== "typed" ? ` in a ${listed.format} file` : ""}`}>
                {readable.length < listed.cameras.length ? `${listed.cameras.length - readable.length} cannot be read; step 2 says why.` : "Every camera can be exported."}
              </Callout>
            )}
          </Step>

          <Step n={2} of={4} title="Cameras" state={camState} word={steps[1].word} active={active === "cameras"} footer={nav("cameras")}
            lead="Every camera that can be read is chosen. Leave out the ones you do not need.">
            {!listed ? <Explain>Step 1 comes first.</Explain> : (
              <>
                <div className="flex gap-2">
                  <Button onClick={() => { setTouchedCams(true); setChosen(readable.map((c) => c.camera)); }}>All</Button>
                  <Button onClick={() => { setTouchedCams(true); setChosen([]); }}>None</Button>
                </div>
                <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                  {listed.cameras.map((c) => {
                    const on = chosen.includes(c.camera);
                    return (
                      <label key={c.camera} className={`${choiceCard(on)} px-4 py-3 flex items-start gap-3 ${c.problem ? "opacity-70 cursor-not-allowed" : ""}`}>
                        <input type="checkbox" className="mt-1 size-4 accent-s-accent" checked={on} disabled={!!c.problem}
                          onChange={(e) => { setTouchedCams(true); setChosen(e.target.checked ? [...chosen, c.camera] : chosen.filter((x) => x !== c.camera)); }} />
                        <span className="grid gap-0.5 min-w-0">
                          <span className="font-semibold">{c.camera}</span>
                          {c.problem ? <span className="text-[14px] text-s-bad">{c.problem}</span>
                            : <span className="text-[14px] text-s-muted">{LENSES.find((l) => l.id === c.model)?.label ?? c.model} · {c.width} × {c.height} px</span>}
                        </span>
                      </label>
                    );
                  })}
                </div>
              </>
            )}
          </Step>

          <Step n={3} of={4} title="What to write" state={targetState} active={active === "targets"} footer={nav("targets")}
            lead="Choose one or both. Each camera gets its own files.">
            <Part>
              <Toggle id="ex-ldc" on={ldcOn} onChange={setLdcOn}><b className="font-semibold">TI LDC mesh</b> for lens correction on a TI Jacinto board</Toggle>
              {ldcOn && (
                <div className="grid gap-4">
                  <Explain>The board corrects each frame from a grid of stored shifts. Closer grid points are more accurate and take more memory. Keeping more of the view leaves more to correct; with a standard lens it can make the shifts too large for the file.</Explain>
                  <div className="grid gap-1 text-[15px] font-medium">Grid points every
                    <Segmented label="Grid point spacing" value={String(factor)} options={SPACING} onChange={(v) => setFactor(Number(v))} />
                  </div>
                  <div className="grid gap-1 text-[15px] font-medium">How much of the view to keep
                    <Range id="ex-balance" value={balance} min={0} max={1} step={0.05} format={(v) => v.toFixed(2)} ends={["0: all of it", "1: crop the edges"]} onChange={setBalance} />
                  </div>
                  <div className="grid gap-1 text-[15px] font-medium">Size of the corrected image (empty: each camera's own size)
                    <SizeBoxes id="ex-out" value={outSize} onChange={(v) => setOutSize(v === "" ? [] : v)} />
                  </div>
                  <MeshPreview rows={preview} checking={checking} error={previewError} selected={shownCam} onSelect={setViewCam} />
                </div>
              )}
            </Part>
            <Part>
              <Toggle id="ex-lut" on={lutOn} onChange={setLutOn}><b className="font-semibold">Isaac Sim lens LUT</b>: two EXR textures, a manifest and a USD camera</Toggle>
              {lutOn && (
                <div className="grid gap-1 text-[15px] font-medium">Texture size (empty: each camera's own size)
                  <SizeBoxes id="ex-tex" value={texSize} onChange={(v) => setTexSize(v === "" ? [] : v)} />
                </div>
              )}
            </Part>
            {spec && shownCam && (
              <Part>
                <LensViewer cameras={chosen.map((c) => ({ name: c, tone: camTone(c) }))} kinds={[...(ldcOn ? ["ldc" as const] : []), ...(lutOn ? ["lut" as const] : [])]}
                  camera={shownCam} onCamera={setViewCam} spec={spec} photos={start === "result"} />
              </Part>
            )}
          </Step>

          <Step n={4} of={4} title="Name and start" state={steps[3].state} active={active === "start"} footer={nav("start")}>
            <Part>
              <label htmlFor="ex-name" className="text-base font-semibold">Name this export</label>
              <input id="ex-name" className={INPUT} value={name} onChange={(e) => setName(e.target.value)} />
              {roots.length > 1 ? (
                <label className="grid gap-1.5 text-base font-semibold">Save it in
                  <select className={INPUT} value={root} onChange={(e) => setRoot(Number(e.target.value))}>
                    {roots.map((r, k) => <option key={r.path} value={k}>{r.path}</option>)}
                  </select>
                </label>
              ) : roots.length === 1 && <div className="text-[14px] text-s-muted break-words">Saved on the computer running Studio, in a new folder with this name, inside <span className="font-studio-mono break-all">{roots[0].path}</span>: <Mono>ldc</Mono>, <Mono>isaac_lut</Mono> and <Mono>exports.json</Mono>.</div>}
            </Part>
            {cutOff.length > 0 && (
              <Callout tone="warn" title="Some meshes would be cut off">
                {cutOff.join(", ")}: shifts too large for the file. You can still start, and those meshes are marked "Do not use". <button type="button" className="underline cursor-pointer" onClick={() => goTo("targets")}>Change the settings in step 3</button>.
              </Callout>
            )}
            {need.length > 0 && (
              <div className="grid gap-1.5" role="status">
                <div className="text-base font-semibold">Before you can start:</div>
                <ul className="grid gap-1">
                  {need.map((n) => (
                    <li key={n.text}><button type="button" className="text-left text-[15px] text-s-accent hover:underline cursor-pointer" onClick={() => goTo(n.step)}>Step {steps.find((s) => s.id === n.step)!.n}: {n.text}</button></li>
                  ))}
                </ul>
              </div>
            )}
            {startError && <Callout tone="bad" title="Not started">{startError}{problems.length > 0 && <ul className="mt-1 list-disc pl-5">{problems.map((p) => <li key={p.key + p.message}>{p.message}</li>)}</ul>}</Callout>}
            <Button kind="primary" className="justify-self-start !min-h-12 !px-6 !text-base" disabled={need.length > 0} busy={starting} onClick={startExport}>Start the export</Button>
            <Explain>The job page shows the same export as one <Mono>ds-msp export</Mono> command for a terminal.</Explain>
          </Step>
        </div>
      </div>

      {picker && (
        <FolderPicker mode="file" title="Choose a calibration file" onClose={() => setPicker(false)}
          onPick={(path) => { setFilePath(path); setPicker(false); }} />
      )}
    </Page>
  );
}

function MeshPreview({ rows, checking, error, selected, onSelect }: {
  rows: ExportCheckRow[] | null; checking: boolean; error: string | null; selected: string; onSelect: (c: string) => void;
}) {
  if (error) return <ErrorBox title="Could not measure the meshes" detail={error} />;
  if (!rows) return <div className="text-[15px] text-s-muted" role="status">{checking ? "Measuring each mesh…" : ""}</div>;
  return (
    <div className="grid gap-2" aria-busy={checking}>
      <div className="text-base font-semibold">What these settings give, measured</div>
      <Explain>How far the mesh sends a corrected pixel from where the lens values say it belongs, in pixels of the camera image (typical / worst), and how much of the corrected image shows the scene.</Explain>
      <div className="overflow-x-auto">
        <table className="w-full text-[15px]">
          <thead><tr className="text-left text-[14px] text-s-muted">
            <th className="pb-1 pr-3 font-medium">Camera</th><th className="pb-1 pr-3 font-medium">Can it be used?</th>
            <th className="pb-1 pr-3 font-medium">Error px</th><th className="pb-1 font-medium">Shows the scene</th>
          </tr></thead>
          <tbody>
            {rows.map((r) => {
              const v = r.ldc ? ldcVerdict(r.ldc) : null;
              return (
                <tr key={r.camera} onClick={() => onSelect(r.camera)} title="Show this camera below"
                  className={`border-t border-s-rule align-top cursor-pointer hover:bg-s-accent-soft/40 ${r.camera === selected ? "bg-s-accent-soft/60" : ""}`}>
                  <td className="py-2 pr-3 font-medium"><button type="button" className="underline-offset-2 hover:underline cursor-pointer" onClick={() => onSelect(r.camera)}>{r.camera}</button></td>
                  <td className="py-2 pr-3">{v && <><Badge tone={v.tone}>{v.word}</Badge>{v.text && <div className="text-[14px] text-s-muted mt-1 max-w-[40ch]">{v.text}</div>}</>}</td>
                  <td className="py-2 pr-3 tnum">{r.ldc?.median_px != null ? num(r.ldc.median_px, 3) : "—"} / {r.ldc?.max_px != null ? num(r.ldc.max_px, 2) : "—"}</td>
                  <td className="py-2 tnum">{r.ldc?.in_view_fraction != null ? `${Math.round(100 * r.ldc.in_view_fraction)} %` : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Page({ children }: { children: React.ReactNode }) {
  return <main className="mx-auto w-full max-w-7xl px-4 sm:px-6 py-6 grid gap-5">{children}</main>;
}
