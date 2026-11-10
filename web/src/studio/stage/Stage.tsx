// The centre of a result page: one dark stage that answers three questions on the user's own
// capture. Photos: how well does the calibration land on what the camera saw? Rig in 3D: where
// are the cameras, and how sure is that? What it means: how exact is a measurement made with it?
// One camera selection and one timeline drive all three.
import { lazy, Suspense, useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { href } from "../../router";
import { api, type CameraDiagnostics, type FrameStat, type PairDoc, type RunDoc, type SceneDoc } from "../api";
import { camName, num } from "../format";
import { STATE_LABEL, type CamStatus } from "../status";
import { MeaningStage } from "./MeaningStage";
import { PhotoStage } from "./PhotoStage";
import { Timeline } from "./Timeline";
import { length, UNIT_CHOICES, useUnit, type Unit } from "./units";

import { readMarked, viewList, writeMarked } from "../marks";

const RigStage = lazy(() => import("./RigStage"));

type Mode = "photos" | "rig" | "meaning";
const MODES: { id: Mode; label: string; question: string }[] = [
  { id: "photos", label: "On your photos", question: "Does the calibration land on what the camera saw?" },
  { id: "rig", label: "The rig in 3D", question: "Where are the cameras, and how sure is that?" },
  { id: "meaning", label: "What it means", question: "How exact is a measurement made with this rig?" },
];
const DOT = { accept: "bg-emerald-400", check: "bg-violet-400", improve: "bg-amber-400", reject: "bg-red-400" } as const;

export function Stage({ doc, runId, cam, statuses }: {
  doc: RunDoc; runId: string; cam: string; statuses: CamStatus[];
}) {
  const [mode, setMode] = useState<Mode>("photos");
  const [frames, setFrames] = useState<FrameStat[]>([]);
  const [frame, setFrame] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [scene, setScene] = useState<SceneDoc | null>(null);
  const [diag, setDiag] = useState<CameraDiagnostics | null>(null);
  const [magnify, setMagnify] = useState<number | null>(null);
  const [showTrust, setShowTrust] = useState(false);
  const [to, setTo] = useState<number | null>(null);
  const [pair, setPair] = useState<PairDoc | null>(null);
  const [wobble, setWobble] = useState(0);
  const [distance, setDistance] = useState<number | null>(null);
  const [pixelNoise, setPixelNoise] = useState(0.5);
  const [unit, setUnit] = useUnit();
  const [marked, setMarked] = useState<string[]>(() => readMarked(runId));

  useEffect(() => { setMarked(readMarked(runId)); }, [runId]);
  const mark = useCallback((next: string[]) => { writeMarked(runId, next); setMarked(next); }, [runId]);

  const camNum = Number(cam);
  const intr = doc.report.intrinsics?.find((i) => i.cam_id === camNum);
  const size = intr ? { width: intr.width, height: intr.height } : null;
  const typical = doc.report.per_camera_errors_px?.[cam]?.median ?? null;
  const ext = doc.report.extrinsics ?? [];

  useEffect(() => { api.scene(runId).then(setScene, () => setScene(null)); }, [runId]);

  useEffect(() => {
    let live = true;
    setPlaying(false);
    api.frames(runId, cam).then((d) => {
      if (!live) return;
      setFrames(d.frames);
      setFrame((old) => (d.frames.some((f) => f.frame === old) ? old : d.frames[0]?.frame ?? null));
    }, () => live && setFrames([]));
    api.diagnostics(runId, cam).then((d) => live && setDiag(d), () => live && setDiag(null));
    return () => { live = false; };
  }, [runId, cam]);

  // measure to the nearest other camera unless the user chose one
  const nearest = useMemo(() => {
    const me = ext.find((e) => e.cam_id === camNum);
    if (!me) return null;
    let best: number | null = null, bestD = Infinity;
    for (const e of ext) {
      if (e.cam_id === camNum) continue;
      const d = Math.hypot(...e.center_in_ref.map((v, j) => v - me.center_in_ref[j]));
      if (d < bestD) { bestD = d; best = e.cam_id; }
    }
    return best;
  }, [ext, camNum]);
  const other = to != null && to !== camNum && ext.some((e) => e.cam_id === to) ? to : nearest;

  useEffect(() => {
    if (other == null) { setPair(null); return; }
    let live = true;
    api.pair(runId, camNum, other).then((p) => live && setPair(p.ok ? p : null), () => live && setPair(null));
    return () => { live = false; };
  }, [runId, camNum, other]);

  const stretch = magnify ?? (size && typical ? Math.max(1, Math.round(size.width / 110 / Math.max(typical, 1e-3))) : 50);
  const baseline = pair?.distance ?? null;
  const dist = distance ?? (baseline ? baseline * 5 : null);
  const current = frames.find((f) => f.frame === frame) ?? null;
  const viewKey = frame != null ? `${camNum}:${frame}` : null;
  const isMarked = viewKey != null && marked.includes(viewKey);
  const markedHere = useMemo(
    () => new Set(marked.filter((k) => k.startsWith(`${camNum}:`)).map((k) => Number(k.split(":")[1]))),
    [marked, camNum]);
  const already = doc.acceptance?.excluded_views ?? [];
  const zones = diag?.lens?.zones;

  return (
    <section className="rounded-xl overflow-hidden border border-slate-800 bg-[#0b1120] text-slate-100" aria-label="Explore this calibration">
      {/* which question */}
      <div className="flex flex-wrap items-stretch gap-x-1 border-b border-white/10 px-2 pt-2" role="tablist">
        {MODES.map((m) => (
          <button key={m.id} type="button" role="tab" aria-selected={mode === m.id} onClick={() => setMode(m.id)}
            className={`px-3.5 py-2.5 text-sm font-medium rounded-t-md border-b-2 ${mode === m.id ? "border-sky-400 text-white bg-white/5" : "border-transparent text-slate-400 hover:text-slate-200"}`}>
            {m.label}
          </button>
        ))}
        <p className="ml-auto self-center hidden md:block text-xs text-slate-400 pr-2">{MODES.find((m) => m.id === mode)?.question}</p>
      </div>

      {/* which camera */}
      <div className="flex gap-1.5 overflow-x-auto px-3 py-2 border-b border-white/10" role="group" aria-label="Camera">
        {statuses.map((s) => {
          const on = s.id === cam;
          return (
            <a key={s.id} href={href({ page: "run", id: runId, cam: s.id })} aria-current={on ? "true" : undefined}
              title={`${camName(s.id)}: ${STATE_LABEL[s.state]}${s.reasons.length ? ` (${s.reasons[0]})` : ""}`}
              className={`shrink-0 inline-flex items-center gap-1.5 rounded-full px-3 min-h-8 text-xs font-medium ${on ? "bg-sky-400 text-slate-900" : "bg-white/10 text-slate-200 hover:bg-white/20"}`}>
              <span className={`size-2 rounded-full ${DOT[s.state]}`} aria-hidden="true" />
              {camName(s.id)}
            </a>
          );
        })}
        {unit === "?" && mode !== "photos" && (
          <span className="ml-auto shrink-0 flex items-center gap-1.5 text-xs text-amber-200">
            Your square size was in
            {(["m", "cm", "mm"] as const).map((u) => (
              <button key={u} type="button" onClick={() => setUnit(u)}
                className="rounded px-2 min-h-7 bg-amber-300/20 hover:bg-amber-300/40 text-amber-100 font-medium">{u}</button>
            ))}
          </span>
        )}
      </div>

      <div className="grid lg:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="min-w-0 p-3">
          {mode === "photos" && (
            <PhotoStage runId={runId} cam={cam} frame={frame} size={size} magnify={stretch} showTrust={showTrust}
              diagnostics={diag} photos={scene?.photos ?? "none"} />
          )}
          {mode === "rig" && (
            <Suspense fallback={<div className="h-[360px] grid place-items-center text-sm text-slate-400">Loading the 3D view…</div>}>
              <RigStage doc={doc} scene={scene} frame={frame} selected={camNum} from={camNum} to={other}
                onPick={(c) => c !== camNum && setTo(c)} wobble={wobble} unit={unit} />
            </Suspense>
          )}
          {mode === "meaning" && (other != null && dist != null
            ? <MeaningStage runId={runId} a={cam} b={String(other)} distance={dist} pixelNoise={pixelNoise} unit={unit} />
            : <div className="h-64 grid place-items-center text-sm text-slate-300 p-6 text-center">This needs two cameras: it shows how exactly the pair can locate a point.</div>)}
        </div>

        {/* the answer, and the few controls that change the view */}
        <aside className="border-t lg:border-t-0 lg:border-l border-white/10 p-4 grid gap-4 content-start text-sm">
          {mode === "photos" && (
            <>
              <Answer>
                {current && typical != null
                  ? current.rms_px > 2 * typical && current.rms_px > 0.2
                    ? <>This photo fits <b>worse than most</b>: {num(current.rms_px)} px here, {num(typical)} px typical for {camName(cam)}.</>
                    : <>The calibration lands within <b>{num(current.rms_px)} px</b> of the corners found in this photo.</>
                  : <>Choose a photo on the timeline.</>}
              </Answer>
              <Control label={`Stretch the error ×${stretch}`}
                help="Each line runs from where a corner was found (white dot) to where the calibration expects it (ring), stretched so you can see it. Short lines in all directions are good. Long lines that point the same way mean something is off.">
                <input type="range" min={0} max={3} step={0.01} value={Math.log10(stretch)} aria-label="Error magnification"
                  onChange={(e) => setMagnify(Math.max(1, Math.round(10 ** Number(e.target.value))))} className="w-full accent-sky-400" />
              </Control>
              {viewKey && (
                <div className="grid gap-1.5">
                  <button type="button" aria-pressed={isMarked}
                    onClick={() => mark(isMarked ? marked.filter((k) => k !== viewKey) : [...marked, viewKey])}
                    className={`rounded-md px-3 min-h-10 text-sm font-medium border ${isMarked ? "bg-amber-400 text-slate-900 border-amber-300 hover:bg-amber-300" : "bg-white/10 text-slate-100 border-white/15 hover:bg-white/20"}`}>
                    {isMarked ? "Marked. Put this photo back" : "Leave this photo out next time"}
                  </button>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    For a photo that is blurred, has glare, or where the board was read wrongly. This result stays as it is; the photo is skipped when you run again.
                  </p>
                </div>
              )}
              <label className="flex items-start gap-2 cursor-pointer">
                <input type="checkbox" checked={showTrust} onChange={(e) => setShowTrust(e.target.checked)} className="mt-1 accent-sky-400" />
                <span>
                  Shade where this lens was never checked
                  <span className="block text-xs text-slate-400 mt-0.5 leading-relaxed">
                    {zones ? zones.answer : "Striped areas had no board in any photo: the lens is a guess there."}
                  </span>
                </span>
              </label>
              <Legend items={[["#34d399", "expected position, close"], ["#f87171", "expected position, far off"], ["#ffffff", "corner as found"]]} />
            </>
          )}
          {mode === "rig" && (
            <>
              <Answer>
                {pair && other != null ? (
                  <>
                    {camName(other)} is <b>{length(pair.distance, unit)}</b> from {camName(cam)}, turned <b>{deg(pair.angle_deg)}</b>.
                  </>
                ) : <>One camera: nothing to measure between.</>}
              </Answer>
              {pair && other != null && (
                <>
                  <PickCamera label="Measure to" value={other} onChange={setTo} cams={ext.map((e) => e.cam_id).filter((c) => c !== camNum)} />
                  <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs tnum">
                    <Row k="to the right" v={length(pair.position[0], unit)} />
                    <Row k="down" v={length(pair.position[1], unit)} />
                    <Row k="forward" v={length(pair.position[2], unit)} />
                    <Row k="yaw (left-right)" v={deg(pair.yaw_deg)} />
                    <Row k="pitch (up-down)" v={deg(pair.pitch_deg)} />
                    <Row k="roll (twist)" v={deg(pair.roll_deg)} />
                    <Row k="known to at least" v={pair.position_sigma != null ? `±${length(pair.position_sigma, unit)}, ±${deg(pair.rotation_sigma_deg, 3)}` : "not recorded"} />
                  </dl>
                  <p className="text-xs text-slate-400 leading-relaxed">Position and angles are of {camName(other)} seen from {camName(cam)}: x right, y down, z forward.</p>
                </>
              )}
              <Control label={wobble > 0 ? `Uncertainty shown ×${wobble}` : "Show the uncertainty"}
                help="Each camera sways by how unsure its position and angle are, enlarged by this factor so you can see it. The reference camera stands still: every other camera is measured from it.">
                <input type="range" min={0} max={4} step={0.02} value={wobble > 0 ? Math.log10(wobble) : 0} aria-label="Uncertainty magnification"
                  onChange={(e) => { const v = Number(e.target.value); setWobble(v <= 0.01 ? 0 : Math.round(10 ** v)); }} className="w-full accent-amber-400" />
              </Control>
              <UnitPicker unit={unit} onChange={setUnit} />
            </>
          )}
          {mode === "meaning" && other != null && (
            <>
              <Answer>Move the sliders to your own task.</Answer>
              <PickCamera label="Second camera" value={other} onChange={setTo} cams={ext.map((e) => e.cam_id).filter((c) => c !== camNum)} />
              {baseline != null && dist != null && (
                <Control label={`Distance to the point: ${length(dist, unit)}`} help="How far away the thing you measure is.">
                  <input type="range" min={Math.log10(baseline * 0.6)} max={Math.log10(baseline * 40)} step={0.01} value={Math.log10(dist)}
                    aria-label="Distance to the point" onChange={(e) => setDistance(10 ** Number(e.target.value))} className="w-full accent-sky-400" />
                </Control>
              )}
              <Control label={`Finding the point in the image: ±${num(pixelNoise)} px`}
                help="How precisely your own detector or matcher finds a point. A printed marker is about 0.1 px, a natural feature 0.5 to 1 px.">
                <input type="range" min={0.05} max={2} step={0.05} value={pixelNoise} aria-label="Pixel noise of your detector"
                  onChange={(e) => setPixelNoise(Number(e.target.value))} className="w-full accent-sky-400" />
              </Control>
              <UnitPicker unit={unit} onChange={setUnit} />
            </>
          )}
        </aside>
      </div>

      {mode !== "meaning" && (
        <div className="border-t border-white/10 px-3 py-3">
          <Timeline frames={frames} frame={frame} onFrame={setFrame} playing={playing} onPlaying={setPlaying} marked={markedHere} />
        </div>
      )}

      {already.length > 0 && (
        <div className="border-t border-white/10 px-4 py-2.5 text-sm text-slate-300">
          {already.length} view{already.length === 1 ? " was" : "s were"} left out of this calibration on request: {viewList(already)}.
        </div>
      )}
      {marked.length > 0 && (
        <div className="border-t border-amber-300/30 bg-amber-400/10 px-4 py-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-sm" role="status">
          <span className="text-amber-100 mr-auto">
            <b>{marked.length} photo{marked.length === 1 ? "" : "s"} marked</b> to leave out: {viewList(marked)}
          </span>
          <button type="button" onClick={() => mark([])} className="text-xs text-slate-300 underline min-h-10 px-1">Clear the marks</button>
          <a href={href({ page: "new", from: runId })}
            className="inline-flex items-center rounded-md px-3.5 min-h-10 text-sm font-semibold bg-amber-400 text-slate-900 hover:bg-amber-300">
            Run again without {marked.length === 1 ? "it" : "them"}
          </a>
        </div>
      )}
    </section>
  );
}

function deg(v: number | null | undefined, digits = 2): string {
  return v == null || !Number.isFinite(v) ? "—" : `${v.toFixed(digits)}°`;
}

function Answer({ children }: { children: ReactNode }) {
  return <p className="text-[15px] leading-snug text-slate-100 [&_b]:text-white [&_b]:font-semibold">{children}</p>;
}

function Control({ label, help, children }: { label: string; help: string; children: ReactNode }) {
  return (
    <div className="grid gap-1.5">
      <div className="text-xs font-medium text-slate-200">{label}</div>
      {children}
      <p className="text-xs text-slate-400 leading-relaxed">{help}</p>
    </div>
  );
}

function Legend({ items }: { items: [string, string][] }) {
  return (
    <ul className="grid gap-1 text-xs text-slate-300">
      {items.map(([c, t]) => (
        <li key={t} className="flex items-center gap-2">
          <span className="size-2.5 rounded-full border border-slate-900" style={{ background: c }} aria-hidden="true" />{t}
        </li>
      ))}
    </ul>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return <><dt className="text-slate-400">{k}</dt><dd className="text-slate-100 text-right">{v}</dd></>;
}

function PickCamera({ label, value, onChange, cams }: { label: string; value: number; onChange: (c: number) => void; cams: number[] }) {
  return (
    <label className="grid gap-1 text-xs font-medium text-slate-200">
      {label}
      <select value={value} onChange={(e) => onChange(Number(e.target.value))}
        className="rounded-md bg-slate-800 border border-white/15 px-2 min-h-9 text-sm text-slate-100">
        {cams.map((c) => <option key={c} value={c}>{camName(c)}</option>)}
      </select>
    </label>
  );
}

function UnitPicker({ unit, onChange }: { unit: Unit; onChange: (u: Unit) => void }) {
  return (
    <label className="grid gap-1 text-xs font-medium text-slate-200">
      My square size was in
      <select value={unit} onChange={(e) => onChange(e.target.value as Unit)}
        className="rounded-md bg-slate-800 border border-white/15 px-2 min-h-9 text-sm text-slate-100">
        {UNIT_CHOICES.map((u) => <option key={u.id} value={u.id}>{u.label}</option>)}
      </select>
      <span className="font-normal text-slate-400 leading-relaxed">Lengths in a result are in that unit. Studio shows them in millimetres once it knows.</span>
    </label>
  );
}
