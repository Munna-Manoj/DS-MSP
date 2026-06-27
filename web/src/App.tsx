import { useEffect, useState, type ReactNode } from "react";
import { Viewer3D } from "./components/Viewer3D";
import { FisheyeRender } from "./components/FisheyeRender";
import { FisheyeOverlay } from "./components/FisheyeImage";
import { Controls } from "./components/Controls";
import { ModelSelector } from "./components/ModelSelector";
import { CAMERAS, thetaMaxOf, type CameraModel, type Params, type Vec3 } from "./lib/cameras";
import { stagesFor } from "./lib/stages";
import type { Surface } from "./lib/raymap";

const SURFACES: { id: Surface; label: string; note: string }[] = [
  { id: "sphere", label: "Sphere", note: "honest for any FOV" },
  { id: "cylinder", label: "Cylinder", note: "panoramic" },
  { id: "plane", label: "Plane", note: "pinhole only" },
];

export default function App() {
  const [model, setModel] = useState<CameraModel>(CAMERAS[0]);
  const [params, setParams] = useState<Params>(CAMERAS[0].defaults);
  const [active, setActive] = useState<Vec3>([1.3, 0.5, 1.6]);
  const [surfaces, setSurfaces] = useState<Surface[]>(["sphere"]);
  const [stage, setStage] = useState<number>(-1); // -1 = show everything
  const [playing, setPlaying] = useState(false);

  const stages = stagesFor(model);
  const fovDeg = (thetaMaxOf(model, params) * 2 * 180) / Math.PI;

  const onParams = (patch: Params) => setParams((p) => ({ ...p, ...patch }));
  const selectModel = (m: CameraModel) => {
    setModel(m);
    setParams(m.defaults);
  };
  const toggleSurface = (s: Surface) =>
    setSurfaces((cur) => (cur.includes(s) ? cur.filter((x) => x !== s) : [...cur, s]));

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setStage((s) => (s >= stages.length - 1 ? 0 : s + 1)), 1500);
    return () => clearInterval(id);
  }, [playing, stages.length]);

  const stepTo = (i: number) => {
    setPlaying(false);
    setStage(i);
  };

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-line">
        <div className="mx-auto max-w-7xl px-6 h-16 flex items-center justify-between">
          <div className="flex items-baseline gap-3">
            <span className="font-display font-extrabold text-lg tracking-tight">DS-MSP</span>
            <span className="text-muted text-sm hidden sm:inline">
              multi-model camera studio · interactive
            </span>
          </div>
          <a href="https://github.com/Munna-Manoj/DS-MSP" target="_blank" rel="noreferrer"
            className="text-sm text-muted hover:text-text transition border border-line rounded-md px-3 py-1.5 hover:border-primary">
            GitHub ↗
          </a>
        </div>
      </header>

      <main className="mx-auto w-full max-w-7xl px-6 py-12 flex-1">
        <section className="max-w-3xl">
          <p className="font-mono text-xs uppercase tracking-[0.18em] text-primary mb-4">
            Eight camera models, one live pipeline
          </p>
          <h1 className="font-display font-extrabold text-4xl sm:text-5xl leading-[1.05] tracking-tight">
            Watch a 3D world<br />
            <span className="text-muted">become a pixel — through any lens.</span>
          </h1>
          <p className="mt-5 text-muted text-lg leading-relaxed max-w-2xl">
            Pick a camera model, drag the point through the scene, and step the projection one stage
            at a time. The image, the wrapped surfaces, and the 3D world are all driven by the same
            math the <code className="font-mono text-text">ds_msp</code> library ships — verified to
            the digit.
          </p>
        </section>

        {/* model rail */}
        <section className="mt-8">
          <h3 className="text-[11px] uppercase tracking-[0.14em] text-faint mb-3">Camera model</h3>
          <ModelSelector selectedId={model.id} onSelect={selectModel} />
        </section>

        {/* step-through strip */}
        <section className="mt-6">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-[11px] uppercase tracking-[0.14em] text-faint">
              Projection pipeline — step through it
            </h3>
            <div className="flex items-center gap-2">
              <button onClick={() => setPlaying((p) => !p)}
                className="font-mono text-[11px] rounded-md border border-line px-3 py-1.5 text-text hover:border-primary transition">
                {playing ? "❚❚ pause" : "▶ play"}
              </button>
              <button onClick={() => { setPlaying(false); setStage(-1); }}
                className={`font-mono text-[11px] rounded-md border px-3 py-1.5 transition ${stage < 0 ? "border-primary text-primary" : "border-line text-muted hover:border-primary"}`}>
                show all
              </button>
            </div>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-px rounded-lg overflow-hidden border border-line bg-line">
            {stages.map((s, i) => {
              const on = stage === i;
              const done = stage < 0 || stage >= i;
              return (
                <button key={i} onClick={() => stepTo(i)}
                  className={`text-left px-4 py-4 transition ${on ? "bg-primary/10" : "bg-ink-2 hover:bg-panel-2"}`}>
                  <div className={`font-mono text-xs tnum ${done ? "text-primary" : "text-faint"}`}>
                    0{i + 1}
                  </div>
                  <div className="mt-1 font-display font-bold text-sm">{s.title}</div>
                  <div className="mt-1 font-mono text-[11px] text-valid leading-snug">{s.eq}</div>
                  <div className="text-[11px] text-faint mt-1 leading-snug">{s.detail}</div>
                </button>
              );
            })}
          </div>
        </section>

        {/* demo */}
        <section className="mt-8 rounded-xl border border-line bg-ink-2 overflow-hidden">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-5 py-3">
            <h2 className="font-display font-bold">{model.name} · projection studio</h2>
            <div className="flex items-center gap-4">
              <span className="font-mono text-[11px] text-faint">FOV ≈ {fovDeg.toFixed(0)}°</span>
              <span className="font-mono text-[11px] text-faint">
                verified vs <span className="text-valid">ds_msp</span>
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-px bg-line">
            {/* 3D scene */}
            <div className="lg:col-span-5 bg-ink relative">
              <PanelLabel>3D world — drag the point · orbit to look around</PanelLabel>
              <div className="h-[340px] sm:h-[500px]">
                <Viewer3D model={model} params={params} active={active}
                  onActiveChange={setActive} surfaces={surfaces} stage={stage} />
              </div>
              {/* surface picker */}
              <div className="absolute bottom-3 left-3 flex gap-1.5">
                {SURFACES.map((s) => {
                  const on = surfaces.includes(s.id);
                  return (
                    <button key={s.id} onClick={() => toggleSurface(s.id)} title={s.note}
                      className={`font-mono text-[10px] rounded-md border px-2.5 py-1.5 backdrop-blur transition ${on ? "border-primary bg-primary/15 text-primary" : "border-line bg-ink/70 text-muted hover:text-text"}`}>
                      {s.label}
                    </button>
                  );
                })}
              </div>
              {/* stage caption */}
              {stage >= 0 && (
                <div className="absolute bottom-3 right-3 max-w-[60%] text-right">
                  <div className="font-mono text-[11px] text-primary">{stages[stage].eq}</div>
                  <div className="text-[10px] text-faint leading-snug">{stages[stage].detail}</div>
                </div>
              )}
            </div>

            {/* fisheye image */}
            <div className="lg:col-span-4 bg-ink-2 relative">
              <PanelLabel>Image this lens forms</PanelLabel>
              <div className="p-4 sm:p-5">
                <div className="relative w-full overflow-hidden rounded-lg border border-line bg-ink"
                  style={{ aspectRatio: `${params.cx} / ${params.cy}` }}>
                  <FisheyeRender model={model} params={params} />
                  <FisheyeOverlay model={model} params={params} active={active} />
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-4 text-[11px] text-muted">
                  <Legend color="#ffc56e" label="90° horizon" />
                  <Legend color="#ff5d6c" label="FOV edge" />
                  <Legend color="#6e8bff" label="picked ray" />
                </div>
              </div>
            </div>

            {/* controls */}
            <div className="lg:col-span-3 bg-panel">
              <PanelLabel>Instrument</PanelLabel>
              <div className="p-5">
                <Controls model={model} params={params} onParams={onParams} active={active} />
              </div>
            </div>
          </div>
        </section>

        <p className="mt-6 text-sm text-faint max-w-2xl">
          Every model's <code className="font-mono text-muted">project</code> /{" "}
          <code className="font-mono text-muted">unproject</code> is a direct port of{" "}
          <code className="font-mono text-muted">ds_msp/models/*_math.py</code>, cross-checked
          against the library to ~10⁻¹² px and ~10⁻⁸ rad.
        </p>
      </main>

      <footer className="border-t border-line">
        <div className="mx-auto max-w-7xl px-6 py-6 text-sm text-faint flex flex-wrap gap-x-6 gap-y-1">
          <span>DS-MSP — Double Sphere &amp; multi-model fisheye camera library.</span>
          <span>Usenko, Demmel &amp; Cremers, “The Double Sphere Camera Model”, 3DV 2018.</span>
        </div>
      </footer>
    </div>
  );
}

function PanelLabel({ children }: { children: ReactNode }) {
  return (
    <div className="absolute top-0 left-0 z-10 font-mono text-[10px] uppercase tracking-[0.1em] text-faint bg-ink/70 backdrop-blur px-3 py-1.5 rounded-br-md pointer-events-none">
      {children}
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: color }} />
      <span className="font-mono tnum">{label}</span>
    </span>
  );
}
