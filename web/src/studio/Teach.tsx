// Settings that show what they do. Each control here replaces a bare text box in "More
// settings" with a picture that moves the way the setting moves the calibration.
import { useEffect, useMemo, useState } from "react";

import { api, type ErrorSpread } from "./api";

/** Seconds since mount, about 30 times a second; fixed when the user asked for less motion. */
function usePhase(): number {
  const [t, setT] = useState(1.1);
  useEffect(() => {
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    let raf = 0;
    let last = 0;
    const t0 = performance.now();
    const tick = (now: number) => {
      if (now - last > 33) { last = now; setT((now - t0) / 1000); }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);
  return t;
}

// -- target shape ------------------------------------------------------------------------------

export type ShapeMode = "rigid" | "bend" | "free" | "off";

const SHAPES: { id: ShapeMode; title: string; text: string }[] = [
  { id: "rigid", title: "Stiff boards",
    text: "Each board keeps its printed shape. How the boards sit relative to one another is measured from all photos together." },
  { id: "bend", title: "Boards may be bowed",
    text: "As stiff boards, and each board may also bow or twist: 3 more numbers per board. For a board on paper, foam or thin plastic." },
  { id: "free", title: "Every corner free",
    text: "Each corner moves on its own. This can hide a lens or detection error inside the target, so set a limit (structure_prior_sigma)." },
  { id: "off", title: "Leave as first estimated",
    text: "The boards stay where the first estimate, averaged over single photos, put them." },
];

const US = [-1, -0.5, 0, 0.5, 1];
const VS = [-1, 0, 1];

/** Where corner (u, v) of board `b` is drawn at time `t`: the printed shape plus what `mode` lets move. */
function corner(mode: ShapeMode | "printed", b: number, u: number, v: number, t: number): [number, number] {
  const cx = b === 0 ? 44 : 124;
  let w = 0;
  let dx = 0;
  const i = (u + 1) * 2 + (v + 1) * 5 + b * 17;
  if (mode === "rigid" && b === 1) {
    w = 6 * Math.sin(t * 1.6) * u;
    dx = 2.5 * Math.sin(t * 1.1);
  } else if (mode === "bend") {
    const s = Math.sin(t * 1.6 + b * 1.3);
    w = 9 * s * (u * u - 1 / 3) + 3.5 * Math.sin(t * 0.9 + b) * u * v + (b === 1 ? 3 * Math.sin(t * 1.1) * u : 0);
  } else if (mode === "free") {
    w = 3.2 * Math.sin(t * 2.3 + 2.4 * i);
    dx = 1.8 * Math.sin(t * 1.9 + 1.1 * i);
  }
  return [cx + 25 * u + 8 * v + dx, 41 + 13 * v - w];
}

function BoardPicture({ mode, t }: { mode: ShapeMode; t: number }) {
  const fine = mode === "free" ? US : [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1];
  const line = (m: ShapeMode | "printed", b: number, pts: [number, number][]) =>
    pts.map(([u, v]) => corner(m, b, u, v, t).map((x) => x.toFixed(1)).join(",")).join(" ");
  const outline: [number, number][] = [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]];
  return (
    <svg viewBox="0 0 168 82" className="w-full h-auto text-s-text" role="img" aria-hidden="true">
      {[0, 1].map((b) => (
        <g key={b}>
          {/* the printed shape, for comparison */}
          <polyline points={line("printed", b, outline)} fill="none" stroke="currentColor" strokeOpacity="0.28" strokeDasharray="3 3" strokeWidth="1" />
          <polygon points={[...VS.map((v) => fine.map((u) => [u, v] as [number, number]))[0],
            ...fine.map((u) => [u, 1] as [number, number]).reverse()].map(([u, v]) => corner(mode, b, u, v, t).map((x) => x.toFixed(1)).join(",")).join(" ")}
            fill="currentColor" fillOpacity="0.06" />
          {VS.map((v) => (
            <polyline key={`r${v}`} points={line(mode, b, fine.map((u) => [u, v] as [number, number]))}
              fill="none" stroke="currentColor" strokeOpacity="0.75" strokeWidth="1.1" />
          ))}
          {US.map((u) => (
            <polyline key={`c${u}`} points={line(mode, b, VS.map((v) => [u, v] as [number, number]))}
              fill="none" stroke="currentColor" strokeOpacity="0.75" strokeWidth="1.1" />
          ))}
          {mode === "free" && US.flatMap((u) => VS.map((v) => {
            const [x, y] = corner(mode, b, u, v, t);
            return <circle key={`${u}${v}`} cx={x} cy={y} r="1.9" fill="#0ea5e9" />;
          }))}
        </g>
      ))}
    </svg>
  );
}

/** `structure_mode`: what may move in the target while the cameras are solved. */
export function TargetShape({ value, fallback, boards, onChange }: {
  value: string | undefined; fallback: string; boards: number; onChange: (v: string) => void;
}) {
  const t = usePhase();
  const current = (value ?? fallback) as ShapeMode;
  return (
    <div className="grid gap-3">
      <div role="radiogroup" aria-label="What may move in the target" className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {SHAPES.map((s) => {
          const on = current === s.id;
          return (
            <button key={s.id} type="button" role="radio" aria-checked={on} onClick={() => onChange(s.id)}
              className={`text-left rounded-lg border p-3 grid gap-2 content-start ${on ? "border-s-accent bg-s-accent/5 ring-1 ring-s-accent" : "border-s-line bg-s-surface hover:bg-s-sunken"}`}>
              <BoardPicture mode={s.id} t={t} />
              <span className="text-sm font-semibold flex flex-wrap items-baseline gap-x-2">
                {s.title}
                {s.id === fallback && <span className="text-xs font-normal text-s-faint">default</span>}
              </span>
              <span className="text-xs text-s-muted leading-relaxed">{s.text}</span>
            </button>
          );
        })}
      </div>
      <p className="text-xs text-s-muted leading-relaxed">
        The dashed outline is the board as printed; the solid one is what this choice lets the calibration move, enlarged.
        {boards <= 1
          ? <b className="text-s-text font-medium"> Your target is one board. These choices act only on a target of several boards fixed together, so nothing changes for this calibration.</b>
          : ` Your target is ${boards} boards fixed together, so this choice takes effect.`}
      </p>
    </div>
  );
}

// -- outlier band ------------------------------------------------------------------------------

/** The inlier band is this many times `noise_bound` (ds_msp.rig.pipeline.TLS_BAND_SIGMAS). */
const BAND_SIGMAS = 3.03;
const R_MIN = 0.01;
const R_MAX = 30;
const PLOT_R = 118;

/** Radius on the plot of an error of `px` pixels: logarithmic, 0.01 px at the centre. */
const plotR = (px: number) => PLOT_R * Math.log10(Math.min(Math.max(px, R_MIN), R_MAX) / R_MIN) / Math.log10(R_MAX / R_MIN);

/** An example capture: most corners near 0.2 px, a few far off. Fixed, so the picture is stable. */
function exampleErrors(): { dx: number[]; dy: number[] } {
  let seed = 7;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  const dx: number[] = [];
  const dy: number[] = [];
  for (let k = 0; k < 170; k++) {
    const far = k >= 160;
    const r = far ? 1.2 + 9 * rnd() ** 2 : 0.22 * Math.sqrt(-2 * Math.log(1 - rnd() * 0.999));
    const a = 2 * Math.PI * rnd();
    dx.push(r * Math.cos(a));
    dy.push(r * Math.sin(a));
  }
  return { dx, dy };
}

/** Corners further than `px` from the fit, from the counted grid (exact on it, interpolated between). */
function countBeyond(s: ErrorSpread, px: number): number {
  const g = s.grid_px;
  if (!g.length || px >= g[g.length - 1]) return 0;
  if (px <= g[0]) return s.above[0];
  let k = 0;
  while (k < g.length - 2 && g[k + 1] <= px) k++;
  const f = Math.log(px / g[k]) / Math.log(g[k + 1] / g[k]);
  return Math.round(s.above[k] + f * (s.above[k + 1] - s.above[k]));
}

const px = (v: number) => (v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2));

/** `noise_bound`: how far a corner may lie from the fit before it is ignored. */
export function NoiseBand({ value, fallback, refRun, onChange }: {
  value: number | undefined; fallback: number; refRun: { id: string; name: string } | null;
  onChange: (v: number | "") => void;
}) {
  const [spread, setSpread] = useState<ErrorSpread | null>(null);
  useEffect(() => {
    setSpread(null);
    if (!refRun) return;
    let live = true;
    api.runErrors(refRun.id).then((s) => { if (live && s.n > 0) setSpread(s); }, () => undefined);
    return () => { live = false; };
  }, [refRun?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const example = useMemo(exampleErrors, []);
  const nb = value ?? fallback;
  const off = !(nb > 0);
  const band = BAND_SIGMAS * nb;
  const pts = spread ?? example;
  const beyond = spread && !off ? countBeyond(spread, band) : null;
  const share = spread && beyond != null ? beyond / spread.n : null;
  const exampleOut = off ? 0 : example.dx.filter((x, k) => Math.hypot(x, example.dy[k]) > band).length;
  const C = 132;

  return (
    <div className="grid gap-4 md:grid-cols-[264px_minmax(0,1fr)] items-start">
      <svg viewBox="0 0 264 264" className="w-full max-w-[264px] h-auto justify-self-center rounded-lg bg-s-sunken text-s-text" role="img"
        aria-label={off ? "Every corner counts" : `Corners further than ${px(band)} pixels from the fit are ignored`}>
        {[0.1, 1, 10].map((r) => (
          <circle key={r} cx={C} cy={C} r={plotR(r)} fill="none" stroke="currentColor" strokeOpacity="0.16" strokeWidth="1" />
        ))}
        {!off && <circle cx={C} cy={C} r={plotR(band)} fill="#f59e0b" fillOpacity="0.10" stroke="#d97706" strokeWidth="2" />}
        {pts.dx.map((x, k) => {
          const y = pts.dy[k];
          const r = Math.hypot(x, y);
          const s = r > 0 ? plotR(r) / r : 0;
          const out = !off && r > band;
          return <circle key={k} cx={C + x * s} cy={C + y * s} r={out ? 3.2 : 2.1}
            fill={out ? "none" : "#0ea5e9"} fillOpacity={out ? 1 : 0.7} stroke={out ? "#dc2626" : "none"} strokeWidth="1.6" />;
        })}
        <path d={`M${C - 5} ${C}H${C + 5}M${C} ${C - 5}V${C + 5}`} stroke="currentColor" strokeWidth="1.2" />
        {[0.1, 1, 10].map((r) => (
          <text key={r} x={C} y={C - plotR(r) - 3} fontSize="10" textAnchor="middle" fill="currentColor" fillOpacity="0.7"
            stroke="var(--s-sunken)" strokeWidth="3" paintOrder="stroke">{r} px</text>
        ))}
        {!off && <text x={C} y={Math.min(256, C + plotR(band) + 13)} fontSize="11" fontWeight="600" textAnchor="middle" fill="#b45309">ignored beyond {px(band)} px</text>}
      </svg>

      <div className="grid gap-3 text-sm">
        <p className="leading-relaxed">
          {off
            ? <>Every corner counts. Corners far from where the calibration expects them get less weight but are never ignored.</>
            : <>A corner more than <b>{px(band)} px</b> from where the calibration expects it is treated as a mistake and <b>ignored</b>.</>}
          {" "}
          {spread && beyond != null && share != null && (
            <>In <b>{refRun?.name}</b> that is <b>{beyond === 0 ? "none" : `about ${beyond.toLocaleString()}`} of its {spread.n.toLocaleString()} corners</b>
              {beyond > 0 ? ` (${share < 0.001 ? "under 0.1" : (100 * share).toFixed(share < 0.1 ? 1 : 0)}%)` : ""}.</>
          )}
          {!spread && !off && <>In this example capture that is {exampleOut} of {example.dx.length} corners.</>}
        </p>
        <div className="grid gap-1.5">
          <label htmlFor="noise_bound" className="text-xs font-medium text-s-muted">
            Expected corner noise: <b className="text-s-text tnum">{off ? "off" : `${px(nb)} px`}</b>
            <span className="font-normal"> · the band is {BAND_SIGMAS} times this</span>
          </label>
          <input id="noise_bound" type="range" min={-1.3} max={0.7} step={0.01} disabled={off}
            value={off ? Math.log10(fallback) : Math.log10(nb)} className="w-full accent-s-accent"
            aria-valuetext={off ? "off" : `${px(nb)} pixels`}
            onChange={(e) => onChange(Number((10 ** Number(e.target.value)).toPrecision(3)))} />
          <div className="flex justify-between text-[11px] text-s-faint tnum"><span>0.05 px · strict</span><span>1 px</span><span>5 px · lenient</span></div>
        </div>
        <label className="flex items-start gap-2 text-sm cursor-pointer">
          <input type="checkbox" className="mt-1 accent-s-accent" checked={off} onChange={(e) => onChange(e.target.checked ? 0 : "")} />
          <span>Never ignore a corner<span className="block text-xs text-s-muted">Writes 0, which switches the band off.</span></span>
        </label>
        {share != null && share > 0.25 && (
          <p className="text-xs text-s-warn leading-relaxed" role="status">
            This is strict for that capture. A camera with fewer than 75% of its corners inside the band fails the coverage check.
          </p>
        )}
        <p className="text-xs text-s-muted leading-relaxed">
          {spread
            ? <>Each dot is one corner of {refRun?.name}: its distance from the centre is its error, on a scale that widens near the centre (rings at 0.1, 1 and 10 px). {spread.n > pts.dx.length ? `${pts.dx.length} corners are drawn, the worst always among them. ` : ""}The count is of that calibration; a different band changes the fit a little.</>
            : <>Each dot is one corner of an example capture: its distance from the centre is its error, on a scale that widens near the centre (rings at 0.1, 1 and 10 px). Once these photos have a result, the dots are your own corners.</>}
          {" "}This value is also the limit of the lens check unless you set <span className="font-studio-mono">max_lens_sigma_px</span>.
        </p>
      </div>
    </div>
  );
}

// -- checks ------------------------------------------------------------------------------------

const GATE: Record<string, [string, string]> = {
  off: ["Skip", "Not checked."],
  warn: ["Warn", "The result is written, with a warning when this check fails."],
  refuse: ["Refuse", "No result is written when this check fails."],
};

/** A check that can be skipped, warn, or refuse the result. */
export function GateChoice({ id, value, fallback, options, onChange }: {
  id: string; value: string | undefined; fallback: string; options: string[]; onChange: (v: string) => void;
}) {
  const current = value ?? fallback;
  return (
    <div className="grid gap-1.5">
      <div role="radiogroup" aria-labelledby={`${id}-label`} className="inline-grid grid-flow-col auto-cols-fr rounded-md border border-s-line overflow-hidden w-full max-w-xs">
        {options.map((o) => {
          const on = current === o;
          return (
            <button key={o} type="button" role="radio" aria-checked={on} onClick={() => onChange(o)}
              className={`min-h-10 px-3 text-sm border-l first:border-l-0 border-s-line ${on ? `font-semibold shadow-[inset_0_0_0_2px_currentColor] ${o === "refuse" ? "bg-s-bad-soft text-s-bad" : o === "warn" ? "bg-s-warn-soft text-s-warn" : "bg-s-sunken text-s-text"}` : "bg-s-surface text-s-muted hover:bg-s-sunken"}`}>
              {GATE[o]?.[0] ?? o}
            </button>
          );
        })}
      </div>
      <span className="text-xs text-s-muted">{GATE[current]?.[1]}{value === undefined ? " This is the default." : ""}</span>
    </div>
  );
}

/** A share between 0 and 1, set as a percentage. */
export function ShareSlider({ id, value, fallback, words, onChange }: {
  id: string; value: number | undefined; fallback: number; words: (pct: number) => string; onChange: (v: number) => void;
}) {
  const v = value ?? fallback;
  return (
    <div className="grid gap-1">
      <input id={id} type="range" min={0} max={1} step={0.01} value={v} className="w-full accent-s-accent"
        aria-valuetext={`${Math.round(v * 100)} percent`} onChange={(e) => onChange(Number(e.target.value))} />
      <span className="text-xs text-s-text">{words(Math.round(v * 100))}</span>
    </div>
  );
}
