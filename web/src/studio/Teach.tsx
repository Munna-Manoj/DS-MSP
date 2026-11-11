// Settings that show what they do. Each control here replaces a bare text box in "More
// settings" with a picture that moves the way the setting moves the calibration.
import { useEffect, useMemo, useState } from "react";

import { api, type ErrorSpread } from "./api";
import { choiceCard } from "./setup/kit";

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
  { id: "rigid", title: "The boards are flat and stiff",
    text: "Each board keeps its printed shape. Studio measures how the boards sit relative to one another." },
  { id: "bend", title: "The boards may be slightly bowed",
    text: "For boards on paper, foam or thin plastic. Each board may also bow or twist a little." },
  { id: "free", title: "Every corner may move",
    text: "Only for a target you do not trust at all. It can hide a lens error inside the target, so set a limit below." },
  { id: "off", title: "Do not adjust the target",
    text: "The boards stay where the first rough estimate put them. For comparing with older results." },
];

// With one board there is no pose between boards to work out: it is flat, or it bows.
const ONE_BOARD: { id: ShapeMode; title: string; text: string }[] = [
  { id: "rigid", title: "It is flat and stiff",
    text: "A board on glass, metal or a thick rigid plate. Studio trusts the printed shape." },
  { id: "bend", title: "It may be slightly bowed",
    text: "A board on paper, foam or thin plastic. Studio also measures how much it bows and twists, and the result shows it." },
];

const US = [-1, -0.5, 0, 0.5, 1];
const VS = [-1, 0, 1];

/** Where corner (u, v) of board `b` is drawn at time `t`: the printed shape plus what `mode` lets move. */
function corner(mode: ShapeMode | "printed", b: number, u: number, v: number, t: number, single = false): [number, number] {
  if (single) {
    // one board, drawn larger and seen more from the side, so a bow is plain to see
    const w = mode === "bend" ? 13 * Math.sin(t * 1.6) * (u * u - 1 / 3) + 4 * Math.sin(t * 0.9) * u * v : 0;
    return [84 + 44 * u + 12 * v, 41 + 15 * v - w];
  }
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

function BoardPicture({ mode, t, single = false }: { mode: ShapeMode; t: number; single?: boolean }) {
  const fine = mode === "free" ? US : [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1];
  const line = (m: ShapeMode | "printed", b: number, pts: [number, number][]) =>
    pts.map(([u, v]) => corner(m, b, u, v, t, single).map((x) => x.toFixed(1)).join(",")).join(" ");
  const outline: [number, number][] = [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]];
  return (
    <svg viewBox="0 0 168 82" className="w-full h-auto text-s-text" role="img" aria-hidden="true">
      {(single ? [0] : [0, 1]).map((b) => (
        <g key={b}>
          {/* the printed shape, for comparison */}
          <polyline points={line("printed", b, outline)} fill="none" stroke="currentColor" strokeOpacity="0.28" strokeDasharray="3 3" strokeWidth="1" />
          <polygon points={[...VS.map((v) => fine.map((u) => [u, v] as [number, number]))[0],
            ...fine.map((u) => [u, 1] as [number, number]).reverse()].map(([u, v]) => corner(mode, b, u, v, t, single).map((x) => x.toFixed(1)).join(",")).join(" ")}
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
            const [x, y] = corner(mode, b, u, v, t, single);
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
  const [hover, setHover] = useState<string | null>(null);
  const single = boards <= 1;
  // for one board only "bend" changes anything; every other value means flat
  const current = (single ? (value === "bend" ? "bend" : "rigid") : value ?? fallback) as ShapeMode;
  return (
    <div className="grid gap-3">
      <div role="radiogroup" aria-label="What may move in the target" className="grid gap-3 sm:grid-cols-2">
        {(single ? ONE_BOARD : SHAPES).map((s) => {
          const on = current === s.id;
          return (
            <button key={s.id} type="button" role="radio" aria-checked={on} onClick={() => onChange(s.id)}
              onPointerEnter={() => setHover(s.id)} onPointerLeave={() => setHover(null)}
              className={`text-left p-4 grid grid-cols-[150px_minmax(0,1fr)] gap-4 items-center ${choiceCard(on)}`}>
              <BoardPicture mode={s.id} single={single} t={on || hover === s.id ? t : 1.1} />
              <span className="grid gap-1">
                <span className="text-base font-semibold flex flex-wrap items-baseline gap-x-2">
                  {s.title}
                  {s.id === fallback && <span className="text-[13px] font-medium text-s-muted">usual choice</span>}
                </span>
                <span className="text-[15px] text-s-muted leading-relaxed">{s.text}</span>
              </span>
            </button>
          );
        })}
      </div>
      <p className="text-[15px] text-s-muted leading-relaxed">The dashed outline is {single ? "the" : "a"} board as printed. The solid one shows, enlarged, what this choice allows to move.</p>
    </div>
  );
}

// -- outlier band ------------------------------------------------------------------------------

/** The inlier band is this many times `noise_bound` (ds_msp.rig.pipeline.TLS_BAND_SIGMAS). */
const BAND_SIGMAS = 3.03;
const R_MIN = 0.01;
const R_MAX = 30;
const PLOT_R = 136;

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

/** `noise_bound`, shown as what it does: the distance beyond which a corner is ignored. */
export function OutlierBand({ value, fallback, refRun, onChange }: {
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
  const beyond = off ? 0 : spread ? countBeyond(spread, band) : example.dx.filter((x, k) => Math.hypot(x, example.dy[k]) > band).length;
  const total = spread ? spread.n : example.dx.length;
  const share = beyond / total;
  const C = 150;

  return (
    <div className="grid gap-5 md:grid-cols-[300px_minmax(0,1fr)] items-start">
      <figure className="grid gap-2 justify-self-center w-full max-w-[300px]">
        <svg viewBox="0 0 300 300" className="w-full h-auto rounded-xl bg-s-sunken text-s-text" role="img"
          aria-label={off ? "No corner is ignored" : `Corners further than ${px(band)} pixels from where they are expected are ignored`}>
          {[0.1, 1, 10].map((r) => (
            <circle key={r} cx={C} cy={C} r={plotR(r)} fill="none" stroke="currentColor" strokeOpacity="0.18" strokeWidth="1" />
          ))}
          {!off && <circle cx={C} cy={C} r={plotR(band)} fill="#10b981" fillOpacity="0.10" stroke="#059669" strokeWidth="2.5" />}
          {pts.dx.map((x, k) => {
            const y = pts.dy[k];
            const r = Math.hypot(x, y);
            const s = r > 0 ? plotR(r) / r : 0;
            const out = !off && r > band;
            return out
              ? <path key={k} d={`M${C + x * s - 4} ${C + y * s - 4}l8 8m0 -8l-8 8`} stroke="#dc2626" strokeWidth="2.2" strokeLinecap="round" />
              : <circle key={k} cx={C + x * s} cy={C + y * s} r="2.4" fill="#0ea5e9" fillOpacity="0.75" />;
          })}
          <path d={`M${C - 6} ${C}H${C + 6}M${C} ${C - 6}V${C + 6}`} stroke="currentColor" strokeWidth="1.5" />
          {!off && <text x={C} y={Math.min(292, C + plotR(band) + 15)} fontSize="13" fontWeight="700" textAnchor="middle" fill="#047857"
            stroke="var(--s-sunken)" strokeWidth="3.5" paintOrder="stroke">your limit: {px(band)} px</text>}
          {[0.1, 1, 10].map((r) => (
            <text key={r} x={C} y={C - plotR(r) - 4} fontSize="12" textAnchor="middle" fill="currentColor" fillOpacity="0.75"
              stroke="var(--s-sunken)" strokeWidth="3.5" paintOrder="stroke">{r} px</text>
          ))}
        </svg>
        <figcaption className="text-[14px] text-s-muted leading-relaxed">
          <span className="inline-flex items-center gap-1.5 mr-3"><span className="size-2.5 rounded-full bg-sky-500" />a corner that is used</span>
          <span className="inline-flex items-center gap-1.5 mr-3"><span className="text-red-600 font-bold leading-none">×</span>a corner that is ignored</span>
          <span className="block mt-1">The centre is where the calibration expects a corner. Each dot is one {spread ? `corner of ${refRun?.name}` : "corner of an example capture"}, drawn as far from the centre as it was found from that place.</span>
        </figcaption>
      </figure>

      <div className="grid gap-4">
        <div className="grid gap-2">
          <label htmlFor="noise_bound" className="text-base font-semibold">Ignore a corner when it is further off than</label>
          <div className="flex items-baseline gap-2">
            <output htmlFor="noise_bound" className="text-3xl font-bold tnum text-s-accent">{off ? "never" : px(band)}</output>
            {!off && <span className="text-base text-s-muted">pixels</span>}
          </div>
          <input id="noise_bound" type="range" min={-0.7} max={1.2} step={0.01} disabled={off}
            value={Math.log10(off ? BAND_SIGMAS * fallback : band)} className="w-full h-2 accent-s-accent cursor-pointer disabled:cursor-not-allowed"
            aria-valuetext={off ? "never" : `${px(band)} pixels`}
            onChange={(e) => onChange(Number((10 ** Number(e.target.value) / BAND_SIGMAS).toPrecision(3)))} />
          <div className="flex justify-between text-[13px] text-s-faint"><span>0.2 px · strict</span><span>3 px · usual</span><span>16 px · lenient</span></div>
        </div>

        <div className={`rounded-lg p-3.5 text-[15px] leading-relaxed ${share > 0.25 ? "bg-s-warn-soft" : "bg-s-info-soft"}`} role="status">
          {off ? <>No corner is ignored. Corners far off still count, only with less weight.</>
            : spread
              ? <>In your result <b>{refRun?.name}</b>, the corner furthest off was <b>{px(spread.max_px ?? 0)} px</b> away. With this setting <b>{beyond === 0 ? "no corner" : `about ${beyond.toLocaleString()} of its ${total.toLocaleString()} corners`}</b> would be ignored{beyond > 0 ? ` (${share < 0.001 ? "under 0.1" : (100 * share).toFixed(share < 0.1 ? 1 : 0)}%)` : ""}.</>
              : <>In this example capture, {beyond} of {total} corners would be ignored. Once these photos have a result, this picture shows your own corners.</>}
          {share > 0.25 && <b className="block mt-1 text-s-warn">That is too strict here: a camera that loses more than a quarter of its corners fails the coverage check.</b>}
        </div>

        <label className="flex items-start gap-2.5 text-base cursor-pointer">
          <input type="checkbox" className="mt-1 size-4 accent-s-accent" checked={off} onChange={(e) => onChange(e.target.checked ? 0 : "")} />
          <span>Never ignore a corner</span>
        </label>
      </div>
    </div>
  );
}
