// Step 2 of the setup: what the calibration board looks like. Studio can read the number of
// squares from the photos; the one thing only the user knows is how big a printed square is.
import { useState } from "react";

import type { BoardCheck, BoardGuess, Values } from "../api";
import { api } from "../api";
import type { Unit } from "../stage/units";
import { Button, Callout, INPUT } from "../ui";
import { TargetShape } from "../Teach";
import { Explain, KeyTag, NumberBox, Range, Stepper, Toggle } from "./kit";

type Set = (key: string, v: unknown) => void;

const num = (v: unknown, fallback: number) => (typeof v === "number" && Number.isFinite(v) ? v : fallback);
const list = (v: unknown): number[] => (Array.isArray(v) ? v.map(Number) : []);
const trim = (v: number) => Number(v.toPrecision(4)).toString();

/** A photo of one camera with the corners Studio found drawn on it. */
export function PhotoThumb({ src, width, height, corners, caption, tone }: {
  src: string; width?: number | null; height?: number | null; corners?: [number, number][];
  caption?: React.ReactNode; tone?: "ok" | "bad" | "plain";
}) {
  const w = width || 4;
  const h = height || 3;
  const r = Math.max(w, h) / 110;
  return (
    <figure className="grid gap-1.5 min-w-0">
      <div className={`relative rounded-lg overflow-hidden bg-s-viewport ring-2 ${tone === "ok" ? "ring-s-ok" : tone === "bad" ? "ring-s-bad" : "ring-transparent"}`}
        style={{ aspectRatio: `${w} / ${h}` }}>
        <img src={src} alt="" loading="lazy" className="absolute inset-0 size-full object-cover" />
        {corners && corners.length > 0 && (
          <svg viewBox={`0 0 ${w} ${h}`} className="absolute inset-0 size-full" aria-hidden="true">
            {corners.map(([u, v], k) => <circle key={k} cx={u} cy={v} r={r} fill="#34d399" stroke="#022c22" strokeWidth={r / 3} />)}
          </svg>
        )}
      </div>
      {caption && <figcaption className="text-[15px] leading-snug min-w-0">{caption}</figcaption>}
    </figure>
  );
}

/** The board as it is printed, with what "squares across", "squares down" and "one square" mean. */
function BoardDrawing({ nx, ny, size, unit, fill }: { nx: number; ny: number; size: number; unit: string; fill: number }) {
  if (!(nx >= 2 && ny >= 2 && nx <= 40 && ny <= 40)) return null;
  const cell = Math.min(300 / nx, 220 / ny);
  const W = nx * cell;
  const H = ny * cell;
  const m = cell * fill;
  const pad = (cell - m) / 2;
  const squares = [];
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      if ((i + j) % 2 === 0) squares.push(<rect key={`b${i}-${j}`} x={i * cell} y={j * cell} width={cell} height={cell} fill="#0f172a" />);
      else squares.push(
        <g key={`m${i}-${j}`}>
          <rect x={i * cell + pad} y={j * cell + pad} width={m} height={m} fill="#0f172a" />
          <rect x={i * cell + pad + m * 0.22} y={j * cell + pad + m * 0.22} width={m * 0.24} height={m * 0.24} fill="#fff" />
          <rect x={i * cell + pad + m * 0.54} y={j * cell + pad + m * 0.5} width={m * 0.24} height={m * 0.26} fill="#fff" />
        </g>);
    }
  const ink = "var(--s-accent)";
  return (
    <svg viewBox={`-34 -30 ${W + 44} ${H + 66}`} className="w-full max-w-[380px] h-auto justify-self-center" role="img"
      aria-label={`A board ${nx} squares across and ${ny} squares down`}>
      <rect x="0" y="0" width={W} height={H} fill="#fff" stroke="#0f172a" strokeWidth="1.5" />
      {squares}
      {/* across */}
      <path d={`M0 -12H${W}M0 -17V-7M${W} -17V-7`} stroke={ink} strokeWidth="1.5" fill="none" />
      <text x={W / 2} y="-18" textAnchor="middle" fontSize="13" fontWeight="600" fill={ink}>{nx} across</text>
      {/* down */}
      <path d={`M-12 0V${H}M-17 0H-7M-17 ${H}H-7`} stroke={ink} strokeWidth="1.5" fill="none" />
      <text transform={`translate(-18 ${H / 2}) rotate(-90)`} textAnchor="middle" fontSize="13" fontWeight="600" fill={ink}>{ny} down</text>
      {/* one square */}
      <rect x="0" y={H - cell} width={cell} height={cell} fill="none" stroke="#f59e0b" strokeWidth="3" />
      <path d={`M0 ${H + 10}H${cell}M0 ${H + 5}V${H + 15}M${cell} ${H + 5}V${H + 15}`} stroke="#d97706" strokeWidth="1.5" fill="none" />
      <text x="0" y={H + 30} fontSize="13" fontWeight="600" fill="#b45309">one square{size > 0 ? ` = ${trim(size)}${unit === "?" ? "" : ` ${unit}`}` : ""}</text>
    </svg>
  );
}

/** White modules of the marker in the close-up, on its 6 by 6 inner grid. */
const BITS = ["101100", "010011", "110101", "001110", "100101", "011010"];

/** How far the measured marker width may sit from the setting and still agree with it. The
 * measurement was within 0.027 of the printed value on the photo sets it was tried on. */
const FILL_AGREE = 0.05;

/** One white square of the board, close up, with its marker: what "marker" and "square" mean,
 * and how big one is against the other. `measured` draws the width read from the photos. */
function MarkerDrawing({ fill, size, unit, measured }: { fill: number; size: number; unit: string; measured: number | null }) {
  const C = 160;                 // the square
  const E = 34;                  // how much of each neighbour shows
  const m = C * fill;
  const pad = (C - m) / 2;
  const mod = m / 8;
  const u = unit === "?" ? "" : ` ${unit}`;
  const ink = "var(--s-accent)";
  const ghost = measured != null && Math.abs(measured - fill) >= 0.04 ? C * measured : null;
  return (
    <svg viewBox={`${-E - 4} ${-E - 46} ${C + 2 * E + 8} ${C + 2 * E + 96}`} className="w-full max-w-[250px] h-auto justify-self-center" role="img"
      aria-label={`One white square of the board with its marker. The marker is ${Math.round(fill * 100)} percent as wide as the square.`}>
      <defs><clipPath id="marker-patch"><rect x={-E} y={-E} width={C + 2 * E} height={C + 2 * E} rx="6" /></clipPath></defs>
      <g clipPath="url(#marker-patch)">
        <rect x={-E} y={-E} width={C + 2 * E} height={C + 2 * E} fill="#fff" />
        <rect x={-C} y="0" width={C} height={C} fill="#0f172a" />
        <rect x={C} y="0" width={C} height={C} fill="#0f172a" />
        <rect x="0" y={-C} width={C} height={C} fill="#0f172a" />
        <rect x="0" y={C} width={C} height={C} fill="#0f172a" />
        {/* the marker: a black frame one module wide around a 6 by 6 pattern */}
        <rect x={pad} y={pad} width={m} height={m} fill="#0f172a" style={{ transition: "all 120ms ease-out" }} />
        {BITS.flatMap((row, j) => [...row].map((bit, i) => bit === "1" &&
          <rect key={`${i}-${j}`} x={pad + (i + 1) * mod} y={pad + (j + 1) * mod} width={mod + 0.4} height={mod + 0.4} fill="#fff" />))}
        {ghost != null && (
          <g>
            <rect x={(C - ghost) / 2} y={(C - ghost) / 2} width={ghost} height={ghost} fill="none" stroke="#fff" strokeWidth="5" />
            <rect x={(C - ghost) / 2} y={(C - ghost) / 2} width={ghost} height={ghost} fill="none" stroke="var(--s-info)" strokeWidth="2.5" strokeDasharray="7 5" />
          </g>
        )}
      </g>
      <rect x={-E} y={-E} width={C + 2 * E} height={C + 2 * E} rx="6" fill="none" stroke="var(--s-field-line)" strokeWidth="1.5" />
      {/* the marker, measured above */}
      <path d={`M${pad} ${pad}V${-E - 12}M${pad + m} ${pad}V${-E - 12}`} stroke="#fff" strokeWidth="4" fill="none" clipPath="url(#marker-patch)" />
      <path d={`M${pad} ${pad}V${-E - 18}M${pad + m} ${pad}V${-E - 18}`} stroke={ink} strokeWidth="1.5" strokeDasharray="4 3" fill="none" />
      <path d={`M${pad} ${-E - 12}H${pad + m}`} stroke={ink} strokeWidth="2" fill="none" />
      <text x={C / 2} y={-E - 24} textAnchor="middle" fontSize="15" fontWeight="700" fill={ink}>
        marker{size > 0 ? ` = ${trim(size * fill)}${u}` : ""}</text>
      {/* the square, measured below */}
      <rect x="0" y="0" width={C} height={C} fill="none" stroke="#f59e0b" strokeWidth="3" />
      <path d={`M0 ${C}V${C + E + 18}M${C} ${C}V${C + E + 18}`} stroke="#d97706" strokeWidth="1.5" strokeDasharray="4 3" fill="none" />
      <path d={`M0 ${C + E + 12}H${C}`} stroke="#d97706" strokeWidth="2" fill="none" />
      <text x={C / 2} y={C + E + 36} textAnchor="middle" fontSize="15" fontWeight="700" fill="#b45309">
        one square{size > 0 ? ` = ${trim(size)}${u}` : ""}</text>
    </svg>
  );
}

export function BoardStep({ values, set, defaults, photos, payload, check, onCheck, checkBusy, checkError, problemOf, unit, setUnit }: {
  unit: Unit; setUnit: (u: Unit) => void;
  values: Values; set: Set; defaults: { nx: number; ny: number; size: number; ls: number; lm: number; shape: string };
  /** The photo folder, when the calibration starts from photos. */
  photos: { root: string; prefix: string | null } | null;
  payload: Values; check: BoardCheck | null; onCheck: (v?: Values) => void; checkBusy: boolean; checkError: string | null;
  problemOf: (key: string) => string | null;
}) {
  const [guess, setGuess] = useState<BoardGuess | null>(null);
  const [guessing, setGuessing] = useState(false);
  const nx = num(values.number_x_square, defaults.nx);
  const ny = num(values.number_y_square, defaults.ny);
  const size = num(values.square_size, defaults.size);
  const nb = num(values.number_board, 1);
  const ls = num(values.length_square, defaults.ls);
  const lm = num(values.length_marker, defaults.lm);
  const fill = Math.min(0.95, Math.max(0.3, lm / ls));
  const perX = list(values.number_x_square_per_board);
  const perY = list(values.number_y_square_per_board);
  const perS = list(values.square_size_per_board);
  const index = list(values.boards_index);
  const differ = perX.length > 0 || perY.length > 0 || perS.length > 0;
  const [printed, setPrinted] = useState(index.length > 0 || "length_marker" in values || "length_square" in values);
  // the marker width as measured in the photos, once Studio has looked
  const [fillRead, setFillRead] = useState<{ fill: number | null; pairs: number; error?: string } | null>(null);
  const [fillBusy, setFillBusy] = useState(false);
  const [rulerBad, setRulerBad] = useState(false);
  const sq = num(values.square_size, 0);
  const setFill = (r: number) => { set("length_square", ls); set("length_marker", Number((ls * r).toPrecision(6))); setRulerBad(false); };
  const fillOff = fillRead?.fill != null && Math.abs(fillRead.fill - lm / ls) > FILL_AGREE;
  const measureFill = () => {
    setFillBusy(true);
    api.findBoard(payload).then(
      (g) => { setFillBusy(false); setFillRead({ fill: g.marker_fill ?? null, pairs: g.marker_fill_pairs ?? 0 }); },
      (e: Error) => { setFillBusy(false); setFillRead({ fill: null, pairs: 0, error: e.message }); },
    );
  };

  const find = () => {
    setGuessing(true);
    setGuess(null);
    api.findBoard(payload).then(
      (g) => {
        setGuessing(false);
        setGuess(g);
        setFillRead({ fill: g.marker_fill ?? null, pairs: g.marker_fill_pairs ?? 0 });
        // markers of another size than the setting: open the place where that is set
        if (g.marker_fill != null && Math.abs(g.marker_fill - lm / ls) > FILL_AGREE) setPrinted(true);
        if (g.ok && g.number_x_square && g.number_y_square) {
          set("number_x_square", g.number_x_square);
          set("number_y_square", g.number_y_square);
          onCheck({ ...payload, number_x_square: g.number_x_square, number_y_square: g.number_y_square });
        }
      },
      (e: Error) => { setGuessing(false); setGuess({ ok: false, number_x_square: null, number_y_square: null, boards_seen: 0, markers: 0, photos: 0, message: e.message }); },
    );
  };

  const setRow = (k: number, key: "x" | "y" | "s", v: number | "") => {
    const fillTo = (arr: number[], d: number) => Array.from({ length: nb }, (_, i) => (Number.isFinite(arr[i]) ? arr[i] : d));
    const X = fillTo(perX, nx);
    const Y = fillTo(perY, ny);
    const S = fillTo(perS, size);
    if (key === "x" && v !== "") X[k] = v;
    if (key === "y" && v !== "") Y[k] = v;
    if (key === "s" && v !== "") S[k] = v;
    set("number_x_square_per_board", X);
    set("number_y_square_per_board", Y);
    set("square_size_per_board", S);
  };
  const allAlike = (on: boolean) => {
    if (on) { set("number_x_square_per_board", ""); set("number_y_square_per_board", ""); set("square_size_per_board", ""); }
    else setRow(0, "x", nx);
  };

  return (
    <>
      {photos && (
        <div className="rounded-xl bg-s-info-soft border border-s-info/25 p-4 flex flex-wrap items-center gap-x-5 gap-y-3">
          <div className="min-w-0 mr-auto">
            <div className="text-base font-semibold text-s-info">Let Studio count the squares</div>
            <div className="text-[15px] text-s-text leading-relaxed">It reads the markers in a few of your photos. Takes a second or two.</div>
          </div>
          <Button kind="primary" onClick={find} busy={guessing}>Find the board in my photos</Button>
        </div>
      )}
      {guess && (guess.ok
        ? <Callout tone="ok" title={guess.message}>
            Read from {guess.markers} markers in {guess.photos} photos. Check the drawing against your board; you only need to enter the size of one square.
            {(guess.boards_seen ?? 1) > nb && <b className="block mt-1">Your target seems to have at least {guess.boards_seen} boards. Set how many are fixed together below.</b>}
          </Callout>
        : <Callout tone="warn" title="Studio could not read the board size">{guess.message}</Callout>)}
      {guess && fillOff && fillRead?.fill != null && (
        <Callout tone="warn" title={`The markers on your board are ${fillRead.fill < lm / ls ? "smaller" : "larger"} than the setting says`}>
          In your photos a marker is about {Math.round(fillRead.fill * 100)} % as wide as its square; Studio is set to {Math.round(lm / ls * 100)} %.
          With the wrong value the corners are found slightly off their true place, and nothing warns about it.
          <div className="mt-2.5 flex flex-wrap items-center gap-3">
            <Button kind="primary" onClick={() => setFill(Math.round(fillRead.fill! * 100) / 100)}>Set it to {Math.round(fillRead.fill * 100)} %</Button>
            <button type="button" className="text-s-accent font-medium hover:underline cursor-pointer"
              onClick={() => document.getElementById("marker-fill")?.scrollIntoView({ behavior: "smooth", block: "center" })}>Compare with the drawing</button>
          </div>
        </Callout>
      )}

      <div className="grid gap-6 md:grid-cols-[minmax(0,1fr)_minmax(0,380px)] items-start">
        <div className="grid gap-5 content-start">
          <div className="flex flex-wrap gap-x-8 gap-y-4">
            <div className="grid gap-1.5">
              <label htmlFor="number_x_square" className="text-base font-semibold">Squares across</label>
              <Stepper id="number_x_square" label="Squares across" value={nx} min={2} max={40} onChange={(v) => set("number_x_square", v)} />
            </div>
            <div className="grid gap-1.5">
              <label htmlFor="number_y_square" className="text-base font-semibold">Squares down</label>
              <Stepper id="number_y_square" label="Squares down" value={ny} min={2} max={40} onChange={(v) => set("number_y_square", v)} />
            </div>
          </div>
          <Explain>Count every square, black and white, along one edge of the board.</Explain>
          {(problemOf("number_x_square") || problemOf("number_y_square")) && <div className="text-[15px] font-medium text-s-bad" role="alert">{problemOf("number_x_square") ?? problemOf("number_y_square")}</div>}

          <div className="grid gap-1.5">
            <label htmlFor="square_size" className="text-base font-semibold">Size of one square</label>
            <Explain className="-mt-0.5">Measure one printed square with a ruler, edge to edge (the orange square in the drawing). Every distance in the result comes from this number, so measure it carefully.</Explain>
            <div className="flex flex-wrap items-center gap-2">
              <NumberBox id="square_size" value={values.square_size} fallback={null} onChange={(v) => set("square_size", v)} width="w-40" />
              <select aria-label="Unit of the square size" className={`${INPUT} !w-auto font-semibold cursor-pointer`} value={unit}
                onChange={(e) => setUnit(e.target.value as Unit)}>
                <option value="?">which unit?</option>
                <option value="mm">millimetres</option>
                <option value="cm">centimetres</option>
                <option value="m">metres</option>
              </select>
            </div>
            {values.square_size == null && <div className="text-[15px] text-s-warn font-medium">Not entered yet. Without it the result has no real-world scale.</div>}
            {problemOf("square_size") && <div className="text-[15px] font-medium text-s-bad" role="alert">{problemOf("square_size")}</div>}
          </div>
        </div>
        <BoardDrawing nx={nx} ny={ny} size={num(values.square_size, 0)} unit={unit} fill={fill} />
      </div>

      {/* several boards ------------------------------------------------------------------- */}
      <div className="border-t border-s-line pt-5 grid gap-4">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <div className="min-w-0 mr-auto">
            <label htmlFor="number_board" className="text-base font-semibold">How many boards are fixed together?</label>
            <Explain>One for a single board. More when several boards are mounted rigidly as one target, for example on the faces of a cube.</Explain>
          </div>
          <Stepper id="number_board" label="Number of boards" value={nb} min={1} max={64}
            onChange={(v) => { set("number_board", v === 1 ? "" : v); if (v === 1) allAlike(true); }} />
        </div>
        {nb > 1 && (
          <>
            <Toggle id="boards-alike" on={!differ} onChange={allAlike}>All {nb} boards are printed the same</Toggle>
            {differ && (
              <div className="grid gap-2">
                {Array.from({ length: nb }, (_, k) => (
                  <div key={k} className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg bg-s-sunken px-3 py-2.5">
                    <span className="text-base font-semibold w-20">Board {k + 1}</span>
                    <Stepper id={`bx${k}`} label={`Squares across, board ${k + 1}`} value={Number.isFinite(perX[k]) ? perX[k] : nx} min={2} max={40} onChange={(v) => setRow(k, "x", v)} />
                    <span className="text-s-muted" aria-hidden="true">×</span>
                    <Stepper id={`by${k}`} label={`Squares down, board ${k + 1}`} value={Number.isFinite(perY[k]) ? perY[k] : ny} min={2} max={40} onChange={(v) => setRow(k, "y", v)} />
                    <span className="text-[15px] text-s-muted">squares, each</span>
                    <NumberBox id={`bs${k}`} value={Number.isFinite(perS[k]) ? perS[k] : size} unit={unit === "?" ? undefined : unit} onChange={(v) => setRow(k, "s", v)} width="w-36" />
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>

      {/* flat or not ---------------------------------------------------------------------- */}
      <div className="border-t border-s-line pt-5 grid gap-3">
        <div className="grid gap-1">
          <div className="flex flex-wrap items-baseline gap-x-3">
            <span className="text-base font-semibold">{nb > 1 ? "How flat and stiff is your target?" : "Is your board really flat?"}</span>
            <KeyTag name="structure_mode" />
            {values.structure_mode != null && <button type="button" onClick={() => { set("structure_mode", ""); set("structure_prior_sigma", ""); }} className="text-[15px] text-s-accent underline cursor-pointer">Back to the usual choice</button>}
          </div>
          <Explain>
            {nb > 1
              ? "Studio works out how the boards are mounted relative to one another. How much else it may change depends on how stiff and flat the target really is."
              : "A board that bows by even a millimetre, treated as flat, pushes that error into the lens values and camera positions."}
          </Explain>
        </div>
        <TargetShape value={typeof values.structure_mode === "string" ? values.structure_mode : undefined} fallback={defaults.shape}
          boards={nb} onChange={(v) => set("structure_mode", v === defaults.shape ? "" : v)} />
        {values.structure_mode === "bend" && (
          <Explain><b className="font-semibold text-s-text">Good to know: </b>this needs photos with the board tilted and at different distances. The result page shows the bow it found; a bow near zero means the board was flat after all.</Explain>
        )}
        {nb > 1 && values.structure_mode === "free" && (
          <label className="grid gap-1.5 text-base font-semibold">
            How far may a corner move from its printed place?
            <span className="text-[15px] font-normal text-s-muted">Without a limit the target can absorb errors that belong to the lens. In the unit of your square size.</span>
            <NumberBox id="structure_prior_sigma" value={values.structure_prior_sigma} fallback={null} unit={unit === "?" ? undefined : unit} onChange={(v) => set("structure_prior_sigma", v)} />
            {problemOf("structure_prior_sigma") && <span className="text-[15px] font-medium text-s-bad" role="alert">{problemOf("structure_prior_sigma")}</span>}
          </label>
        )}
        {problemOf("structure_mode") && <div className="text-[15px] font-medium text-s-bad" role="alert">{problemOf("structure_mode")}</div>}
      </div>

      {/* check ---------------------------------------------------------------------------- */}
      {photos && (
        <div className="border-t border-s-line pt-5 grid gap-4">
          <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
            <div className="min-w-0 mr-auto">
              <div className="text-base font-semibold">See it found in your photos</div>
              <Explain>Studio looks for this board in a few photos of each camera and marks every corner it finds.</Explain>
            </div>
            <Button onClick={() => onCheck()} busy={checkBusy}>{check ? "Look again" : "Look for this board"}</Button>
          </div>
          {checkError && <Callout tone="bad" title="Could not check the board">{checkError}</Callout>}
          {check && <Callout tone={check.found === 0 ? "bad" : "ok"} title={check.message} />}
          {check && check.found > 0 && (
            <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 lg:grid-cols-4">
              {check.cameras.map((c) => (
                <PhotoThumb key={c.index} src={api.thumbUrl(photos.root, c.index, c.sample?.photo ?? 0, photos.prefix)}
                  width={c.sample?.width} height={c.sample?.height} corners={c.sample?.corners} tone={c.found > 0 ? "ok" : "bad"}
                  caption={<><b className="font-semibold">{c.folder}</b><span className={`block ${c.found > 0 ? "text-s-ok" : "text-s-bad"} font-medium`}>
                    {c.found > 0 ? `${c.corners_max} of ${c.corners_of} corners found` : `not found in ${c.tried} photos`}</span></>} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* printed differently -------------------------------------------------------------- */}
      <div className="border-t border-s-line pt-4 grid gap-4">
        <Toggle id="board-printed" on={printed} onChange={(on) => {
          setPrinted(on);
          if (!on) { set("length_marker", ""); set("length_square", ""); set("boards_index", ""); }
        }}>My board was not printed with the usual settings</Toggle>
        {printed && (
          <div className="grid gap-5">
            <div className="grid gap-4">
              <div className="grid gap-1.5">
                <div className="flex flex-wrap items-baseline gap-x-3"><span className="text-base font-semibold">How much of a white square does its marker fill?</span><KeyTag name="length_marker / length_square" /></div>
                <Explain>Every white square of the board holds one marker. Drag the slider until the drawing looks like your board, or measure one marker with a ruler. Most boards are printed at 75 %.</Explain>
              </div>
              <div className="grid gap-x-8 gap-y-5 sm:grid-cols-[minmax(0,250px)_minmax(0,1fr)] items-start">
                <MarkerDrawing fill={fill} size={sq} unit={unit} measured={fillRead?.fill ?? null} />
                <div className="grid gap-5 content-start min-w-0">
                  <Range id="marker-fill" value={Math.round(fill * 100)} min={30} max={95} step={1} unit="%" words="of the square's width"
                    marks={[{ at: 75, label: "usual" }, ...(fillRead?.fill != null && Math.abs(fillRead.fill - 0.75) > 0.03
                      ? [{ at: Math.min(95, Math.max(30, Math.round(fillRead.fill * 100))), label: "in your photos", tone: "info" as const }] : [])]}
                    onChange={(v) => setFill(v / 100)} />

                  <div className="grid gap-1.5">
                    <label htmlFor="marker-width" className="text-[15px] font-semibold">Or measure one marker with a ruler</label>
                    {sq > 0 ? (
                      <>
                        <div className="flex flex-wrap items-center gap-x-3 gap-y-2 text-[15px]">
                          <NumberBox id="marker-width" value={Number((lm / ls * sq).toPrecision(5))} unit={unit === "?" ? undefined : unit} width="w-36"
                            onChange={(v) => {
                              if (v === "") return;
                              if (v > 0 && v < sq) setFill(v / sq); else setRulerBad(true);
                            }} />
                          <span className="text-s-muted">wide, in a square of {trim(sq)}{unit === "?" ? "" : ` ${unit}`}</span>
                        </div>
                        <Explain>The black frame of one marker, edge to edge, in the unit of the square size.</Explain>
                        {rulerBad && <div className="text-[15px] font-medium text-s-bad" role="alert">A marker is narrower than its square. Check this number and the size of one square.</div>}
                      </>
                    ) : <Explain>Enter the size of one square above; then you can type the width of a marker here.</Explain>}
                  </div>

                  {photos && !fillRead && (
                    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
                      <Button onClick={measureFill} busy={fillBusy}>Measure it in my photos</Button>
                      <span className="text-[15px] text-s-muted">Studio compares the markers with the gaps between them.</span>
                    </div>
                  )}
                  {fillRead && (fillRead.fill == null
                    ? <Callout tone="info" title="Studio could not measure it in the sample photos">
                        {fillRead.error ?? "Too few neighbouring markers could be read, or the readings disagree."} Use the drawing or a ruler.
                      </Callout>
                    : fillOff
                      ? <Callout tone="warn" title={`In your photos a marker fills about ${Math.round(fillRead.fill * 100)} % of its square, not ${Math.round(lm / ls * 100)} %`}>
                          Read from {fillRead.pairs} pairs of neighbouring markers (the dashed outline in the drawing). In tests the reading was within 3 points of the printed value.
                          <div className="mt-2.5"><Button kind="primary" onClick={() => setFill(Math.round(fillRead.fill! * 100) / 100)}>Set it to {Math.round(fillRead.fill * 100)} %</Button></div>
                        </Callout>
                      : <Callout tone="ok" title="Your photos agree with this value">
                          Studio measured about {Math.round(fillRead.fill * 100)} % on {fillRead.pairs} pairs of neighbouring markers. In tests the reading was within 3 points of the printed value.
                        </Callout>)}
                </div>
              </div>
              <Explain>Why it matters: Studio finds the board's corners from where the markers are. With a wrong value the corners are still found, but slightly off their true place, and nothing warns about it.</Explain>
            </div>
            {nb > 1 && (
              <div className="grid gap-2">
                <div className="flex flex-wrap items-baseline gap-x-3"><span className="text-base font-semibold">Which printed board is which?</span><KeyTag name="boards_index" /></div>
                <Explain>Boards from one generated set are numbered 0, 1, 2 and so on, and their markers do not repeat. Change these only if your target uses other boards of the set, for example boards 3, 4 and 7.</Explain>
                <div className="flex flex-wrap gap-3">
                  {Array.from({ length: nb }, (_, k) => (
                    <div key={k} className="grid gap-1">
                      <label htmlFor={`bi${k}`} className="text-[15px] font-medium">Board {k + 1} is number</label>
                      <Stepper id={`bi${k}`} label={`Printed number of board ${k + 1}`} value={Number.isFinite(index[k]) ? index[k] : k} min={0} max={199}
                        onChange={(v) => set("boards_index", Array.from({ length: nb }, (_, i) => (i === k ? v : Number.isFinite(index[i]) ? index[i] : i)))} />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </>
  );
}
