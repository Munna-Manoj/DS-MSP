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

  const find = () => {
    setGuessing(true);
    setGuess(null);
    api.findBoard(payload).then(
      (g) => {
        setGuessing(false);
        setGuess(g);
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
            <div className="grid gap-1.5">
              <div className="flex flex-wrap items-baseline gap-x-3"><span className="text-base font-semibold">How much of a white square does its marker fill?</span><KeyTag name="length_marker / length_square" /></div>
              <Explain>Compare the drawing with your board. The usual board has markers three quarters as wide as a square. A wrong value makes Studio miss corners.</Explain>
              <Range id="marker-fill" value={Math.round(fill * 100)} min={30} max={95} step={1} unit="%" ends={["small markers", "markers nearly fill the square"]}
                onChange={(v) => { set("length_square", ls); set("length_marker", Number((ls * v / 100).toPrecision(4))); }} />
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
