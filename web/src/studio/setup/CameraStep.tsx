// Step 3 of the setup: one card per camera, with a photo it took, and its lens type. Studio can
// measure each lens on the photos and choose the type itself.
import type { ReactNode } from "react";

import { api, type IntrinsicsScan, type LensAdvice, type LensHealth } from "../api";
import { int, num } from "../format";
import { Button, Callout } from "../ui";
import { choiceCard, Explain, Part, Segmented } from "./kit";

export const LENSES: { id: string; label: string; short: string; hint: string }[] = [
  { id: "radtan", label: "Standard lens", short: "Standard", hint: "ordinary lenses, up to about 100° wide" },
  { id: "kb", label: "Fisheye", short: "Fisheye", hint: "wide-angle and fisheye lenses; the safe choice for them" },
  { id: "ds", label: "Fisheye, double sphere", short: "Double sphere", hint: "fisheye lenses, also beyond 180°" },
  { id: "eucm", label: "Fisheye, EUCM", short: "EUCM", hint: "fisheye lenses, when your other software expects EUCM" },
  { id: "ucm", label: "Fisheye, UCM", short: "UCM", hint: "the simplest fisheye model; never for an ordinary lens" },
];
const lensLabel = (id: string) => LENSES.find((l) => l.id === id)?.label ?? id;

/** The values each lens type needs, in the order of the lens file. */
const DISTORTION: Record<string, string[]> = {
  radtan: ["k1", "k2", "p1", "p2", "k3"], kb: ["k1", "k2", "k3", "k4"], ds: ["xi", "alpha"], eucm: ["alpha", "beta"], ucm: ["alpha"],
};
export const lensFields = (model: string) => ["fx", "fy", "cx", "cy", ...(DISTORTION[model] ?? [])];

/** Where the lens values come from: worked out from the photos, a lens file, or typed in. */
export type LensMode = "auto" | "file" | "typed";
/** What has been typed on each camera's card, by camera and field. */
export type Typed = Record<number, Record<string, string>>;

export interface CameraInfo { index: number; name: string; photos: number | null; width: number | null; height: number | null }

/** The lens type Studio picks when asked to choose: a standard lens under 100° across, a
 * fisheye above. A fisheye type the user already chose for a wide lens is kept when it fits. */
export function chosenFor(a: LensAdvice): string | null {
  if (a.verdict === "unknown" || a.fov_h_deg == null) return a.verdict === "misfit" ? "kb" : null;
  if (a.fov_h_deg < 100) return "radtan";
  return a.verdict === "ok" && ["kb", "ds", "eucm"].includes(a.chosen) ? a.chosen : "kb";
}

function AdviceLine({ a, current, onUse, typing }: { a: LensAdvice | "busy" | undefined; current: string; onUse: (m: string) => void; typing: boolean }) {
  if (!a) return null;
  if (a === "busy") return <div className="text-[15px] text-s-info flex items-center gap-2"><span className="size-3.5 rounded-full border-2 border-current border-r-transparent animate-spin" aria-hidden="true" />Measuring this lens…</div>;
  if (a.chosen !== current) return null;                      // the choice changed since the check
  const better = a.suggested && a.suggested !== current ? a.suggested : null;
  const tone = a.verdict === "ok" ? "text-s-ok" : a.verdict === "misfit" ? "text-s-bad" : a.verdict === "caution" ? "text-s-warn" : "text-s-muted";
  const head = a.fov_h_deg != null ? `${Math.round(a.fov_h_deg)}° wide` : null;
  return (
    <div className="grid gap-1.5 text-[15px] leading-snug">
      <div className={`font-semibold ${tone}`}>
        {a.verdict === "ok" ? "✓ " : a.verdict === "misfit" ? "✗ " : a.verdict === "caution" ? "! " : ""}
        {a.verdict === "ok" ? `${head ? `${head} · ` : ""}this type fits (${num(a.chosen_px)} px)`
          : a.verdict === "misfit" ? `${head ? `${head} · ` : ""}this type cannot fit this lens`
          : a.verdict === "caution" ? `${head ? `${head} · ` : ""}fits the photos, but is the wrong kind` : "Could not measure this lens"}
      </div>
      {a.verdict !== "ok" && <div className="text-s-muted">{a.message}</div>}
      {typing && (a.verdict === "misfit" || a.verdict === "caution") && <div className="text-s-text">You can keep this type and enter the values you computed. Studio will not stop you, but check the result closely.</div>}
      {better && (a.verdict === "misfit" || a.verdict === "caution") && (
        <button type="button" onClick={() => onUse(better)}
          className="justify-self-start rounded-md bg-s-accent text-white font-semibold px-3 min-h-9 hover:brightness-110 cursor-pointer">Switch to {lensLabel(better)}</button>
      )}
    </div>
  );
}

export function CameraStep({ cameras, models, setModel, setAll, advice, onMeasure, measuring, canMeasure, measureWhat, photos, samples,
  lens, health, onLensBrowse, onLensRemove, fixed, onFixed, problem, lensProblem, lensError, lensReading,
  lensMode, onLensMode, strict, typed, onTyped }: {
  lensMode: LensMode; onLensMode: (m: LensMode) => void; strict: boolean;
  typed: Typed; onTyped: (k: number, field: string, value: string) => void;
  /** Why the file just chosen was not used, and the name of a file being read. */
  lensError: string | null; lensReading: string | null;
  cameras: CameraInfo[]; models: string[]; setModel: (k: number, m: string) => void; setAll: (m: string) => void;
  advice: Record<number, LensAdvice | "busy">; onMeasure: () => void; measuring: boolean; canMeasure: boolean;
  measureWhat: "photos" | "saved detections";
  photos: { root: string; prefix: string | null } | null;
  /** Per camera, the photo and corners of the board check, when it ran. */
  samples: Record<number, { photo: number; width: number; height: number; corners: [number, number][] } | null>;
  lens: IntrinsicsScan | null; health: Record<number, LensHealth | "busy">; onLensBrowse: () => void; onLensRemove: () => void;
  fixed: boolean; onFixed: (v: boolean) => void; problem: string | null; lensProblem: string | null;
}) {
  const n = cameras.length;
  const measured = Object.values(advice).filter((a): a is LensAdvice => a !== "busy" && a.verdict !== "unknown");
  const widths = measured.map((a) => a.fov_h_deg).filter((v): v is number => v != null);
  const alike = n > 0 && models.slice(0, n).every((m) => (m ?? "radtan") === (models[0] ?? "radtan"));
  const misfit = Object.values(health).some((h) => h !== "busy" && h.verdict === "misfit");

  return (
    <>
      {/* where the lens values come from: this decides what the cards below ask for -------- */}
      <Part>
        <div className="grid gap-2">
          <div className="text-base font-semibold">Where do the lens values come from?</div>
          {strict && <Explain>Strict checking keeps lens values exactly as you give them. It does not work them out from these photos.</Explain>}
          <div className="grid gap-3 sm:grid-cols-3" role="radiogroup" aria-label="Where the lens values come from">
            {([{ id: "auto", title: "Work them out", text: "Studio computes them from these photos.", off: strict },
              { id: "file", title: "From a lens file", text: "A calibrated_cameras_data.yml from MC-Calib or DS-MSP.", off: false },
              { id: "typed", title: "I'll type them in", text: "No such file: fill in each camera's card below.", off: false }] as { id: LensMode; title: string; text: string; off: boolean }[]).map((o) => {
              const on = lensMode === o.id && !o.off;
              return (
                <button key={o.id} type="button" role="radio" aria-checked={on} disabled={o.off} onClick={() => onLensMode(o.id)}
                  className={`text-left p-3.5 grid gap-1 ${o.off ? "rounded-xl border-2 border-s-rule opacity-50 cursor-not-allowed" : choiceCard(on)}`}>
                  <span className="text-base font-semibold">{o.title}</span>
                  <span className="text-[15px] text-s-muted leading-relaxed">{o.off ? "Not available with strict checking." : o.text}</span>
                </button>
              );
            })}
          </div>
        </div>
        {lensMode === "file" && (
          <div className="flex flex-wrap items-center justify-end gap-3">
            {lens && <Button kind="quiet" onClick={onLensRemove}>Remove</Button>}
            <Button onClick={onLensBrowse}>{lens ? "Choose another file" : "Choose the file"}</Button>
          </div>
        )}
        {lensReading && <Callout tone="info">Reading {lensReading}…</Callout>}
        {lensError && (
          <Callout tone="bad" title="That file was not used">
            {lensError}{lens ? ` Studio keeps using ${lens.path.split("/").pop()}.` : ""}
          </Callout>
        )}
      </Part>

      {canMeasure && (
        <div className="rounded-xl bg-s-info-soft border border-s-info/25 p-4 flex flex-wrap items-center gap-x-5 gap-y-3">
          <div className="min-w-0 mr-auto">
            <div className="text-base font-semibold text-s-info">Let Studio choose the lens types</div>
            <div className="text-[15px] text-s-text leading-relaxed">
              It measures how wide each lens sees on a few of your {measureWhat} and picks the type that fits. About a second per camera.
            </div>
          </div>
          <Button kind="primary" onClick={onMeasure} busy={measuring}>{measured.length ? "Measure again" : "Measure my lenses"}</Button>
        </div>
      )}
      {measured.length > 0 && !measuring && widths.length > 0 && (
        <Callout tone="ok" title={`Measured ${measured.length} of ${n} lenses`}>
          They see between {Math.round(Math.min(...widths))}° and {Math.round(Math.max(...widths))}° across. Each camera below now has the type that fits it; you can still change any of them.
        </Callout>
      )}

      {n > 1 && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <span className="text-base font-semibold">Set every camera to</span>
          <Segmented label="Lens type for every camera" value={(alike ? models[0] ?? "radtan" : "") as string}
            options={LENSES.slice(0, 2).map((l) => ({ id: l.id, title: l.label }))} onChange={setAll} />
          {!alike && <span className="text-[15px] text-s-muted">The cameras now have different types.</span>}
        </div>
      )}

      <div className={`grid gap-4 grid-cols-1 sm:grid-cols-2 ${lensMode === "typed" ? "" : "xl:grid-cols-3"}`}>
        {cameras.map((c) => {
          const current = models[c.index] ?? "radtan";
          const s = samples[c.index];
          const w = s?.width ?? c.width ?? 4;
          const h = s?.height ?? c.height ?? 3;
          const r = Math.max(w, h) / 110;
          return (
            <article key={c.index} className="rounded-xl border border-s-rule bg-s-surface overflow-hidden grid content-start transition hover:border-s-accent/45 hover:shadow-md focus-within:border-s-accent focus-within:ring-4 focus-within:ring-s-accent/10">
              {photos ? (
                <div className="relative bg-s-viewport" style={{ aspectRatio: `${w} / ${h}` }}>
                  <img src={api.thumbUrl(photos.root, c.index, s?.photo ?? 0, photos.prefix)} alt={`A photo taken by ${c.name}`} loading="lazy" className="absolute inset-0 size-full object-cover" />
                  {s && s.corners.length > 0 && (
                    <svg viewBox={`0 0 ${w} ${h}`} className="absolute inset-0 size-full" aria-hidden="true">
                      {s.corners.map(([u, v], k) => <circle key={k} cx={u} cy={v} r={r} fill="#34d399" stroke="#022c22" strokeWidth={r / 3} />)}
                    </svg>
                  )}
                </div>
              ) : (
                <div className="h-2 bg-s-accent/70" aria-hidden="true" />
              )}
              <div className="p-3.5 grid gap-2.5">
                <div className="flex flex-wrap items-baseline gap-x-3">
                  <h3 className="text-base font-bold">{c.name}</h3>
                  <span className="text-[15px] text-s-muted tnum">
                    {c.photos != null ? `${int(c.photos)} photos` : ""}{c.width ? ` · ${c.width} × ${c.height}` : ""}
                  </span>
                </div>
                <label className="grid gap-1">
                  <span className="text-[15px] font-medium text-s-muted">Lens type</span>
                  <select className="w-full rounded-lg border-[1.5px] border-s-field-line bg-s-field px-3 min-h-11 text-base font-semibold cursor-pointer outline-none hover:border-s-strong/60 focus:border-s-accent focus:ring-4 focus:ring-s-accent/20"
                    aria-label={`Lens type of ${c.name}`} value={current} onChange={(e) => setModel(c.index, e.target.value)}>
                    {LENSES.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
                  </select>
                </label>
                <AdviceLine a={advice[c.index]} current={current} onUse={(m) => setModel(c.index, m)} typing={lensMode === "typed"} />
                {lensMode === "typed" && (
                  <div className="grid gap-2 border-t border-s-rule pt-2.5">
                    <div className="text-[15px] font-semibold">Lens values of this camera</div>
                    <div className="grid grid-cols-2 gap-2">
                      {[...lensFields(current), "width", "height"].map((f) => {
                        const fallback = f === "width" ? c.width : f === "height" ? c.height : null;
                        const text = typed[c.index]?.[f] ?? (fallback != null ? String(fallback) : "");
                        const bad = text.trim() === "" || !Number.isFinite(Number(text));
                        return (
                          <label key={f} className="grid gap-0.5 text-[14px] font-medium text-s-muted">
                            <span className="font-studio-mono">{f}{f === "width" || f === "height" ? " (px)" : ""}</span>
                            <input inputMode="decimal" aria-label={`${f} of ${c.name}`} value={text} onChange={(e) => onTyped(c.index, f, e.target.value)}
                              className={`w-full min-w-0 rounded-md border-[1.5px] bg-s-field px-2 min-h-10 text-[14px] font-studio-mono outline-none focus:border-s-accent ${bad ? "border-s-warn" : "border-s-field-line"}`} />
                          </label>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </article>
          );
        })}
      </div>

      <details className="group">
        <summary className="cursor-pointer text-base font-semibold text-s-accent hover:underline">Which lens type is which?</summary>
        <dl className="mt-3 grid gap-2 sm:grid-cols-[auto_minmax(0,1fr)] gap-x-5 text-[15px]">
          {LENSES.map((l) => <Row key={l.id} k={l.label}>{l.hint}</Row>)}
        </dl>
        <Explain className="mt-3">A type that cannot bend enough for the lens still gives a normal-looking error and a wrong rig. If you are unsure about a wide lens, choose Fisheye.</Explain>
      </details>

      {/* what was read or typed, and what to do with it ------------------------------------ */}
      {lens && (
        <Part>
          <>
            {lensMode === "typed" ? <Callout tone="ok" title="Your values are complete">Each camera is being checked against your {measureWhat}. They are saved with the result.</Callout> : <Callout tone="ok" title={`Using ${lens.path.split("/").pop()}`}>
              It holds lens values for {lens.cameras.length} camera{lens.cameras.length === 1 ? "" : "s"}. The lens types of the cameras above were set from it, and each camera's values are checked against your {measureWhat} below.
              <span className="block font-studio-mono text-[13px] text-s-muted break-all mt-1">{lens.path}</span>
            </Callout>}
            <ul className="grid gap-2">
              {lens.cameras.slice(0, 30).map((c) => {
                const h = health[c.index];
                const bad = h && h !== "busy" && h.verdict === "misfit";
                return (
                  <li key={c.index} className={`rounded-lg px-3.5 py-2.5 text-[15px] flex flex-wrap items-baseline gap-x-4 gap-y-1 ${bad ? "bg-s-bad-soft" : "bg-s-sunken"}`}>
                    <b className="font-semibold">camera {c.index}</b>
                    <span className="text-s-muted tnum">{c.model ? lensLabel(c.model) : "lens type not stated"} · focal length {num(c.fx, 0)} px</span>
                    {h === "busy" && <span className="text-s-info">checking…</span>}
                    {h && h !== "busy" && (
                      <span className={`font-medium ${h.verdict === "ok" ? "text-s-ok" : h.verdict === "misfit" ? "text-s-bad" : "text-s-muted"}`}>
                        {h.verdict === "ok" ? "✓ " : h.verdict === "misfit" ? "✗ " : ""}{h.message}
                      </span>
                    )}
                  </li>
                );
              })}
              {lens.cameras.length > 30 && <li className="text-[15px] text-s-muted">and {lens.cameras.length - 30} more</li>}
            </ul>
            {misfit && (
              <Callout tone="bad" title={`Some of these values do not fit your ${measureWhat}`}>
                Keeping them fixed would give a wrong rig. {lensMode === "typed" ? "Check what you typed" : "Check that the file belongs to these cameras"}{strict ? "." : ", or let Studio refine them."}
              </Callout>
            )}
            {lens.cameras.length < n && <Callout tone="warn">The file covers {lens.cameras.length} of your {n} cameras. The others are worked out from the {measureWhat}.</Callout>}
            <div className="grid gap-1.5">
              <div className="text-base font-semibold">What should Studio do with them?</div>
              <div className="grid gap-3 sm:grid-cols-2">
                {[...(strict ? [] : [{ v: false, title: "Start from them and improve them", text: "The lens values are adjusted to fit this capture. Choose this unless you trust the file more than these photos." }]),
                  { v: true, title: "Keep them exactly as they are", text: "Only the camera positions are worked out. Every camera must be in the file." }].map((o) => {
                  const on = fixed === o.v;
                  return (
                    <button key={o.title} type="button" role="radio" aria-checked={on} onClick={() => onFixed(o.v)}
                      className={`text-left p-4 grid gap-1 ${choiceCard(on)}`}>
                      <span className="text-base font-semibold flex items-center gap-2">
                        <span className={`size-5 rounded-full border-2 grid place-items-center ${on ? "border-s-accent" : "border-s-field-line"}`} aria-hidden="true">{on && <span className="size-2.5 rounded-full bg-s-accent" />}</span>
                        {o.title}
                      </span>
                      <span className="text-[15px] text-s-muted leading-relaxed">{o.text}</span>
                    </button>
                  );
                })}
              </div>
              {lensProblem && <div className="text-[15px] font-medium text-s-bad" role="alert">{lensProblem}</div>}
            </div>
          </>
        </Part>
      )}
      {problem && <div className="text-[15px] font-medium text-s-bad" role="alert">{problem}</div>}
    </>
  );
}

function Row({ k, children }: { k: string; children: ReactNode }) {
  return <><dt className="font-semibold">{k}</dt><dd className="text-s-muted">{children}</dd></>;
}
