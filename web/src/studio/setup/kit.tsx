// The pieces of the setup page. Each kind of thing has one look, so a reader can tell them
// apart at a glance: a box to fill is white with a firm outline, an explanation is plain grey
// text with no box, something Studio worked out sits on a tinted panel with a mark, and
// anything that can be clicked has a border or a solid colour and reacts to the pointer.
import { useEffect, useState, type ReactNode } from "react";

import { INPUT } from "../ui";

/** One numbered step of the setup, with whether it is done. */
export function Step({ n, title, state, aside, children }: {
  n: number | string; title: ReactNode; state?: "done" | "todo" | "optional"; aside?: ReactNode; children: ReactNode;
}) {
  return (
    <section className="rounded-2xl bg-s-surface shadow-[0_1px_2px_rgb(15_23_42/0.06),0_4px_16px_rgb(15_23_42/0.05)] border border-s-line min-w-0">
      <header className="flex flex-wrap items-center gap-3 px-5 sm:px-6 pt-5">
        <span className={`size-9 shrink-0 rounded-full grid place-items-center text-base font-bold ${state === "done" ? "bg-s-ok text-white" : "bg-s-accent text-white"}`} aria-hidden="true">
          {state === "done" ? "✓" : n}
        </span>
        <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
        {state === "optional" && <span className="rounded-full bg-s-sunken text-s-muted text-[13px] font-medium px-2.5 py-0.5">optional</span>}
        {aside && <div className="ml-auto text-[15px] text-s-muted">{aside}</div>}
      </header>
      <div className="px-5 sm:px-6 pb-6 pt-4 grid gap-5 min-w-0">{children}</div>
    </section>
  );
}

/** An explanation: plain text, never in a box. */
export function Explain({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <p className={`text-[15px] leading-relaxed text-s-muted max-w-[68ch] ${className}`}>{children}</p>;
}

/** Whole numbers chosen with − and +, or typed. */
export function Stepper({ id, value, min = 1, max = 99, onChange, label }: {
  id: string; value: number; min?: number; max?: number; onChange: (v: number) => void; label: string;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => { setText((old) => (Number(old) === value ? old : String(value))); }, [value]);
  const clamp = (v: number) => Math.min(max, Math.max(min, Math.round(v)));
  const btn = "size-12 shrink-0 grid place-items-center text-2xl leading-none font-medium text-s-accent bg-s-surface " +
    "hover:bg-s-accent-soft active:bg-s-accent active:text-white disabled:text-s-faint disabled:hover:bg-s-surface cursor-pointer disabled:cursor-not-allowed";
  return (
    <div className="inline-flex items-stretch rounded-lg border-[1.5px] border-s-field-line bg-s-field overflow-hidden shadow-[inset_0_1px_2px_rgb(15_23_42/0.06)] focus-within:border-s-accent focus-within:ring-4 focus-within:ring-s-accent/20">
      <button type="button" className={`${btn} border-r border-s-line`} aria-label={`One fewer: ${label}`} disabled={value <= min} onClick={() => onChange(clamp(value - 1))}>−</button>
      <input id={id} inputMode="numeric" aria-label={label} value={text}
        className="w-16 text-center text-xl font-semibold tnum bg-transparent outline-none"
        onChange={(e) => { setText(e.target.value); const v = Number(e.target.value); if (e.target.value.trim() !== "" && Number.isFinite(v)) onChange(clamp(v)); }}
        onBlur={() => setText(String(value))} />
      <button type="button" className={`${btn} border-l border-s-line`} aria-label={`One more: ${label}`} disabled={value >= max} onClick={() => onChange(clamp(value + 1))}>+</button>
    </div>
  );
}

/** An on/off switch with its meaning in words beside it. */
export function Toggle({ id, on, onChange, children }: { id: string; on: boolean; onChange: (v: boolean) => void; children?: ReactNode }) {
  return (
    <label htmlFor={id} className="inline-flex items-center gap-3 cursor-pointer select-none">
      <input id={id} type="checkbox" role="switch" checked={on} onChange={(e) => onChange(e.target.checked)} className="peer sr-only" />
      <span className={`relative w-12 h-7 rounded-full transition-colors peer-focus-visible:ring-4 peer-focus-visible:ring-s-accent/30 ${on ? "bg-s-accent" : "bg-s-field-line"}`} aria-hidden="true">
        <span className={`absolute top-0.5 left-0.5 size-6 rounded-full bg-white shadow transition-transform ${on ? "translate-x-5" : ""}`} />
      </span>
      {children && <span className="text-base">{children}</span>}
    </label>
  );
}

/** A few choices side by side; one is always selected. */
export function Segmented<T extends string>({ label, value, options, onChange }: {
  label: string; value: T; options: { id: T; title: string; tone?: "bad" | "warn" }[]; onChange: (v: T) => void;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-lg border-[1.5px] border-s-field-line/70 bg-s-sunken p-1 gap-1">
      {options.map((o) => {
        const on = value === o.id;
        const lit = o.tone === "bad" ? "bg-s-bad text-white" : o.tone === "warn" ? "bg-s-warn text-white" : "bg-s-accent text-white";
        return (
          <button key={o.id} type="button" role="radio" aria-checked={on} onClick={() => onChange(o.id)}
            className={`min-h-10 px-4 rounded-md text-[15px] font-semibold cursor-pointer transition ${on ? `${lit} shadow` : "text-s-muted hover:bg-s-surface hover:text-s-text"}`}>
            {o.title}
          </button>
        );
      })}
    </div>
  );
}

/** A number chosen on a slider, shown large with its unit; `words` says what the value means. */
export function Range({ id, value, min, max, step, unit, format, words, onChange, ends }: {
  id: string; value: number; min: number; max: number; step: number; unit?: string;
  format?: (v: number) => string; words?: ReactNode; onChange: (v: number) => void; ends?: [string, string];
}) {
  return (
    <div className="grid gap-1.5">
      <div className="flex items-baseline gap-2">
        <output htmlFor={id} className="text-2xl font-bold tnum text-s-accent">{format ? format(value) : value}</output>
        {unit && <span className="text-base text-s-muted">{unit}</span>}
        {words && <span className="text-[15px] text-s-text ml-2">{words}</span>}
      </div>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-2 accent-s-accent cursor-pointer" />
      {ends && <div className="flex justify-between text-[13px] text-s-faint"><span>{ends[0]}</span><span>{ends[1]}</span></div>}
    </div>
  );
}

/** A number typed in, with its unit written inside the box. Empty means "use the default". */
export function NumberBox({ id, value, fallback, unit, onChange, decimal = true, width = "w-44" }: {
  id: string; value: unknown; fallback?: number | null; unit?: string; onChange: (v: number | "") => void;
  decimal?: boolean; width?: string;
}) {
  const [text, setText] = useState(value == null || value === "" ? "" : String(value));
  useEffect(() => {
    setText((old) => (value == null || value === "" ? (old.trim() === "" || Number(old) === fallback ? old : "") : Number(old) === value ? old : String(value)));
  }, [value, fallback]);
  return (
    <div className={`relative ${width}`}>
      <input id={id} className={`${INPUT} tnum font-semibold ${unit ? "pr-14" : ""}`} inputMode={decimal ? "decimal" : "numeric"}
        value={text} placeholder={fallback != null ? String(fallback) : ""}
        onChange={(e) => {
          setText(e.target.value);
          const t = e.target.value.trim();
          const n = Number(t);
          if (t === "") onChange("");
          else if (Number.isFinite(n)) onChange(n);
        }} />
      {unit && <span className="absolute right-3.5 top-1/2 -translate-y-1/2 text-[15px] text-s-muted pointer-events-none">{unit}</span>}
    </div>
  );
}

/** The name a setting has in calib_param.yml, for people who edit the file by hand. */
export function KeyTag({ name }: { name: string }) {
  return <span className="font-studio-mono text-[13px] text-s-faint" title="The name of this setting in calib_param.yml">{name}</span>;
}

export function Mono({ children }: { children: ReactNode }) {
  return <code className="font-studio-mono text-[0.92em] bg-s-sunken rounded px-1 py-0.5">{children}</code>;
}
