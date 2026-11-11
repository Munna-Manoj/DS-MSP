// Where the user is in setting up a calibration: the way they started, the steps under it, and
// the parts of the open step under that. Each step keeps its number whatever its state.
import type { ReactNode } from "react";

import { Button } from "../ui";
import { STEP_WORD, type StepState } from "./kit";

export type StepId = "data" | "board" | "lens" | "rules" | "more" | "start";
export type ItemState = "ok" | "todo" | "optional";

export interface TrackedStep {
  id: StepId; n: number; title: string; state: StepState; note: string;
  /** Says the state in other words than usual. */
  word?: string;
  items?: { label: string; state: ItemState }[];
}

const TICK = "M5 12.5l4.5 4.5L19 7.5";

/** A step's number in a circle, in the colour of its state. */
export function StepDot({ step, on, size = "size-9" }: { step: TrackedStep; on: boolean; size?: string }) {
  const fill = step.state === "done" ? "bg-s-ok text-white"
    : step.state === "warn" ? "bg-s-warn text-white"
    : step.state === "fix" ? "bg-s-bad text-white"
    : on ? "bg-s-accent text-white"
    : step.state === "todo" ? "bg-s-surface text-s-accent border-2 border-s-accent"
    : "bg-s-surface text-s-muted border-2 border-s-field-line";
  return (
    <span className={`relative shrink-0 rounded-full grid place-items-center text-[15px] font-bold ${size} ${fill}`} aria-hidden="true">
      {step.n}
      {step.state === "done" && (
        <span className="absolute -right-1 -bottom-1 size-4 rounded-full bg-s-surface grid place-items-center">
          <svg viewBox="0 0 24 24" className="size-3.5 text-s-ok" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round"><path d={TICK} /></svg>
        </span>
      )}
    </span>
  );
}

function ItemMark({ state }: { state: ItemState }) {
  if (state === "ok")
    return <svg viewBox="0 0 24 24" className="size-4 mt-0.5 shrink-0 text-s-ok" fill="none" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={TICK} /></svg>;
  return <span className={`size-3 mt-1 mx-0.5 shrink-0 rounded-full border-2 ${state === "todo" ? "border-s-accent" : "border-s-field-line"}`} aria-hidden="true" />;
}

export function Tracker({ from, icon, onChangeFrom, prelude, steps, active, onGo, headline, done, total }: {
  from: string; icon: string; onChangeFrom: () => void;
  /** What has to be chosen before the steps open, when something does. */
  prelude: string | null;
  steps: TrackedStep[]; active: StepId; onGo: (id: StepId) => void;
  headline: ReactNode; done: number; total: number;
}) {
  const share = total ? Math.round((done / total) * 100) : 0;
  const current = steps.find((s) => s.id === active);
  return (
    <nav aria-label="Setup progress" className="rounded-2xl bg-s-surface border border-s-rule shadow-[0_1px_2px_rgb(15_23_42/0.06),0_8px_24px_rgb(15_23_42/0.06)] overflow-hidden">
      <div className="flex items-center gap-3 px-4 py-3.5 bg-s-accent-soft border-b border-s-accent/20">
        <span className="size-10 shrink-0 rounded-xl grid place-items-center bg-s-accent text-white" aria-hidden="true">
          <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d={icon} /></svg>
        </span>
        <div className="min-w-0 mr-auto">
          <div className="text-[14px] text-s-muted">Starting from</div>
          <div className="text-base font-semibold leading-tight truncate">{from}</div>
        </div>
        <Button kind="quiet" className="!px-2 !min-h-9" onClick={onChangeFrom}>Change</Button>
      </div>

      <div className="px-4 pt-4 grid gap-2" role="status">
        <div className="text-base font-semibold leading-snug">{headline}</div>
        <div className="h-2 rounded-full bg-s-sunken overflow-hidden" role="progressbar" aria-valuemin={0} aria-valuemax={total} aria-valuenow={done} aria-label="Required steps done">
          <div className="h-full rounded-full bg-s-ok transition-all duration-300" style={{ width: `${share}%` }} />
        </div>
        <div className="text-[14px] text-s-muted tnum">{done} of {total} required steps done</div>
      </div>

      {/* phone and tablet: a row of numbers with the open step named under it */}
      <div className="lg:hidden px-4 py-4 grid gap-2">
        <ol className="flex items-center gap-2 overflow-x-auto py-1 px-1">
          {steps.map((s) => (
            <li key={s.id}>
              <button type="button" disabled={!!prelude} aria-current={s.id === active ? "step" : undefined} aria-label={`Step ${s.n}: ${s.title}. ${s.word ?? STEP_WORD[s.state]}`} onClick={() => onGo(s.id)}
                className={`rounded-full cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 ${s.id === active && !prelude ? "ring-4 ring-s-accent/30" : ""}`}>
                <StepDot step={s} on={s.id === active && !prelude} />
              </button>
            </li>
          ))}
        </ol>
        <div className="text-[15px] font-semibold">{prelude ?? current?.title}</div>
      </div>

      {/* wide screens: the whole list */}
      <div className="hidden lg:block p-2 pt-3">
        {prelude && (
          <div className="rounded-xl bg-s-accent-soft px-3 py-2.5 mb-1 text-[15px] font-semibold text-s-accent" aria-current="step">{prelude}</div>
        )}
        <ol className={`grid gap-0.5 ${prelude ? "opacity-55" : ""}`}>
          {steps.map((s, k) => {
            const on = s.id === active && !prelude;
            return (
              <li key={s.id} className="relative">
                {k < steps.length - 1 && <span className="absolute left-[29px] top-11 bottom-[-10px] w-0.5 bg-s-line" aria-hidden="true" />}
                <button type="button" disabled={!!prelude} aria-current={on ? "step" : undefined} onClick={() => onGo(s.id)}
                  className={`relative w-full text-left rounded-xl px-3 py-2 flex items-start gap-3 transition cursor-pointer disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-s-accent/30 ${on ? "bg-s-accent-soft shadow-[inset_3px_0_0_var(--s-accent)]" : "hover:bg-s-sunken enabled:hover:translate-x-0.5"}`}>
                  <StepDot step={s} on={on} />
                  <span className="min-w-0 grid">
                    <span className={`text-[15px] leading-tight mt-0.5 ${on ? "font-bold text-s-accent" : "font-semibold"}`}>{s.title}</span>
                    <span className={`text-[14px] leading-snug mt-0.5 ${s.state === "fix" ? "text-s-bad font-medium" : s.state === "warn" ? "text-s-warn font-medium" : s.state === "todo" ? "text-s-accent" : "text-s-muted"}`}>{s.note}</span>
                  </span>
                </button>
                {on && s.items && s.items.length > 0 && (
                  <ul className="relative ml-[52px] mr-2 mt-1 mb-2 grid gap-1 border-l-2 border-s-accent/30 pl-3">
                    {s.items.map((it) => (
                      <li key={it.label} className={`flex items-start gap-2 text-[14px] leading-snug ${it.state === "todo" ? "font-medium text-s-text" : "text-s-muted"}`}>
                        <ItemMark state={it.state} />
                        <span>{it.label}{it.state === "optional" ? " (optional)" : ""}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            );
          })}
        </ol>
      </div>
    </nav>
  );
}
