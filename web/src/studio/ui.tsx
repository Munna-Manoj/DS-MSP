import type React from "react";
import type { ReactNode } from "react";

type Tone = "ok" | "warn" | "bad" | "info" | "plain";

const TONE: Record<Tone, string> = {
  ok: "bg-s-ok-soft text-s-ok",
  warn: "bg-s-warn-soft text-s-warn",
  bad: "bg-s-bad-soft text-s-bad",
  info: "bg-s-accent-soft text-s-accent",
  plain: "bg-s-sunken text-s-muted",
};

export function Badge({ tone, children }: { tone: Tone; children: ReactNode }) {
  return (
    <span className={`inline-flex items-center rounded px-2 py-0.5 text-xs font-medium whitespace-nowrap ${TONE[tone]}`}>
      {children}
    </span>
  );
}

export function Card({ title, aside, children, className = "" }: {
  title?: ReactNode; aside?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`rounded-lg border border-s-rule bg-s-surface min-w-0 ${className}`}>
      {title && (
        <div className="flex items-baseline gap-3 px-4 pt-4">
          <h2 className="text-[15px] font-semibold">{title}</h2>
          {aside && <div className="ml-auto text-xs text-s-faint">{aside}</div>}
        </div>
      )}
      <div className="p-4 min-w-0">{children}</div>
    </section>
  );
}

export function Stat({ k, v, sub }: { k: ReactNode; v: ReactNode; sub?: ReactNode }) {
  return (
    <div className="grid gap-0.5 min-w-0">
      <div className="text-xs text-s-faint">{k}</div>
      <div className="text-[15px] font-semibold tnum">{v}</div>
      {sub && <div className="text-xs text-s-muted">{sub}</div>}
    </div>
  );
}

export function Note({ children }: { children: ReactNode }) {
  return <p className="text-[15px] leading-relaxed text-s-muted max-w-prose">{children}</p>;
}

export function Loading({ what }: { what: string }) {
  return <div className="text-s-muted text-sm" role="status">Loading {what}…</div>;
}

export function ErrorBox({ title, detail }: { title: string; detail: string }) {
  return (
    <div role="alert" className="rounded-lg border border-s-bad/40 bg-s-bad-soft text-s-bad p-4 text-sm">
      <div className="font-semibold">{title}</div>
      <div className="font-studio-mono text-xs mt-1 break-all">{detail}</div>
    </div>
  );
}

type ButtonKind = "primary" | "secondary" | "danger" | "quiet";

const BUTTON: Record<ButtonKind, string> = {
  primary: "bg-s-accent text-white shadow-sm hover:brightness-110 border border-transparent",
  secondary: "bg-s-surface text-s-text border-[1.5px] border-s-field-line/70 shadow-sm hover:bg-s-sunken hover:border-s-strong/50",
  danger: "bg-s-surface text-s-bad border border-s-bad/40 hover:bg-s-bad-soft",
  quiet: "text-s-accent hover:underline border border-transparent",
};

export function Button({ kind = "secondary", busy, className = "", children, ...rest }:
  React.ButtonHTMLAttributes<HTMLButtonElement> & { kind?: ButtonKind; busy?: boolean }) {
  return (
    <button
      type="button"
      {...rest}
      disabled={rest.disabled || busy}
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-4 min-h-11 text-[15px] font-semibold transition cursor-pointer active:translate-y-px
        disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-2 focus-visible:outline-offset-2
        focus-visible:outline-s-accent ${BUTTON[kind]} ${className}`}
    >
      {busy && <span className="size-3.5 rounded-full border-2 border-current border-r-transparent animate-spin" aria-hidden="true" />}
      {children}
    </button>
  );
}

export const INPUT =
  "w-full rounded-lg border-[1.5px] border-s-field-line bg-s-field px-3.5 min-h-12 text-base text-s-text " +
  "shadow-[inset_0_1px_2px_rgb(15_23_42/0.06)] outline-none transition placeholder:text-s-faint " +
  "hover:border-s-strong/60 focus:border-s-accent focus:ring-4 focus:ring-s-accent/20 " +
  "disabled:bg-s-sunken disabled:text-s-faint disabled:border-s-rule";

/** A labelled form control with its help text and, when set, its problem. */
export function Field({ label, help, problem, htmlFor, children, className = "" }: {
  label: ReactNode; help?: ReactNode; problem?: string | null; htmlFor: string;
  children: ReactNode; className?: string;
}) {
  return (
    <div className={`grid gap-1.5 content-start min-w-0 ${className}`}>
      <label htmlFor={htmlFor} className="text-base font-semibold">{label}</label>
      {help && <div className="text-[15px] text-s-muted leading-relaxed -mt-0.5">{help}</div>}
      {children}
      {problem && <div className="text-[15px] font-medium text-s-bad" role="alert">{problem}</div>}
    </div>
  );
}

export function Callout({ tone, title, children }: {
  tone: "ok" | "warn" | "bad" | "info"; title?: ReactNode; children?: ReactNode;
}) {
  const box = { ok: "border-s-ok/40 bg-s-ok-soft", warn: "border-s-warn/40 bg-s-warn-soft",
    bad: "border-s-bad/40 bg-s-bad-soft", info: "border-s-info/30 bg-s-info-soft" }[tone];
  const head = { ok: "text-s-ok", warn: "text-s-warn", bad: "text-s-bad", info: "text-s-info" }[tone];
  const dot = { ok: "bg-s-ok", warn: "bg-s-warn", bad: "bg-s-bad", info: "bg-s-info" }[tone];
  const mark = { ok: "✓", warn: "!", bad: "×", info: "i" }[tone];
  return (
    <div className={`rounded-lg border p-3.5 text-[15px] leading-relaxed flex gap-3 ${box}`} role={tone === "bad" ? "alert" : "status"}>
      <span className={`mt-0.5 size-5 shrink-0 rounded-full grid place-items-center text-xs font-bold text-white ${dot}`} aria-hidden="true">{mark}</span>
      <div className="min-w-0">
        {title && <div className={`font-semibold ${head}`}>{title}</div>}
        {children && <div className={`text-s-text ${title ? "mt-0.5" : ""}`}>{children}</div>}
      </div>
    </div>
  );
}
