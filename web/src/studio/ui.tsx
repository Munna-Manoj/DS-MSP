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
    <section className={`rounded-lg border border-s-line bg-s-surface min-w-0 ${className}`}>
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
  return <p className="text-xs leading-relaxed text-s-muted max-w-prose">{children}</p>;
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
  primary: "bg-s-accent text-white hover:opacity-90 border border-transparent",
  secondary: "bg-s-surface text-s-text border border-s-line hover:bg-s-sunken",
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
      className={`inline-flex items-center justify-center gap-2 rounded-md px-3.5 min-h-10 text-sm font-medium transition
        disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-2 focus-visible:outline-offset-2
        focus-visible:outline-s-accent ${BUTTON[kind]} ${className}`}
    >
      {busy && <span className="size-3.5 rounded-full border-2 border-current border-r-transparent animate-spin" aria-hidden="true" />}
      {children}
    </button>
  );
}

export const INPUT =
  "w-full rounded-md border border-s-line bg-s-surface px-3 min-h-10 text-sm outline-none " +
  "focus:border-s-accent disabled:bg-s-sunken disabled:text-s-faint";

/** A labelled form control with its help text and, when set, its problem. */
export function Field({ label, help, problem, htmlFor, children, className = "" }: {
  label: ReactNode; help?: ReactNode; problem?: string | null; htmlFor: string;
  children: ReactNode; className?: string;
}) {
  return (
    <div className={`grid gap-1 content-start min-w-0 ${className}`}>
      <label htmlFor={htmlFor} className="text-sm font-medium">{label}</label>
      {children}
      {problem
        ? <div className="text-xs text-s-bad" role="alert">{problem}</div>
        : help && <div className="text-xs text-s-muted leading-relaxed">{help}</div>}
    </div>
  );
}

export function Callout({ tone, title, children }: {
  tone: "ok" | "warn" | "bad" | "info"; title?: ReactNode; children?: ReactNode;
}) {
  const box = { ok: "border-s-ok/40 bg-s-ok-soft", warn: "border-s-warn/40 bg-s-warn-soft",
    bad: "border-s-bad/40 bg-s-bad-soft", info: "border-s-line bg-s-sunken" }[tone];
  const head = { ok: "text-s-ok", warn: "text-s-warn", bad: "text-s-bad", info: "text-s-text" }[tone];
  return (
    <div className={`rounded-lg border p-3 text-sm ${box}`} role={tone === "bad" ? "alert" : "status"}>
      {title && <div className={`font-semibold ${head}`}>{title}</div>}
      {children && <div className={`text-s-text ${title ? "mt-1" : ""}`}>{children}</div>}
    </div>
  );
}
