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
      <div className="p-4">{children}</div>
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
