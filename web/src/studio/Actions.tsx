// Layer 1: at most three things to do next, worst first, each pointing at its evidence.

import { useState } from "react";
import { href } from "../router";
import type { RunDoc, VerdictAction } from "./api";
import { camName } from "./format";
import { STATE_LABEL, STATE_TONE } from "./status";
import { Badge, Card, Note } from "./ui";

const EVIDENCE_LABEL: Record<string, string> = {
  lens: "uncertainty map", frames: "worst photos", residuals: "error map", rig: "rig view",
  links: "camera pairs", tech: "technical details",
};

export function Actions({ doc, runId }: { doc: RunDoc; runId: string }) {
  const [all, setAll] = useState(false);
  const acts = doc.verdict?.actions ?? [];
  const max = doc.verdict?.max_shown ?? 3;
  if (!doc.verdict) return null;
  if (acts.length === 0)
    return (
      <Card title="Next steps">
        <Note>Nothing to fix. Every check the calibration records passed.</Note>
      </Card>
    );
  const shown = all ? acts : acts.slice(0, max);
  return (
    <Card title="Next steps" aside={acts.length > max ? `${max} of ${acts.length}` : undefined}>
      <ol className="grid gap-2">
        {shown.map((a, i) => <ActionRow key={i} a={a} runId={runId} />)}
      </ol>
      {acts.length > max && (
        <button type="button" onClick={() => setAll(!all)} className="mt-3 text-sm text-s-accent hover:underline">
          {all ? "Show the first three" : `Show all ${acts.length}`}
        </button>
      )}
    </Card>
  );
}

function ActionRow({ a, runId }: { a: VerdictAction; runId: string }) {
  const cam = a.cams.length === 1 ? a.cams[0] : undefined;
  const target = a.evidence && ["lens", "frames", "residuals"].includes(a.evidence) && cam != null
    ? `${href({ page: "run", id: runId, cam })}` : undefined;
  return (
    <li className="flex gap-3 items-start rounded-md bg-s-sunken px-3 py-2">
      <Badge tone={STATE_TONE[a.level]}>{STATE_LABEL[a.level]}</Badge>
      <div className="grid gap-0.5 min-w-0">
        <div className="text-sm">{a.text}</div>
        <div className="text-xs text-s-muted">
          {a.cams.length > 0 && <span>{a.cams.slice(0, 6).map(camName).join(", ")}{a.cams.length > 6 ? ` and ${a.cams.length - 6} more` : ""}</span>}
          {a.evidence && (
            <span>
              {a.cams.length > 0 ? " · " : ""}see{" "}
              {target ? <a className="text-s-accent hover:underline" href={target}>{EVIDENCE_LABEL[a.evidence]}</a> : EVIDENCE_LABEL[a.evidence]}
            </span>
          )}
        </div>
      </div>
    </li>
  );
}
