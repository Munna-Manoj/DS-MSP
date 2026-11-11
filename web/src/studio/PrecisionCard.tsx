// How far this result is from the precision the user needs, and what more photos of the same kind
// would do. The answer comes first; each limit is one bar with the limit marked on it.
import { useEffect, useState } from "react";

import { api, type PrecisionDoc, type RunDoc } from "./api";
import { camName, int, num } from "./format";
import { NumberBox } from "./setup/kit";
import { length, useUnit } from "./stage/units";
import { Callout, Card, Note } from "./ui";

const pairName = (key: string) => { const [a, b] = key.split("-"); return `${camName(a)} and ${camName(b)}`; };

/** The measured value on a scale where the limit sits at a fixed place. */
function Bar({ label, have, want, ratio, pair }: { label: string; have: string; want: string; ratio: number; pair: string }) {
  const over = ratio > 1;
  // The limit stays at 40% of the width; a result up to 2.5 times the limit fits the bar.
  const fill = Math.min(100, ratio * 40);
  return (
    <div className="grid gap-1.5">
      <div className="flex flex-wrap items-baseline gap-x-3">
        <span className="text-base font-semibold">{label}</span>
        <span className={`text-[15px] font-semibold tnum ${over ? "text-s-bad" : "text-s-ok"}`}>
          {have} · {over ? `${num(ratio, ratio >= 10 ? 0 : 1)}× what you asked for` : "within what you asked for"}
        </span>
      </div>
      <div className="relative h-4 rounded bg-s-sunken" role="img" aria-label={`${label}: ${have}, limit ${want}`}>
        <div className={`absolute inset-y-0 left-0 rounded ${over ? "bg-s-bad/70" : "bg-s-ok/70"}`} style={{ width: `${fill}%` }} />
        <div className="absolute -inset-y-1 w-0.5 bg-s-text" style={{ left: "40%" }} />
      </div>
      <div className="text-[14px] text-s-muted tnum">You asked for {want}. Weakest pair: {pair}.</div>
    </div>
  );
}

export function PrecisionCard({ doc, runId }: { doc: RunDoc; runId: string }) {
  const [unit] = useUnit();
  const limits = doc.acceptance?.extrinsic_uncertainty?.limits;
  const [rot, setRot] = useState<number | "">(limits?.max_rotation_sigma_deg ?? "");
  const [ctr, setCtr] = useState<number | "">(limits?.max_centre_sigma ?? "");
  const [gap, setGap] = useState<PrecisionDoc | null>(null);

  useEffect(() => {
    let live = true;
    const t = window.setTimeout(() => {
      api.precision(runId, rot === "" ? null : rot, ctr === "" ? null : ctr).then(
        (d) => { if (live) setGap(d); }, () => { if (live) setGap(null); });
    }, 250);
    return () => { live = false; window.clearTimeout(t); };
  }, [runId, rot, ctr]);

  if (!doc.acceptance?.extrinsic_uncertainty?.pairs || Object.keys(doc.acceptance.extrinsic_uncertainty.pairs).length === 0) return null;
  const asked = rot !== "" || ctr !== "";
  const wr = gap?.worst_rotation, wc = gap?.worst_centre;
  const over = gap?.n_over ?? 0;
  const pairs = gap?.n_pairs ?? 0;
  const undetermined = gap?.n_undetermined ?? 0;

  return (
    <Card title="How close is this to what you need?">
      <div className="grid gap-4">
        <div className="flex flex-wrap gap-x-8 gap-y-3">
          <label className="grid gap-1.5 text-[15px] font-semibold">Angle between two cameras, to within
            <NumberBox id="need-rotation" value={rot} fallback={null} unit="°" onChange={setRot} />
          </label>
          <label className="grid gap-1.5 text-[15px] font-semibold">Position of one camera from another, to within
            <NumberBox id="need-centre" value={ctr} fallback={null} unit={unit === "?" ? undefined : unit} onChange={setCtr} />
          </label>
        </div>
        {!asked && <Note>Type what your application needs, in the unit of your square size for position. Studio shows how far this result is from it.</Note>}
        {asked && gap?.ok && (
          <>
            {undetermined > 0
              ? <Callout tone="bad" title={`${int(undetermined)} of ${int(pairs)} camera pairs are not determined by these photos`}>More photos like these will not help. Take photos that both cameras in each pair see at the same time.</Callout>
              : over === 0
                ? <Callout tone="ok" title={pairs === 1 ? "This result is within what you asked for" : `All ${int(pairs)} camera pairs are within what you asked for`} />
                : <Callout tone="warn" title={pairs === 1 ? "This result is not yet as precise as you asked" : `${int(over)} of ${int(pairs)} camera pairs are not yet as precise as you asked`} />}
            {rot !== "" && wr?.rotation != null && <Bar label="Angle" have={`±${num(wr.rotation, 3)}°`} want={`±${num(rot, 3)}°`} ratio={wr.rotation / rot} pair={pairName(wr.pair)} />}
            {ctr !== "" && wc?.centre != null && <Bar label="Position" have={`±${length(wc.centre, unit)}`} want={`±${length(ctr, unit)}`} ratio={wc.centre / ctr} pair={pairName(wc.pair)} />}
            {gap.frames_needed != null && gap.times != null && (
              <div className="rounded-xl bg-s-info-soft border border-s-info/25 p-4 grid gap-1.5">
                <div className="text-base font-semibold text-s-info">With more photos like these: about {int(gap.frames_needed)} moments in total</div>
                <div className="text-[15px] text-s-text leading-relaxed">You have {int(gap.frames ?? 0)}, so that is about {num(gap.times, gap.times >= 10 ? 0 : 1)} times as many. This assumes the new photos are as varied as these, and it is rough on a small capture.</div>
                <div className="text-[15px] text-s-text leading-relaxed">Photos that add new angles and reach new parts of the image can need far fewer. Studio cannot yet suggest where to hold the board.</div>
              </div>
            )}
            <Note>One standard deviation from photo noise only. On real captures the true error has been 2 to 3 times larger, so leave a margin.</Note>
          </>
        )}
      </div>
    </Card>
  );
}
