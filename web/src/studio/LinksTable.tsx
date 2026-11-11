import { useState } from "react";
import type { RunDoc } from "./api";
import { camName, num } from "./format";
import { Card, Note } from "./ui";

const SHOWN = 12;

export function LinksTable({ doc }: { doc: RunDoc }) {
  const [all, setAll] = useState(false);
  const unc = doc.acceptance?.extrinsic_uncertainty;
  const pairs = Object.entries(unc?.pairs ?? {})
    .map(([key, p]) => {
      const [a, b] = key.split("-");
      return { key, a, b, rot: p.rotation_sigma_deg, pos: p.centre_sigma };
    })
    .sort((x, y) => (y.rot ?? Infinity) - (x.rot ?? Infinity));
  if (!unc || pairs.length === 0)
    return (
      <Card title="How well cameras agree">
        <Note>This result has no extrinsic uncertainty (the uncertainty check was off).</Note>
      </Card>
    );
  const unit = unc.units?.centre ?? "scene length units";
  const rows = all ? pairs : pairs.slice(0, SHOWN);
  return (
    <Card title="How well cameras agree" aside={`${pairs.length} pairs, weakest first`}>
      <div className="grid gap-3">
        <div className="overflow-x-auto">
          <table className="w-full text-sm tnum">
            <thead>
              <tr className="text-left text-xs text-s-faint border-b border-s-rule">
                <th className="py-2 pr-4 font-medium">Cameras</th>
                <th className="py-2 pr-4 font-medium text-right">Angle uncertainty</th>
                <th className="py-2 font-medium text-right">Position uncertainty</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((p) => (
                <tr key={p.key} className="border-b border-s-rule last:border-0">
                  <td className="py-1.5 pr-4">{camName(p.a)} and {camName(p.b)}</td>
                  <td className="py-1.5 pr-4 text-right">{p.rot == null ? "unbounded" : `at least ±${num(p.rot, 3)}°`}</td>
                  <td className="py-1.5 text-right">{p.pos == null ? "unbounded" : `at least ±${num(p.pos, 4)}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {pairs.length > SHOWN && (
          <button type="button" onClick={() => setAll(!all)} className="justify-self-start text-sm text-s-accent hover:underline">
            {all ? "Show fewer" : `Show all ${pairs.length} pairs`}
          </button>
        )}
        <Note>
          One standard deviation from photo noise only, so read it as a lower bound: on real
          captures the true error has been 2 to 3 times larger. Position is in {unit}.
        </Note>
      </div>
    </Card>
  );
}
