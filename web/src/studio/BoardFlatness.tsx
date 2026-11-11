// Board flatness: how far each solved board departs from a plane, in target units.

import type { RunDoc } from "./api";
import { int, num } from "./format";
import { length, useUnit } from "./stage/units";
import { Callout, Card, Note } from "./ui";

/** When the calibration was asked to measure the board's bow, say what it found, up front. */
export function BowFinding({ doc }: { doc: RunDoc }) {
  const [unit] = useUnit();
  const bf = doc.acceptance?.diagnostics?.board_flatness;
  const mode = (doc.acceptance as { structure_mode?: string } | null)?.structure_mode;
  if (!bf || mode !== "bend") return null;
  const bows = Object.values(bf).flatMap((per) => Object.values(per)).map((v) => v.sag_ptp ?? 0);
  if (!bows.length) return null;
  const worst = Math.max(...bows);
  // without a known unit, a fixed number of decimals would show a small bow as zero
  const shown = unit === "?" ? `${Number(worst.toPrecision(2))} (in the unit of your square size)` : length(worst, unit);
  return (
    <Callout tone="info" title={bows.length === 1
      ? `Your board bows by ${shown} from its lowest to its highest point`
      : `Your ${bows.length} boards bow by up to ${shown} from lowest to highest point`}>
      You asked Studio to measure this, and the calibration takes the bow into account. A bow of a small fraction of a millimetre can also come from corner noise on a board that is flat.
    </Callout>
  );
}

export function BoardFlatness({ doc }: { doc: RunDoc }) {
  const bf = doc.acceptance?.diagnostics?.board_flatness;
  if (!bf) return null;
  const rows = Object.entries(bf).flatMap(([o, per]) => Object.entries(per).map(([b, v]) => ({ o, b, ...v })));
  if (!rows.length) return null;
  const mode = (doc.acceptance as { structure_mode?: string } | null)?.structure_mode;
  const bent = rows.some((r) => (r.sag_ptp ?? 0) > 1e-9);
  return (
    <Card title="Calibration board" aside={mode ? `target refined: ${mode}` : undefined}>
      {!bent ? (
        <Note>
          Each board was held flat in this calibration ({rows.length} board{rows.length === 1 ? "" : "s"}).
          To measure whether a board bows, choose "may be slightly bowed" for it when you set up the calibration.
        </Note>
      ) : (
        <div className="grid gap-2">
          <table className="text-sm tnum">
            <thead>
              <tr className="text-left text-xs text-s-faint">
                <th className="pr-6 py-1 font-medium">Board</th>
                <th className="pr-6 py-1 font-medium text-right">Bow, peak to valley</th>
                <th className="pr-6 py-1 font-medium text-right">RMS</th>
                <th className="py-1 font-medium text-right">Corners</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={`${r.o}-${r.b}`} className="border-t border-s-rule">
                  <td className="pr-6 py-1 font-studio-mono">{r.b}</td>
                  <td className="pr-6 py-1 text-right">{num(r.sag_ptp, 3)}</td>
                  <td className="pr-6 py-1 text-right">{num(r.rms, 3)}</td>
                  <td className="py-1 text-right">{int(r.n_corners)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <Note>In the board's length units (those of square_size). A bow comparable to the corner noise is fine; a larger one means the board is not flat: remount it on a rigid flat backing.</Note>
        </div>
      )}
    </Card>
  );
}
