// Board flatness: how far each solved board departs from a plane, in target units.

import type { RunDoc } from "./api";
import { int, num } from "./format";
import { Card, Note } from "./ui";

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
          To measure a bent board, calibrate with structure_mode: bend.
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
                <tr key={`${r.o}-${r.b}`} className="border-t border-s-line">
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
