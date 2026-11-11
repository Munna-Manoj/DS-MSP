// The LDC meshes and Isaac LUTs a result or an export holds: one row per camera, with what was
// measured, what it means, and the files. A mesh whose shifts were cut off is marked as unusable.
import { useState } from "react";

import { api, type ExportSummary, type LdcRow } from "./api";
import { LensViewer } from "./LensViewer";
import { num } from "./format";
import { Badge, Card, Note } from "./ui";

/** What a measured mesh means, in words. Only a cut-off mesh is called wrong: that is a fact
 * about the file, not a threshold. */
export function ldcVerdict(r: { q3_overflow?: boolean; n_invalid_nodes?: number; max_px?: number | null }):
  { tone: "ok" | "warn" | "bad"; word: string; text: string } {
  if (r.q3_overflow)
    return { tone: "bad", word: "Do not use", text: "Some shifts were too large for the file and were cut off, so the edges are corrected wrongly. Keep less of the view, or make the corrected image smaller." };
  if ((r.n_invalid_nodes ?? 0) > 0)
    return { tone: "warn", word: "Check the edges", text: `${r.n_invalid_nodes} grid points lie outside what this lens sees; those parts of the image stay black. Keeping less of the view removes them.` };
  return { tone: "ok", word: "Usable", text: r.max_px != null && r.max_px > 1 ? "Closer grid points would lower the worst error." : "" };
}

function LdcLine({ r, id }: { r: LdcRow; id: string }) {
  if (r.problem)
    return (
      <tr className="border-t border-s-rule align-top">
        <td className="py-2 pr-3 font-medium">{r.camera}</td>
        <td className="py-2 pr-3" colSpan={4}><Badge tone="bad">Not written</Badge> <span className="text-[14px]">{r.problem}</span></td>
      </tr>
    );
  const v = ldcVerdict(r);
  return (
    <tr className="border-t border-s-rule align-top">
      <td className="py-2 pr-3 font-medium">{r.camera}</td>
      <td className="py-2 pr-3"><Badge tone={v.tone}>{v.word}</Badge>{v.text && <div className="text-[14px] text-s-muted mt-1 max-w-[42ch]">{v.text}</div>}</td>
      <td className="py-2 pr-3 tnum">{r.median_px != null ? num(r.median_px, 3) : "—"} / {r.max_px != null ? num(r.max_px, 2) : "—"}</td>
      <td className="py-2 pr-3 tnum">{r.in_view_fraction != null ? `${Math.round(100 * r.in_view_fraction)} %` : "—"}</td>
      <td className="py-2 whitespace-nowrap">
        {r.record && <a className="text-s-accent hover:underline" href={api.exportFileUrl(id, `ldc/${r.record}`)}>record</a>}
        {r.mesh && <> · <a className="text-s-accent hover:underline" href={api.exportFileUrl(id, `ldc/${r.mesh}`)}>mesh</a></>}
      </td>
    </tr>
  );
}

export function ExportsPanel({ summary, id, photos = false }: { summary: ExportSummary; id: string; photos?: boolean }) {
  const ldc = summary.ldc;
  const lut = summary.isaac_lut;
  const names = [...new Set([...(ldc?.cameras ?? []).filter((r) => r.record), ...(lut?.cameras ?? []).filter((r) => r.manifest)].map((r) => r.camera))];
  const [cam, setCam] = useState("");
  const shown = names.includes(cam) ? cam : names[0] ?? "";
  const kinds = [...(ldc?.cameras.some((r) => r.camera === shown && r.record) ? ["ldc" as const] : []),
    ...(lut?.cameras.some((r) => r.camera === shown && r.manifest) ? ["lut" as const] : [])];
  const tone = (n: string) => {
    const r = ldc?.cameras.find((x) => x.camera === n);
    return r?.record ? ldcVerdict(r).tone : "plain" as const;
  };
  return (
    <>

      {ldc && (
        <Card title="TI LDC meshes" aside={`${ldc.cameras.filter((r) => r.record).length} of ${ldc.cameras.length} written`}>
          <div className="grid gap-3 min-w-0">
            <Note>
              The error is how far the stored mesh sends each corrected pixel from where the lens values say it belongs, in pixels
              of the camera image: the typical value, then the worst. The share is how much of the corrected image shows the scene;
              the rest is black. Not yet checked on a TI board.
            </Note>
            <div className="overflow-x-auto">
              <table className="w-full text-[15px]">
                <thead><tr className="text-left text-[14px] text-s-muted">
                  <th className="pb-1 pr-3 font-medium">Camera</th><th className="pb-1 pr-3 font-medium">Can it be used?</th>
                  <th className="pb-1 pr-3 font-medium">Error px, typical / worst</th><th className="pb-1 pr-3 font-medium">Shows the scene</th>
                  <th className="pb-1 font-medium">Files</th>
                </tr></thead>
                <tbody>{ldc.cameras.map((r) => <LdcLine key={r.camera} r={r} id={id} />)}</tbody>
              </table>
            </div>
          </div>
        </Card>
      )}
      {lut && (
        <Card title="Isaac Sim lens LUTs" aside={`${lut.cameras.filter((r) => r.manifest).length} of ${lut.cameras.length} written`}>
          <ul className="grid gap-1.5 text-[15px]">
            {lut.cameras.map((r) => (
              <li key={r.camera} className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                <span className="font-medium">{r.camera}</span>
                {r.manifest
                  ? <><Badge tone="ok">Written</Badge><a className="text-s-accent hover:underline" href={api.exportFileUrl(id, `isaac_lut/${r.manifest}`)}>manifest</a>
                    {r.texture_size && <span className="text-s-muted tnum">{r.texture_size[0]} × {r.texture_size[1]} texture</span>}</>
                  : <><Badge tone="bad">Not written</Badge><span className="text-[14px]">{r.problem}</span></>}
              </li>
            ))}
          </ul>
        </Card>
      )}
      {shown && kinds.length > 0 && (
        <Card title="See the exports" aside="drawn from the files">
          <LensViewer cameras={names.map((n) => ({ name: n, tone: tone(n) }))} kinds={kinds} camera={shown} onCamera={setCam} filesId={id} photos={photos} />
        </Card>
      )}
    </>
  );
}
