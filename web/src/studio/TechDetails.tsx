import type { RunDoc } from "./api";
import { api } from "./api";
import { bytes, num } from "./format";
import { Card, Note } from "./ui";

export function TechDetails({ doc }: { doc: RunDoc }) {
  const acc = doc.acceptance;
  const stages = acc?.solver?.stages ?? [];
  const resolved = doc.manifest?.configuration?.resolved ?? {};
  const scalar = Object.entries(resolved).filter(([, v]) => v === null || typeof v !== "object");
  const gates: [string, string | undefined][] = [
    ["Profile", acc?.profile],
    ["Lens check", acc?.lens_gate],
    ["Split check", acc?.crossval_gate],
    ["Physical accuracy", acc?.accuracy_status?.toLowerCase().replace(/_/g, " ")],
  ];
  return (
    <Card title="Technical details">
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="grid gap-4 content-start">
          <div className="grid gap-1.5">
            <h3 className="text-sm font-semibold">Checks</h3>
            <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1 text-sm">
              {gates.map(([k, v]) => (
                <div key={k} className="contents">
                  <dt className="text-s-faint">{k}</dt>
                  <dd>{v ?? "—"}</dd>
                </div>
              ))}
            </dl>
          </div>
          {stages.length > 0 && (
            <div className="grid gap-1.5">
              <h3 className="text-sm font-semibold">Solver stages</h3>
              <table className="text-sm tnum">
                <tbody>
                  {stages.map((s, i) => (
                    <tr key={`${s.stage}-${i}`} className="border-t border-s-rule">
                      <td className="pr-4 py-1 font-studio-mono text-xs">{s.stage}</td>
                      <td className="pr-4 py-1">{s.status?.toLowerCase().replace(/_/g, " ")}</td>
                      <td className="pr-4 py-1 text-right text-s-muted">{s.iterations} it</td>
                      <td className="py-1 text-right text-s-muted">{num(s.rms, 3)} px</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {(acc?.config_warnings ?? []).length > 0 && (
            <div className="grid gap-1.5">
              <h3 className="text-sm font-semibold">Config warnings</h3>
              <ul className="grid gap-1 text-xs text-s-muted list-disc pl-4">
                {acc!.config_warnings!.map((w) => <li key={w}>{w}</li>)}
              </ul>
            </div>
          )}
        </div>
        <div className="grid gap-4 content-start">
          <div className="grid gap-1.5">
            <h3 className="text-sm font-semibold">Files</h3>
            <ul className="grid gap-1 text-sm">
              {doc.files.map((f) => (
                <li key={f.name} className="flex gap-3">
                  <a className="font-studio-mono text-xs text-s-accent hover:underline break-all" href={api.fileUrl(doc.id, f.name)} target="_blank" rel="noreferrer">
                    {f.name}
                  </a>
                  <span className="ml-auto text-xs text-s-faint tnum">{bytes(f.bytes)}</span>
                </li>
              ))}
            </ul>
          </div>
          {scalar.length > 0 && (
            <details>
              <summary className="cursor-pointer text-sm font-semibold text-s-accent">Resolved configuration</summary>
              <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5 text-xs font-studio-mono">
                {scalar.map(([k, v]) => (
                  <div key={k} className="contents">
                    <dt className="text-s-faint">{k}</dt>
                    <dd className="break-all">{v === null ? "—" : String(v)}</dd>
                  </div>
                ))}
              </dl>
            </details>
          )}
          <Note>Run id {doc.manifest?.run_id ?? "—"}. Studio reads these files and never changes them.</Note>
        </div>
      </div>
    </Card>
  );
}
