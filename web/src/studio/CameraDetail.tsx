import type { RunDoc } from "./api";
import { camName, int, num, px } from "./format";
import { findingsFor, STATE_LABEL, STATE_TONE, type CamStatus } from "./status";
import { Badge, Card, Note, Stat } from "./ui";

export function CameraDetail({ doc, cam, status }: { doc: RunDoc; cam: string; status?: CamStatus }) {
  const intr = doc.report.intrinsics?.find((i) => String(i.cam_id) === cam);
  const err = doc.report.per_camera_errors_px?.[cam];
  const lens = doc.acceptance?.lens?.cameras?.[cam];
  const cv = doc.acceptance?.crossval;
  const cvCam = cv?.cameras?.[cam];
  const cov = doc.acceptance?.coverage?.[cam];
  const geo = doc.report.audit?.coverage?.[cam];
  const extr = doc.report.extrinsics?.find((e) => String(e.cam_id) === cam);
  const atBound = new Set(lens?.params_at_bound ?? []);
  const camFindings = findingsFor(doc, cam);
  const tone = STATE_TONE[status?.state ?? "accept"];

  return (
    <Card
      title={
        <span className="flex items-center gap-3">
          {camName(cam)}
          {status && <Badge tone={tone}>{STATE_LABEL[status.state]}</Badge>}
        </span>
      }
      aside={intr ? `${intr.model} · ${intr.width} × ${intr.height}` : undefined}
    >
      <div className="grid gap-5">
        {status && status.reasons.length > 0 && (
          <ul className="grid gap-1 text-sm">
            {status.reasons.map((r) => (
              <li key={r} className="flex gap-2"><span aria-hidden className="text-s-warn">●</span>{r}</li>
            ))}
          </ul>
        )}

        <div className="grid gap-4 grid-cols-2 sm:grid-cols-3 lg:grid-cols-6">
          <Stat k="Photos used" v={lens ? int(lens.frames) : "—"} sub={cov ? `${int(cov.n_observed)} corners` : lens ? undefined : "not recorded"} />
          <Stat k="Typical error" v={px(err?.median)} sub={`95% under ${px(err?.p95)}`} />
          <Stat k="Largest error" v={px(err?.max, 1)} sub={err?.inlier_frac != null ? `${num(err.inlier_frac * 100, 1)}% inliers` : undefined} />
          <Stat
            k="Lens uncertainty"
            v={!lens ? "—" : lens.sigma_median_widened_px == null ? "not determined" : `at least ±${num(lens.sigma_median_widened_px)} px`}
            sub={lens ? "photo noise only" : "not recorded in this result"}
          />
          <Stat
            k="Board tilt seen"
            v={geo?.tilt_range_deg != null ? `${num(geo.tilt_range_deg, 0)}°` : "—"}
            sub="range across photos"
          />
          <Stat
            k="Position in rig"
            v={extr ? (extr.is_ref ? "reference" : `${num(extr.baseline)} from ref`) : "—"}
            
            sub={extr && !extr.is_ref ? `scene units · ${num(extr.rot_deg, 1)}° turned` : undefined}
          />
        </div>

        {cv && (
          <div className="text-sm">
            <span className="font-medium">Split check: </span>
            {!cv.applicable || !cvCam?.applicable ? (
              <span className="text-s-muted">not applicable ({cv.reason ?? `${cvCam?.frames_halves?.join(" / ")} frames per half`})</span>
            ) : (
              <span>
                halves agree to ≈{px(cvCam.error_estimate_px)}
                {cvCam.ok === false && <span className="text-s-warn"> (over the {px(cv.limit_px)} limit)</span>}
                <span className="text-s-muted"> · two halves of one capture; a second capture is the stronger test</span>
              </span>
            )}
          </div>
        )}

        {camFindings.length > 0 && (
          <div className="grid gap-1.5">
            <h3 className="text-sm font-semibold">What the audit found</h3>
            <ul className="grid gap-1.5 text-sm">
              {camFindings.map((f, i) => (
                <li key={i} className="rounded-md bg-s-sunken px-3 py-2">
                  <span className="font-studio-mono text-xs text-s-faint">{f.kind}</span>
                  <div>{f.message ?? "weakly determined"}</div>
                </li>
              ))}
            </ul>
          </div>
        )}

        {intr && (
          <details className="group">
            <summary className="cursor-pointer text-sm font-semibold text-s-accent">Lens parameters</summary>
            <div className="mt-3 overflow-x-auto">
              <table className="text-sm tnum">
                <thead>
                  <tr className="text-left text-xs text-s-faint">
                    <th className="pr-6 py-1 font-medium">Parameter</th>
                    <th className="pr-6 py-1 font-medium text-right">Value</th>
                    <th className="pr-6 py-1 font-medium text-right">± 1σ</th>
                    <th className="py-1 font-medium" />
                  </tr>
                </thead>
                <tbody>
                  {intr.params.map((p) => (
                    <tr key={p.name} className="border-t border-s-line">
                      <td className="pr-6 py-1 font-studio-mono">{p.name}</td>
                      <td className="pr-6 py-1 text-right font-studio-mono">{p.value.toPrecision(7)}</td>
                      <td className="pr-6 py-1 text-right font-studio-mono text-s-muted">{p.sigma == null ? "—" : p.sigma.toPrecision(2)}</td>
                      <td className="py-1">{atBound.has(p.name) && <Badge tone="warn">at its limit</Badge>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-2">
              <Note>
                The ± value is one standard deviation from photo noise only; the real error can be
                larger. A parameter at its limit means the photos could not pin it down.
              </Note>
            </div>
          </details>
        )}
      </div>
    </Card>
  );
}
