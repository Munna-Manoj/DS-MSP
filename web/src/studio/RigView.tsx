// 3D view of the calibrated rig: one frustum per camera, in the reference camera's frame.
// T_c_ref maps reference coordinates to camera coordinates (X_c = R X_ref + t), so the
// camera centre is -Rᵀt and a camera-frame point p sits at Rᵀp + C in the reference frame.
// Camera frames are x right, y down, z forward; the view shows y up, so (x, y, z) -> (x, -y, -z).
import { Html, Line, OrbitControls } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useState } from "react";
import type { RunDoc } from "./api";
import { camName } from "./format";
import { cameraStatus, camIds } from "./status";

type V3 = [number, number, number];

const toView = (p: number[]): V3 => [p[0], -p[1], -p[2]];

function frustum(T: number[][], depth: number, aspect: number) {
  const R = [T[0].slice(0, 3), T[1].slice(0, 3), T[2].slice(0, 3)];
  const t = [T[0][3], T[1][3], T[2][3]];
  const C = [0, 1, 2].map((j) => -(R[0][j] * t[0] + R[1][j] * t[1] + R[2][j] * t[2]));
  const toRef = (p: number[]) => [0, 1, 2].map((j) => R[0][j] * p[0] + R[1][j] * p[1] + R[2][j] * p[2] + C[j]);
  const w = depth * 0.6;
  const h = w / aspect;
  const corners = [[-w, -h, depth], [w, -h, depth], [w, h, depth], [-w, h, depth]].map(toRef);
  const up = toRef([0, -h * 1.5, depth]);
  const c = toView(C);
  const k = corners.map(toView);
  const segs: V3[] = [c, k[0], c, k[1], c, k[2], c, k[3], k[0], k[1], k[1], k[2], k[2], k[3], k[3], k[0], k[0], toView(up), toView(up), k[1]];
  return { centre: c, segs };
}

function ViewSetter({ mode, centre, extent }: { mode: "orbit" | "top"; centre: V3; extent: number }) {
  const { camera } = useThree();
  useEffect(() => {
    const d = extent * 1.8 + 1e-6;
    if (mode === "top") camera.position.set(centre[0], centre[1] + d, centre[2] + 1e-3 * d);
    else camera.position.set(centre[0] + d * 0.7, centre[1] + d * 0.6, centre[2] + d * 0.9);
    camera.lookAt(...centre);
    camera.updateProjectionMatrix();
  }, [mode, centre, extent, camera]);
  return null;
}

export default function RigView({ doc, selected }: { doc: RunDoc; selected: number }) {
  const [mode, setMode] = useState<"orbit" | "top">("orbit");
  const cams = useMemo(() => {
    const ext = doc.report.extrinsics ?? [];
    const intr = new Map((doc.report.intrinsics ?? []).map((i) => [i.cam_id, i]));
    const centres = ext.map((e) => e.center_in_ref);
    let extent = 0;
    for (const a of centres) for (const b of centres) extent = Math.max(extent, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
    extent = extent || 1;
    // Frustum size from the typical gap between neighbouring cameras, so a long chain of
    // cameras still shows each one at a readable size.
    const nn = centres
      .map((a, i) => Math.min(...centres.filter((_, j) => j !== i).map((b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]))))
      .filter((d) => Number.isFinite(d) && d > 0)
      .sort((x, y) => x - y);
    const gap = nn.length ? nn[Math.floor(nn.length / 2)] : extent;
    const depth = Math.min(Math.max(gap * 0.6, extent * 0.02), extent * 0.25);
    const status = new Map(camIds(doc).map((c) => [Number(c), cameraStatus(doc, c).state]));
    return {
      extent,
      list: ext.map((e) => {
        const i = intr.get(e.cam_id);
        const f = frustum(e.T_c_ref, depth, i ? i.width / i.height : 4 / 3);
        return { id: e.cam_id, ref: e.is_ref, state: status.get(e.cam_id) ?? "accept", ...f };
      }),
    };
  }, [doc]);
  const centre = useMemo<V3>(() => {
    const n = cams.list.length || 1;
    const s = cams.list.reduce((a, c) => [a[0] + c.centre[0], a[1] + c.centre[1], a[2] + c.centre[2]], [0, 0, 0]);
    return [s[0] / n, s[1] / n, s[2] / n];
  }, [cams]);
  const color = (c: { id: number; state: string }) =>
    c.id === selected ? "#38bdf8" : c.state === "reject" ? "#ef4444" : c.state === "improve" ? "#f59e0b" : c.state === "check" ? "#a78bfa" : "#cbd5e1";
  const labelled = (c: { id: number; ref: boolean }) => cams.list.length <= 8 || c.id === selected || c.ref;

  return (
    <div className="relative h-[340px] sm:h-[420px] lg:h-[560px] rounded-md overflow-hidden" style={{ background: "radial-gradient(120% 100% at 50% 40%, #334155, #0b1120)" }}>
      <Canvas camera={{ fov: 45, near: cams.extent * 1e-3, far: cams.extent * 100 }} dpr={[1, 2]}>
        <ViewSetter mode={mode} centre={centre} extent={cams.extent} />
        <OrbitControls target={centre} makeDefault enableDamping />
        {cams.list.map((c) => (
          <group key={c.id}>
            <Line points={c.segs} segments color={color(c)} lineWidth={c.id === selected ? 2.2 : 1.2} />
            {labelled(c) && (
              <Html position={c.centre} center distanceFactor={undefined} zIndexRange={[10, 0]}>
                <span className="pointer-events-none select-none whitespace-nowrap rounded px-1 text-[10px] font-studio-mono"
                  style={{ color: color(c), background: "rgba(15,23,42,0.7)" }}>
                  {camName(c.id)}{c.ref ? " ref" : ""}
                </span>
              </Html>
            )}
          </group>
        ))}
      </Canvas>
      <div className="absolute top-2 left-2 flex gap-1" role="group" aria-label="View">
        {(["orbit", "top"] as const).map((m) => (
          <button key={m} type="button" onClick={() => setMode(m)} aria-pressed={mode === m}
            className={`rounded px-2 py-1 text-xs font-medium ${mode === m ? "bg-white/90 text-slate-900" : "bg-slate-900/70 text-slate-200 hover:bg-slate-800"}`}>
            {m === "orbit" ? "Turntable" : "Top view"}
          </button>
        ))}
      </div>
      <div className="absolute bottom-2 left-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-slate-200">
        <span><span style={{ color: "#38bdf8" }}>■</span> selected</span>
        <span><span style={{ color: "#ef4444" }}>■</span> not ready</span>
        <span><span style={{ color: "#f59e0b" }}>■</span> improve</span>
        <span><span style={{ color: "#a78bfa" }}>■</span> check</span>
        <span>Top view looks along the reference camera's down axis.</span>
      </div>
    </div>
  );
}
