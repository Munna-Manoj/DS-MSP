// The rig in 3D, in the reference camera's frame: every camera, the board where it was in the
// selected photo (scrub the timeline to replay the capture), a ruler between two cameras, and,
// on request, each camera wobbling by its own uncertainty, enlarged until it can be seen.
// Camera frames are x right, y down, z forward; the view shows y up, so (x, y, z) -> (x, -y, -z).
import { Html, Line, OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, type ReactNode } from "react";
import { DoubleSide, type Group } from "three";
import type { RunDoc, SceneDoc } from "../api";
import { camName } from "../format";
import { cameraStatus, camIds } from "../status";
import { boardColor, boardName } from "./boards";
import { length, type Unit } from "./units";

type V3 = [number, number, number];
const toView = (p: number[]): V3 => [p[0], -p[1], -p[2]];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dist = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

function frustum(T: number[][], depth: number, aspect: number) {
  const R = [T[0].slice(0, 3), T[1].slice(0, 3), T[2].slice(0, 3)];
  const t = [T[0][3], T[1][3], T[2][3]];
  const C = [0, 1, 2].map((j) => -(R[0][j] * t[0] + R[1][j] * t[1] + R[2][j] * t[2]));
  const toRef = (p: number[]) => [0, 1, 2].map((j) => R[0][j] * p[0] + R[1][j] * p[1] + R[2][j] * p[2] + C[j]);
  const w = depth * 0.6, h = w / aspect;
  const centre = toView(C);
  const k = [[-w, -h, depth], [w, -h, depth], [w, h, depth], [-w, h, depth]].map((p) => sub(toView(toRef(p)), centre));
  const up = sub(toView(toRef([0, -h * 1.5, depth])), centre);
  const o: V3 = [0, 0, 0];
  const segs: V3[] = [o, k[0], o, k[1], o, k[2], o, k[3], k[0], k[1], k[1], k[2], k[2], k[3], k[3], k[0], k[0], up, up, k[1]];
  return { centre, segs };
}

function applyT(T: number[][], p: number[]): number[] {
  return [0, 1, 2].map((i) => T[i][0] * p[0] + T[i][1] * p[1] + T[i][2] * p[2] + T[i][3]);
}

function ViewSetter({ centre, extent, back }: { centre: V3; extent: number; back: V3 }) {
  const { camera } = useThree();
  useEffect(() => {
    // Stand behind the cameras, a little above and to one side, looking at what they look at.
    const d = extent * 2.9 + 1e-6;            // 45 degree lens: the whole sphere fits
    const side: V3 = [back[2], 0, -back[0]];
    camera.position.set(
      centre[0] + d * (0.72 * back[0] + 0.42 * side[0]),
      centre[1] + d * (0.72 * back[1] + 0.48),
      centre[2] + d * (0.72 * back[2] + 0.42 * side[2]));
    camera.lookAt(...centre);
    camera.updateProjectionMatrix();
  }, [centre, extent, back, camera]);
  return null;
}

/** One camera. With `wobble`, it sways by its position and angle uncertainty times `wobble`. */
function CameraNode({ centre, segs, color, width, sigmaPos, sigmaRad, wobble, seed, onPick, radius, children }: {
  centre: V3; segs: V3[]; color: string; width: number; sigmaPos: number; sigmaRad: number;
  wobble: number; seed: number; onPick: () => void; radius: number; children?: ReactNode;
}) {
  const ref = useRef<Group>(null);
  useFrame(({ clock }) => {
    const g = ref.current;
    if (!g) return;
    const t = clock.elapsedTime, k = wobble;
    const a = sigmaPos * k, r = Math.min(sigmaRad * k, 0.6);
    g.position.set(
      centre[0] + a * Math.sin(1.7 * t + seed),
      centre[1] + a * Math.sin(2.3 * t + 2 * seed),
      centre[2] + a * Math.sin(1.1 * t + 3 * seed));
    g.rotation.set(r * Math.sin(1.9 * t + seed), r * Math.sin(1.3 * t + 4 * seed), r * Math.sin(2.1 * t + 5 * seed));
  });
  return (
    <group ref={ref} position={centre}>
      <Line points={segs} segments color={color} lineWidth={width} />
      <mesh onClick={(e) => { e.stopPropagation(); onPick(); }}>
        <sphereGeometry args={[radius, 12, 12]} />
        <meshBasicMaterial transparent opacity={0} depthWrite={false} />
      </mesh>
      {children}
    </group>
  );
}

export default function RigStage({ doc, scene, frame, selected, from, to, onPick, wobble, unit }: {
  doc: RunDoc; scene: SceneDoc | null; frame: number | null; selected: number;
  from: number; to: number | null; onPick: (cam: number) => void; wobble: number; unit: Unit;
}) {
  const cams = useMemo(() => {
    const ext = doc.report.extrinsics ?? [];
    const intr = new Map((doc.report.intrinsics ?? []).map((i) => [i.cam_id, i]));
    const centres = ext.map((e) => e.center_in_ref);
    let span = 0;
    for (const a of centres) for (const b of centres) span = Math.max(span, dist(a, b));
    const nn = centres
      .map((a, i) => Math.min(...centres.filter((_, j) => j !== i).map((b) => dist(a, b))))
      .filter((d) => Number.isFinite(d) && d > 0).sort((x, y) => x - y);
    const gap = nn.length ? nn[Math.floor(nn.length / 2)] : span || 1;
    const depth = Math.min(Math.max(gap * 0.55, (span || gap) * 0.02), (span || gap) * 0.25);
    const refId = doc.report.capture?.ref_cam_id ?? ext.find((e) => e.is_ref)?.cam_id ?? 0;
    const pairs = doc.acceptance?.extrinsic_uncertainty?.pairs ?? {};
    const status = new Map(camIds(doc).map((c) => [Number(c), cameraStatus(doc, c).state]));
    return {
      depth, span: span || gap,
      list: ext.map((e) => {
        const i = intr.get(e.cam_id);
        const p = pairs[`${Math.min(refId, e.cam_id)}-${Math.max(refId, e.cam_id)}`];
        return {
          id: e.cam_id, ref: e.is_ref, state: status.get(e.cam_id) ?? "accept",
          sigmaPos: e.is_ref ? 0 : p?.centre_sigma ?? 0,
          sigmaRad: e.is_ref ? 0 : ((p?.rotation_sigma_deg ?? 0) * Math.PI) / 180,
          ...frustum(e.T_c_ref, depth, i ? i.width / i.height : 4 / 3),
        };
      }),
    };
  }, [doc]);

  const board = useMemo(() => {
    const f = scene?.frames.find((x) => x.frame === frame);
    if (!scene || !f) return null;
    return scene.boards.map((b, k) => {
      const q = b.outline.map((p) => toView(applyT(f.T, p)));
      const tri = new Float32Array([...q[0], ...q[1], ...q[2], ...q[0], ...q[2], ...q[3]]);
      const dots = new Float32Array(b.points.flatMap((p) => toView(applyT(f.T, p))));
      // boards can share a plane (the two faces of one board): name each at a different corner
      return { id: b.id, loop: [...q, q[0]], tri, dots, tag: q[k % 4] };
    });
  }, [scene, frame]);

  const view = useMemo(() => {
    const pts: V3[] = cams.list.map((c) => c.centre);
    for (const f of scene?.frames ?? []) pts.push(toView([f.T[0][3], f.T[1][3], f.T[2][3]]));
    // the middle of the box around every camera and every board position, so all of it shows
    const centre: V3 = [0, 1, 2].map((j) => {
      const v = pts.map((p) => p[j]);
      return (Math.min(...v) + Math.max(...v)) / 2;
    }) as V3;
    const extent = Math.max(...pts.map((p) => dist(p, centre)), cams.span * 0.5, 1e-6) + cams.depth;
    // from the middle of the boards back to the middle of the cameras
    const mean = (list: V3[]): V3 => [0, 1, 2].map((j) => list.reduce((a, p) => a + p[j], 0) / (list.length || 1)) as V3;
    const camMid = mean(cams.list.map((c) => c.centre));
    const boards = pts.slice(cams.list.length);
    let back: V3 = boards.length ? sub(camMid, mean(boards)) : [0.5, 0, 0.85];
    const len = Math.hypot(...back) || 1;
    back = [back[0] / len, back[1] / len, back[2] / len];
    return { centre, extent, back };
  }, [cams, scene]);

  const a = cams.list.find((c) => c.id === from), b = cams.list.find((c) => c.id === to);
  const color = (c: { id: number; state: string }) =>
    c.id === to ? "#fbbf24" : c.id === selected || c.id === from ? "#38bdf8"
      : c.state === "reject" ? "#ef4444" : c.state === "improve" ? "#f59e0b" : "#cbd5e1";

  return (
    <div className="relative h-[360px] sm:h-[460px] lg:h-[540px]">
      <Canvas camera={{ fov: 45, near: view.extent * 1e-3, far: view.extent * 100 }} dpr={[1, 2]}>
        <ViewSetter centre={view.centre} extent={view.extent} back={view.back} />
        <OrbitControls target={view.centre} makeDefault enableDamping />
        {cams.list.map((c, k) => (
          <CameraNode key={c.id} centre={c.centre} segs={c.segs} color={color(c)} width={c.id === selected || c.id === from || c.id === to ? 2.4 : 1.3}
            sigmaPos={c.sigmaPos} sigmaRad={c.sigmaRad} wobble={wobble} seed={1 + k * 1.37} radius={cams.depth * 0.7}
            onPick={() => onPick(c.id)}>
            {(cams.list.length <= 10 || c.id === selected || c.id === from || c.id === to || c.ref) && (
              <Html center zIndexRange={[10, 0]}>
                <span className="pointer-events-none select-none whitespace-nowrap rounded px-1 text-[10px] font-studio-mono"
                  style={{ color: color(c), background: "rgba(15,23,42,0.7)" }}>
                  {camName(c.id)}{c.ref ? " ref" : ""}
                </span>
              </Html>
            )}
          </CameraNode>
        ))}
        {board?.map((bd) => (
          <group key={bd.id}>
            <mesh>
              <bufferGeometry>
                <bufferAttribute attach="attributes-position" count={6} array={bd.tri} itemSize={3} />
              </bufferGeometry>
              <meshBasicMaterial color="#f8fafc" transparent opacity={0.92} side={DoubleSide} />
            </mesh>
            <Line points={bd.loop} color={boardColor(bd.id)} lineWidth={3} />
            <Html position={bd.tag} center zIndexRange={[15, 0]}>
              <span className="pointer-events-none select-none whitespace-nowrap rounded px-1.5 py-0.5 text-[11px] font-bold"
                style={{ color: "#0f172a", background: boardColor(bd.id) }}>{boardName(bd.id)}</span>
            </Html>
            <points>
              <bufferGeometry>
                <bufferAttribute attach="attributes-position" count={bd.dots.length / 3} array={bd.dots} itemSize={3} />
              </bufferGeometry>
              <pointsMaterial color="#0f172a" size={4} sizeAttenuation={false} />
            </points>
          </group>
        ))}
        {a && b && (
          <group>
            <Line points={[a.centre, b.centre]} color="#fbbf24" lineWidth={2} dashed dashSize={cams.depth * 0.25} gapSize={cams.depth * 0.15} />
            <Html position={[(a.centre[0] + b.centre[0]) / 2, (a.centre[1] + b.centre[1]) / 2, (a.centre[2] + b.centre[2]) / 2]} center zIndexRange={[20, 0]}>
              <span className="pointer-events-none select-none whitespace-nowrap rounded px-1.5 py-0.5 text-[11px] font-semibold"
                style={{ color: "#0f172a", background: "#fbbf24" }}>
                {length(dist(a.centre, b.centre), unit)}
              </span>
            </Html>
          </group>
        )}
      </Canvas>
      <div className="absolute bottom-2 left-2 right-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-300 pointer-events-none">
        <span>Drag to turn, scroll to zoom. Click a camera to measure to it.</span>
        {board ? <span>The boards are where the target was in the selected photo, each outlined in its colour.</span> : <span>No board position is recorded for this photo.</span>}
      </div>
    </div>
  );
}
