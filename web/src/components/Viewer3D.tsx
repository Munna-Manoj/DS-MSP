import { useEffect, useMemo, useRef, type ReactNode } from "react";
import { Canvas, useThree, useFrame } from "@react-three/fiber";
import { OrbitControls, TransformControls, Line, Html, useTexture } from "@react-three/drei";
import * as THREE from "three";
import { zoneColor, type CameraModel, type Params, type Vec3 } from "../lib/cameras";
import {
  buildSurfaceMesh,
  computeRayGrid,
  flipY,
  surfaceIndices,
  type Surface,
} from "../lib/raymap";
import { ENV_URL } from "../lib/assets";

const SURFACE_N = 96;

interface Props {
  model: CameraModel;
  params: Params;
  active: Vec3;
  onActiveChange: (p: Vec3) => void;
  surfaces: Surface[];
  stage: number; // -1 = show everything; otherwise reveal up to this stage
}

const SURFACE_TINT: Record<Surface, string> = {
  sphere: "#6e8bff",
  cylinder: "#4ed7c0",
  plane: "#ffc56e",
};

function Environment360() {
  const tex = useTexture(ENV_URL);
  const { scene } = useThree();
  useEffect(() => {
    tex.mapping = THREE.EquirectangularReflectionMapping;
    tex.colorSpace = THREE.SRGBColorSpace;
    const pBg = scene.background;
    const pEnv = scene.environment;
    scene.background = tex;
    scene.environment = tex;
    return () => {
      scene.background = pBg;
      scene.environment = pEnv;
    };
  }, [tex, scene]);
  return null;
}

// A captured-image surface (sphere / cylinder / plane): every pixel placed along
// its true world bearing, so it coincides with the 3D world it samples. Where the
// model can't see, the mesh simply has a hole — that absence is the lesson.
function SurfaceShell({
  model,
  params,
  surface,
  pano,
  visible,
}: {
  model: CameraModel;
  params: Params;
  surface: Surface;
  pano: THREE.Texture;
  visible: boolean;
}) {
  const geom = useMemo(() => {
    const grid = computeRayGrid(model, params, SURFACE_N);
    const m = buildSurfaceMesh(grid, surface);
    const idx = surfaceIndices(m);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(m.positions, 3));
    g.setAttribute("uv", new THREE.BufferAttribute(m.uvs, 2));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.computeVertexNormals();
    return g;
  }, [model, params, surface]);

  useEffect(() => () => geom.dispose(), [geom]);
  if (!visible) return null;

  return (
    <mesh geometry={geom} renderOrder={1}>
      <meshBasicMaterial
        map={pano}
        side={THREE.DoubleSide}
        transparent
        opacity={0.96}
        toneMapped={false}
      />
    </mesh>
  );
}

function Dot({ p, color, r = 0.05 }: { p: Vec3; color: string; r?: number }) {
  return (
    <mesh position={p}>
      <sphereGeometry args={[r, 18, 18]} />
      <meshBasicMaterial color={color} toneMapped={false} />
    </mesh>
  );
}

function Tag({ p, children }: { p: Vec3; children: ReactNode }) {
  return (
    <Html position={p} center className="pointer-events-none select-none">
      <span className="font-mono text-[10px] text-text bg-ink/80 backdrop-blur px-1.5 py-0.5 rounded whitespace-nowrap border border-line">
        {children}
      </span>
    </Html>
  );
}

// where a bearing lands on a given surface (mirror of raymap.placeOnSurface,
// used only for the single picked-ray marker so we don't rebuild a whole grid)
function landOn(d: Vec3, s: Surface): Vec3 | null {
  if (s === "sphere") return [d[0], d[1], d[2]];
  if (s === "cylinder") {
    const h = Math.hypot(d[0], d[2]) || 1e-9;
    const yy = d[1] / h;
    return Math.abs(yy) < 3 && d[2] > -0.999 ? [d[0] / h, yy, d[2] / h] : null;
  }
  if (d[2] <= 0.08) return null;
  const px = d[0] / d[2];
  const py = d[1] / d[2];
  return Math.hypot(px, py) < 2.4 ? [px, py, 1] : null;
}

// An expanding/fading halo marking the step the play head is currently on, so
// each auto-step visibly "lights up" the element it introduces.
function Pulse({ p, color }: { p: Vec3; color: string }) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame((state) => {
    const t = (state.clock.elapsedTime % 1.1) / 1.1; // 0..1 loop
    const m = ref.current;
    if (!m) return;
    m.scale.setScalar(0.08 + t * 0.55);
    (m.material as THREE.MeshBasicMaterial).opacity = 0.55 * (1 - t);
  });
  return (
    <mesh ref={ref} position={p}>
      <sphereGeometry args={[1, 20, 20]} />
      <meshBasicMaterial color={color} transparent opacity={0.5} wireframe depthWrite={false} />
    </mesh>
  );
}

function PipelineViz({
  model,
  params,
  active,
  surfaces,
  stage,
}: {
  model: CameraModel;
  params: Params;
  active: Vec3;
  surfaces: Surface[];
  stage: number;
}) {
  // active is a y-up world point; the model projects in the camera (y-down) frame.
  const proj = model.project(flipY(active), params);
  const color = zoneColor[proj.zone];
  const n = Math.hypot(...active) || 1;
  const bearing: Vec3 = [active[0] / n, active[1] / n, active[2] / n];

  const show = (s: number) => stage < 0 || stage >= s;
  const cur = (s: number) => stage === s; // the step the play head just revealed
  const firstLand = surfaces.map((s) => landOn(bearing, s)).find((l) => l) ?? null;

  return (
    <>
      {/* ① world ray */}
      {show(0) && (
        <Line points={[[0, 0, 0], active]} color={color} lineWidth={cur(0) ? 4 : 2.5} />
      )}
      {cur(0) && <Pulse p={active} color={color} />}

      {/* ② bearing on the unit sphere */}
      {show(1) && <Dot p={bearing} color={color} r={cur(1) ? 0.075 : 0.05} />}
      {show(1) && (
        <Tag p={[bearing[0] * 1.16, bearing[1] * 1.16 + 0.12, bearing[2] * 1.16]}>
          ② bearing b
        </Tag>
      )}
      {cur(1) && <Pulse p={bearing} color={color} />}

      {/* ③ landing on each shown surface + the pixel */}
      {show(2) &&
        surfaces.map((s) => {
          const land = landOn(bearing, s);
          if (!land) return null;
          return (
            <group key={s}>
              <Line points={[bearing, land]} color={SURFACE_TINT[s]} lineWidth={cur(2) ? 2.6 : 1.4} dashed dashSize={0.07} gapSize={0.05} />
              <Dot p={land} color={SURFACE_TINT[s]} r={cur(2) ? 0.07 : 0.045} />
              {show(3) && proj.valid && (
                <Tag p={[land[0], land[1] - 0.16, land[2]]}>
                  {s} · ({proj.u.toFixed(0)}, {proj.v.toFixed(0)})
                </Tag>
              )}
            </group>
          );
        })}
      {(cur(2) || cur(3)) && firstLand && <Pulse p={firstLand} color={color} />}
    </>
  );
}

function Target({
  active,
  onActiveChange,
  color,
}: {
  active: Vec3;
  onActiveChange: (p: Vec3) => void;
  color: string;
}) {
  const ref = useRef<THREE.Mesh>(null);
  useEffect(() => {
    if (ref.current) ref.current.position.set(active[0], active[1], active[2]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return (
    <TransformControls
      mode="translate"
      size={0.6}
      onObjectChange={() => {
        const p = ref.current;
        if (p) onActiveChange([p.position.x, p.position.y, p.position.z]);
      }}
    >
      <mesh ref={ref}>
        <sphereGeometry args={[0.1, 32, 32]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.6} metalness={0.2} roughness={0.3} />
      </mesh>
    </TransformControls>
  );
}

function Rig({ model, params, active, onActiveChange, surfaces, stage }: Props) {
  const pano = useTexture(ENV_URL);
  useEffect(() => {
    pano.colorSpace = THREE.SRGBColorSpace;
  }, [pano]);
  const color = zoneColor[model.project(flipY(active), params).zone];

  return (
    <>
      {/* reference bearing cage */}
      <mesh>
        <sphereGeometry args={[1.0, 36, 24]} />
        <meshBasicMaterial wireframe color="#46557a" transparent opacity={0.12} />
      </mesh>

      {(["sphere", "cylinder", "plane"] as Surface[]).map((s) => (
        <SurfaceShell
          key={s}
          model={model}
          params={params}
          surface={s}
          pano={pano}
          visible={surfaces.includes(s)}
        />
      ))}

      <PipelineViz model={model} params={params} active={active} surfaces={surfaces} stage={stage} />

      {/* optical centre */}
      <mesh>
        <sphereGeometry args={[0.05, 16, 16]} />
        <meshBasicMaterial color="#e6eaf2" toneMapped={false} />
      </mesh>
      <Tag p={[0, -0.18, 0]}>camera centre O</Tag>

      <Target active={active} onActiveChange={onActiveChange} color={color} />
    </>
  );
}

export function Viewer3D(props: Props) {
  return (
    <Canvas
      camera={{ position: [3.6, 2.2, 4.4], fov: 45 }}
      dpr={[1, 2]}
      gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}
    >
      <Environment360 />
      <ambientLight intensity={0.4} />
      <Rig {...props} />
      <OrbitControls makeDefault enablePan={false} enableDamping dampingFactor={0.08} minDistance={2.2} maxDistance={11} />
    </Canvas>
  );
}
