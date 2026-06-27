// The ray grid is the spine of the whole visualiser. For the current model and
// params we unproject a grid of image pixels into their true world bearings
// (using the verified ports in cameras.ts). Everything else — the raw image, the
// wrapped sphere/cylinder/plane — is then just a different way of *placing* and
// *colouring* those same rays, so they are all guaranteed to agree with each
// other and with the 3D world the camera sees.

import type { CameraModel, Params, Vec3 } from "./cameras";

/** The camera models use the computer-vision image frame (x-right, y-DOWN,
 *  z-forward), but the 3D scene and the equirectangular panorama are y-UP. Flip y
 *  to move a bearing/point between the two frames so the synthesized image, the
 *  wrapped surfaces, and the y-up world all read upright and agree. Self-inverse,
 *  so this one helper converts both directions (camera↔world). */
export function flipY(d: Vec3): Vec3 {
  return [d[0], -d[1], d[2]];
}

/** Equirectangular lookup matching three.js' equirectUv (so our surfaces line up
 *  with scene.background). u = atan2(z,x)/2π+0.5, v = asin(y)/π+0.5. */
export function equirectUv(d: Vec3): [number, number] {
  const u = Math.atan2(d[2], d[0]) / (2 * Math.PI) + 0.5;
  const v = Math.asin(Math.max(-1, Math.min(1, d[1]))) / Math.PI + 0.5;
  return [u, v];
}

export interface RayGrid {
  n: number;
  /** RGBA float per node: [rx, ry, rz, valid?1:0]. Row j=0 is image-top (v=0). */
  data: Float32Array;
}

/** Unproject an n×n grid of pixels to world bearings for the given model/params. */
export function computeRayGrid(model: CameraModel, p: Params, n: number): RayGrid {
  const W = p.cx * 2;
  const H = p.cy * 2;
  const data = new Float32Array(n * n * 4);
  for (let j = 0; j < n; j++) {
    const v = (j / (n - 1)) * H;
    for (let i = 0; i < n; i++) {
      const u = (i / (n - 1)) * W;
      const { ray, valid } = model.unproject(u, v, p);
      // camera frame (y-down) -> world (y-up) so the image/surfaces sit upright
      // in the y-up scene and sample the panorama the right way up.
      const w = flipY(ray);
      const k = (j * n + i) * 4;
      data[k] = w[0];
      data[k + 1] = w[1];
      data[k + 2] = w[2];
      data[k + 3] = valid ? 1 : 0;
    }
  }
  return { n, data };
}

export type Surface = "plane" | "sphere" | "cylinder";

export interface SurfaceMesh {
  positions: Float32Array; // xyz per node
  uvs: Float32Array; // equirect uv per node, for texturing
  valid: Uint8Array; // per node
  n: number;
}

// Where a bearing lands on each candidate image surface. The sphere is the
// honest one for wide FOV (every direction has a home); the plane only holds the
// forward hemisphere and blows the rim up — which is the lesson.
const SPHERE_R = 1.0;
const CYL_R = 1.0;
const PLANE_Z = 1.0;
const PLANE_CLIP = 2.4; // how far off-axis the flat plane is allowed to stretch

function placeOnSurface(d: Vec3, s: Surface): { pos: Vec3; ok: boolean } {
  if (s === "sphere") {
    return { pos: [d[0] * SPHERE_R, d[1] * SPHERE_R, d[2] * SPHERE_R], ok: true };
  }
  if (s === "cylinder") {
    const horiz = Math.hypot(d[0], d[2]) || 1e-9;
    const yy = (d[1] / horiz) * CYL_R;
    return {
      pos: [(d[0] / horiz) * CYL_R, yy, (d[2] / horiz) * CYL_R],
      ok: Math.abs(yy) < 3.0 && d[2] > -0.999,
    };
  }
  // plane: normalized image plane at z = PLANE_Z; only the forward cone reaches it
  if (d[2] <= 0.08) return { pos: [0, 0, 0], ok: false };
  const px = (d[0] / d[2]) * PLANE_Z;
  const py = (d[1] / d[2]) * PLANE_Z;
  return { pos: [px, py, PLANE_Z], ok: Math.hypot(px, py) < PLANE_CLIP };
}

/** Build a textured surface mesh (sphere/cylinder/plane) from a ray grid. */
export function buildSurfaceMesh(grid: RayGrid, surface: Surface): SurfaceMesh {
  const { n, data } = grid;
  const positions = new Float32Array(n * n * 3);
  const uvs = new Float32Array(n * n * 2);
  const valid = new Uint8Array(n * n);
  for (let idx = 0; idx < n * n; idx++) {
    const k = idx * 4;
    const d: Vec3 = [data[k], data[k + 1], data[k + 2]];
    const nodeOk = data[k + 3] > 0.5;
    const { pos, ok } = placeOnSurface(d, surface);
    positions[idx * 3] = pos[0];
    positions[idx * 3 + 1] = pos[1];
    positions[idx * 3 + 2] = pos[2];
    const [uu, vv] = equirectUv(d);
    uvs[idx * 2] = uu;
    uvs[idx * 2 + 1] = vv;
    valid[idx] = nodeOk && ok ? 1 : 0;
  }
  return { positions, uvs, valid, n };
}

/** Triangle indices for the grid, skipping any quad touching an invalid node so
 *  un-captured directions simply aren't meshed (truthful, no dark fill). Also
 *  skips quads that straddle the equirect seam to avoid smeared triangles. */
export function surfaceIndices(mesh: SurfaceMesh): Uint32Array {
  const { n, valid, uvs } = mesh;
  const out: number[] = [];
  for (let j = 0; j < n - 1; j++) {
    for (let i = 0; i < n - 1; i++) {
      const a = j * n + i;
      const b = j * n + i + 1;
      const c = (j + 1) * n + i;
      const dd = (j + 1) * n + i + 1;
      if (!valid[a] || !valid[b] || !valid[c] || !valid[dd]) continue;
      // seam guard: if u-coords span more than half the texture, skip
      const us = [uvs[a * 2], uvs[b * 2], uvs[c * 2], uvs[dd * 2]];
      if (Math.max(...us) - Math.min(...us) > 0.5) continue;
      out.push(a, b, c, b, dd, c);
    }
  }
  return new Uint32Array(out);
}
