# DS-MSP — interactive studio (`web/`)

A small Vite + React + TypeScript + three.js app that teaches wide-FOV camera
projection live: pick any of the six models the library ships, drag a 3D point,
and step its projection onto a sphere / cylinder / plane.

**Live demo:** https://munna-manoj.github.io/DS-MSP/studio/

## Not part of the Python package

This folder is **learning tooling, not the library**. It is excluded from the
PyPI distribution (`MANIFEST.in` prunes `web/`; the wheel only contains
`ds_msp*`), so `pip install ds-msp` never downloads any of it. Cloning the repo
pulls only the small source here — `node_modules/` and `dist/` are git-ignored.

## How it stays correct

Every model's `project` / `unproject` in [`src/lib/cameras.ts`](src/lib/cameras.ts)
is a faithful port of `ds_msp/models/*_math.py`. The whole visualisation is then
driven by a *ray grid* — each pixel unprojected to its true world bearing — so
the raw image, the wrapped surfaces, and the 3D world are all guaranteed to agree.

The ports are cross-checked against the Python library:

```bash
# from the repo root
python3 web/scripts/dump_reference.py          # dump ds_msp reference values
cd web
node_modules/.bin/esbuild scripts/verify.mjs --bundle --format=esm \
  --platform=node --outfile=/tmp/verify.bundle.mjs
node /tmp/verify.bundle.mjs                     # → VERIFY PASS (~1e-12 px, ~1e-8 rad)
```

## Develop

```bash
cd web
npm install
npm run dev      # http://localhost:5173/
npm run build    # type-check + production bundle into web/dist (base /DS-MSP/)
npm run preview  # serve the production bundle locally
```

## Credits

Environment panorama `public/env.jpg` — "Venice Sunset" by Greg Zaal / Poly Haven,
released under **CC0** (public domain). https://polyhaven.com/a/venice_sunset
