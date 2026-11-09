import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Two builds of one app:
// - `npm run build` (mode "production"): the public demo on GitHub Pages under
//   "/DS-MSP/studio/". There is no server there, so only the Learn page works.
// - `npm run build:studio` (mode "studio"): the interface `ds-msp studio` serves, written into
//   the Python package (ds_msp/studio/static) with relative asset paths.
// Dev (`npm run dev`) proxies /api to a local `ds-msp studio --no-browser` on port 8765.
export default defineConfig(({ command, mode }) => {
  const studio = mode === "studio";
  return {
    base: studio ? "./" : command === "build" ? "/DS-MSP/studio/" : "/",
    plugins: [react(), tailwindcss()],
    define: { __HAS_SERVER__: JSON.stringify(studio || command === "serve") },
    build: studio
      ? { outDir: "../ds_msp/studio/static", emptyOutDir: false, chunkSizeWarningLimit: 1500 }
      : { chunkSizeWarningLimit: 1500 },
    server: { proxy: { "/api": "http://127.0.0.1:8765" } },
  };
});
