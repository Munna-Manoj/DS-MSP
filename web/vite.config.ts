import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev serves at "/", production builds under the GitHub Pages project subpath
// "/DS-MSP/studio/" -- the docs site (MkDocs) owns the site root; this studio is
// deployed alongside it as a subpath. Assets are loaded via import.meta.env.BASE_URL
// so both work without depending on a trailing slash.
export default defineConfig(({ command }) => ({
  base: command === "build" ? "/DS-MSP/studio/" : "/",
  plugins: [react(), tailwindcss()],
}));
