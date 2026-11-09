// Empty ds_msp/studio/static before a studio build, keeping its tracked README.md.
import { readdirSync, rmSync } from "node:fs";
import { join } from "node:path";

const dir = new URL("../../ds_msp/studio/static/", import.meta.url).pathname;
for (const name of readdirSync(dir)) {
  if (name !== "README.md") rmSync(join(dir, name), { recursive: true, force: true });
}
