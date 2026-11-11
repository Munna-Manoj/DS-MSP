// Hash routes, so the app works from any base path and needs no server-side routing:
//   #/            run list            #/run/<id>    one run            #/learn   model demos
//   #/new         set up a calibration (#/new/from/<id> starts from that folder's settings)
//   #/job/<id>    a calibration started from Studio, while it runs and after
//   #/repeat/<id> a result's fingerprint, and repeating it exactly
//   #/export      LDC meshes and Isaac LUTs for cameras you have (#/export/from/<id>: a result's)
import { useEffect, useState } from "react";

export type Route =
  | { page: "runs" }
  | { page: "run"; id: string; cam?: string }
  | { page: "new"; from?: string }
  | { page: "job"; id: string }
  | { page: "repeat"; id: string }
  | { page: "export"; from?: string }
  | { page: "learn" };

export function parseHash(hash: string): Route {
  const h = hash.replace(/^#\/?/, "");
  if (h.startsWith("learn")) return { page: "learn" };
  if (h.startsWith("new")) {
    const from = h.startsWith("new/from/") ? decodeURIComponent(h.slice(9)) : undefined;
    return { page: "new", from };
  }
  if (h.startsWith("export")) {
    const from = h.startsWith("export/from/") ? decodeURIComponent(h.slice(12)) : undefined;
    return { page: "export", from };
  }
  if (h.startsWith("job/")) return { page: "job", id: decodeURIComponent(h.slice(4)) };
  if (h.startsWith("repeat/")) return { page: "repeat", id: decodeURIComponent(h.slice(7)) };
  if (h.startsWith("run/")) {
    const [id, cam] = h.slice(4).split("/cam/");
    return { page: "run", id: decodeURIComponent(id), cam: cam ? decodeURIComponent(cam) : undefined };
  }
  return { page: __HAS_SERVER__ ? "runs" : "learn" };
}

export function href(r: Route): string {
  if (r.page === "learn") return "#/learn";
  if (r.page === "new") return r.from ? `#/new/from/${encodeURIComponent(r.from)}` : "#/new";
  if (r.page === "export") return r.from ? `#/export/from/${encodeURIComponent(r.from)}` : "#/export";
  if (r.page === "job") return `#/job/${encodeURIComponent(r.id)}`;
  if (r.page === "repeat") return `#/repeat/${encodeURIComponent(r.id)}`;
  if (r.page === "run")
    return `#/run/${encodeURIComponent(r.id)}${r.cam != null ? `/cam/${encodeURIComponent(r.cam)}` : ""}`;
  return "#/";
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  useEffect(() => {
    const on = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}
