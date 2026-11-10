// Set up a calibration and start it. The form holds only the settings the user changed; the
// server writes them as calib_param.yml into a new result folder and runs the rig CLI on it,
// so a run started here and the same file run from a terminal are the same calibration.
import type React from "react";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { href } from "../router";
import {
  api, ApiError, type BoardCheck, type CaptureHistory, type ConfigCheck, type ConfigSchema,
  type IntrinsicsScan, type KeypointsScan, type LensAdvice, type LensHealth, type Problem,
  type RunSummary, type ScanDoc, type Values,
} from "./api";
import { FolderPicker } from "./FolderPicker";
import { readMarked, viewWords, writeMarked } from "./marks";
import { int, num, when } from "./format";
import { BoardStep, PhotoThumb } from "./setup/BoardStep";
import { CameraStep, chosenFor, type CameraInfo } from "./setup/CameraStep";
import { Explain, Mono, NumberBox, Step } from "./setup/kit";
import { Settings } from "./setup/Settings";
import { useUnit } from "./stage/units";
import { Button, Callout, ErrorBox, INPUT, Loading } from "./ui";

/** Form values for a lens check: with saved detections as the source, the photo folder is not sent. */
function viewValues(values: Values, source: string): Values {
  if (source !== "detections") return values;
  const out = { ...values };
  delete out.root_path;
  return out;
}

// A picker either fills one setting, or loads a whole starting point ("@config", ...).
type Picker = { mode: "folder" | "file"; key: string; title: string } | null;
type Source = "photos" | "config" | "detections" | "result";

// Each icon is drawn on a 24 x 24 grid with the current text colour.
const SOURCES: { id: Source; title: string; text: string; icon: string }[] = [
  { id: "photos", title: "My photos", text: "A folder of photos of the board, one subfolder per camera.",
    icon: "M3 7a2 2 0 0 1 2-2h3l1.5-2h5L16 5h3a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2zM12 9a4 4 0 1 0 0 8 4 4 0 0 0 0-8z" },
  { id: "config", title: "A settings file", text: "A calib_param.yml you already have. It fills in this page.",
    icon: "M6 2h8l5 5v13a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2zM14 2v5h5M8 12h8M8 16h6" },
  { id: "detections", title: "Corners found earlier", text: "A saved corners file. The photos are not read again.",
    icon: "M4 4h16v16H4zM4 9.3h16M4 14.6h16M9.3 4v16M14.6 4v16" },
  { id: "result", title: "A finished result", text: "Repeat it exactly, or change one thing and run it again.",
    icon: "M4 12a8 8 0 1 0 2.3-5.6M4 4v4h4M8.5 12.5l2.5 2.5 4.5-5" },
];

function defaultName(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `calibration_${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}`;
}

export function NewRunPage({ from }: { from?: string }) {
  const [schema, setSchema] = useState<ConfigSchema | null>(null);
  const [fatal, setFatal] = useState<string | null>(null);
  const [canRun, setCanRun] = useState(true);
  const [values, setValues] = useState<Values>({});
  const [models, setModels] = useState<string[]>([]);
  const [scan, setScan] = useState<ScanDoc | null>(null);
  const [scanning, setScanning] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);
  const [board, setBoard] = useState<BoardCheck | null>(null);
  const [boardBusy, setBoardBusy] = useState(false);
  const [boardError, setBoardError] = useState<string | null>(null);
  const [picker, setPicker] = useState<Picker>(null);
  const [source, setSource] = useState<Source>("photos");
  const [kp, setKp] = useState<KeypointsScan | null>(null);
  const [lens, setLens] = useState<IntrinsicsScan | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [results, setResults] = useState<RunSummary[] | null>(null);
  const [dragging, setDragging] = useState(false);
  const [advice, setAdvice] = useState<Record<number, LensAdvice | "busy">>({});
  const [health, setHealth] = useState<Record<number, LensHealth | "busy">>({});
  const [hist, setHist] = useState<CaptureHistory | null>(null);
  const [fromName, setFromName] = useState<string | null>(null);
  const [reuse, setReuse] = useState(true);
  const [name, setName] = useState(defaultName);
  const [root, setRoot] = useState(0);
  const [roots, setRoots] = useState<{ name: string; path: string }[]>([]);
  const [problems, setProblems] = useState<Problem[]>([]);
  const [startError, setStartError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [preview, setPreview] = useState<ConfigCheck | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [unit, setUnit] = useUnit();
  const fileInput = useRef<HTMLInputElement>(null);
  const summary = useRef<HTMLDivElement>(null);

  const set = useCallback((key: string, v: unknown) => {
    setValues((old) => {
      const next = { ...old };
      if (v === "" || v == null) delete next[key];
      else next[key] = v;
      return next;
    });
    setProblems((old) => old.filter((p) => p.key !== key));
    setPreview(null);
  }, []);

  const rootPath = typeof values.root_path === "string" ? values.root_path : "";
  const nCam = scan?.number_camera || Number(values.number_camera) || 0;

  const runScan = useCallback((path: string, prefix?: string) => {
    setScanning(true);
    setScanError(null);
    setBoard(null);
    api.scanPhotos(path, prefix).then(
      (doc) => {
        setScan(doc);
        setScanning(false);
        setValues((old) => {
          const next: Values = { ...old };
          if (doc.number_camera) next.number_camera = doc.number_camera;
          else delete next.number_camera;
          if (doc.cam_prefix && doc.cam_prefix !== "Cam_") next.cam_prefix = doc.cam_prefix;
          else delete next.cam_prefix;
          return next;
        });
        setModels((old) => Array.from({ length: doc.number_camera }, (_, k) => old[k] ?? old[0] ?? "radtan"));
      },
      (e: Error) => { setScanning(false); setScan(null); setScanError(e.message); },
    );
  }, []);

  const loadDetections = useCallback((path: string) => {
    setFileError(null);
    api.sourceKeypoints(path).then(
      (doc) => {
        setKp(doc);
        setScan(null);
        setBoard(null);
        setValues((old) => {
          const next: Values = { ...old, keypoints_path: doc.path, number_camera: doc.number_camera };
          delete next.root_path;
          delete next.cam_prefix;
          if (doc.number_board > 1) next.number_board = doc.number_board;
          return next;
        });
        setModels((old) => Array.from({ length: doc.number_camera }, (_, k) => old[k] ?? old[0] ?? "radtan"));
      },
      (e: Error) => { setKp(null); setFileError(e.message); },
    );
  }, []);

  const loadLens = useCallback((path: string) => {
    setFileError(null);
    api.sourceIntrinsics(path).then(
      (doc) => {
        setLens(doc);
        setValues((old) => ({ ...old, cam_params_path: doc.path }));
        // the file states each lens type: take them, so the choice below matches the file
        setModels((old) => {
          const n = Math.max(old.length, doc.cameras.length);
          return Array.from({ length: n }, (_, k) => doc.cameras.find((c) => c.index === k)?.model ?? old[k] ?? "radtan");
        });
      },
      (e: Error) => { setLens(null); setFileError(e.message); },
    );
  }, []);

  // Fill the form from settings read out of a file or an earlier run.
  const adopt = useCallback((incoming: Values, label: string, ignored: string[]) => {
    const next: Values = { ...incoming };
    delete next.save_path;
    delete next.webviewer;
    let given = Array.isArray(next.camera_models) ? (next.camera_models as string[]) : [];
    delete next.camera_models;
    // MC-Calib's way of naming lens types (0 = standard, 1 = fisheye) becomes this page's choice
    const old = Array.isArray(next.distortion_per_camera) && next.distortion_per_camera.length
      ? (next.distortion_per_camera as number[]) : next.distortion_model != null ? [Number(next.distortion_model)] : [];
    if (!given.length && old.length) given = old.map((v) => (Number(v) === 1 ? "kb" : "radtan"));
    delete next.distortion_per_camera;
    delete next.distortion_model;
    if (next.refine_structure === true && next.structure_mode == null) next.structure_mode = "free";
    delete next.refine_structure;
    setValues(next);
    setModels(given);
    setScan(null);
    setBoard(null);
    setProblems([]);
    setPreview(null);
    setKp(null);
    setLens(null);
    setNotice(`${label}${ignored.length ? ` Keys DS-MSP does not use were left out: ${ignored.join(", ")}.` : ""}`);
    const detections = typeof next.keypoints_path === "string" && next.keypoints_path !== "";
    setSource(detections ? "detections" : "photos");
    if (detections) loadDetections(next.keypoints_path as string);
    else if (typeof next.root_path === "string" && next.root_path)
      runScan(next.root_path, typeof next.cam_prefix === "string" ? next.cam_prefix : undefined);
    if (typeof next.cam_params_path === "string" && next.cam_params_path) loadLens(next.cam_params_path);
  }, [runScan, loadDetections, loadLens]);

  useEffect(() => {
    api.health().then((h) => setCanRun(h.can_run !== false), () => undefined);
    api.schema().then(setSchema, (e: Error) => setFatal(e.message));
    api.browse(null).then((d) => setRoots(d.dirs.map((x) => ({ name: x.name, path: x.path }))), () => undefined);
  }, []);

  useEffect(() => {
    if (!from) return;
    api.configOf(from).then(
      (doc) => {
        // photos marked on that result's page are left out, on top of those it already left out
        const before = Array.isArray(doc.values.exclude_views) ? (doc.values.exclude_views as string[]) : [];
        const marks = readMarked(from);
        const all = [...new Set([...before, ...marks])];
        adopt(all.length ? { ...doc.values, exclude_views: all } : doc.values,
          `Started from the settings of ${doc.name}${marks.length ? `, leaving out the ${marks.length} photo${marks.length === 1 ? "" : "s"} you marked` : ""}.${doc.changed ? " Its settings file was edited after that run, so these are the settings as they are now." : ""}`,
          doc.unknown);
        setName(`${doc.name}_${marks.length ? "cleaned" : "again"}`);
        writeMarked(from, []);
        setFromName(doc.name);
        const k = Number(from.split(":", 1)[0]);
        if (Number.isInteger(k)) setRoot(k);
      },
      (e: Error) => setNotice(`Could not read the settings of that run: ${e.message}`),
    );
  }, [from, adopt]);

  const payloadRef = useRef<Values>({});
  const payload = useMemo<Values>(() => {
    const out: Values = { ...values };
    if (nCam) out.number_camera = nCam;
    if (models.length) out.camera_models = models.slice(0, nCam || models.length);
    return out;
  }, [values, models, nCam]);
  payloadRef.current = payload;

  const problemOf = (key: string) => problems.find((p) => p.key === key)?.message ?? null;
  const excluded = Array.isArray(values.exclude_views) ? (values.exclude_views as string[]) : [];

  // Measure each camera's lens on a few of its photos, one camera at a time. With `choose`, a
  // camera whose type does not suit what was measured is switched and measured once more.
  const runAdvice = useCallback(async (vals: Values, n: number, choose = false) => {
    setAdvice(Object.fromEntries(Array.from({ length: n }, (_, k) => [k, "busy" as const])));
    const picked = Array.from({ length: n }, (_, k) => (vals.camera_models as string[] | undefined)?.[k] ?? "radtan");
    for (let k = 0; k < n; k++) {
      try {
        let a = await api.lensAdvice(viewValues({ ...vals, camera_models: picked }, source), k);
        const better = choose ? chosenFor(a) : null;
        if (better && better !== picked[k]) {
          picked[k] = better;
          setModels((old) => { const next = Array.from({ length: n }, (_, i) => old[i] ?? "radtan"); next[k] = better; return next; });
          a = await api.lensAdvice(viewValues({ ...vals, camera_models: picked }, source), k);
        }
        setAdvice((old) => ({ ...old, [k]: a }));
      } catch {
        setAdvice((old) => { const next = { ...old }; delete next[k]; return next; });
      }
    }
  }, [source]);

  // A lens file is checked against the photos, or the saved detections, as soon as both are known.
  const lensPath = lens?.path ?? null;
  const scanned = scan?.number_camera ?? 0;
  // What the lens checks look at: the photo folder, or the detections file when there are no photos.
  const viewsFrom = source === "detections" ? (kp?.path ?? null) : source === "photos" ? (rootPath || null) : null;
  const viewCams = source === "detections" ? (kp?.number_camera ?? 0) : scanned;
  useEffect(() => {
    if (!lensPath || !viewCams || !viewsFrom) { setHealth({}); return; }
    let live = true;
    const cams = (lens?.cameras ?? []).map((c) => c.index).filter((c) => c < viewCams);
    setHealth(Object.fromEntries(cams.map((c) => [c, "busy" as const])));
    (async () => {
      for (const c of cams) {
        try {
          const h = await api.lensHealth(viewValues(payloadRef.current, source), c);
          if (live) setHealth((old) => ({ ...old, [c]: h }));
        } catch {
          if (live) setHealth((old) => { const next = { ...old }; delete next[c]; return next; });
        }
      }
    })();
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lensPath, viewCams, viewsFrom, values.number_x_square, values.number_y_square, values.number_board]);

  // What was already calibrated from these photos, and whether its detections fit these settings.
  useEffect(() => {
    if (!scanned || !rootPath || source !== "photos") { setHist(null); return; }
    let live = true;
    const t = window.setTimeout(() => {
      api.captureHistory(payloadRef.current).then((h) => live && setHist(h), () => live && setHist(null));
    }, 250);
    return () => { live = false; window.clearTimeout(t); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scanned, rootPath, source, values.number_x_square, values.number_y_square, values.number_board, values.cam_prefix]);

  function pickFolder(path: string, ref: string) {
    const key = picker?.key ?? "root_path";
    setPicker(null);
    set(key, path);
    if (key === "root_path") {
      const k = Number(ref.split(":", 1)[0]);
      if (Number.isInteger(k)) setRoot(k);
      runScan(path);
    }
  }

  function pickFile(path: string) {
    const key = picker?.key ?? "";
    setPicker(null);
    if (key === "@config") {
      api.sourceConfig(path).then(
        (doc) => adopt(doc.values, `Settings loaded from ${doc.path}.`, doc.unknown),
        (e: Error) => setFileError(e.message),
      );
    } else if (key === "@detections") loadDetections(path);
    else if (key === "@lens") loadLens(path);
    else set(key, path);
  }

  function chooseSource(next: Source) {
    setSource(next);
    setFileError(null);
    if (next === "result" && results == null) api.runs().then(setResults, () => setResults([]));
    if (next === "photos" && kp) {                 // back from detections to photos
      setKp(null);
      setValues((old) => { const v = { ...old }; delete v.keypoints_path; delete v.number_camera; return v; });
      setModels([]);
    }
  }

  function importFile(file: File) {
    file.text().then((text) => api.importConfig(text)).then(
      (doc) => adopt(doc.values, `Settings loaded from ${file.name}.`, doc.unknown),
      (e: Error) => setNotice(`Could not read ${file.name}: ${e.message}`),
    );
  }

  function runBoardCheck(vals?: Values) {
    setBoardBusy(true);
    setBoardError(null);
    api.checkBoard(vals ?? payload).then(
      (doc) => { setBoard(doc); setBoardBusy(false); },
      (e: Error) => { setBoardError(e.message); setBoardBusy(false); },
    );
  }

  function loadPreview() {
    setPreviewBusy(true);
    api.checkConfig(payload).then(
      (doc) => { setPreview(doc); setPreviewBusy(false); setProblems(doc.problems); },
      (e: Error) => { setPreviewBusy(false); setStartError(e.message); },
    );
  }

  function start() {
    // The strict profile cannot run without its two limits: say so here, by the fields,
    // before the calibration's own loader refuses the file in its own words.
    if (payload.profile === "production") {
      const missing = ["max_rotation_sigma_deg", "max_centre_sigma"].filter((k) => payload[k] == null);
      if (missing.length) {
        setProblems(missing.map((key) => ({ key, message: "Strict checking needs this limit." })));
        setStartError("Strict checking needs both limits. Fill them in under step 4, or choose the standard checking.");
        return;
      }
    }
    setStarting(true);
    setStartError(null);
    const reused = reuse && source === "photos" ? hist?.reusable ?? null : null;
    api.startJob(root, name, payload, reused).then(
      (job) => { window.location.hash = href({ page: "job", id: job.id }); },
      (e: Error) => {
        setStarting(false);
        setStartError(e.message);
        setProblems(e instanceof ApiError ? e.problems : []);
        summary.current?.scrollIntoView({ behavior: "smooth", block: "center" });
      },
    );
  }

  if (fatal)
    return <Page><ErrorBox title="Could not load the settings form" detail={fatal} /></Page>;
  if (!schema) return <Page><Loading what="settings" /></Page>;
  if (!canRun)
    return (
      <Page>
        <Callout tone="info" title="This Studio only shows results">
          It was started with <Mono>--read-only</Mono>. Start it without that option to set up and run calibrations here.
        </Callout>
      </Page>
    );

  const props = schema.properties;
  const nx = Number(values.number_x_square ?? props.number_x_square.default);
  const ny = Number(values.number_y_square ?? props.number_y_square.default);
  const nb = Number(values.number_board ?? 1) || 1;
  const production = values.profile === "production";
  const usesKeypoints = typeof values.keypoints_path === "string" && values.keypoints_path !== "";
  const photosReady = (!!scan && scan.number_camera > 0) || (usesKeypoints && nCam > 0);
  const sized = values.square_size != null;
  const ready = photosReady && name.trim() !== "" && !scanning;
  const blocking = scan?.problems ?? [];
  const photoSource = source === "photos" && rootPath && scan?.number_camera ? { root: rootPath, prefix: scan.cam_prefix } : null;
  const cameras: CameraInfo[] = Array.from({ length: nCam }, (_, k) => {
    const d = kp?.cameras.find((x) => x.index === k);
    const c = scan?.cameras[k];
    return { index: k, name: c?.folder ?? `camera ${k}`, photos: c?.images ?? d?.frames ?? null, width: c?.width ?? d?.width ?? null, height: c?.height ?? d?.height ?? null };
  });
  const samples = Object.fromEntries((board?.cameras ?? []).map((c) => [c.index, c.sample]));
  const misfit = Object.values(advice).some((a) => a !== "busy" && a.verdict === "misfit");
  const setModel = (k: number, m: string) => setModels((old) => { const next = Array.from({ length: nCam }, (_, i) => old[i] ?? "radtan"); next[k] = m; return next; });
  const reusedName = hist?.runs.find((h) => h.id === hist.reusable)?.name;

  return (
    <Page
      onDragOver={(e) => { if (e.dataTransfer.types.includes("Files")) { e.preventDefault(); setDragging(true); } }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const f = e.dataTransfer.files?.[0];
        if (f && /\.(ya?ml|txt)$/i.test(f.name)) importFile(f);
        else if (f) setFileError(`${f.name} is not a settings file. Drop a calib_param.yml here. Corner files and lens files are chosen with their own buttons, because the calibration reads them from disk.`);
      }}
    >
      {dragging && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-s-accent/25 backdrop-blur-sm pointer-events-none" aria-hidden="true">
          <div className="rounded-2xl bg-s-surface border-4 border-dashed border-s-accent px-10 py-8 text-2xl font-bold">Drop your settings file to fill in this page</div>
        </div>
      )}
      <header className="grid gap-2">
        <a href={href({ page: "runs" })} className="text-[15px] font-medium text-s-accent hover:underline justify-self-start">← All results</a>
        <h1 className="text-3xl sm:text-4xl font-bold tracking-tight">New calibration</h1>
        <p className="text-lg text-s-muted max-w-[60ch]">Three things are needed: your photos, your board, and the kind of lens on each camera. Studio can work out most of it from the photos.</p>
        <input ref={fileInput} type="file" accept=".yml,.yaml,.txt" className="sr-only"
          aria-label="Load settings from a calib_param.yml file on your own computer"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) importFile(f); e.target.value = ""; }} />
      </header>

      <fieldset className="grid gap-3">
        <legend className="text-lg font-semibold mb-3">What do you have?</legend>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {SOURCES.map((o) => {
            const on = source === o.id;
            return (
              <label key={o.id}
                className={`relative rounded-2xl border-2 p-4 cursor-pointer grid grid-cols-[auto_minmax(0,1fr)] gap-x-3.5 gap-y-1 items-start transition
                  ${on ? "border-s-accent bg-s-surface shadow-[0_0_0_4px_rgb(3_105_161/0.12)]" : "border-transparent bg-s-surface shadow-sm hover:border-s-field-line"}`}>
                <input type="radio" name="source" checked={on} onChange={() => chooseSource(o.id)} className="sr-only" />
                <span className={`row-span-2 size-12 rounded-xl grid place-items-center ${on ? "bg-s-accent text-white" : "bg-s-accent-soft text-s-accent"}`} aria-hidden="true">
                  <svg viewBox="0 0 24 24" className="size-6" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d={o.icon} /></svg>
                </span>
                <span className="text-[17px] font-semibold leading-tight pt-0.5">{o.title}</span>
                <span className="text-[15px] text-s-muted leading-snug">{o.text}</span>
                {on && <span className="absolute top-2.5 right-3 text-s-accent font-bold" aria-hidden="true">✓</span>}
              </label>
            );
          })}
        </div>
      </fieldset>

      {notice && <Callout tone="info">{notice}</Callout>}
      {fileError && <ErrorBox title="Could not use that file" detail={fileError} />}

      {source === "config" && (
        <Step n="→" title="Load your settings file">
          <Explain>Everything the file sets is filled in below, and you can still change any of it before starting. You can also drop the file anywhere on this page.</Explain>
          <div className="flex flex-wrap gap-3">
            <Button kind="primary" onClick={() => setPicker({ mode: "file", key: "@config", title: "Choose a calib_param.yml" })}>Choose a file on this computer</Button>
            <Button onClick={() => fileInput.current?.click()}>Upload one from my device</Button>
          </div>
        </Step>
      )}

      {source === "result" && (
        <Step n="→" title="Choose the result to start from" aside={results ? `${results.length} here` : undefined}>
          {!results && <Loading what="results" />}
          {results && results.length === 0 && <Explain>There are no results yet in the folders Studio was started with.</Explain>}
          {results && results.length > 0 && (
            <ul className="grid divide-y divide-s-line -my-2">
              {results.slice(0, 40).map((r) => (
                <li key={r.id} className="flex flex-wrap items-center gap-x-4 gap-y-2 py-3">
                  <div className="min-w-0 mr-auto">
                    <div className="text-base font-semibold break-all">{r.path || r.name}</div>
                    <div className="text-[15px] text-s-muted">{when(r.created_utc)} · {int(r.n_cameras)} cameras · {(r.models ?? []).join(", ") || "—"}</div>
                  </div>
                  <a href={href({ page: "repeat", id: r.id })} className={LINK_BUTTON}>Repeat exactly</a>
                  <a href={href({ page: "new", from: r.id })} className={LINK_BUTTON}>Change something first</a>
                </li>
              ))}
            </ul>
          )}
        </Step>
      )}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_22rem] items-start">
        <div className="grid gap-6 min-w-0">
          {/* 1 ---------------------------------------------------------------------------- */}
          {source === "detections" ? (
            <Step n={1} title="Your saved corners" state={kp ? "done" : "todo"}>
              <Explain>The <Mono>detected_keypoints_data.yml</Mono> file an earlier calibration wrote. Its corners are used as they are, so the photos are not needed.</Explain>
              <div className="flex flex-wrap gap-3">
                <input id="keypoints_path" aria-label="Corners file" className={`${INPUT} font-studio-mono !text-sm flex-1 min-w-60`} readOnly placeholder="No file chosen yet"
                  value={typeof values.keypoints_path === "string" ? values.keypoints_path : ""} />
                <Button kind={kp ? "secondary" : "primary"} onClick={() => setPicker({ mode: "file", key: "@detections", title: "Choose a corners file" })}>{kp ? "Choose another file" : "Choose the file"}</Button>
              </div>
              {problemOf("keypoints_path") && <div className="text-[15px] font-medium text-s-bad" role="alert">{problemOf("keypoints_path")}</div>}
              {kp && (
                <Callout tone={kp.problems.length ? "warn" : "ok"}
                  title={`${kp.number_camera} camera${kp.number_camera === 1 ? "" : "s"}, ${int(kp.cameras.reduce((a, c) => a + c.sightings, 0))} board views`}>
                  {kp.number_board > 1 ? `${kp.number_board} boards. ` : ""}In step 2, describe the board exactly as it was when these corners were found.
                </Callout>
              )}
              {kp?.problems.map((p) => <Callout key={p} tone="warn">{p}</Callout>)}
              {kp?.reused_target && nb > 1 && (
                <Callout tone="warn" title="A saved target sits next to this file">
                  With several boards, the calibration will load <Mono>{kp.reused_target}</Mono> instead of working the target out again from these corners. Move the corners file to an empty folder if you want it worked out again.
                </Callout>
              )}
            </Step>
          ) : (
            <Step n={1} title="Your photos" state={scan && scan.number_camera > 0 ? "done" : "todo"}>
              {!rootPath && !scanning && (
                <div className="rounded-xl border-2 border-dashed border-s-field-line bg-s-sunken p-6 grid gap-5 sm:grid-cols-[minmax(0,1fr)_auto] items-center">
                  <div className="grid gap-3">
                    <Explain>Choose the folder that holds one subfolder per camera. Photos taken at the same moment have the same number in their file name.</Explain>
                    <Button kind="primary" className="justify-self-start min-h-12 !text-base px-5" onClick={() => setPicker({ mode: "folder", key: "root_path", title: "Folder with the photos" })}>Choose the photo folder</Button>
                  </div>
                  <pre className="font-studio-mono text-sm leading-relaxed text-s-muted bg-s-surface rounded-lg border border-s-line px-4 py-3" aria-label="Example folder layout">{"my_rig/\n├─ Cam_001/\n│   ├─ 0001.png\n│   └─ 0002.png\n└─ Cam_002/\n    ├─ 0001.png\n    └─ 0002.png"}</pre>
                </div>
              )}
              {rootPath && (
                <div className="flex flex-wrap gap-3">
                  <input id="root_path" aria-label="Folder with the photos" className={`${INPUT} font-studio-mono !text-sm flex-1 min-w-60`} readOnly value={rootPath} />
                  <Button onClick={() => setPicker({ mode: "folder", key: "root_path", title: "Folder with the photos" })}>Choose another folder</Button>
                </div>
              )}
              {problemOf("root_path") && <div className="text-[15px] font-medium text-s-bad" role="alert">{problemOf("root_path")}</div>}
              {scanning && <Loading what="the folder" />}
              {scanError && <ErrorBox title="Could not look at this folder" detail={scanError} />}
              {scan && scan.number_camera > 0 && (
                <>
                  <Callout tone={blocking.length ? "warn" : "ok"}
                    title={`Found ${scan.number_camera} camera${scan.number_camera === 1 ? "" : "s"} and ${int(scan.cameras.reduce((a, c) => a + c.images, 0))} photos`}>
                    {scan.number_camera > 1 && <>{int(scan.shared_frames)} moments were photographed by two or more cameras at once.</>}
                  </Callout>
                  <div className="grid gap-4 grid-cols-[repeat(auto-fill,minmax(150px,1fr))]">
                    {scan.cameras.slice(0, 24).map((c) => (
                      <PhotoThumb key={c.index} src={api.thumbUrl(rootPath, c.index, 0, scan.cam_prefix)} width={c.width} height={c.height}
                        caption={<><b className="font-semibold">{c.folder}</b><span className="block text-s-muted">{int(c.images)} photos</span></>} />
                    ))}
                  </div>
                  {scan.cameras.length > 24 && <Explain>and {scan.cameras.length - 24} more cameras.</Explain>}
                </>
              )}
              {blocking.map((p) => <Callout key={p} tone="warn">{p}</Callout>)}

              {excluded.length > 0 && (
                <div className="border-t border-s-line pt-5 grid gap-3">
                  <div className="text-base font-semibold">{excluded.length} photo{excluded.length === 1 ? " is" : "s are"} left out of this calibration</div>
                  <Explain>They take no part at all. Click one to put it back.</Explain>
                  <ul className="flex flex-wrap gap-2">
                    {excluded.map((k) => (
                      <li key={k}>
                        <button type="button" aria-label={`Put ${viewWords(k)} back`} onClick={() => set("exclude_views", excluded.filter((x) => x !== k))}
                          className="inline-flex items-center gap-2 rounded-full border-[1.5px] border-s-warn/50 bg-s-warn-soft pl-3.5 pr-3 min-h-10 text-[15px] font-medium cursor-pointer hover:border-s-warn">
                          {viewWords(k)} <span aria-hidden="true" className="text-lg leading-none">×</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                  {problemOf("exclude_views") && <div className="text-[15px] font-medium text-s-bad" role="alert">{problemOf("exclude_views")}</div>}
                </div>
              )}

              {hist && hist.runs.length > 0 && (
                <div className="border-t border-s-line pt-5 grid gap-3">
                  <div className="text-base font-semibold">You calibrated these photos before</div>
                  {hist.reusable ? (
                    <label className={`flex items-start gap-3.5 cursor-pointer rounded-xl border-2 p-4 ${reuse ? "border-s-accent bg-s-accent-soft" : "border-s-line"}`}>
                      <input type="checkbox" checked={reuse} onChange={(e) => setReuse(e.target.checked)} className="mt-1 size-5 accent-s-accent" />
                      <span className="grid gap-1">
                        <span className="text-base font-semibold">Skip finding the corners again</span>
                        <span className="text-[15px] text-s-muted leading-relaxed">
                          <b className="text-s-text">{reusedName}</b> already found them in these same photos with this board, so the new calibration starts in seconds. That result is copied from, never changed. Lens types and every other setting can still differ.
                        </span>
                      </span>
                    </label>
                  ) : (
                    <Explain>The corners will be found again, because for {hist.runs[0].name} {hist.runs[0].reuse?.why ?? "the saved corners do not fit these settings"}.</Explain>
                  )}
                  <details>
                    <summary className="cursor-pointer text-[15px] font-semibold text-s-accent hover:underline">Show the {hist.runs.length} earlier result{hist.runs.length === 1 ? "" : "s"}</summary>
                    <ul className="grid divide-y divide-s-line mt-2">
                      {hist.runs.slice(0, 8).map((h) => (
                        <li key={h.id} className="flex flex-wrap items-baseline gap-x-4 gap-y-1 py-2.5 text-[15px]">
                          <a href={href({ page: "run", id: h.id })} className="font-semibold text-s-accent hover:underline break-all">{h.name}</a>
                          <span className="text-s-muted mr-auto">{when(h.created_utc)}{h.median_px != null && ` · typical error ${num(h.median_px)} px`}</span>
                          {h.has_config && <a href={href({ page: "new", from: h.id })} className="text-s-accent underline">Use its settings</a>}
                        </li>
                      ))}
                    </ul>
                  </details>
                </div>
              )}
            </Step>
          )}

          {/* 2 ---------------------------------------------------------------------------- */}
          <Step n={2} title="Your calibration board" state={sized && (board ? board.found > 0 : true) ? "done" : "todo"} aside="ChArUco board">
            <BoardStep values={values} set={set} payload={payload} photos={photoSource} unit={unit} setUnit={setUnit}
              defaults={{ nx: Number(props.number_x_square.default), ny: Number(props.number_y_square.default), size: Number(props.square_size.default),
                ls: Number(props.length_square.default), lm: Number(props.length_marker.default) }}
              check={board} onCheck={runBoardCheck} checkBusy={boardBusy} checkError={boardError} problemOf={problemOf} />
          </Step>

          {/* 3 ---------------------------------------------------------------------------- */}
          <Step n={3} title="Your cameras and their lenses" state={nCam > 0 && !misfit ? "done" : "todo"} aside={nCam ? `${nCam} camera${nCam === 1 ? "" : "s"}` : undefined}>
            {nCam === 0
              ? <Explain>{source === "detections" ? "Choose the corners file in step 1. The cameras are read from it." : "Choose the photo folder in step 1. The cameras are read from it."}</Explain>
              : <CameraStep cameras={cameras} models={models} setModel={setModel} setAll={(m) => setModels(Array.from({ length: nCam }, () => m))}
                  advice={advice} onMeasure={() => void runAdvice(payload, nCam, true)} measuring={Object.values(advice).some((a) => a === "busy")}
                  canMeasure={!!viewsFrom && viewCams > 0} measureWhat={source === "detections" ? "saved detections" : "photos"}
                  photos={photoSource} samples={samples} lens={lens} health={health}
                  onLensBrowse={() => setPicker({ mode: "file", key: "@lens", title: "Choose a lens file" })}
                  onLensRemove={() => { setLens(null); set("cam_params_path", ""); set("fix_intrinsic", ""); }}
                  fixed={values.fix_intrinsic === true} onFixed={(v) => set("fix_intrinsic", v ? true : "")}
                  problem={problemOf("camera_models")} lensProblem={problemOf("fix_intrinsic")} />}
          </Step>

          {/* 4 ---------------------------------------------------------------------------- */}
          <Step n={4} title="What should happen with a doubtful result?" state="done">
            <div className="grid gap-3 sm:grid-cols-2">
              {[{ id: "", title: "Give it to me with warnings", tag: "standard", text: "You always get a result. If a camera is weak or a check fails, the result page says so clearly. Best while you are still improving your capture." },
                { id: "production", title: "Refuse it", tag: "strict", text: "No result is written unless every check passes and the cameras are known as precisely as you ask below. Best when the result goes straight into a product." }].map((o) => {
                const on = (values.profile ?? "") === o.id || (o.id === "" && values.profile === "default");
                return (
                  <button key={o.id || "standard"} type="button" role="radio" aria-checked={on} onClick={() => set("profile", o.id)}
                    className={`text-left rounded-xl border-2 p-4 grid gap-1.5 cursor-pointer transition ${on ? "border-s-accent bg-s-accent-soft" : "border-s-line bg-s-surface hover:border-s-field-line"}`}>
                    <span className="text-[17px] font-semibold flex items-center gap-2.5">
                      <span className={`size-5 rounded-full border-2 grid place-items-center ${on ? "border-s-accent" : "border-s-field-line"}`} aria-hidden="true">{on && <span className="size-2.5 rounded-full bg-s-accent" />}</span>
                      {o.title}
                      <span className="rounded-full bg-s-surface border border-s-line text-s-muted text-[13px] font-medium px-2 py-0.5">{o.tag}</span>
                    </span>
                    <span className="text-[15px] text-s-muted leading-relaxed">{o.text}</span>
                  </button>
                );
              })}
            </div>
            {problemOf("profile") && <div className="text-[15px] font-medium text-s-bad" role="alert">{problemOf("profile")}</div>}
            <div className="grid gap-3">
              <div className="text-base font-semibold">How precisely must the cameras be known? {production ? <span className="text-s-bad">Needed for strict checking.</span> : <span className="font-normal text-s-muted">Optional.</span>}</div>
              <Explain>Between any two cameras, Studio estimates how uncertain their angle and their distance are. {production ? "A result less certain than these limits is refused." : "With limits, the result warns when a pair is less certain than you need."}</Explain>
              <div className="flex flex-wrap gap-x-8 gap-y-4">
                <label className="grid gap-1.5 text-base font-semibold">Angle between two cameras, to within
                  <NumberBox id="max_rotation_sigma_deg" value={values.max_rotation_sigma_deg} fallback={null} unit="°" onChange={(v) => set("max_rotation_sigma_deg", v)} />
                  <span className="text-[15px] font-normal text-s-muted">0.05° is a strict start.</span>
                  {problemOf("max_rotation_sigma_deg") && <span className="text-[15px] font-medium text-s-bad" role="alert">{problemOf("max_rotation_sigma_deg")}</span>}
                </label>
                <label className="grid gap-1.5 text-base font-semibold">Position of one camera from another, to within
                  <NumberBox id="max_centre_sigma" value={values.max_centre_sigma} fallback={null} unit={unit === "?" ? undefined : unit} onChange={(v) => set("max_centre_sigma", v)} />
                  <span className="text-[15px] font-normal text-s-muted">{unit === "?" ? "In the unit of your square size." : `In ${unit}, like your square size.`}</span>
                  {problemOf("max_centre_sigma") && <span className="text-[15px] font-medium text-s-bad" role="alert">{problemOf("max_centre_sigma")}</span>}
                </label>
              </div>
            </div>
          </Step>

          {/* 5 ---------------------------------------------------------------------------- */}
          <Settings schema={schema} values={values} problems={problems} set={set} boards={nb} source={source} strict={production}
            refRun={from && fromName ? { id: from, name: fromName }
              : source === "photos" && hist?.runs.length ? { id: hist.runs[0].id, name: hist.runs[0].name } : null}
            onBrowse={(key) => setPicker({ mode: "file", key, title: "Choose the file" })} />
        </div>

        {/* start -------------------------------------------------------------------------- */}
        <aside ref={summary} className="grid gap-5 lg:sticky lg:top-4">
          <section className="rounded-2xl bg-s-surface border border-s-line shadow-[0_1px_2px_rgb(15_23_42/0.06),0_8px_24px_rgb(15_23_42/0.08)] p-5 grid gap-4">
            <h2 className="text-xl font-semibold tracking-tight">Ready to start?</h2>
            <ul className="grid gap-2.5">
              <Check ok={photosReady}>{photosReady ? `${nCam} camera${nCam === 1 ? "" : "s"}${usesKeypoints ? ", from saved corners" : ""}` : source === "detections" ? "Choose the corners file" : "Choose the photo folder"}</Check>
              <Check ok={sized} warn={!sized}>{sized ? `Board ${nx} × ${ny}, squares of ${String(values.square_size)}` : `Board ${nx} × ${ny}: enter the size of one square`}</Check>
              <Check ok={nCam > 0 && !misfit} warn={misfit}>{misfit ? "A lens type does not fit its photos (step 3)" : nCam > 0 ? lensSummary(models, nCam) : "Lens types"}</Check>
              <Check ok>{production ? "A doubtful result is refused" : "A doubtful result comes with warnings"}</Check>
              {source === "photos" && hist?.reusable && reuse && <Check ok>Corners reused from {reusedName}</Check>}
              {excluded.length > 0 && <Check ok>{`${excluded.length} photo${excluded.length === 1 ? "" : "s"} left out`}</Check>}
            </ul>
            <div className="grid gap-1.5 border-t border-s-line pt-4">
              <label htmlFor="run-name" className="text-base font-semibold">Name this calibration</label>
              <input id="run-name" className={INPUT} value={name} onChange={(e) => { setName(e.target.value); setProblems((o) => o.filter((p) => p.key !== "name")); }} />
              {problemOf("name") && <div className="text-[15px] font-medium text-s-bad" role="alert">{problemOf("name")}</div>}
              {roots.length > 1 ? (
                <label className="grid gap-1.5 text-base font-semibold mt-2">Save the result in
                  <select id="run-root" className={INPUT} value={root} onChange={(e) => setRoot(Number(e.target.value))}>
                    {roots.map((r, k) => <option key={r.path} value={k}>{r.path}</option>)}
                  </select>
                </label>
              ) : roots.length === 1 && <div className="text-[14px] text-s-muted break-all">The result is saved in a new folder with this name, inside {roots[0].path}</div>}
            </div>
            {startError && <Callout tone="bad" title="Not started">{startError}</Callout>}
            {problems.filter((p) => p.key !== "").length > 0 && (
              <Callout tone="bad" title="Some settings need attention">
                <ul className="grid gap-1 mt-1">
                  {problems.filter((p) => p.key !== "").map((p) => <li key={p.key}><Mono>{p.key}</Mono> {p.message}</li>)}
                </ul>
              </Callout>
            )}
            <Button kind="primary" onClick={start} busy={starting} disabled={!ready} className="min-h-14 !text-lg">Start calibration</Button>
            {!ready && !scanning && <Explain>{photosReady ? "Give the calibration a name." : source === "detections" ? "Choose the corners file to continue." : "Choose the folder with your photos to continue."}</Explain>}
          </section>

          <details className="rounded-2xl bg-s-surface border border-s-line p-5 group">
            <summary className="cursor-pointer text-base font-semibold text-s-accent hover:underline">Prefer a terminal? See the settings file</summary>
            <div className="grid gap-3 mt-3">
              <Explain>Starting writes your choices as <Mono>calib_param.yml</Mono> into the result folder. The same file runs with <Mono>ds-msp-calibrate-rig --config</Mono>.</Explain>
              <Button onClick={loadPreview} busy={previewBusy}>Check and show the file</Button>
              {preview && (
                <>
                  {preview.ok
                    ? <Callout tone="ok" title="These settings are valid" />
                    : <Callout tone="bad" title="These settings have problems">
                        <ul className="grid gap-1 mt-1">{preview.problems.map((p, k) => <li key={k}>{p.key && <Mono>{p.key}</Mono>} {p.message}</li>)}</ul>
                      </Callout>}
                  {preview.warnings.map((w) => <Callout key={w} tone="warn">{w}</Callout>)}
                  <pre className="font-studio-mono text-[13px] leading-relaxed bg-s-sunken border border-s-line rounded-lg p-3 overflow-x-auto max-h-80">{preview.yaml}</pre>
                </>
              )}
            </div>
          </details>
        </aside>
      </div>

      {picker && (
        <FolderPicker mode={picker.mode} title={picker.title} onClose={() => setPicker(null)}
          onPick={(path, ref) => (picker.mode === "folder" ? pickFolder(path, ref) : pickFile(path))} />
      )}
    </Page>
  );
}

const LINK_BUTTON = "inline-flex items-center rounded-lg px-4 min-h-11 text-[15px] font-semibold border-[1.5px] border-s-field-line/70 bg-s-surface shadow-sm hover:bg-s-sunken hover:border-s-text/50";

function Page({ children, ...drop }: { children: ReactNode } & Pick<React.HTMLAttributes<HTMLElement>, "onDragOver" | "onDragLeave" | "onDrop">) {
  return <main {...drop} className="mx-auto w-full max-w-7xl px-4 sm:px-6 py-8 grid gap-7 text-base">{children}</main>;
}

function lensSummary(models: string[], n: number): string {
  const counts = new Map<string, number>();
  for (let k = 0; k < n; k++) counts.set(models[k] ?? "radtan", (counts.get(models[k] ?? "radtan") ?? 0) + 1);
  const word = (m: string) => (m === "radtan" ? "standard" : m === "kb" ? "fisheye" : m);
  return [...counts].map(([m, c]) => (counts.size === 1 ? `All lenses ${word(m)}` : `${c} ${word(m)}`)).join(", ");
}

function Check({ ok, warn, children }: { ok: boolean; warn?: boolean; children: ReactNode }) {
  return (
    <li className="flex items-start gap-2.5 text-base">
      <span className={`mt-0.5 size-5 shrink-0 rounded-full grid place-items-center text-xs font-bold ${warn ? "bg-s-warn text-white" : ok ? "bg-s-ok text-white" : "border-2 border-s-field-line"}`} aria-hidden="true">
        {warn ? "!" : ok ? "✓" : ""}
      </span>
      <span className={warn ? "text-s-warn font-medium" : ok ? "" : "text-s-muted"}>{children}</span>
    </li>
  );
}
