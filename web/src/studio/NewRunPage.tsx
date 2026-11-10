// Set up a calibration and start it. The form holds only the settings the user changed; the
// server writes them as calib_param.yml into a new result folder and runs the rig CLI on it,
// so a run started here and the same file run from a terminal are the same calibration.
import type React from "react";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { href } from "../router";
import {
  api, ApiError, type BoardCheck, type ConfigCheck, type ConfigSchema, type IntrinsicsScan,
  type KeypointsScan, type Problem, type RunSummary, type ScanDoc, type SchemaProp, type Values,
} from "./api";
import { FolderPicker } from "./FolderPicker";
import { int, num, when } from "./format";
import { Badge, Button, Callout, Card, ErrorBox, Field, INPUT, Loading, Note } from "./ui";

const LENSES: { id: string; label: string; hint: string }[] = [
  { id: "radtan", label: "Standard lens (RadTan)", hint: "ordinary lenses, up to about 120°" },
  { id: "kb", label: "Fisheye (KB)", hint: "wide and fisheye lenses; a safe first choice for them" },
  { id: "ds", label: "Fisheye (Double Sphere)", hint: "fisheye lenses, also past 180°" },
  { id: "eucm", label: "Fisheye (EUCM)", hint: "fisheye lenses" },
  { id: "ucm", label: "Fisheye (UCM)", hint: "simplest fisheye model; not for ordinary lenses" },
];

// Settings with their own place in the form, or set by Studio itself.
const PLACED = new Set([
  "root_path", "cam_prefix", "number_camera", "camera_models", "number_x_square",
  "number_y_square", "square_size", "number_board", "profile", "max_rotation_sigma_deg",
  "max_centre_sigma", "save_path", "webviewer", "keypoints_path", "cam_params_path", "fix_intrinsic",
]);
const FILE_KEYS = new Set(["object_path", "accuracy_profile_path"]);

// A picker either fills one setting, or loads a whole starting point ("@config", ...).
type Picker = { mode: "folder" | "file"; key: string; title: string } | null;
type Source = "photos" | "config" | "detections" | "result";

const SOURCES: { id: Source; title: string; text: string }[] = [
  { id: "photos", title: "Photos", text: "A folder of calibration photos, one subfolder per camera." },
  { id: "config", title: "A config file", text: "A calib_param.yml from DS-MSP or MC-Calib. It fills in everything it sets." },
  { id: "detections", title: "Saved detections", text: "A detected_keypoints_data.yml from an earlier run. The photos are not read again." },
  { id: "result", title: "A finished result", text: "Repeat a calibration exactly, or change one thing and run it again." },
];

function defaultName(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `calibration_${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}`;
}

function shown(v: unknown): string {
  if (v == null) return "not set";
  if (Array.isArray(v)) return v.length ? v.join(", ") : "empty";
  if (typeof v === "boolean") return v ? "yes" : "no";
  return String(v);
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
  const [name, setName] = useState(defaultName);
  const [root, setRoot] = useState(0);
  const [roots, setRoots] = useState<{ name: string; path: string }[]>([]);
  const [problems, setProblems] = useState<Problem[]>([]);
  const [startError, setStartError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [preview, setPreview] = useState<ConfigCheck | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
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
    const given = Array.isArray(next.camera_models) ? (next.camera_models as string[]) : [];
    delete next.camera_models;
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
        adopt(doc.values, `Started from the settings of ${doc.name}.`, doc.unknown);
        setName(`${doc.name}_again`);
        const k = Number(from.split(":", 1)[0]);
        if (Number.isInteger(k)) setRoot(k);
      },
      (e: Error) => setNotice(`Could not read the settings of that run: ${e.message}`),
    );
  }, [from, adopt]);

  const payload = useMemo<Values>(() => {
    const out: Values = { ...values };
    if (nCam) out.number_camera = nCam;
    if (models.length) out.camera_models = models.slice(0, nCam || models.length);
    return out;
  }, [values, models, nCam]);

  const problemOf = (key: string) => problems.find((p) => p.key === key)?.message ?? null;

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

  function runBoardCheck() {
    setBoardBusy(true);
    setBoardError(null);
    api.checkBoard(payload).then(
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
        setProblems(missing.map((key) => ({ key, message: "Production checks need this limit." })));
        setStartError("Production checks need both uncertainty limits. Fill them in under step 4, or choose Standard.");
        return;
      }
    }
    setStarting(true);
    setStartError(null);
    api.startJob(root, name, payload).then(
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
          It was started with <code className="font-studio-mono">--read-only</code>. Start it
          without that option to set up and run calibrations here.
        </Callout>
      </Page>
    );

  const props = schema.properties;
  const nx = Number(values.number_x_square ?? props.number_x_square.default);
  const ny = Number(values.number_y_square ?? props.number_y_square.default);
  const sq = Number(values.square_size ?? props.square_size.default);
  const production = values.profile === "production";
  const usesKeypoints = typeof values.keypoints_path === "string" && values.keypoints_path !== "";
  const photosReady = (!!scan && scan.number_camera > 0) || (usesKeypoints && nCam > 0);
  const ready = photosReady && name.trim() !== "" && !scanning;
  const blocking = scan?.problems ?? [];

  return (
    <Page
      onDragOver={(e) => { if (e.dataTransfer.types.includes("Files")) { e.preventDefault(); setDragging(true); } }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const f = e.dataTransfer.files?.[0];
        if (f && /\.(ya?ml|txt)$/i.test(f.name)) importFile(f);
        else if (f) setFileError(`${f.name} is not a .yml file. Drop a calib_param.yml here; choose detections and lens files with Browse, so the calibration can read them from disk.`);
      }}
    >
      {dragging && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-s-accent/20 pointer-events-none" aria-hidden="true">
          <div className="rounded-xl bg-s-surface border-2 border-dashed border-s-accent px-8 py-6 text-lg font-semibold">Drop the calib_param.yml to load it</div>
        </div>
      )}
      <div className="flex flex-wrap items-end gap-4">
        <div className="grid gap-1 min-w-0">
          <a href={href({ page: "runs" })} className="text-xs text-s-accent hover:underline">← All results</a>
          <h1 className="text-2xl font-semibold">New calibration</h1>
          <p className="text-s-muted text-sm max-w-prose">
            Tell DS-MSP where the photos are, what the board looks like and which lenses the
            cameras have. Everything else has a default.
          </p>
        </div>
        <input
          ref={fileInput}
          type="file"
          accept=".yml,.yaml,.txt"
          className="sr-only"
          aria-label="Load settings from a calib_param.yml file on your own computer"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) importFile(f); e.target.value = ""; }}
        />
      </div>

      <fieldset className="grid gap-2">
        <legend className="text-sm font-medium mb-2">What are you starting from?</legend>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          {SOURCES.map((o) => {
            const on = source === o.id;
            return (
              <label key={o.id}
                className={`rounded-lg border p-3 cursor-pointer grid gap-1 content-start ${on ? "border-s-accent bg-s-accent-soft" : "border-s-line bg-s-surface hover:bg-s-sunken"}`}>
                <span className="flex items-center gap-2 font-medium text-sm">
                  <input type="radio" name="source" checked={on} onChange={() => chooseSource(o.id)} />
                  {o.title}
                </span>
                <span className="text-xs text-s-muted leading-relaxed">{o.text}</span>
              </label>
            );
          })}
        </div>
      </fieldset>

      {notice && <Callout tone="info">{notice}</Callout>}
      {fileError && <ErrorBox title="Could not use that file" detail={fileError} />}

      {source === "config" && (
        <Card title="Load a config file">
          <div className="grid gap-3">
            <Note>
              Every setting in the file is filled in below, and you can change any of them before
              starting. Paths written relative to the file are resolved against its own folder.
            </Note>
            <div className="flex flex-wrap gap-2">
              <Button kind="primary" onClick={() => setPicker({ mode: "file", key: "@config", title: "Choose a calib_param.yml" })}>
                Choose a file on this computer
              </Button>
              <Button onClick={() => fileInput.current?.click()}>Upload one from my device</Button>
            </div>
            <Note>You can also drop a <Mono>.yml</Mono> file anywhere on this page.</Note>
          </div>
        </Card>
      )}

      {source === "result" && (
        <Card title="Start from a finished result" aside={results ? `${results.length} here` : undefined}>
          {!results && <Loading what="results" />}
          {results && results.length === 0 && <Note>No results under the folders Studio was started with.</Note>}
          {results && results.length > 0 && (
            <ul className="grid divide-y divide-s-line">
              {results.slice(0, 40).map((r) => (
                <li key={r.id} className="flex flex-wrap items-center gap-x-4 gap-y-2 py-2.5">
                  <div className="min-w-0 mr-auto">
                    <div className="font-medium text-sm break-all">{r.path || r.name}</div>
                    <div className="text-xs text-s-muted">{when(r.created_utc)} · {int(r.n_cameras)} cameras · {(r.models ?? []).join(", ") || "—"}</div>
                  </div>
                  <a href={href({ page: "repeat", id: r.id })} className="inline-flex items-center rounded-md px-3 min-h-9 text-sm font-medium border border-s-line bg-s-surface hover:bg-s-sunken">Repeat exactly</a>
                  <a href={href({ page: "new", from: r.id })} className="inline-flex items-center rounded-md px-3 min-h-9 text-sm font-medium border border-s-line bg-s-surface hover:bg-s-sunken">Change settings</a>
                </li>
              ))}
            </ul>
          )}
        </Card>
      )}

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_20rem] items-start">
        <div className="grid gap-5 min-w-0">
          {/* 1 ---------------------------------------------------------------------------- */}
          {source === "detections" ? (
            <Card title="1. Saved detections">
              <div className="grid gap-3">
                <Field htmlFor="keypoints_path" label="Detections file" problem={problemOf("keypoints_path")}
                  help={<>The <Mono>detected_keypoints_data.yml</Mono> an earlier calibration wrote. The corners in it are used as they are, so the photos are not needed.</>}>
                  <div className="flex gap-2">
                    <input id="keypoints_path" className={`${INPUT} font-studio-mono text-xs`} readOnly placeholder="No file chosen"
                      value={typeof values.keypoints_path === "string" ? values.keypoints_path : ""} />
                    <Button kind={kp ? "secondary" : "primary"} onClick={() => setPicker({ mode: "file", key: "@detections", title: "Choose a detections file" })}>Browse</Button>
                  </div>
                </Field>
                {kp && (
                  <Callout tone={kp.problems.length ? "warn" : "ok"}
                    title={`${kp.number_camera} camera${kp.number_camera === 1 ? "" : "s"}, ${int(kp.cameras.reduce((a, c) => a + c.sightings, 0))} board views`}>
                    {kp.number_board > 1 ? `${kp.number_board} boards. ` : ""}Set the board below exactly as it was when these corners were found.
                  </Callout>
                )}
                {kp?.problems.map((p) => <Callout key={p} tone="warn">{p}</Callout>)}
                {kp?.reused_target && Number(values.number_board ?? 1) > 1 && (
                  <Callout tone="warn" title="A saved target sits next to this file">
                    A calibration with several boards will load <Mono>{kp.reused_target}</Mono> instead of
                    rebuilding the target from these detections. Move the detections file to an empty
                    folder if you want the target rebuilt.
                  </Callout>
                )}
              </div>
            </Card>
          ) : (
          <Card title="1. Photos">
            <div className="grid gap-3">
              <Field
                htmlFor="root_path"
                label="Folder with the photos"
                problem={problemOf("root_path")}
                help={<>One subfolder per camera inside it: <Mono>Cam_001</Mono>, <Mono>Cam_002</Mono> and so on. Photos taken at the same moment carry the same number in their file name.</>}
              >
                <div className="flex gap-2">
                  <input id="root_path" className={`${INPUT} font-studio-mono text-xs`} readOnly value={rootPath}
                    placeholder="No folder chosen" />
                  <Button kind={rootPath ? "secondary" : "primary"}
                    onClick={() => setPicker({ mode: "folder", key: "root_path", title: "Folder with the photos" })}>
                    Browse
                  </Button>
                </div>
              </Field>
              {scanning && <Loading what="the folder" />}
              {scanError && <ErrorBox title="Could not look at this folder" detail={scanError} />}
              {scan && scan.number_camera > 0 && (
                <Callout tone={blocking.length ? "warn" : "ok"}
                  title={`Found ${scan.number_camera} camera${scan.number_camera === 1 ? "" : "s"} and ${int(scan.cameras.reduce((a, c) => a + c.images, 0))} photos`}>
                  {scan.number_camera > 1 && <>{int(scan.shared_frames)} moments were photographed by two or more cameras. </>}
                  Folders start with <Mono>{scan.cam_prefix}</Mono>.
                </Callout>
              )}
              {blocking.map((p) => <Callout key={p} tone="warn">{p}</Callout>)}
            </div>
          </Card>
          )}

          {/* 2 ---------------------------------------------------------------------------- */}
          <Card title="2. Calibration board" aside="ChArUco, 6×6 markers">
            <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_12rem]">
              <div className="grid gap-3 sm:grid-cols-2 content-start">
                <NumberField id="number_x_square" label="Squares across" value={values.number_x_square}
                  prop={props.number_x_square} problem={problemOf("number_x_square")} onChange={set} />
                <NumberField id="number_y_square" label="Squares down" value={values.number_y_square}
                  prop={props.number_y_square} problem={problemOf("number_y_square")} onChange={set} />
                <NumberField id="square_size" label="Size of one square" value={values.square_size}
                  prop={props.square_size} problem={problemOf("square_size")} onChange={set} decimal
                  help="Measure a printed square. Use any unit (m, cm or mm): every length in the result is in that unit." />
                <NumberField id="number_board" label="Number of boards" value={values.number_board}
                  prop={props.number_board} problem={problemOf("number_board")} onChange={set}
                  help="More than one when several boards are fixed together as one target." />
              </div>
              <BoardDrawing nx={nx} ny={ny} sq={sq} />
            </div>
            <div className="mt-4 grid gap-3">
              <div className="flex flex-wrap items-center gap-3">
                <Button onClick={runBoardCheck} busy={boardBusy} disabled={source === "detections" || !rootPath || !scan?.number_camera}>
                  Look for this board in my photos
                </Button>
                <span className="text-xs text-s-muted">{source === "detections" ? "Needs photos: with saved detections the board is not looked for again." : "Tries a few photos per camera. Takes a few seconds."}</span>
              </div>
              {boardError && <ErrorBox title="Could not check the board" detail={boardError} />}
              {board && (
                <Callout tone={board.found === 0 ? "bad" : board.found < board.tried ? "info" : "ok"} title={board.message}>
                  <ul className="grid gap-0.5 text-xs text-s-muted mt-1">
                    {board.cameras.map((c) => (
                      <li key={c.index} className="tnum">
                        <Mono>{c.folder}</Mono>: found in {c.found} of {c.tried} photos
                        {c.found > 0 && `, up to ${c.corners_max} of ${c.corners_of} corners`}
                      </li>
                    ))}
                  </ul>
                </Callout>
              )}
            </div>
          </Card>

          {/* 3 ---------------------------------------------------------------------------- */}
          <Card title="3. Cameras" aside={nCam ? `${nCam} in the rig` : undefined}>
            {nCam === 0 && (
              <Note>{source === "detections" ? "Choose the detections file first: the cameras are read from it." : "Choose the photo folder first: the cameras are read from it."}</Note>
            )}
            {nCam > 0 && (
              <div className="grid gap-3">
                {nCam > 1 && (
                  <div className="flex flex-wrap items-center gap-2 text-sm">
                    <label htmlFor="all-lenses" className="text-s-muted">Set every camera to</label>
                    <select id="all-lenses" className={`${INPUT} w-auto`} value=""
                      onChange={(e) => { if (e.target.value) setModels(Array.from({ length: nCam }, () => e.target.value)); }}>
                      <option value="">Choose a lens type…</option>
                      {LENSES.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
                    </select>
                  </div>
                )}
                <div className="overflow-x-auto rounded-lg border border-s-line">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-s-faint text-xs uppercase tracking-wide border-b border-s-line">
                        <th className="px-3 py-2 font-medium">Camera</th>
                        <th className="px-3 py-2 font-medium text-right">Photos</th>
                        <th className="px-3 py-2 font-medium">Image size</th>
                        <th className="px-3 py-2 font-medium">Lens type</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Array.from({ length: nCam }, (_, k) => {
                        const d = kp?.cameras.find((x) => x.index === k);
                        const c = scan?.cameras[k] ?? (d ? { folder: `camera ${k}`, images: d.frames, width: d.width, height: d.height } : undefined);
                        return (
                          <tr key={k} className="border-b border-s-line last:border-0">
                            <td className="px-3 py-2 font-studio-mono text-xs">{c?.folder ?? `camera ${k}`}</td>
                            <td className="px-3 py-2 text-right tnum">{c ? int(c.images) : "—"}</td>
                            <td className="px-3 py-2 tnum text-s-muted whitespace-nowrap">
                              {c?.width ? `${c.width} × ${c.height}` : "—"}
                            </td>
                            <td className="px-3 py-1.5">
                              <select className={`${INPUT} min-w-52`} aria-label={`Lens type of ${c?.folder ?? `camera ${k}`}`}
                                value={models[k] ?? "radtan"}
                                onChange={(e) => setModels((old) => {
                                  const next = Array.from({ length: nCam }, (_, i) => old[i] ?? "radtan");
                                  next[k] = e.target.value;
                                  return next;
                                })}>
                                {LENSES.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
                              </select>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                {problemOf("camera_models") && <div className="text-xs text-s-bad" role="alert">{problemOf("camera_models")}</div>}
                <div className="rounded-lg border border-s-line p-3 grid gap-3">
                  <div className="flex flex-wrap items-center gap-3">
                    <div className="mr-auto min-w-0">
                      <div className="text-sm font-medium">Lens values I already have</div>
                      <div className="text-xs text-s-muted leading-relaxed">
                        A lens file from an earlier calibration or the factory (<Mono>calibrated_cameras_data.yml</Mono> format). Optional.
                      </div>
                    </div>
                    {lens && <Button kind="quiet" onClick={() => { setLens(null); set("cam_params_path", ""); set("fix_intrinsic", ""); }}>Remove</Button>}
                    <Button onClick={() => setPicker({ mode: "file", key: "@lens", title: "Choose a lens file" })}>{lens ? "Change" : "Browse"}</Button>
                  </div>
                  {lens && (
                    <>
                      <ul className="grid gap-0.5 text-xs text-s-muted tnum">
                        {lens.cameras.slice(0, 12).map((c) => (
                          <li key={c.index}><Mono>camera {c.index}</Mono>: {c.model ?? "lens type not stated"}, focal {num(c.fx, 1)} px, centre {num(c.cx, 1)}, {num(c.cy, 1)}</li>
                        ))}
                        {lens.cameras.length > 12 && <li>and {lens.cameras.length - 12} more</li>}
                      </ul>
                      {lens.cameras.length < nCam && <Callout tone="warn">The file has {lens.cameras.length} of {nCam} cameras. The others are worked out from the photos.</Callout>}
                      <div className="grid gap-2 sm:grid-cols-2">
                        {[
                          { v: false, title: "Start from them and refine", text: "The lenses are adjusted to fit these photos. The lens type of each camera can still be changed above." },
                          { v: true, title: "Keep them fixed", text: "Only the camera positions are worked out. Needs every camera in the file, with the lens type the file states." },
                        ].map((o) => {
                          const on = (values.fix_intrinsic === true) === o.v;
                          return (
                            <label key={o.title} className={`rounded-lg border p-3 cursor-pointer grid gap-1 ${on ? "border-s-accent bg-s-accent-soft" : "border-s-line hover:bg-s-sunken"}`}>
                              <span className="flex items-center gap-2 font-medium text-sm">
                                <input type="radio" name="fix_intrinsic" checked={on} onChange={() => set("fix_intrinsic", o.v ? true : "")} />
                                {o.title}
                              </span>
                              <span className="text-xs text-s-muted leading-relaxed">{o.text}</span>
                            </label>
                          );
                        })}
                      </div>
                      {problemOf("fix_intrinsic") && <div className="text-xs text-s-bad" role="alert">{problemOf("fix_intrinsic")}</div>}
                    </>
                  )}
                </div>
                <details className="text-sm">
                  <summary className="cursor-pointer text-s-accent">Which lens type should I pick?</summary>
                  <ul className="mt-2 grid gap-1 text-s-muted">
                    {LENSES.map((l) => <li key={l.id}><span className="text-s-text font-medium">{l.label}</span>: {l.hint}.</li>)}
                    <li>A lens type that cannot bend enough for the lens gives a wrong rig with a normal-looking error. When unsure about a wide lens, use Fisheye (KB).</li>
                  </ul>
                </details>
              </div>
            )}
          </Card>

          {/* 4 ---------------------------------------------------------------------------- */}
          <Card title="4. How strict should the checks be?">
            <div className="grid gap-3 sm:grid-cols-2">
              {[
                { id: "", title: "Standard", text: "Calibrates and warns about weak cameras. The result is always written, with the warnings in it." },
                { id: "production", title: "Production", text: "Refuses to write a result that fails a check: an unconverged solve, a weak lens, or a camera position less certain than the limits you set." },
              ].map((o) => {
                const on = (values.profile ?? "") === o.id || (o.id === "" && values.profile === "default");
                return (
                  <label key={o.id || "standard"}
                    className={`rounded-lg border p-3 cursor-pointer grid gap-1 ${on ? "border-s-accent bg-s-accent-soft" : "border-s-line hover:bg-s-sunken"}`}>
                    <span className="flex items-center gap-2 font-medium text-sm">
                      <input type="radio" name="profile" checked={on} onChange={() => set("profile", o.id)} />
                      {o.title}
                    </span>
                    <span className="text-xs text-s-muted leading-relaxed">{o.text}</span>
                  </label>
                );
              })}
            </div>
            {production && (
              <div className="grid gap-3 sm:grid-cols-2 mt-4">
                <NumberField id="max_rotation_sigma_deg" label="Largest rotation uncertainty, degrees" decimal
                  value={values.max_rotation_sigma_deg} prop={props.max_rotation_sigma_deg}
                  problem={problemOf("max_rotation_sigma_deg")} onChange={set}
                  help="Between any two cameras, 1σ. Production needs this limit; 0.05 is a strict start." />
                <NumberField id="max_centre_sigma" label="Largest position uncertainty" decimal
                  value={values.max_centre_sigma} prop={props.max_centre_sigma}
                  problem={problemOf("max_centre_sigma")} onChange={set}
                  help="Of one camera seen from another, 1σ, in the unit of the square size." />
              </div>
            )}
            {problemOf("profile") && <div className="text-xs text-s-bad mt-2" role="alert">{problemOf("profile")}</div>}
          </Card>

          {/* 5 ---------------------------------------------------------------------------- */}
          <Advanced schema={schema} values={values} problems={problems} set={set}
            onBrowse={(key) => setPicker({ mode: "file", key, title: `File for ${key}` })} />
        </div>

        {/* start -------------------------------------------------------------------------- */}
        <div ref={summary} className="grid gap-4 lg:sticky lg:top-4">
          <Card title="Start">
            <div className="grid gap-3">
              <Field htmlFor="run-name" label="Name of this calibration" problem={problemOf("name")}
                help="A folder with this name is created for the result.">
                <input id="run-name" className={INPUT} value={name} onChange={(e) => { setName(e.target.value); setProblems((o) => o.filter((p) => p.key !== "name")); }} />
              </Field>
              {roots.length > 1 && (
                <Field htmlFor="run-root" label="Save it in">
                  <select id="run-root" className={INPUT} value={root} onChange={(e) => setRoot(Number(e.target.value))}>
                    {roots.map((r, k) => <option key={r.path} value={k}>{r.path}</option>)}
                  </select>
                </Field>
              )}
              {roots.length === 1 && <div className="text-xs text-s-muted break-all">Saved in <Mono>{roots[0].path}</Mono></div>}
              <ul className="grid gap-1.5 text-sm">
                <Check ok={photosReady}>{photosReady ? `${nCam} camera${nCam === 1 ? "" : "s"}${usesKeypoints ? ", from saved detections" : ""}` : source === "detections" ? "Choose the detections file" : "Choose the photo folder"}</Check>
                <Check ok>{`Board ${nx} × ${ny}, squares of ${sq}`}</Check>
                <Check ok={nCam > 0}>{nCam > 0 ? lensSummary(models, nCam) : "Lens types"}</Check>
                <Check ok>{production ? "Production checks" : "Standard checks"}</Check>
              </ul>
              {startError && <Callout tone="bad" title="Not started">{startError}</Callout>}
              {problems.filter((p) => p.key !== "").length > 0 && (
                <Callout tone="bad" title="These settings need attention">
                  <ul className="grid gap-0.5 mt-1 text-xs">
                    {problems.filter((p) => p.key !== "").map((p) => <li key={p.key}><Mono>{p.key}</Mono> {p.message}</li>)}
                  </ul>
                </Callout>
              )}
              <Button kind="primary" onClick={start} busy={starting} disabled={!ready} className="min-h-11">
                Start calibration
              </Button>
              {!ready && !scanning && <Note>{photosReady ? "Give the calibration a name." : source === "detections" ? "Choose the detections file to continue." : "Choose the folder with the photos to continue."}</Note>}
            </div>
          </Card>

          <Card title="Config file">
            <div className="grid gap-3">
              <Note>
                Starting writes these settings as <Mono>calib_param.yml</Mono> in the result folder.
                The same file runs from a terminal with <Mono>ds-msp-calibrate-rig --config</Mono>.
              </Note>
              <Button onClick={loadPreview} busy={previewBusy}>Check and show the file</Button>
              {preview && (
                <>
                  {preview.ok
                    ? <Badge tone="ok">The settings are valid</Badge>
                    : <Callout tone="bad" title="The settings have problems">
                        <ul className="grid gap-0.5 mt-1 text-xs">
                          {preview.problems.map((p, k) => <li key={k}>{p.key && <Mono>{p.key}</Mono>} {p.message}</li>)}
                        </ul>
                      </Callout>}
                  {preview.warnings.map((w) => <Callout key={w} tone="warn">{w}</Callout>)}
                  <pre className="font-studio-mono text-[11px] leading-relaxed bg-s-sunken border border-s-line rounded-lg p-3 overflow-x-auto max-h-80">
                    {preview.yaml}
                  </pre>
                </>
              )}
            </div>
          </Card>
        </div>
      </div>

      {picker && (
        <FolderPicker mode={picker.mode} title={picker.title} onClose={() => setPicker(null)}
          onPick={(path, ref) => (picker.mode === "folder" ? pickFolder(path, ref) : pickFile(path))} />
      )}
    </Page>
  );
}

function Page({ children, ...drop }: { children: ReactNode } & Pick<React.HTMLAttributes<HTMLElement>, "onDragOver" | "onDragLeave" | "onDrop">) {
  return <main {...drop} className="mx-auto w-full max-w-7xl px-4 sm:px-6 py-6 grid gap-5">{children}</main>;
}

function Mono({ children }: { children: ReactNode }) {
  return <code className="font-studio-mono text-[0.92em]">{children}</code>;
}

function lensSummary(models: string[], n: number): string {
  const counts = new Map<string, number>();
  for (let k = 0; k < n; k++) counts.set(models[k] ?? "radtan", (counts.get(models[k] ?? "radtan") ?? 0) + 1);
  return [...counts].map(([m, c]) => (counts.size === 1 ? `All ${m}` : `${c} ${m}`)).join(", ");
}

function Check({ ok, children }: { ok: boolean; children: ReactNode }) {
  return (
    <li className="flex items-start gap-2">
      <span className={`mt-0.5 size-4 shrink-0 rounded-full grid place-items-center text-[10px] font-bold ${ok ? "bg-s-ok-soft text-s-ok" : "bg-s-sunken text-s-faint"}`} aria-hidden="true">
        {ok ? "✓" : "·"}
      </span>
      <span className={ok ? "" : "text-s-muted"}>{children}</span>
    </li>
  );
}

function NumberField({ id, label, value, prop, problem, onChange, help, decimal }: {
  id: string; label: string; value: unknown; prop: SchemaProp; problem: string | null;
  onChange: (key: string, v: unknown) => void; help?: string; decimal?: boolean;
}) {
  // Text is kept while typing so that "0." or an empty box is not rewritten under the cursor.
  const [text, setText] = useState(value == null ? "" : String(value));
  const def = prop.default;
  useEffect(() => {
    setText((old) => {
      if (value == null) return old.trim() === "" || Number(old) === def ? old : "";
      return Number(old) === value || old === value ? old : String(value);
    });
  }, [value, def]);
  return (
    <Field htmlFor={id} label={label} problem={problem}
      help={help ?? (def != null ? `Default ${String(def)}` : undefined)}>
      <input id={id} className={`${INPUT} tnum`} inputMode={decimal ? "decimal" : "numeric"}
        value={text} placeholder={def != null ? String(def) : ""}
        onChange={(e) => {
          const t = e.target.value.trim();
          setText(e.target.value);
          const n = Number(t);
          onChange(id, t === "" ? "" : Number.isFinite(n) ? (n === def ? "" : n) : t);
        }} />
    </Field>
  );
}

function BoardDrawing({ nx, ny, sq }: { nx: number; ny: number; sq: number }) {
  const okShape = Number.isInteger(nx) && Number.isInteger(ny) && nx >= 2 && ny >= 2 && nx <= 40 && ny <= 40;
  if (!okShape) return <div className="text-xs text-s-muted self-center">The drawing appears once the square counts are whole numbers from 2 to 40.</div>;
  const cell = 160 / Math.max(nx, ny);
  const cells = [];
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++)
      if ((i + j) % 2 === 0) cells.push(<rect key={`${i}-${j}`} x={i * cell} y={j * cell} width={cell} height={cell} />);
  const size = Number.isFinite(sq) && sq > 0;
  const fmt = (v: number) => Number(v.toPrecision(4)).toString();
  return (
    <figure className="grid gap-2 justify-items-center content-start">
      <svg viewBox={`-1 -1 ${nx * cell + 2} ${ny * cell + 2}`} className="w-40 max-h-40" role="img"
        aria-label={`Board with ${nx} squares across and ${ny} down`}>
        <rect x="0" y="0" width={nx * cell} height={ny * cell} fill="#fff" stroke="#111827" strokeWidth="1" />
        <g fill="#111827">{cells}</g>
      </svg>
      <figcaption className="text-xs text-s-muted text-center leading-relaxed">
        {nx} × {ny} squares{size && <>, {fmt(nx * sq)} × {fmt(ny * sq)} overall in your unit</>}
      </figcaption>
    </figure>
  );
}

function Advanced({ schema, values, problems, set, onBrowse }: {
  schema: ConfigSchema; values: Values; problems: Problem[];
  set: (key: string, v: unknown) => void; onBrowse: (key: string) => void;
}) {
  const keys = Object.keys(schema.properties).filter((k) => !PLACED.has(k));
  const changed = keys.filter((k) => k in values);
  const bad = problems.filter((p) => keys.includes(p.key)).length;
  return (
    <details className="rounded-lg border border-s-line bg-s-surface" open={bad > 0 || undefined}>
      <summary className="cursor-pointer px-4 py-3.5 flex flex-wrap items-center gap-3">
        <h2 className="text-[15px] font-semibold">5. More settings</h2>
        <span className="text-xs text-s-muted">The defaults suit most rigs.</span>
        {changed.length > 0 && <Badge tone="info">{changed.length} changed</Badge>}
        {bad > 0 && <Badge tone="bad">{bad} to fix</Badge>}
      </summary>
      <div className="px-4 pb-4 grid gap-6">
        {schema["x-groups"].map((g) => {
          const inGroup = keys.filter((k) => schema.properties[k]["x-group"] === g.id);
          if (!inGroup.length) return null;
          return (
            <section key={g.id} className="grid gap-3">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-s-faint">{g.title}</h3>
              <div className="grid gap-x-5 gap-y-4 md:grid-cols-2">
                {inGroup.map((k) => (
                  <AdvancedField key={k} id={k} prop={schema.properties[k]} value={values[k]}
                    problem={problems.find((p) => p.key === k)?.message ?? null}
                    set={set} onBrowse={FILE_KEYS.has(k) ? () => onBrowse(k) : undefined} />
                ))}
              </div>
            </section>
          );
        })}
      </div>
    </details>
  );
}

function AdvancedField({ id, prop, value, problem, set, onBrowse }: {
  id: string; prop: SchemaProp; value: unknown; problem: string | null;
  set: (key: string, v: unknown) => void; onBrowse?: () => void;
}) {
  const types = Array.isArray(prop.type) ? prop.type : [prop.type];
  const kind = types.find((t) => t !== "null") ?? "string";
  const isSet = value !== undefined;
  const label = (
    <span className="flex flex-wrap items-baseline gap-x-2">
      <span className="font-studio-mono text-[13px]">{id}</span>
      {isSet && <button type="button" className="text-xs text-s-accent hover:underline font-normal" onClick={() => set(id, "")}>Reset</button>}
    </span>
  );
  const help = <>{prop.description} <span className="text-s-faint">Default: {shown(prop.default)}.</span></>;
  const cls = `${INPUT} ${isSet ? "border-s-accent" : ""}`;
  let control;
  if (kind === "boolean" || prop.enum) {
    const options = kind === "boolean" ? ["true", "false"] : (prop.enum ?? []).map(String);
    const current = value === undefined ? "" : String(value);
    control = (
      <select id={id} className={cls} value={current}
        onChange={(e) => set(id, e.target.value === "" ? "" : kind === "boolean" ? e.target.value === "true" : e.target.value)}>
        <option value="">Default ({shown(prop.default)})</option>
        {options.map((o) => <option key={o} value={o}>{kind === "boolean" ? (o === "true" ? "yes" : "no") : o}</option>)}
      </select>
    );
  } else {
    const text = value === undefined ? "" : Array.isArray(value) ? value.join(", ") : String(value);
    control = (
      <div className="flex gap-2">
        <input id={id} className={`${cls} ${kind === "string" ? "font-studio-mono text-xs" : "tnum"}`} value={text}
          inputMode={kind === "integer" ? "numeric" : kind === "number" ? "decimal" : undefined}
          placeholder={kind === "array" ? "values separated by commas" : shown(prop.default)}
          onChange={(e) => set(id, e.target.value)} />
        {onBrowse && <Button onClick={onBrowse}>Browse</Button>}
      </div>
    );
  }
  return <Field htmlFor={id} label={label} help={help} problem={problem}>{control}</Field>;
}
