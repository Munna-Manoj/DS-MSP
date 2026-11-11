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
import { CameraStep, chosenFor, lensFields, type CameraInfo, type LensMode, type Typed } from "./setup/CameraStep";
import { choiceCard, Explain, Mono, NumberBox, Part, Step, STEP_WORD } from "./setup/kit";
import { LineupStep } from "./setup/LineupStep";
import { Settings, settingsChanged } from "./setup/Settings";
import { StepDot, Tracker, type StepId, type TrackedStep } from "./setup/Tracker";
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
  { id: "config", title: "A settings file", text: "A calib_param.yml you already have. It fills in the steps for you.",
    icon: "M6 2h8l5 5v13a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2zM14 2v5h5M8 12h8M8 16h6" },
  { id: "detections", title: "Corners found earlier", text: "A saved corners file. The photos are not read again.",
    icon: "M4 4h16v16H4zM4 9.3h16M4 14.6h16M9.3 4v16M14.6 4v16" },
  { id: "result", title: "A finished result", text: "Repeat it exactly, or change one thing and run it again.",
    icon: "M4 12a8 8 0 1 0 2.3-5.6M4 4v4h4M8.5 12.5l2.5 2.5 4.5-5" },
];

/** The step that holds each setting, so a problem can lead to where it is fixed. */
const KEY_STEP: Record<string, StepId> = Object.fromEntries([
  ...["root_path", "cam_prefix", "number_camera", "keypoints_path", "exclude_views"].map((k) => [k, "data"]),
  ...["number_x_square", "number_y_square", "square_size", "number_board", "length_square", "length_marker", "boards_index",
    "number_x_square_per_board", "number_y_square_per_board", "square_size_per_board", "structure_mode", "structure_prior_sigma"].map((k) => [k, "board"]),
  ...["camera_models", "cam_params_path", "fix_intrinsic"].map((k) => [k, "lens"]),
  ...["profile", "max_rotation_sigma_deg", "max_centre_sigma"].map((k) => [k, "rules"]),
  ["name", "start"],
]) as Record<string, StepId>;
const stepOfKey = (key: string): StepId => KEY_STEP[key] ?? "more";

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
  // Nothing but the choice of how to start is shown until it has been made.
  const [started, setStarted] = useState(!!from);
  const [active, setActive] = useState<StepId>("data");
  // Settings were just loaded: open the first step that still needs something once they are read.
  const [landing, setLanding] = useState(false);
  const firstOpen = useRef<StepId>("data");
  const [kp, setKp] = useState<KeypointsScan | null>(null);
  const [lens, setLens] = useState<IntrinsicsScan | null>(null);
  const [lensMode, setLensMode] = useState<LensMode>("auto");
  const [typed, setTyped] = useState<Typed>({});
  const [lineupBusy, setLineupBusy] = useState(false);
  const [lineupError, setLineupError] = useState<string | null>(null);
  // The folders of the lineup being changed, so changing it starts from what was chosen.
  const [lastLineup, setLastLineup] = useState<string[] | null>(null);
  // Where every chooser opens: the folder the user started from. They can still browse away.
  const [startRef, setStartRef] = useState<string | null>(null);
  // A file that could not be used is reported where it was chosen, not at the top of the page.
  const [fileError, setFileError] = useState<{ where: "lens" | "detections" | "config"; message: string } | null>(null);
  const [reading, setReading] = useState<{ where: "lens" | "detections"; name: string } | null>(null);
  const noticeRef = useRef<HTMLDivElement>(null);
  const showTop = () => window.setTimeout(() => noticeRef.current?.scrollIntoView({ behavior: "smooth", block: "center" }), 60);
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
  const [anywhere, setAnywhere] = useState(false);
  const [problems, setProblems] = useState<Problem[]>([]);
  const [startError, setStartError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [preview, setPreview] = useState<ConfigCheck | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [unit, setUnit] = useUnit();
  const fileInput = useRef<HTMLInputElement>(null);

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
  const pending = source === "photos" && !!scan?.needs_lineup;
  const nCam = pending ? 0 : scan?.number_camera || Number(values.number_camera) || 0;

  const runScan = useCallback((path: string, prefix?: string) => {
    setScanning(true);
    setScanError(null);
    setBoard(null);
    const pattern = payloadRef.current.frame_pattern;
    api.scanPhotos(path, prefix, typeof pattern === "string" && pattern ? pattern : undefined).then(
      (doc) => {
        setScan(doc);
        setScanning(false);
        setValues((old) => {
          const next: Values = { ...old };
          if (doc.number_camera && !doc.needs_lineup) next.number_camera = doc.number_camera;
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
    setReading({ where: "detections", name: path.split("/").pop() ?? path });
    api.sourceKeypoints(path).then(
      (doc) => {
        setReading(null);
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
      // a file that does not fit leaves the one chosen before in place
      (e: Error) => { setReading(null); setFileError({ where: "detections", message: e.message }); },
    );
  }, []);

  const loadLens = useCallback((path: string) => {
    setFileError(null);
    setReading({ where: "lens", name: path.split("/").pop() ?? path });
    api.sourceIntrinsics(path).then(
      (doc) => {
        setReading(null);
        setLens(doc);
        setProblems((old) => old.filter((p) => p.key !== "cam_params_path"));
        setLensMode("file");
        setValues((old) => ({ ...old, cam_params_path: doc.path }));
        // the file states each lens type: take them, so the choice below matches the file
        setModels((old) => {
          const n = Math.max(old.length, doc.cameras.length);
          return Array.from({ length: n }, (_, k) => doc.cameras.find((c) => c.index === k)?.model ?? old[k] ?? "radtan");
        });
      },
      (e: Error) => { setReading(null); setFileError({ where: "lens", message: e.message }); },
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
    setLensMode("auto");
    setTyped({});
    setNotice(`${label}${ignored.length ? ` Keys DS-MSP does not use were left out: ${ignored.join(", ")}.` : ""}`);
    const detections = typeof next.keypoints_path === "string" && next.keypoints_path !== "";
    setSource(detections ? "detections" : "photos");
    setStarted(true);
    setLanding(true);
    if (detections) loadDetections(next.keypoints_path as string);
    else if (typeof next.root_path === "string" && next.root_path)
      runScan(next.root_path, typeof next.cam_prefix === "string" ? next.cam_prefix : undefined);
    if (typeof next.cam_params_path === "string" && next.cam_params_path) loadLens(next.cam_params_path);
  }, [runScan, loadDetections, loadLens]);

  useEffect(() => {
    api.health().then((h) => setCanRun(h.can_run !== false), () => undefined);
    api.schema().then(setSchema, (e: Error) => setFatal(e.message));
    api.browse(null).then((d) => { setRoots(d.dirs.map((x) => ({ name: x.name, path: x.path }))); setAnywhere(d.anywhere === true); }, () => undefined);
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
    if (out.profile === "production") out.fix_intrinsic = true;   // strict: lens values are never computed
    return out;
  }, [values, models, nCam]);
  payloadRef.current = payload;

  // Complete, valid typed values for every camera, or null.
  const typedCams = useMemo(() => {
    if (lensMode !== "typed" || !nCam) return null;
    const out: Record<string, unknown>[] = [];
    for (let k = 0; k < nCam; k++) {
      const model = models[k] ?? "radtan";
      const card = typed[k] ?? {};
      const known = kp?.cameras.find((x) => x.index === k) ?? scan?.cameras[k];
      const row: Record<string, unknown> = { model };
      for (const f of [...lensFields(model), "width", "height"]) {
        const text = card[f] ?? (f === "width" ? known?.width : f === "height" ? known?.height : undefined);
        if (text == null || String(text).trim() === "" || !Number.isFinite(Number(text))) return null;
        row[f] = Number(text);
      }
      out.push(row);
    }
    return out;
  }, [lensMode, nCam, models, typed, kp, scan]);
  const typedKey = typedCams ? JSON.stringify(typedCams) : null;

  // Typed values become a lens file as soon as they are complete, so they get the same checks.
  useEffect(() => {
    if (lensMode !== "typed") return;
    setLens(null);
    setFileError(null);
    setValues((old) => { const v = { ...old }; delete v.cam_params_path; return v; });
    if (!typedKey) return;
    let live = true;
    const t = window.setTimeout(() => {
      api.lensValues(JSON.parse(typedKey) as Record<string, unknown>[]).then(
        (doc) => { if (live) { setLens(doc); setValues((old) => ({ ...old, cam_params_path: doc.path })); setProblems((old) => old.filter((p) => p.key !== "cam_params_path")); } },
        (e: Error) => { if (live) setFileError({ where: "lens", message: e.message }); },
      );
    }, 400);
    return () => { live = false; window.clearTimeout(t); };
  }, [lensMode, typedKey]);

  function chooseLensMode(next: LensMode) {
    setLensMode(next);
    setLens(null);
    setFileError(null);
    setValues((old) => {
      const v = { ...old };
      delete v.cam_params_path;
      // someone typing in values usually wants them kept; they can still choose to refine
      if (next === "typed") v.fix_intrinsic = true; else delete v.fix_intrinsic;
      return v;
    });
  }

  // A new file-name pattern changes which photos line up, so the folder is read again.
  const framePattern = typeof values.frame_pattern === "string" ? values.frame_pattern : "";
  const lastPattern = useRef(framePattern);
  useEffect(() => {
    if (framePattern === lastPattern.current) return;
    lastPattern.current = framePattern;
    if (!rootPath || source !== "photos") return;
    const t = window.setTimeout(() => runScan(rootPath, typeof values.cam_prefix === "string" ? values.cam_prefix : undefined), 400);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [framePattern]);

  const goTo = useCallback((id: StepId) => {
    setActive(id);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  useEffect(() => {
    if (!landing || !schema || scanning || reading) return;
    setLanding(false);
    goTo(firstOpen.current);
  }, [landing, schema, scanning, reading, goTo]);

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

  // While typing values, each camera's chosen type is compared with its photos, and never changed.
  const modelsKey = models.slice(0, nCam).join();
  useEffect(() => {
    if (lensMode !== "typed" || !nCam || !(source === "detections" ? kp?.path : rootPath)) return;
    const t = window.setTimeout(() => void runAdvice(payloadRef.current, nCam, false), 500);
    return () => window.clearTimeout(t);
  }, [lensMode, modelsKey, nCam, source, kp?.path, rootPath, runAdvice]);

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
        if (!live) return;
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
    if (key === "@save") {
      api.addRoot(path).then(
        (r) => api.browse(null).then((d) => { setRoots(d.dirs.map((x) => ({ name: x.name, path: x.path }))); setRoot(r.index); }),
        (e: Error) => setStartError(e.message),
      );
      return;
    }
    set(key, path);
    if (key === "root_path") {
      const k = Number(ref.split(":", 1)[0]);
      if (Number.isInteger(k)) setRoot(k);
      setStartRef(ref);
      setLastLineup(null);
      runScan(path);
    }
  }

  function applyLineup(folders: string[]) {
    if (!scan || !rootPath) return;
    setLineupBusy(true);
    setLineupError(null);
    api.stageLineup(rootPath, folders).then(
      (doc) => {
        setLineupBusy(false);
        setLastLineup(folders);
        setScan(null);                                 // the cards belong to the folder being left
        setValues((old) => ({ ...old, root_path: doc.root_path }));
        runScan(doc.root_path);
      },
      (e: Error) => { setLineupBusy(false); setLineupError(e.message); },
    );
  }

  function changeLineup(root: string) {
    setScan(null);
    setValues((old) => ({ ...old, root_path: root }));
    runScan(root);
  }

  /** Fill the page from a settings file on disk, saying which of its paths were not valid here. */
  function adoptFile(doc: Awaited<ReturnType<typeof api.sourceConfig>>) {
    const moved = doc.repointed ?? [], gone = doc.missing ?? [];
    const notes = [
      moved.length ? `It was written on another computer, so ${moved.map((m) => m.key).join(" and ")} now point${moved.length === 1 ? "s" : ""} to what sits next to it.` : "",
      gone.length ? `${gone.map((m) => m.key).join(" and ")} could not be found here: choose ${gone.length === 1 ? "it" : "them"} in the steps.` : "",
    ].filter(Boolean).join(" ");
    adopt(doc.values, `Settings loaded from ${doc.path.split("/").pop()}.${notes ? ` ${notes}` : ""}`, doc.unknown);
    showTop();
  }

  function useFound(f: NonNullable<ScanDoc["found"]>[number]) {
    if (f.kind === "lens") loadLens(f.path);
    else if (f.kind === "corners") { setSource("detections"); loadDetections(f.path); }
    else api.sourceConfig(f.path).then(adoptFile, (e: Error) => { setFileError({ where: "config", message: e.message }); showTop(); });
  }

  function pickFile(path: string, ref: string) {
    const key = picker?.key ?? "";
    setPicker(null);
    // no photo folder yet: the folder of the first file chosen is the place to start from
    if (!rootPath && ref) setStartRef((old) => old ?? ref);
    if (key === "@config") {
      api.sourceConfig(path).then(
        (doc) => adoptFile(doc),
        (e: Error) => { setFileError({ where: "config", message: e.message }); showTop(); },
      );
    } else if (key === "@detections") loadDetections(path);
    else if (key === "@lens") loadLens(path);
    else set(key, path);
  }

  function begin(next: Source) {
    chooseSource(next);
    setStarted(true);
    setActive("data");
    window.scrollTo({ top: 0 });
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
      (doc) => {
        const paths = ["root_path", "cam_params_path", "keypoints_path"].some((k) => doc.values[k]);
        adopt(doc.values, `Settings loaded from a copy of ${file.name}.${paths ? " Studio does not know where the original is, so check the folders and files in the steps and choose again any it cannot find." : ""}`, doc.unknown);
        showTop();
      },
      (e: Error) => { setFileError({ where: "config", message: `Could not read ${file.name}: ${e.message}` }); showTop(); },
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
    setStarting(true);
    setStartError(null);
    const reused = reuse && source === "photos" ? hist?.reusable ?? null : null;
    api.startJob(root, name, payload, reused).then(
      (job) => { window.location.hash = href({ page: "job", id: job.id }); },
      (e: Error) => {
        setStarting(false);
        setStartError(e.message);
        setProblems(e instanceof ApiError ? e.problems : []);
        goTo("start");
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
  const photosReady = (!!scan && !pending && scan.number_camera > 0) || (usesKeypoints && nCam > 0);
  const sized = values.square_size != null;
  const blocking = pending ? [] : scan?.problems ?? [];
  const keeps = production || values.fix_intrinsic === true;
  const limitsMissing = production && (values.max_rotation_sigma_deg == null || values.max_centre_sigma == null);
  // Everything that stops a run, by step, in the order of the page. The server checks the same again.
  const need: { step: StepId; text: string }[] = [
    ...(pending ? [{ step: "data" as const, text: "Say which folder is which camera" }] : []),
    ...(!pending && !photosReady ? [{ step: "data" as const, text: source === "detections" ? "Choose the corners file" : "Choose the folder with your photos" }] : []),
    ...(blocking.length ? [{ step: "data" as const, text: "Fix the problems with the photos" }] : []),
    ...(!sized ? [{ step: "board" as const, text: "Enter the size of one square" }] : []),
    ...(nCam > 0 && lensMode === "typed" && !typedCams ? [{ step: "lens" as const, text: "Fill in every value on each camera's card" }]
      : nCam > 0 && keeps && !values.cam_params_path ? [{ step: "lens" as const, text: production ? "Strict checking needs your lens values: choose a file or type them in" : "Choose the lens file, or type the values in" }]
      : nCam > 0 && keeps && lens && lens.cameras.length < nCam ? [{ step: "lens" as const, text: "Lens values are needed for every camera" }] : []),
    ...(limitsMissing ? [{ step: "rules" as const, text: "Strict checking needs both precision limits" }] : []),
    ...(name.trim() === "" ? [{ step: "start" as const, text: "Give the calibration a name" }] : []),
  ];
  const missing = need.map((x) => x.text);
  const ready = missing.length === 0 && !scanning;
  const shownRoot = scan?.lineup?.root ?? rootPath;
  const opensAt = startRef ?? (anywhere && shownRoot ? `abs:${shownRoot}` : null);
  const originOf = (camera: string) => scan?.lineup?.cameras.find((c) => c.camera === camera)?.folder ?? null;
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

  const fixing = (id: StepId) => problems.some((p) => p.key !== "" && stepOfKey(p.key) === id);
  const open = (id: StepId) => need.filter((x) => x.step === id);
  const stateOf = (id: StepId, otherwise: TrackedStep["state"]): TrackedStep["state"] => (fixing(id) ? "fix" : open(id).length ? "todo" : otherwise);
  const unitWord = unit === "?" ? "" : ` ${unit}`;
  const lensBad = misfit || Object.values(health).some((h) => h !== "busy" && h.verdict === "misfit");
  // Lens types start as "standard" without anyone having looked, so that alone is not "done".
  const lensChosen = lensMode !== "auto" || models.slice(0, nCam).some((m) => m !== "radtan") || Object.values(advice).some((a) => a !== "busy");
  const nChanged = settingsChanged(values);
  const limitsSet = values.max_rotation_sigma_deg != null && values.max_centre_sigma != null;
  const steps: TrackedStep[] = [
    { id: "data", n: 1, title: source === "detections" ? "Saved corners" : "Photos", state: stateOf("data", "done"),
      note: scanning ? "Reading the folder…" : photosReady ? `${nCam} camera${nCam === 1 ? "" : "s"}${usesKeypoints ? ", from saved corners" : `, ${int(scan?.cameras.reduce((a, c) => a + c.images, 0) ?? 0)} photos`}` : open("data")[0]?.text ?? "",
      items: source === "detections" ? [{ label: "Corners file", state: kp ? "ok" : "todo" }] : [
        { label: "Photo folder", state: rootPath ? "ok" : "todo" },
        ...(pending || scan?.lineup ? [{ label: "Folders matched to cameras", state: pending ? "todo" as const : "ok" as const }] : []),
        ...(scan && !pending ? [{ label: "Photos can be read", state: blocking.length ? "todo" as const : "ok" as const }] : []),
      ] },
    { id: "board", n: 2, title: "Board", state: stateOf("board", board?.found === 0 ? "warn" : "done"),
      note: board?.found === 0 ? "Not found in the photos" : sized ? `${nx} × ${ny}, squares of ${String(values.square_size)}${unitWord}` : "Needs the size of one square",
      items: [{ label: `${nx} × ${ny} squares`, state: "ok" }, { label: "Size of one square", state: sized ? "ok" : "todo" },
        ...(photoSource ? [{ label: "Seen in the photos", state: board && board.found > 0 ? "ok" as const : "optional" as const }] : [])] },
    { id: "lens", n: 3, title: "Lenses", state: nCam === 0 ? "waiting" : stateOf("lens", lensBad ? "warn" : lensChosen ? "done" : "usual"),
      note: nCam === 0 ? "After step 1" : open("lens")[0] ? "Lens values needed" : lensBad ? "A lens does not fit its photos" : `${lensSummary(models, nCam)}${lensChosen ? "" : ", not measured"}`,
      items: nCam === 0 ? [] : [{ label: "Lens type of each camera", state: "ok" },
        keeps || lensMode === "typed" ? { label: "Lens values", state: open("lens").length ? "todo" : "ok" }
          : { label: lens ? "Lens values from a file" : "Lens values worked out from the photos", state: "ok" }] },
    { id: "rules", n: 4, title: "Doubtful results", state: stateOf("rules", production || limitsSet ? "done" : "usual"),
      note: limitsMissing ? "Needs both limits" : production ? "Refuse them" : "Give them with warnings",
      items: [{ label: production ? "Refuse a doubtful result" : "Warn about a doubtful result", state: "ok" },
        { label: "Precision limits", state: limitsSet ? "ok" : production ? "todo" : "optional" }] },
    { id: "more", n: 5, title: "More settings", state: fixing("more") ? "fix" : nChanged ? "done" : "usual",
      note: nChanged ? `${nChanged} changed` : "Optional" },
    { id: "start", n: 6, title: "Name and start", state: stateOf("start", ready ? "done" : "waiting"), word: ready ? "Ready" : undefined,
      note: ready ? "Ready" : scanning ? "Reading the folder…" : open("start")[0] ? "Needs a name" : "After the steps above" },
  ];
  const required = steps.filter((s) => ["data", "board", "lens"].includes(s.id) || (s.id !== "start" && ["todo", "fix"].includes(s.state)));
  const remaining = required.filter((s) => ["todo", "fix", "waiting"].includes(s.state)).length;
  firstOpen.current = steps.find((s) => s.state === "todo" || s.state === "fix")?.id ?? "start";
  const prelude = source === "config" ? "Load your settings file" : source === "result" ? "Choose the result to start from" : null;
  const shownSteps = prelude ? steps.map((s) => ({ ...s, state: "waiting" as const, note: source === "config" ? "Filled in from the file" : "If you change something", items: [] })) : steps;
  const from_ = SOURCES.find((o) => o.id === source) ?? SOURCES[0];
  const headline = prelude ? "Choose this first, and the steps open"
    : ready ? "Ready to calibrate"
    : remaining > 0 ? `${remaining} more step${remaining === 1 ? "" : "s"} before you can calibrate`
    : scanning ? "Reading the folder…" : "Give the calibration a name";
  const stepOf = (id: StepId) => steps.find((s) => s.id === id) as TrackedStep;
  const at = (id: StepId) => !prelude && active === id;

  /** Back and next, and what this step still needs. */
  const stepNav = (id: StepId) => {
    const k = steps.findIndex((s) => s.id === id);
    const prev = steps[k - 1], next = steps[k + 1];
    const left = open(id);
    return (
      <div className="flex flex-wrap items-center gap-x-4 gap-y-3">
        {prev ? <Button onClick={() => goTo(prev.id)}>← Back</Button> : <Button onClick={() => setStarted(false)}>← Change how to start</Button>}
        {left.length > 0 && <div className="text-[15px] font-medium text-s-accent" role="status">Still needed here: {left.map((x) => x.text).join("; ")}</div>}
        {next && <Button kind={left.length ? "secondary" : "primary"} className="ml-auto" onClick={() => goTo(next.id)}>Next: {next.title} →</Button>}
      </div>
    );
  };

  const notices = (
      <div ref={noticeRef} className="grid gap-3 empty:hidden">
        {notice && <Callout tone="info">{notice}</Callout>}
        {fileError?.where === "config" && <Callout tone="bad" title="That file was not used">{fileError.message}</Callout>}
      </div>
  );

  return (
    <Page
      onDragOver={(e) => { if (e.dataTransfer.types.includes("Files")) { e.preventDefault(); setDragging(true); } }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const f = e.dataTransfer.files?.[0];
        if (f && /\.(ya?ml|txt)$/i.test(f.name)) importFile(f);
        else if (f) { setFileError({ where: "config", message: `${f.name} is not a settings file. Drop a calib_param.yml here. Corner files and lens files are chosen with their own buttons, because the calibration reads them from disk.` }); showTop(); }
      }}
    >
      {dragging && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-s-accent/25 backdrop-blur-sm pointer-events-none" aria-hidden="true">
          <div className="rounded-2xl bg-s-surface border-4 border-dashed border-s-accent px-10 py-8 text-2xl font-bold">Drop a copy of your settings file to fill in this page</div>
        </div>
      )}
      <header className="grid gap-2">
        <a href={href({ page: "runs" })} className="text-[15px] font-medium text-s-accent hover:underline justify-self-start">← All results</a>
        <h1 className="text-3xl sm:text-4xl font-bold tracking-tight">New calibration</h1>
        {!started && <p className="text-lg text-s-muted max-w-[60ch]">Choose how you want to start. Studio then takes you through what it needs, one step at a time.</p>}
        <input ref={fileInput} type="file" accept=".yml,.yaml,.txt" className="sr-only"
          aria-label="Load settings from a calib_param.yml file on your own computer"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) importFile(f); e.target.value = ""; }} />
      </header>

      {!started && (
        <section className="rounded-2xl bg-s-surface border border-s-rule shadow-[0_1px_2px_rgb(15_23_42/0.06),0_8px_24px_rgb(15_23_42/0.06)] overflow-hidden">
          <header className="px-5 sm:px-6 py-4 bg-s-accent-soft border-b border-s-accent/20">
            <h2 className="text-xl font-semibold tracking-tight">What do you have?</h2>
            <p className="text-[15px] text-s-muted mt-0.5">Pick one. You can change it later.</p>
          </header>
          <div className="p-5 sm:p-6 grid gap-4 sm:grid-cols-2" role="group" aria-label="How to start">
            {SOURCES.map((o) => (
              <button key={o.id} type="button" onClick={() => begin(o.id)}
                className={`group text-left p-5 grid grid-cols-[auto_minmax(0,1fr)_auto] gap-x-4 gap-y-1 items-center ${choiceCard(false)}`}>
                <span className="row-span-2 size-14 rounded-xl grid place-items-center bg-s-accent-soft text-s-accent transition group-hover:bg-s-accent group-hover:text-white" aria-hidden="true">
                  <svg viewBox="0 0 24 24" className="size-7" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d={o.icon} /></svg>
                </span>
                <span className="text-lg font-semibold leading-tight">{o.title}</span>
                <span className="row-span-2 text-2xl text-s-faint transition group-hover:text-s-accent group-hover:translate-x-1" aria-hidden="true">→</span>
                <span className="text-[15px] text-s-muted leading-snug">{o.text}</span>
              </button>
            ))}
          </div>
        </section>
      )}

      {!started && notices}

      <div className={started ? "grid gap-6 lg:grid-cols-[19rem_minmax(0,1fr)] items-start" : "hidden"}>
        <aside className="lg:sticky lg:top-4 min-w-0">
          <Tracker from={from_.title} icon={from_.icon} onChangeFrom={() => setStarted(false)} prelude={prelude} steps={shownSteps} active={active} onGo={goTo}
            headline={headline} done={prelude ? 0 : required.length - remaining} total={required.length} />
        </aside>
        <div className="grid gap-6 min-w-0">
          {started && notices}
        {source === "config" && (
          <Step title="Load your settings file">
            <Explain>Everything the file sets is filled in for you, and you can still change any of it in the steps before starting.</Explain>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="grid gap-2 content-start">
                <Button kind="primary" onClick={() => setPicker({ mode: "file", key: "@config", title: "Choose a calib_param.yml" })}>Choose a file where Studio runs</Button>
                <Explain><b className="font-semibold text-s-text">Recommended.</b> For a file on the computer running Studio. Studio knows where the file is, so it can find the photos and lens file it points to, even if the file was written on another computer.</Explain>
              </div>
              <div className="grid gap-2 content-start">
                <Button onClick={() => fileInput.current?.click()}>Upload a copy from this device</Button>
                <Explain>For a file only on the device you are using now. You can also drop it anywhere on this page. Studio gets only a copy of the text, so you choose again any photo folder or lens file it cannot find.</Explain>
              </div>
            </div>
          </Step>
        )}

        {source === "result" && (
          <Step title="Choose the result to start from" lead={results ? `${results.length} here` : undefined}>
            {!results && <Loading what="results" />}
            {results && results.length === 0 && <Explain>There are no results yet in the folders Studio was started with.</Explain>}
            {results && results.length > 0 && (
              <ul className="grid divide-y divide-s-line -my-2">
                {results.slice(0, 40).map((r) => (
                  <li key={r.id} className="flex flex-wrap items-center gap-x-4 gap-y-2 py-3 px-3 -mx-3 rounded-lg transition hover:bg-s-accent-soft/40">
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

          {/* 1 ---------------------------------------------------------------------------- */}
          {source === "detections" ? (
            <Step n={1} of={6} title="Your saved corners" lead="The corners an earlier calibration found." state={stepOf("data").state} active={at("data")} footer={stepNav("data")}>
              <Explain>The <Mono>detected_keypoints_data.yml</Mono> file an earlier calibration wrote. Its corners are used as they are, so the photos are not needed.</Explain>
              <div className="flex flex-wrap gap-3">
                <input id="keypoints_path" aria-label="Corners file" className={`${INPUT} font-studio-mono !text-sm flex-1 min-w-60`} readOnly placeholder="No file chosen yet"
                  value={typeof values.keypoints_path === "string" ? values.keypoints_path : ""} />
                <Button kind={kp ? "secondary" : "primary"} onClick={() => setPicker({ mode: "file", key: "@detections", title: "Choose a corners file" })}>{kp ? "Choose another file" : "Choose the file"}</Button>
              </div>
              {problemOf("keypoints_path") && <div className="text-[15px] font-medium text-s-bad" role="alert">{problemOf("keypoints_path")}</div>}
              {reading?.where === "detections" && <Callout tone="info">Reading {reading.name}…</Callout>}
              {fileError?.where === "detections" && (
                <Callout tone="bad" title="That file was not used">{fileError.message}{kp ? ` Studio keeps using ${kp.path.split("/").pop()}.` : ""}</Callout>
              )}
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
            <Step n={1} of={6} title="Your photos" lead="The folder with one subfolder per camera." state={stepOf("data").state} active={at("data")} footer={stepNav("data")}>
              {!rootPath && !scanning && (
                <div className="grid gap-5 sm:grid-cols-[minmax(0,1fr)_auto] items-center">
                  <div className="grid gap-3">
                    <Explain>Choose the folder, on the computer running Studio, that holds one subfolder per camera. Photos taken at the same moment have the same number in their file name.</Explain>
                    <Button kind="primary" className="justify-self-start min-h-12 !text-base px-5" onClick={() => setPicker({ mode: "folder", key: "root_path", title: "Folder with the photos" })}>Choose the photo folder</Button>
                  </div>
                  <pre className="font-studio-mono text-sm leading-relaxed text-s-muted bg-s-sunken rounded-lg px-4 py-3" aria-label="Example folder layout">{"my_rig/\n├─ Cam_001/\n│   ├─ 0001.png\n│   └─ 0002.png\n└─ Cam_002/\n    ├─ 0001.png\n    └─ 0002.png"}</pre>
                </div>
              )}
              {rootPath && (
                <div className="flex flex-wrap gap-3">
                  <input id="root_path" aria-label="Folder with the photos" className={`${INPUT} font-studio-mono !text-sm flex-1 min-w-60`} readOnly value={shownRoot} />
                  <Button onClick={() => setPicker({ mode: "folder", key: "root_path", title: "Folder with the photos" })}>Choose another folder</Button>
                </div>
              )}
              {(problemOf("root_path") ?? problemOf("number_camera")) && <div className="text-[15px] font-medium text-s-bad" role="alert">{problemOf("root_path") ?? problemOf("number_camera")}</div>}
              {scanning && <Loading what="the folder" />}
              {scanError && <ErrorBox title="Could not look at this folder" detail={scanError} />}
              {pending && scan && <LineupStep key={rootPath} root={rootPath} folders={scan.folders} previous={lastLineup} onApply={applyLineup} busy={lineupBusy} error={lineupError} />}
              {scan?.lineup && <div className="flex flex-wrap items-center gap-x-4 gap-y-2"><Explain>Your folders are linked under the names below and were not changed.</Explain><Button kind="quiet" onClick={() => changeLineup(scan.lineup!.root)}>Change which folder is which camera</Button></div>}
              {scan?.notes.map((n) => <Callout key={n} tone="info">{n}</Callout>)}
              {scan && !pending && scan.number_camera > 0 && (
                <>
                  <Callout tone={blocking.length ? "warn" : "ok"}
                    title={`Found ${scan.number_camera} camera${scan.number_camera === 1 ? "" : "s"} and ${int(scan.cameras.reduce((a, c) => a + c.images, 0))} photos`}>
                    {scan.number_camera > 1 && <>{int(scan.shared_frames)} moments were photographed by two or more cameras at once.</>}
                  </Callout>
                  <div className="grid gap-4 grid-cols-[repeat(auto-fill,minmax(150px,1fr))]">
                    {scan.cameras.slice(0, 24).map((c) => (
                      <PhotoThumb key={c.index} src={api.thumbUrl(rootPath, c.index, 0, scan.cam_prefix)} width={c.width} height={c.height}
                        caption={<><b className="font-semibold">{originOf(c.folder) ? `${originOf(c.folder)} → ${c.folder}` : c.folder}</b><span className="block text-s-muted">{int(c.images)} photos</span></>} />
                    ))}
                  </div>
                  {scan.cameras.length > 24 && <Explain>and {scan.cameras.length - 24} more cameras.</Explain>}
                  {(scan.found ?? []).length > 0 && (
                    <div className="rounded-xl bg-s-info-soft border border-s-info/25 p-4 grid gap-3">
                      <div>
                        <div className="text-base font-semibold text-s-info">This folder also holds files Studio can use</div>
                        <div className="text-[15px] text-s-text">Use one to fill in this page instead of typing it. Nothing is used until you choose.</div>
                      </div>
                      <ul className="grid gap-2">
                        {(scan.found ?? []).map((f) => {
                          const inUse = (f.kind === "lens" && lens?.path === f.path);
                          return (
                            <li key={f.path} className="flex flex-wrap items-center gap-x-4 gap-y-2">
                              <div className="min-w-0 mr-auto">
                                <div className="text-[15px] font-semibold break-all">{f.name}</div>
                                <div className="text-[14px] text-s-muted">{FOUND_WORDS[f.kind].what}</div>
                              </div>
                              {inUse ? <span className="text-[15px] font-medium text-s-ok">In use</span>
                                : <Button onClick={() => useFound(f)}>{FOUND_WORDS[f.kind].action}</Button>}
                            </li>
                          );
                        })}
                      </ul>
                    </div>
                  )}
                </>
              )}
              {blocking.map((p) => <Callout key={p} tone="warn">{p}</Callout>)}

              {excluded.length > 0 && (
                <Part>
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
                </Part>
              )}

              {hist && hist.runs.length > 0 && (
                <Part>
                  <div className="text-base font-semibold">You calibrated these photos before</div>
                  {hist.reusable ? (
                    <label className={`flex items-start gap-3.5 p-4 ${choiceCard(reuse)}`}>
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
                </Part>
              )}
            </Step>
          )}

          {/* 2 ---------------------------------------------------------------------------- */}
          <Step n={2} of={6} title="Your calibration board" lead="What your ChArUco board looks like, and how big it is." state={stepOf("board").state} active={at("board")} footer={stepNav("board")}>
            <BoardStep values={values} set={set} payload={payload} photos={photoSource} unit={unit} setUnit={setUnit}
              defaults={{ nx: Number(props.number_x_square.default), ny: Number(props.number_y_square.default), size: Number(props.square_size.default),
                ls: Number(props.length_square.default), lm: Number(props.length_marker.default), shape: String(props.structure_mode.default) }}
              check={board} onCheck={runBoardCheck} checkBusy={boardBusy} checkError={boardError} problemOf={problemOf} />
          </Step>

          {/* 3 ---------------------------------------------------------------------------- */}
          <Step n={3} of={6} title="Your cameras and their lenses" lead={nCam ? `${nCam} camera${nCam === 1 ? "" : "s"}: the kind of lens on each, and where its values come from.` : "The kind of lens on each camera."} state={stepOf("lens").state} active={at("lens")} footer={stepNav("lens")}>
            {nCam === 0
              ? <Explain>{source === "detections" ? "Choose the corners file in step 1. The cameras are read from it." : "Choose the photo folder in step 1. The cameras are read from it."}</Explain>
              : <CameraStep cameras={cameras} models={models} setModel={setModel} setAll={(m) => setModels(Array.from({ length: nCam }, () => m))}
                  advice={advice} onMeasure={() => void runAdvice(payload, nCam, true)} measuring={Object.values(advice).some((a) => a === "busy")}
                  canMeasure={!!viewsFrom && viewCams > 0} measureWhat={source === "detections" ? "saved detections" : "photos"}
                  photos={photoSource} samples={samples} lens={lens} health={health}
                  onLensBrowse={() => setPicker({ mode: "file", key: "@lens", title: "Choose a lens file" })}
                  onLensRemove={() => { setLens(null); setFileError(null); set("cam_params_path", ""); set("fix_intrinsic", ""); }}
                  lensMode={lensMode} onLensMode={chooseLensMode} strict={production} typed={typed}
                  onTyped={(k, f, v) => setTyped((old) => ({ ...old, [k]: { ...old[k], [f]: v } }))}
                  lensError={fileError?.where === "lens" ? fileError.message : null}
                  lensReading={reading?.where === "lens" ? reading.name : null}
                  fixed={keeps} onFixed={(v) => set("fix_intrinsic", v ? true : "")}
                  problem={problemOf("camera_models") ?? problemOf("cam_params_path")} lensProblem={problemOf("fix_intrinsic")} />}
          </Step>

          {/* 4 ---------------------------------------------------------------------------- */}
          <Step n={4} of={6} title="What should happen with a doubtful result?" lead="Already set to the usual choice. Change it if the result goes straight into a product." state={stepOf("rules").state} active={at("rules")} footer={stepNav("rules")}>
            <div className="grid gap-3 sm:grid-cols-2">
              {[{ id: "", title: "Give it to me with warnings", tag: "standard", text: "You always get a result. If a camera is weak or a check fails, the result page says so clearly. Best while you are still improving your capture." },
                { id: "production", title: "Refuse it", tag: "strict", text: "No result is written unless every check passes and the cameras are known as precisely as you ask below. Needs your own lens values for every camera, kept exactly as given. Best when the result goes straight into a product." }].map((o) => {
                const on = (values.profile ?? "") === o.id || (o.id === "" && values.profile === "default");
                return (
                  <button key={o.id || "standard"} type="button" role="radio" aria-checked={on} onClick={() => set("profile", o.id)}
                    className={`text-left p-4 grid gap-1.5 ${choiceCard(on)}`}>
                    <span className="text-[17px] font-semibold flex items-center gap-2.5">
                      <span className={`size-5 rounded-full border-2 grid place-items-center ${on ? "border-s-accent" : "border-s-field-line"}`} aria-hidden="true">{on && <span className="size-2.5 rounded-full bg-s-accent" />}</span>
                      {o.title}
                      <span className="rounded-full bg-s-surface border border-s-rule text-s-muted text-[13px] font-medium px-2 py-0.5">{o.tag}</span>
                    </span>
                    <span className="text-[15px] text-s-muted leading-relaxed">{o.text}</span>
                  </button>
                );
              })}
            </div>
            {problemOf("profile") && <div className="text-[15px] font-medium text-s-bad" role="alert">{problemOf("profile")}</div>}
            <Part>
              <div className="text-base font-semibold">How precisely must the cameras be known? {production ? <span className="text-s-bad">Needed for strict checking.</span> : <span className="font-normal text-s-muted">Optional.</span>}</div>
              <Explain>Between any two cameras, Studio estimates how uncertain their angle and their distance are. {production ? "A result less certain than these limits is refused." : "With limits, the result warns when a pair is less certain than you need."} The result page shows how close you came and what more photos would do.</Explain>
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
            </Part>
          </Step>

          {/* 5 ---------------------------------------------------------------------------- */}
          <Step n={5} of={6} title="More settings" lead="Optional. A normal calibration needs none of these." state={stepOf("more").state} active={at("more")} footer={stepNav("more")}>
            <Settings schema={schema} values={values} problems={problems} set={set} source={source} strict={production}
              refRun={from && fromName ? { id: from, name: fromName }
                : source === "photos" && hist?.runs.length ? { id: hist.runs[0].id, name: hist.runs[0].name } : null}
              onBrowse={(key) => setPicker({ mode: "file", key, title: "Choose the file" })} />
          </Step>

          {/* 6 ---------------------------------------------------------------------------- */}
          <Step n={6} of={6} title="Name and start" lead="Check your choices, name this calibration and start it." state={stepOf("start").state} word={stepOf("start").word} active={at("start")} footer={stepNav("start")}>
            <Part>
              <div className="text-base font-semibold">Your choices</div>
              <ul className="grid gap-1 -mx-2">
                {steps.filter((x) => x.id !== "start").map((x) => (
                  <li key={x.id}>
                    <button type="button" onClick={() => goTo(x.id)} className="w-full text-left rounded-lg px-2 py-2 flex flex-wrap items-center gap-x-3 gap-y-1 cursor-pointer transition hover:bg-s-accent-soft/50 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-s-accent/30">
                      <StepDot step={x} on={false} size="size-8" />
                      <span className="text-base font-semibold w-36">{x.title}</span>
                      <span className="text-[15px] text-s-muted mr-auto">{x.note}</span>
                      <span className={`text-[15px] font-medium ${x.state === "done" ? "text-s-ok" : x.state === "warn" ? "text-s-warn" : x.state === "fix" ? "text-s-bad" : x.state === "todo" ? "text-s-accent" : "text-s-muted"}`}>{STEP_WORD[x.state]}</span>
                    </button>
                  </li>
                ))}
                {source === "photos" && hist?.reusable && reuse && <li className="px-2 py-1 text-[15px] text-s-muted">Corners reused from {reusedName}</li>}
                {excluded.length > 0 && <li className="px-2 py-1 text-[15px] text-s-muted">{`${excluded.length} photo${excluded.length === 1 ? "" : "s"} left out`}</li>}
              </ul>
            </Part>
            <Part>
              <label htmlFor="run-name" className="text-base font-semibold">Name this calibration</label>
              <input id="run-name" className={INPUT} value={name} onChange={(e) => { setName(e.target.value); setProblems((o) => o.filter((p) => p.key !== "name")); }} />
              {problemOf("name") && <div className="text-[15px] font-medium text-s-bad" role="alert">{problemOf("name")}</div>}
              {roots.length > 1 ? (
                <label className="grid gap-1.5 text-base font-semibold">Save the result in
                  <select id="run-root" className={INPUT} value={root} onChange={(e) => setRoot(Number(e.target.value))}>
                    {roots.map((r, k) => <option key={r.path} value={k}>{r.path}</option>)}
                  </select>
                </label>
              ) : roots.length === 1 && <div className="text-[14px] text-s-muted break-words">The result is saved on the computer running Studio, in a new folder with this name, inside <span className="font-studio-mono break-all">{roots[0].path}</span></div>}
              {anywhere && (
                <button type="button" className="justify-self-start text-[15px] font-medium text-s-accent hover:underline cursor-pointer"
                  onClick={() => setPicker({ mode: "folder", key: "@save", title: "Save the result in" })}>Save somewhere else</button>
              )}
            </Part>
            {startError && <Callout tone="bad" title="Not started">{startError}</Callout>}
            {problems.filter((p) => p.key !== "").length > 0 && (
              <Callout tone="bad" title="Some settings need attention">
                <ul className="grid gap-1 mt-1">
                  {problems.filter((p) => p.key !== "").map((p) => (
                    <li key={p.key}>
                      <button type="button" className="text-left cursor-pointer hover:underline" onClick={() => goTo(stepOfKey(p.key))}>
                        <b className="font-semibold">Step {stepOf(stepOfKey(p.key)).n}:</b> <Mono>{p.key}</Mono> {p.message}
                      </button>
                    </li>
                  ))}
                </ul>
              </Callout>
            )}
            {!scanning && need.length > 0 && (
              <div className="grid gap-1.5" role="status">
                <div className="text-base font-semibold">Before you can start:</div>
                <ul className="grid gap-1">
                  {need.map((m) => (
                    <li key={m.text}>
                      <button type="button" className="text-left text-[15px] text-s-accent font-medium hover:underline cursor-pointer" onClick={() => goTo(m.step)}>
                        Step {stepOf(m.step).n}: {m.text}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            <Button kind="primary" onClick={start} busy={starting} disabled={!ready} className="min-h-14 !text-lg">Start calibration</Button>
            <details>
              <summary className="cursor-pointer text-base font-semibold text-s-accent hover:underline">Prefer a terminal? See the settings file</summary>
              <div className="grid gap-3 mt-3">
                <Explain>Starting writes your choices as <Mono>calib_param.yml</Mono> into the result folder. The same file runs with <Mono>ds-msp-calibrate-rig --config</Mono>.</Explain>
                <Button className="justify-self-start" onClick={loadPreview} busy={previewBusy}>Check and show the file</Button>
                {preview && (
                  <>
                    {preview.ok
                      ? <Callout tone="ok" title="These settings are valid" />
                      : <Callout tone="bad" title="These settings have problems">
                          <ul className="grid gap-1 mt-1">{preview.problems.map((p, k) => <li key={k}>{p.key && <Mono>{p.key}</Mono>} {p.message}</li>)}</ul>
                        </Callout>}
                    {preview.warnings.map((w) => <Callout key={w} tone="warn">{w}</Callout>)}
                    <pre className="font-studio-mono text-[13px] leading-relaxed bg-s-sunken border border-s-rule rounded-lg p-3 overflow-x-auto max-h-80">{preview.yaml}</pre>
                  </>
                )}
              </div>
            </details>
          </Step>
        </div>
      </div>

      {picker && (
        <FolderPicker mode={picker.mode} title={picker.title} start={opensAt} onClose={() => setPicker(null)}
          onPick={(path, ref) => (picker.mode === "folder" ? pickFolder(path, ref) : pickFile(path, ref))} />
      )}
    </Page>
  );
}

const LINK_BUTTON = "inline-flex items-center rounded-lg px-4 min-h-11 text-[15px] font-semibold border-[1.5px] border-s-field-line/70 bg-s-surface shadow-sm hover:bg-s-sunken hover:border-s-strong/50";

const FOUND_WORDS = {
  settings: { what: "Settings: the board, the lens types and the rest of this page", action: "Use these settings" },
  lens: { what: "Lens values for the cameras", action: "Use these lens values" },
  corners: { what: "Corners found earlier: the photos would not be read again", action: "Use these corners" },
};

function Page({ children, ...drop }: { children: ReactNode } & Pick<React.HTMLAttributes<HTMLElement>, "onDragOver" | "onDragLeave" | "onDrop">) {
  return <main {...drop} className="mx-auto w-full max-w-7xl px-4 sm:px-6 py-8 grid gap-7 text-base">{children}</main>;
}

function lensSummary(models: string[], n: number): string {
  const counts = new Map<string, number>();
  for (let k = 0; k < n; k++) counts.set(models[k] ?? "radtan", (counts.get(models[k] ?? "radtan") ?? 0) + 1);
  const word = (m: string) => (m === "radtan" ? "standard" : m === "kb" ? "fisheye" : m);
  return [...counts].map(([m, c]) => (counts.size === 1 ? `All lenses ${word(m)}` : `${c} ${word(m)}`)).join(", ");
}
