// See an LDC mesh or an Isaac LUT at work, one camera at a time. The server draws the pictures
// with the same code that writes and checks the files (ds_msp.studio.lensview), so what is shown
// is what the board or Isaac Sim gets. Cameras are picked from a row of chips; with many cameras
// the row wraps, and Previous / Next step through them.
import { useEffect, useState } from "react";

import { api, type ExportSpec, type LensViewDoc } from "./api";
import { Segmented, Toggle } from "./setup/kit";
import { Button, ErrorBox } from "./ui";

export type CamTone = "ok" | "warn" | "bad" | "plain";
type Kind = "ldc" | "lut";

const DOT: Record<CamTone, string> = { ok: "bg-s-ok", warn: "bg-s-warn", bad: "bg-s-bad", plain: "bg-s-faint" };

export function LensViewer({ cameras, kinds, camera, onCamera, spec, filesId, photos }: {
  cameras: { name: string; tone: CamTone; label?: string }[];
  kinds: Kind[];
  camera: string; onCamera: (name: string) => void;
  /** Preview what the spec would write; or show the files written in this folder. */
  spec?: ExportSpec | null; filesId?: string;
  /** Whether a camera's own photos can be shown (a rig result). */
  photos: boolean;
}) {
  const [kind, setKind] = useState<Kind>(kinds[0] ?? "ldc");
  const [input, setInput] = useState<"room" | "photo">("room");
  const [frame, setFrame] = useState<number | null>(null);
  const [overlay, setOverlay] = useState(false);
  const [doc, setDoc] = useState<LensViewDoc | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { if (!kinds.includes(kind) && kinds[0]) setKind(kinds[0]); }, [kinds, kind]);
  useEffect(() => { setFrame(null); }, [camera]);

  const specKey = JSON.stringify(spec ?? null);
  useEffect(() => {
    if (!camera || (!spec && !filesId)) return;
    let live = true;
    setBusy(true); setError(null);
    const width = Math.min(1400, Math.round((kind === "ldc" ? 0.5 : 0.9) * Math.min(window.innerWidth, 1200) * (window.devicePixelRatio || 1)));
    const opts = { camera, kind, photo: kind === "ldc" && input === "photo", frame, overlay: kind === "ldc" && overlay, width };
    const t = window.setTimeout(() => {
      const call = filesId ? api.runExportView(filesId, opts) : api.exportView(spec!, opts);
      call.then((d) => { if (live) setDoc(d); }, (e: Error) => { if (live) setError(e.message); })
        .finally(() => { if (live) setBusy(false); });
    }, 250);
    return () => { live = false; window.clearTimeout(t); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera, kind, input, frame, overlay, specKey, filesId]);

  const k = cameras.findIndex((c) => c.name === camera);
  const frames = doc?.frames ?? [];
  const fk = doc?.frame != null ? frames.indexOf(doc.frame) : -1;

  return (
    <div className="grid gap-4 min-w-0">
      <div className="grid gap-2">
        <div className="text-base font-semibold">See it, one camera at a time</div>
        <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Camera to show">
          {cameras.map((c) => (
            <button key={c.name} type="button" aria-pressed={c.name === camera} onClick={() => onCamera(c.name)}
              title={c.label}
              className={`inline-flex items-center gap-1.5 rounded-full border px-3 min-h-9 text-[14px] font-medium cursor-pointer transition focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-s-accent/30 ${
                c.name === camera ? "border-s-accent bg-s-accent text-white" : "border-s-rule bg-s-surface hover:border-s-accent/60 hover:bg-s-accent-soft/50"}`}>
              <span className={`size-2.5 rounded-full ${DOT[c.tone]} ${c.name === camera ? "ring-2 ring-white" : ""}`} aria-hidden="true" />
              {c.name}
            </button>
          ))}
        </div>
        {cameras.length > 1 && (
          <div className="flex gap-2">
            <Button disabled={k <= 0} onClick={() => onCamera(cameras[k - 1].name)}>Previous camera</Button>
            <Button disabled={k < 0 || k >= cameras.length - 1} onClick={() => onCamera(cameras[k + 1].name)}>Next camera</Button>
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
        {kinds.length > 1 && (
          <Segmented label="What to show" value={kind} onChange={(v) => setKind(v as Kind)}
            options={[{ id: "ldc", title: "On the TI board (LDC)" }, { id: "lut", title: "In Isaac Sim (LUT)" }]} />
        )}
        {kind === "ldc" && (
          <Segmented label="Picture to correct" value={input} onChange={(v) => setInput(v as "room" | "photo")}
            options={photos ? [{ id: "room", title: "Test room" }, { id: "photo", title: "A photo" }] : [{ id: "room", title: "Test room" }]} />
        )}
        {kind === "ldc" && <Toggle id="lv-overlay" on={overlay} onChange={setOverlay}>Mark where the mesh is less accurate</Toggle>}
      </div>

      {error && <ErrorBox title="Could not draw this camera" detail={error} />}
      {doc && (
        <div className={`grid gap-3 ${busy ? "opacity-60" : ""} transition-opacity`} aria-busy={busy}>
          {doc.kind === "ldc" && doc.images.camera && doc.images.corrected ? (
            <div className="grid gap-3 md:grid-cols-2">
              <Figure img={doc.images.camera} caption={`What ${doc.camera} sees${doc.used_photo ? `: photo ${doc.frame}` : ": the test room"}`} />
              <Figure img={doc.images.corrected} caption="After the board's correction" />
            </div>
          ) : doc.images.render ? (
            <Figure img={doc.images.render} caption={`The test room as Isaac Sim renders it through ${doc.camera}'s lens`} />
          ) : null}
          {kind === "ldc" && input === "photo" && frames.length > 1 && (
            <div className="flex items-center gap-2 text-[14px] text-s-muted">
              <Button disabled={fk <= 0} onClick={() => setFrame(frames[fk - 1])}>Previous photo</Button>
              <Button disabled={fk < 0 || fk >= frames.length - 1} onClick={() => setFrame(frames[fk + 1])}>Next photo</Button>
              <span className="tnum">{fk + 1} of {frames.length}</span>
            </div>
          )}
          {kind === "ldc" && input === "photo" && !doc.used_photo && (
            <p className="text-[14px] text-s-muted">No photo of this camera could be read, so the test room is shown.</p>
          )}
          <p className="text-[14px] text-s-muted leading-relaxed max-w-[80ch]">
            {doc.kind === "ldc"
              ? <>{doc.used_photo ? "After a correct correction, edges that are straight in the scene, like the board's, are straight in the picture." : "The test room is a box around the camera with a grid on every wall and a red cross in front. After a correct correction its lines are straight."} Black is where the corrected image shows nothing. {overlay && doc.overlay && (doc.overlay.over_half_px === 0
                ? <b className="font-semibold text-s-ok">Nothing to mark: the mesh places every visible pixel within 0.5 px.</b>
                : <>Amber: the mesh misplaces the picture by more than 0.5 px there ({pct(doc.overlay.over_half_px)} of the picture); red: by more than 2 px ({pct(doc.overlay.over_2_px)}).</>)}</>
              : <>The same test room, seen through the ray directions the LUT stores. Floor at the bottom and the red cross ahead means the axes are right.</>}
            {" "}{doc.from_files ? "Drawn from the files written." : "A preview: nothing is written yet."} Simulated on this computer, not on a board.
          </p>
        </div>
      )}
      {!doc && !error && <div className="text-[15px] text-s-muted" role="status">Drawing…</div>}
    </div>
  );
}

const pct = (v: number) => (v > 0 && v < 0.001 ? "under 0.1 %" : `${(100 * v).toFixed(1)} %`);

function Figure({ img, caption }: { img: { src: string; width: number; height: number }; caption: string }) {
  return (
    <figure className="grid gap-1.5 min-w-0">
      <img src={img.src} width={img.width} height={img.height} alt={caption}
        className="w-full h-auto rounded-lg border border-s-rule bg-black" />
      <figcaption className="text-[14px] font-medium text-s-text">{caption}</figcaption>
    </figure>
  );
}
