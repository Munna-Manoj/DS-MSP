// A folder (or file) chooser. A browser cannot hand a page the path of a folder on disk, so the
// server lists them (GET api/browse): any folder on this computer, or, for a Studio limited to
// them, the folders it was started with.
import { useEffect, useRef, useState } from "react";
import { api, type BrowseDoc } from "./api";
import { Button, ErrorBox, Loading } from "./ui";

export function FolderPicker({ mode, title, start, onPick, onClose }: {
  mode: "folder" | "file";
  title: string;
  /** Folder to open first, as a browse ref; the list of roots when absent. */
  start?: string | null;
  onPick: (path: string, ref: string) => void;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [ref, setRef] = useState<string | null>(start ?? null);
  // The folder that was open last, to go back to when the next one cannot be opened.
  const [shown, setShown] = useState<string | null>(null);
  const [doc, setDoc] = useState<BrowseDoc | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [typed, setTyped] = useState("");
  const [anywhere, setAnywhere] = useState(false);

  useEffect(() => {
    const d = dialog.current;
    if (d && !d.open) d.showModal();
  }, []);

  useEffect(() => {
    let live = true;
    setDoc(null);
    setError(null);
    api.browse(ref).then(
      (d) => {
        if (!live) return;
        setAnywhere(d.anywhere === true);
        // one served folder and nowhere else to go: open it instead of showing a list of one
        if (ref == null && d.dirs.length === 1 && !d.anywhere) setRef(d.dirs[0].ref);
        else { setDoc(d); setShown(ref); }
      },
      (e: Error) => {
        if (!live) return;
        if (ref != null && ref === start) setRef(null);   // the folder to open first has gone
        else setError(e.message);
      },
    );
    return () => { live = false; };
  }, [ref, start]);

  const up = doc && doc.ref != null ? doc.parent : undefined;
  return (
    <dialog
      ref={dialog}
      onClose={onClose}
      aria-label={title}
      className="m-auto w-[min(44rem,calc(100vw-2rem))] max-h-[calc(100vh-4rem)] rounded-xl border border-s-rule
        bg-s-surface text-s-text p-0 backdrop:bg-black/50"
    >
      <div className="flex flex-col max-h-[calc(100vh-4rem)]">
        <div className="px-5 pt-4 pb-3 border-b border-s-rule grid gap-1">
          <h2 className="text-lg font-semibold">{title}</h2>
          <p className="text-[14px] text-s-muted">Showing the computer running Studio, which may not be the device you are using now.</p>
          <div className="font-studio-mono text-xs text-s-muted break-all min-h-4">
            {doc?.path ?? (anywhere ? "Places on that computer" : "Folders this Studio was started with")}
          </div>
          {anywhere && (
            <form className="flex gap-2 mt-1.5" onSubmit={(e) => { e.preventDefault(); if (typed.trim()) setRef(`abs:${typed.trim()}`); }}>
              <input aria-label="Go to a folder by its path" value={typed} onChange={(e) => setTyped(e.target.value)} placeholder="Or paste a folder path"
                className="flex-1 min-w-0 rounded-md border border-s-field-line bg-s-field px-3 min-h-10 text-sm font-studio-mono outline-none focus:border-s-accent" />
              <Button type="submit">Go</Button>
            </form>
          )}
        </div>

        <div className="overflow-y-auto px-2 py-2 min-h-48">
          {error && (
            <div className="p-3 grid gap-3">
              <ErrorBox title="Could not open this folder" detail={error} />
              <Button className="justify-self-start" onClick={() => { setError(null); if (ref === shown) setRef(null); else setRef(shown); }}>Go back</Button>
            </div>
          )}
          {!doc && !error && <div className="p-3"><Loading what="folder" /></div>}
          {doc && (
            <ul className="grid">
              {up !== undefined && (
                <li>
                  <button type="button" onClick={() => setRef(up)} className={ROW}>
                    <span aria-hidden="true">↑</span>
                    <span className="text-s-muted">{up == null ? "All folders" : "Up one folder"}</span>
                  </button>
                </li>
              )}
              {doc.places?.map((d) => (
                <li key={d.ref}>
                  <button type="button" onClick={() => setRef(d.ref)} className={ROW}>
                    <FolderIcon />
                    <span className="font-medium text-left">{d.name}</span>
                    <span className="ml-auto text-xs text-s-faint font-studio-mono break-all">{d.path}</span>
                  </button>
                </li>
              ))}
              {doc.dirs.map((d) => (
                <li key={d.ref}>
                  <button type="button" disabled={d.locked} onClick={() => setRef(d.ref)} className={`${ROW} ${d.locked ? "opacity-50 cursor-not-allowed hover:bg-transparent" : ""}`}>
                    <FolderIcon />
                    <span className="font-medium break-all text-left">{d.name}</span>
                    <span className="ml-auto text-xs text-s-faint whitespace-nowrap">
                      {d.locked ? "no permission" : d.result ? "calibration result" : d.images > 0 ? `${d.images} images` : ""}
                    </span>
                  </button>
                </li>
              ))}
              {mode === "file" && doc.files.map((f) => (
                <li key={f.path}>
                  <button type="button" onClick={() => onPick(f.path, doc.ref ?? "")} className={ROW}>
                    <FileIcon />
                    <span className="break-all text-left">{f.name}</span>
                    <span className="ml-auto text-xs text-s-accent">Choose</span>
                  </button>
                </li>
              ))}
              {doc.dirs.length === 0 && (mode === "folder" || doc.files.length === 0) && (
                <li className="px-3 py-6 text-sm text-s-muted">
                  {mode === "folder"
                    ? doc.images > 0
                      ? `No folders inside. This folder holds ${doc.images} images itself.`
                      : "No folders inside."
                    : "No .yml, .yaml or .json files here."}
                </li>
              )}
            </ul>
          )}
        </div>

        <div className="px-5 py-3 border-t border-s-rule flex flex-wrap items-center gap-3">
          <p className="text-xs text-s-muted mr-auto max-w-sm">
            {anywhere ? "Nothing is changed in a folder you only read from."
              : <>Only folders given to <code className="font-studio-mono">ds-msp studio --root</code> are listed.</>}
          </p>
          <Button onClick={() => dialog.current?.close()}>Cancel</Button>
          {mode === "folder" && (
            <Button
              kind="primary"
              disabled={!doc || doc.path == null}
              onClick={() => doc?.path && onPick(doc.path, doc.ref ?? "")}
            >
              Use this folder
            </Button>
          )}
        </div>
      </div>
    </dialog>
  );
}

const ROW =
  "w-full flex items-center gap-3 rounded-md px-3 min-h-11 text-sm hover:bg-s-sunken " +
  "focus-visible:outline-2 focus-visible:outline-s-accent";

function FolderIcon() {
  return (
    <svg viewBox="0 0 20 20" className="size-4 shrink-0 text-s-accent" fill="none" stroke="currentColor"
      strokeWidth="1.6" aria-hidden="true">
      <path d="M2.5 5.5a1 1 0 0 1 1-1h4l1.5 2h7.5a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1h-13a1 1 0 0 1-1-1z" />
    </svg>
  );
}

function FileIcon() {
  return (
    <svg viewBox="0 0 20 20" className="size-4 shrink-0 text-s-muted" fill="none" stroke="currentColor"
      strokeWidth="1.6" aria-hidden="true">
      <path d="M5 2.5h6.5L15.5 6.5v11h-10.5z M11.5 2.5v4h4" />
    </svg>
  );
}
