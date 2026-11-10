// A folder (or file) chooser over the folders Studio was started with. A browser cannot hand a
// page the path of a folder on disk, so the server lists them (GET api/browse).
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
  const [doc, setDoc] = useState<BrowseDoc | null>(null);
  const [error, setError] = useState<string | null>(null);

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
        // one served folder: open it straight away instead of showing a list of one
        if (ref == null && d.dirs.length === 1) setRef(d.dirs[0].ref);
        else setDoc(d);
      },
      (e: Error) => live && setError(e.message),
    );
    return () => { live = false; };
  }, [ref]);

  const up = doc && doc.ref != null ? doc.parent : undefined;
  return (
    <dialog
      ref={dialog}
      onClose={onClose}
      aria-label={title}
      className="m-auto w-[min(44rem,calc(100vw-2rem))] max-h-[calc(100vh-4rem)] rounded-xl border border-s-line
        bg-s-surface text-s-text p-0 backdrop:bg-black/50"
    >
      <div className="flex flex-col max-h-[calc(100vh-4rem)]">
        <div className="px-5 pt-4 pb-3 border-b border-s-line grid gap-1">
          <h2 className="text-lg font-semibold">{title}</h2>
          <div className="font-studio-mono text-xs text-s-muted break-all min-h-4">
            {doc?.path ?? "Folders this Studio was started with"}
          </div>
        </div>

        <div className="overflow-y-auto px-2 py-2 min-h-48">
          {error && <div className="p-3"><ErrorBox title="Could not open this folder" detail={error} /></div>}
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
              {doc.dirs.map((d) => (
                <li key={d.ref}>
                  <button type="button" onClick={() => setRef(d.ref)} className={ROW}>
                    <FolderIcon />
                    <span className="font-medium break-all text-left">{d.name}</span>
                    <span className="ml-auto text-xs text-s-faint whitespace-nowrap">
                      {d.result ? "calibration result" : d.images > 0 ? `${d.images} images` : ""}
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

        <div className="px-5 py-3 border-t border-s-line flex flex-wrap items-center gap-3">
          <p className="text-xs text-s-muted mr-auto max-w-sm">
            Only folders given to <code className="font-studio-mono">ds-msp studio --root</code> are
            listed.
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
