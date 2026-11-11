// Which folder is which camera, when the folders are not named Cam_001, Cam_002, ... Each folder
// is a card with one of its photos and the name it will go by. The folders are never renamed:
// Studio links them under the new names.
import { useState } from "react";

import { api, type ScanDoc } from "../api";
import { int } from "../format";
import { Button, Callout } from "../ui";
import { Explain, Mono } from "./kit";

type Folder = ScanDoc["folders"][number];

const camName = (k: number) => `Cam_${String(k + 1).padStart(3, "0")}`;

export function LineupStep({ root, folders, previous, onApply, busy, error }: {
  root: string; folders: Folder[];
  /** The folders of the lineup being changed, in camera order. */
  previous: string[] | null;
  onApply: (folders: string[]) => void; busy: boolean; error: string | null;
}) {
  const usable = folders.filter((f) => f.images > 0);
  const empty = folders.filter((f) => f.images === 0);
  const [order, setOrder] = useState<string[]>(() => {
    const first = (previous ?? []).filter((n) => usable.some((f) => f.name === n));
    return [...first, ...usable.map((f) => f.name).filter((n) => !first.includes(n))];
  });
  const [used, setUsed] = useState<Set<string>>(() => new Set(previous ?? usable.filter((f) => f.proposed).map((f) => f.name)));
  const [dragged, setDragged] = useState<string | null>(null);

  const cameras = order.filter((n) => used.has(n));
  const skipped = order.filter((n) => !used.has(n));
  const byName = new Map(folders.map((f) => [f.name, f]));

  // Cameras come first, so moving one card only ever changes the camera numbers.
  const place = (name: string, at: number) => setOrder(() => {
    const rest = cameras.filter((n) => n !== name);
    rest.splice(at, 0, name);
    return [...rest, ...skipped];
  });
  const toggle = (name: string) => setUsed((old) => {
    const next = new Set(old);
    if (next.has(name)) next.delete(name); else next.add(name);
    return next;
  });

  const card = (name: string) => {
    const f = byName.get(name) as Folder;
    const k = cameras.indexOf(name);
    const on = k >= 0;
    const target = on ? camName(k) : null;
    const same = target === name;
    return (
      <li key={name} draggable={on}
        onDragStart={() => setDragged(name)} onDragEnd={() => setDragged(null)}
        onDragOver={(e) => { if (on && dragged) e.preventDefault(); }}
        onDrop={(e) => { e.preventDefault(); if (on && dragged && dragged !== name) place(dragged, k); setDragged(null); }}
        className={`rounded-xl border-2 overflow-hidden bg-s-surface grid content-start transition ${
          !on ? "border-dashed border-s-rule" : same ? "border-s-ok/60" : "border-s-warn/70"} ${on ? "cursor-grab hover:shadow-md hover:-translate-y-0.5" : "hover:border-s-accent/50"} ${dragged === name ? "opacity-50" : ""}`}>
        <div className="relative bg-s-viewport" style={{ aspectRatio: `${f.width ?? 4} / ${f.height ?? 3}` }}>
          <img src={api.folderThumbUrl(root, name)} alt={`A photo from ${name}`} loading="lazy" draggable={false}
            className={`absolute inset-0 size-full object-contain ${on ? "" : "grayscale opacity-40"}`} />
          {on && <span className={`absolute top-2 left-2 size-9 rounded-full grid place-items-center text-base font-bold text-white shadow ${same ? "bg-s-ok" : "bg-s-warn"}`}>{k + 1}</span>}
        </div>
        <div className="p-3 grid gap-2">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[15px]">
            <Mono>{name}</Mono>
            {on && !same && <><span aria-hidden="true" className="text-s-warn font-bold">→</span><span className="sr-only">will be read as</span>
              <span className="rounded-md bg-s-warn-soft text-s-warn font-semibold font-studio-mono text-sm px-2 py-0.5">{target}</span></>}
            {on && same && <span className="text-s-ok font-medium">✓ already right</span>}
          </div>
          <div className="text-[14px] text-s-muted tnum">
            {int(f.images)} photo{f.images === 1 ? "" : "s"}{f.width ? ` · ${f.width} × ${f.height}` : ""}{!on && f.reason ? ` · ${f.reason}` : ""}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {on && cameras.length > 1 && <>
              <Button kind="quiet" disabled={k === 0} aria-label={`Move ${name} earlier`} onClick={() => place(name, k - 1)}>←</Button>
              <Button kind="quiet" disabled={k === cameras.length - 1} aria-label={`Move ${name} later`} onClick={() => place(name, k + 1)}>→</Button>
            </>}
            <Button kind="quiet" className="ml-auto" onClick={() => toggle(name)}>{on ? "Not a camera" : "Use as a camera"}</Button>
          </div>
        </div>
      </li>
    );
  };

  return (
    <div className="grid gap-4">
      <Callout tone="warn" title="Say which folder is which camera">
        These folders are not numbered the way the calibration reads them. Your photos are fine. Check the order below: it sets the camera numbers, and lens values go with those numbers. Drag a card to move it.
      </Callout>
      <ul className="grid gap-4 grid-cols-[repeat(auto-fill,minmax(200px,1fr))]" aria-label="Camera folders">
        {cameras.map(card)}
        {skipped.map(card)}
      </ul>
      {empty.length > 0 && <Explain>Left out because they hold no photos: {empty.map((f) => f.name).join(", ")}.</Explain>}
      {error && <Callout tone="bad" title="That lineup was not used">{error}</Callout>}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
        <Button kind="primary" busy={busy} disabled={cameras.length === 0} onClick={() => onApply(cameras)} className="min-h-12 !text-base px-5">
          Use these {cameras.length} camera{cameras.length === 1 ? "" : "s"}
        </Button>
        <Explain>Your folders are not renamed or moved. Studio links them under these names for the calibration.</Explain>
      </div>
    </div>
  );
}
