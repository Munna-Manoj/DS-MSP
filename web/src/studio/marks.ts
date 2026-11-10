// Photos marked on a result page to be left out of the next run. The marks live in the
// browser session until the setup page picks them up and writes them as `exclude_views`.
import { camName } from "./format";

const key = (runId: string) => `dsmsp.exclude.${runId}`;

/** The marks of a result, as `exclude_views` entries ("cam:frame"). */
export function readMarked(runId: string): string[] {
  try {
    const raw = window.sessionStorage.getItem(key(runId));
    const list = raw ? (JSON.parse(raw) as unknown) : [];
    return Array.isArray(list) ? list.filter((x): x is string => typeof x === "string") : [];
  } catch {
    return [];
  }
}

export function writeMarked(runId: string, marks: string[]): void {
  try {
    if (marks.length) window.sessionStorage.setItem(key(runId), JSON.stringify(marks));
    else window.sessionStorage.removeItem(key(runId));
  } catch {
    /* a private window: the marks last while the page is open */
  }
}

/** "0:12" reads "cam00 frame 12"; "0:12:1" reads "cam00 frame 12 board 1". */
export function viewWords(entry: string): string {
  const [c, f, b] = entry.split(":");
  return `${camName(c)} frame ${f}${b != null ? ` board ${b}` : ""}`;
}

export function viewList(entries: string[], most = 6): string {
  return entries.slice(0, most).map(viewWords).join(", ") + (entries.length > most ? ` and ${entries.length - most} more` : "");
}
