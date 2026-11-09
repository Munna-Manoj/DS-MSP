// Number formatting for the viewer: fixed decimals for display, scientific only when tiny.

export function num(v: number | null | undefined, digits = 2): string {
  if (v == null || !Number.isFinite(v)) return "—";
  if (v !== 0 && Math.abs(v) < Math.pow(10, -digits)) return v.toExponential(1);
  return v.toFixed(digits);
}

export function px(v: number | null | undefined, digits = 2): string {
  return v == null ? "—" : `${num(v, digits)} px`;
}

export function int(v: number | null | undefined): string {
  return v == null ? "—" : v.toLocaleString("en-US");
}

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} kB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function when(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

export function camName(id: string | number): string {
  return `cam${String(id).padStart(2, "0")}`;
}
