// Lengths in a result are in the unit of the board's square size, which the result does not
// name. The user says once which unit that was; lengths are then shown in millimetres or metres.
import { useCallback, useState } from "react";

export type Unit = "m" | "cm" | "mm" | "?";
const KEY = "dsmsp.unit";
const TO_MM: Record<Unit, number | null> = { m: 1000, cm: 10, mm: 1, "?": null };

export function useUnit(): [Unit, (u: Unit) => void] {
  const [unit, setUnit] = useState<Unit>(() => {
    try {
      const v = window.localStorage.getItem(KEY);
      return v === "m" || v === "cm" || v === "mm" ? v : "?";
    } catch {
      return "?";
    }
  });
  const set = useCallback((u: Unit) => {
    setUnit(u);
    try {
      window.localStorage.setItem(KEY, u);
    } catch {
      /* private window: the choice lasts for this page only */
    }
  }, []);
  return [unit, set];
}

function digits(v: number): number {
  const a = Math.abs(v);
  return a >= 100 ? 0 : a >= 10 ? 1 : a >= 0.1 ? 2 : 3;
}

/** A length for reading: "1.24 mm", "2.50 m", or "0.0012 units" when the unit is not known. */
export function length(v: number | null | undefined, unit: Unit): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const k = TO_MM[unit];
  if (k == null) return `${v.toFixed(Math.abs(v) >= 10 ? 2 : 4)} units`;
  const mm = v * k;
  if (Math.abs(mm) >= 1000) return `${(mm / 1000).toFixed(digits(mm / 1000))} m`;
  if (Math.abs(mm) < 0.01 && mm !== 0) return `${(mm * 1000).toFixed(digits(mm * 1000))} µm`;
  return `${mm.toFixed(digits(mm))} mm`;
}

/** Scene units in one millimetre, or null when the unit is not known. */
export function oneMillimetre(unit: Unit): number | null {
  const k = TO_MM[unit];
  return k == null ? null : 1 / k;
}

export const UNIT_CHOICES: { id: Unit; label: string }[] = [
  { id: "?", label: "not sure" },
  { id: "m", label: "metres" },
  { id: "cm", label: "centimetres" },
  { id: "mm", label: "millimetres" },
];
