import { ITEM_FIELDS } from "@/lib/api/client";

export type FieldStatus = "same" | "diff" | "onlyA" | "onlyB";

export type FieldRow = {
  path: string;
  a: unknown;
  b: unknown;
  status: FieldStatus;
  /** Поле есть в схеме Item, то есть Python его читает. */
  known: boolean;
};

const MAX_DEPTH = 3;

/** Поле-за-полем сравнение двух JSON: вложенные объекты разворачиваются в пути вида attributes.brand. */
export function diffFields(a: Record<string, unknown>, b: Record<string, unknown>): FieldRow[] {
  const flatA = flatten(a);
  const flatB = flatten(b);
  // Порядок полей как в A, поля только из B — следом; вложенные (attributes.*) держатся вместе.
  const top = [...new Set([...Object.keys(a), ...Object.keys(b)])];
  const paths = [...new Set([...flatA.keys(), ...flatB.keys()])].sort(
    (x, y) => top.indexOf(x.split(".")[0]) - top.indexOf(y.split(".")[0]),
  );

  return paths.map((path) => {
    const inA = flatA.has(path);
    const inB = flatB.has(path);
    const valueA = flatA.get(path);
    const valueB = flatB.get(path);
    const status: FieldStatus = !inA
      ? "onlyB"
      : !inB
        ? "onlyA"
        : JSON.stringify(valueA) === JSON.stringify(valueB)
          ? "same"
          : "diff";
    return { path, a: valueA, b: valueB, status, known: ITEM_FIELDS.has(path.split(".")[0]) };
  });
}

function flatten(obj: Record<string, unknown>, prefix = "", depth = 0, out = new Map<string, unknown>()) {
  for (const [key, value] of Object.entries(obj)) {
    const path = prefix ? `${prefix}.${key}` : key;
    const isObject = typeof value === "object" && value !== null && !Array.isArray(value);
    if (isObject && depth < MAX_DEPTH && Object.keys(value).length > 0) {
      flatten(value as Record<string, unknown>, path, depth + 1, out);
    } else {
      out.set(path, value);
    }
  }
  return out;
}
