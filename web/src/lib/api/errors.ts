import { formatBytes } from "@/lib/format";

import type { ValidationError } from "./client";

const SIDE: Record<string, string> = { item_a: "A", item_b: "B" };

/** Превращает ответ Python-API с ошибкой в понятный текст. */
export function describeError(status: number, body: unknown): string {
  const detail = (body as { detail?: unknown } | undefined)?.detail;

  // Ошибки валидации Pydantic: [{loc: ["body", "item_a", "image_url", ...], msg}, ...]
  if (Array.isArray(detail)) {
    const lines = new Map<string, string>();
    for (const err of detail as ValidationError[]) {
      const [, side, field] = err.loc.map(String);
      const where = `${SIDE[side] ?? side}${field ? ` · ${field}` : ""}`;
      lines.set(where, `${where}: ${translate(field, err.msg)}`);
    }
    return [...lines.values()].join("\n");
  }

  // Картинку не удалось прочитать: "<url>: <причина>"
  if (typeof detail === "string") {
    const tooLarge = detail.match(/image is larger than (\d+) bytes/);
    if (tooLarge) return `Картинка больше ${formatBytes(Number(tooLarge[1]))} — это лимит ITEM_COMPARE_IMAGE_MAX_BYTES`;
    if (detail.endsWith("not a readable image")) return "Python не смог прочитать картинку: Pillow не знает этот формат";
    if (detail.endsWith("invalid base64")) return "Картинка повреждена: невалидный base64";
    return detail;
  }

  return `Python API ответил ошибкой ${status}`;
}

function translate(field: string | undefined, msg: string): string {
  if (msg.includes("item needs image_url")) return "нужна картинка (image_url) или текст (title, description, attributes)";
  if (field === "image_url") return "должна быть ссылка http(s) или data:image/…;base64";
  return msg;
}
