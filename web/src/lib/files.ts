import type { Item } from "@/lib/api/client";

export type Slot =
  | { type: "image"; name: string; size: number; dataUrl: string; width?: number; height?: number }
  | { type: "json"; name: string; size: number; data: Record<string, unknown>; text: string };

const IMAGE_EXT = /\.(png|jpe?g|gif|webp|bmp|tiff?|avif)$/i;

export const ACCEPT = "image/*,.json,application/json";

/** Читает перетащенный файл: картинку — в data: URI, JSON — в объект товара. */
export async function readSlot(file: File): Promise<Slot> {
  if (file.type === "application/json" || /\.json$/i.test(file.name)) {
    return parseJson(await file.text(), file.name);
  }
  if (file.type.startsWith("image/") || IMAGE_EXT.test(file.name)) {
    const dataUrl = withImageMime(await readAsDataUrl(file), file.name);
    return { type: "image", name: file.name, size: file.size, dataUrl, ...(await imageSize(dataUrl)) };
  }
  throw new Error(`${file.name}: поддерживаются только картинки и .json`);
}

export function parseJson(text: string, name: string): Slot {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch (err) {
    throw new Error(`${name}: невалидный JSON — ${(err as Error).message}`);
  }
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    throw new Error(`${name}: нужен один товар — JSON-объект { … }`);
  }
  return { type: "json", name, size: new Blob([text]).size, data: data as Record<string, unknown>, text };
}

export function toItem(slot: Slot): Item {
  // JSON уходит как есть: Python сам проверит его по схеме Item и отбросит лишние поля.
  return slot.type === "image" ? { image_url: slot.dataUrl } : (slot.data as Item);
}

function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error(`${file.name}: не удалось прочитать файл`));
    reader.readAsDataURL(file);
  });
}

// Python принимает только data:image/...;base64 — файлам без MIME-типа подставляем его по расширению.
function withImageMime(dataUrl: string, name: string): string {
  if (dataUrl.startsWith("data:image/")) return dataUrl;
  const ext = name.split(".").pop()?.toLowerCase() ?? "png";
  return `data:image/${ext === "jpg" ? "jpeg" : ext};base64,${dataUrl.slice(dataUrl.indexOf(",") + 1)}`;
}

function imageSize(src: string): Promise<{ width?: number; height?: number }> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
    // Браузер может не уметь показать формат (например, TIFF), а Pillow в Python — уметь.
    img.onerror = () => resolve({});
    img.src = src;
  });
}
