"use client";

/* eslint-disable @next/next/no-img-element -- превью из data: URI и чужих ссылок, next/image тут не нужен */

import { useState } from "react";

import { ITEM_FIELDS } from "@/lib/api/client";
import type { Slot } from "@/lib/files";
import { formatBytes } from "@/lib/format";
import { ImageIcon } from "./icons";

export function SlotPreview({ slot }: { slot: Slot }) {
  return slot.type === "image" ? <ImagePreview slot={slot} /> : <JsonPreview slot={slot} />;
}

function ImagePreview({ slot }: { slot: Extract<Slot, { type: "image" }> }) {
  return (
    <div className="flex h-full flex-col">
      <div className="checker flex min-h-0 flex-1 items-center justify-center rounded-xl p-3">
        {slot.width ? (
          <img src={slot.dataUrl} alt={slot.name} className="max-h-full max-w-full rounded-md object-contain shadow-sm" />
        ) : (
          <div className="flex flex-col items-center gap-2 px-6 text-center text-sm text-zinc-500">
            <ImageIcon className="size-8 text-zinc-400" />
            Браузер не может показать превью — Python всё равно попробует прочитать файл
          </div>
        )}
      </div>
      <Meta
        items={[slot.width && slot.height ? `${slot.width}×${slot.height}` : null, formatBytes(slot.size), "картинка"]}
      />
    </div>
  );
}

function JsonPreview({ slot }: { slot: Extract<Slot, { type: "json" }> }) {
  const [raw, setRaw] = useState(false);
  const { data } = slot;
  const title = str(data.title);
  const description = str(data.description);
  const image = str(data.image_url);
  const attributes = isRecord(data.attributes) ? Object.entries(data.attributes) : [];
  const ignored = Object.keys(data).filter((key) => !ITEM_FIELDS.has(key));

  return (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1 overflow-auto rounded-xl bg-zinc-50 p-4 dark:bg-zinc-950/60">
        {raw ? (
          <pre className="font-mono text-xs leading-relaxed whitespace-pre-wrap break-all text-zinc-700 dark:text-zinc-300">
            {JSON.stringify(data, (_, v) => (typeof v === "string" && v.length > 200 ? `${v.slice(0, 200)}…` : v), 2)}
          </pre>
        ) : (
          <div className="flex gap-4">
            {image && /^(https?:|data:image\/)/.test(image) && (
              <div className="checker flex size-24 shrink-0 items-center justify-center overflow-hidden rounded-lg sm:size-28">
                <img src={image} alt="" className="max-h-full max-w-full object-contain" />
              </div>
            )}
            <div className="min-w-0 flex-1">
              {str(data.id) && <div className="mb-1 truncate font-mono text-xs text-zinc-500">{str(data.id)}</div>}
              <div className={`font-semibold leading-snug ${title ? "" : "text-zinc-400 italic"}`}>
                {title ?? "без названия"}
              </div>
              {attributes.length > 0 && (
                <div className="mt-2.5 flex flex-wrap gap-1.5">
                  {attributes.map(([key, value]) => (
                    <span
                      key={key}
                      className="rounded-md bg-white px-2 py-0.5 text-xs ring-1 ring-zinc-200 dark:bg-zinc-900 dark:ring-zinc-800"
                    >
                      <span className="text-zinc-500">{key}:</span> {String(value)}
                    </span>
                  ))}
                </div>
              )}
              {description && <p className="mt-2.5 line-clamp-4 text-sm text-zinc-600 dark:text-zinc-400">{description}</p>}
            </div>
          </div>
        )}
      </div>
      <div className="flex items-center justify-between gap-3">
        <Meta
          items={[
            formatBytes(slot.size),
            ignored.length ? `не учитывается: ${ignored.join(", ")}` : "все поля учитываются",
          ]}
        />
        <button
          type="button"
          onClick={() => setRaw(!raw)}
          className="mt-3 shrink-0 rounded-md px-2 py-0.5 font-mono text-xs text-zinc-500 ring-1 ring-zinc-200 transition hover:text-zinc-900 dark:ring-zinc-800 dark:hover:text-zinc-100"
        >
          {raw ? "карточка" : "{ } json"}
        </button>
      </div>
    </div>
  );
}

function Meta({ items }: { items: (string | null)[] }) {
  return (
    <div className="mt-3 truncate text-xs text-zinc-500" title={items.filter(Boolean).join(" · ")}>
      {items.filter(Boolean).join(" · ")}
    </div>
  );
}

function str(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
