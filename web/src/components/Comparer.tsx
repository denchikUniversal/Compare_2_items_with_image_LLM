"use client";

import { useEffect, useEffectEvent, useRef, useState } from "react";

import { api } from "@/lib/api/client";
import { describeError } from "@/lib/api/errors";
import { parseJson, readSlot, toItem, type Slot } from "@/lib/files";
import { Breakdown } from "./Breakdown";
import { DropSlot } from "./DropSlot";
import { FieldDiff } from "./FieldDiff";
import { AlertIcon } from "./icons";
import { ScoreRing, type Run } from "./ScoreRing";

type Pair = [Slot | null, Slot | null];

const EXAMPLE: [string, string] = [
  JSON.stringify(
    {
      id: "sku-1001",
      title: "Apple iPhone 15 Pro 128GB Natural Titanium",
      attributes: { brand: "Apple", storage: "128GB", color: "Natural Titanium" },
      price: 999,
    },
    null,
    2,
  ),
  JSON.stringify(
    {
      id: "mp-77812",
      title: "iPhone 15 Pro 128 GB, natural titanium",
      attributes: { brand: "Apple", storage: "128 GB", color: "natural titanium" },
      description: "Brand new, sealed box",
      price: 949,
      seller: "TechStore",
    },
    null,
    2,
  ),
];

export function Comparer() {
  const [slots, setSlots] = useState<Pair>([null, null]);
  const [run, setRun] = useState<Run>({ status: "idle" });
  const [notice, setNotice] = useState<string | null>(null);
  const [pageDragging, setPageDragging] = useState(false);
  // Ссылки нужны обработчикам, которые срабатывают асинхронно (чтение файлов, ответ API).
  const current = useRef<Pair>([null, null]);
  const lastResult = useRef<Run>({ status: "idle" });
  const inflight = useRef<AbortController | null>(null);
  const dragDepth = useRef(0);

  function update(pair: Pair) {
    current.current = pair;
    setSlots(pair);
    void compare(pair);
  }

  async function compare([a, b]: Pair) {
    inflight.current?.abort();
    if (!a || !b) {
      lastResult.current = { status: "idle" };
      setRun(lastResult.current);
      return;
    }
    const controller = new AbortController();
    inflight.current = controller;
    const prev = lastResult.current.status === "done" ? lastResult.current.result : undefined;
    setRun({ status: "loading", prev });

    let next: Run;
    try {
      const { data, error, response } = await api.POST("/v1/compare", {
        body: { item_a: toItem(a), item_b: toItem(b) },
        signal: controller.signal,
      });
      next = data ? { status: "done", result: data } : { status: "error", message: describeError(response.status, error) };
    } catch {
      if (controller.signal.aborted) return;
      next = { status: "error", message: "Не удалось связаться с сервером Next.js" };
    }
    if (controller.signal.aborted) return;
    lastResult.current = next;
    setRun(next);
  }

  /** Раскладывает файлы по слотам: два файла — сразу в A и B, один — в указанный или первый свободный. */
  async function addSlots(read: Promise<Slot>[], target?: 0 | 1) {
    if (read.length === 0) return;
    const settled = await Promise.allSettled(read.slice(0, 2));
    const ok = settled.flatMap((r) => (r.status === "fulfilled" ? [r.value] : []));
    const errors = settled.flatMap((r) => (r.status === "rejected" ? [(r.reason as Error).message] : []));
    if (read.length > 2) errors.push(`Файлов ${read.length} — взял первые два`);
    setNotice(errors.length ? errors.join("\n") : null);
    if (ok.length === 0) return;

    const pair: Pair = [...current.current];
    if (ok.length === 2) {
      [pair[0], pair[1]] = ok;
    } else {
      const index = target ?? (pair[0] === null ? 0 : 1);
      pair[index] = ok[0];
    }
    update(pair);
  }

  function addFiles(files: File[], target?: 0 | 1) {
    void addSlots(files.map(readSlot), target);
  }

  function clear(index?: 0 | 1) {
    const pair: Pair = index === undefined ? [null, null] : [...current.current];
    if (index !== undefined) pair[index] = null;
    setNotice(null);
    update(pair);
  }

  // Ctrl/⌘+V: картинка из буфера, JSON-текст или ссылка на картинку.
  const onPaste = useEffectEvent((event: ClipboardEvent) => {
    const files = [...(event.clipboardData?.files ?? [])];
    if (files.length) {
      event.preventDefault();
      addFiles(files);
      return;
    }
    const text = event.clipboardData?.getData("text/plain").trim();
    if (!text) return;
    if (/^https?:\/\/\S+$/.test(text)) {
      void addSlots([Promise.resolve(parseJson(JSON.stringify({ image_url: text }), "ссылка из буфера"))]);
    } else if (text.startsWith("{")) {
      void addSlots([Promise.resolve().then(() => parseJson(text, "JSON из буфера"))]);
    }
  });

  // Файл, брошенный мимо слотов (хоть на шапку), тоже принимаем, а не даём браузеру открыть его.
  const onPageDrag = useEffectEvent((event: globalThis.DragEvent) => {
    event.preventDefault();
    if (event.type === "dragenter") dragDepth.current += 1;
    if (event.type === "dragleave") dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (event.type === "drop") {
      dragDepth.current = 0;
      addFiles([...(event.dataTransfer?.files ?? [])]);
    }
    if (event.type !== "dragover") setPageDragging(dragDepth.current > 0);
  });

  useEffect(() => {
    const drag = ["dragenter", "dragleave", "dragover", "drop"] as const;
    window.addEventListener("paste", onPaste);
    drag.forEach((type) => window.addEventListener(type, onPageDrag));
    return () => {
      window.removeEventListener("paste", onPaste);
      drag.forEach((type) => window.removeEventListener(type, onPageDrag));
    };
  }, []);

  // Слот сам останавливает всплытие drop, поэтому счётчик страницы сбрасываем здесь.
  const slotDrop = (files: File[], index: 0 | 1) => {
    dragDepth.current = 0;
    setPageDragging(false);
    addFiles(files, index);
  };

  const [a, b] = slots;
  const filled = slots.filter(Boolean).length;

  return (
    <div className="flex flex-1 flex-col">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-zinc-500">
          Два файла сразу разложатся по A и B. Можно вставить из буфера —{" "}
          <kbd className="rounded border border-zinc-300 px-1 font-mono text-xs dark:border-zinc-700">⌘V</kbd>
        </p>
        <div className="flex gap-2">
          <ToolbarButton
            onClick={() => void addSlots(EXAMPLE.map((text, i) => Promise.resolve(parseJson(text, `пример-${i + 1}.json`))))}
          >
            Пример
          </ToolbarButton>
          <ToolbarButton onClick={() => clear()} disabled={filled === 0}>
            Очистить
          </ToolbarButton>
        </div>
      </div>

      {notice && <Alert tone="warn">{notice}</Alert>}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_15rem_minmax(0,1fr)] lg:items-center">
        <DropSlot index={0} slot={a} pageDragging={pageDragging} onFiles={slotDrop} onClear={clear} />
        <div className="order-last flex justify-center py-2 lg:order-none">
          <ScoreRing run={run} filled={filled} />
        </div>
        <DropSlot index={1} slot={b} pageDragging={pageDragging} onFiles={slotDrop} onClear={clear} />
      </div>

      <div className="mt-6 space-y-6">
        {run.status === "error" && <Alert tone="error">{run.message}</Alert>}
        {run.status === "done" && <Breakdown result={run.result} />}
        {a?.type === "json" && b?.type === "json" && <FieldDiff a={a.data} b={b.data} />}
      </div>
    </div>
  );
}

function ToolbarButton(props: React.ComponentProps<"button">) {
  return (
    <button
      type="button"
      {...props}
      className="rounded-lg border border-zinc-200 bg-white px-3 py-1.5 text-sm font-medium shadow-sm transition hover:bg-zinc-50 disabled:pointer-events-none disabled:opacity-40 dark:border-zinc-800 dark:bg-zinc-900 dark:hover:bg-zinc-800"
    />
  );
}

function Alert({ tone, children }: { tone: "warn" | "error"; children: React.ReactNode }) {
  const style =
    tone === "error"
      ? "border-rose-500/30 bg-rose-500/5 text-rose-700 dark:text-rose-300"
      : "border-amber-500/30 bg-amber-500/5 text-amber-800 dark:text-amber-300";
  return (
    <div role="alert" className={`mb-4 flex gap-3 rounded-xl border px-4 py-3 text-sm ${style}`}>
      <AlertIcon className="mt-0.5 size-4 shrink-0" />
      <div className="whitespace-pre-line">{children}</div>
    </div>
  );
}
