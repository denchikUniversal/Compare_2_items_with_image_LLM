"use client";

import { useRef, useState, type DragEvent } from "react";

import { ACCEPT, type Slot } from "@/lib/files";
import { BracesIcon, CloseIcon, ImageIcon, ReplaceIcon, UploadIcon } from "./icons";
import { SlotPreview } from "./SlotPreview";

export const SIDES = [
  { label: "A", badge: "bg-sky-500", ring: "ring-sky-500/60", tint: "bg-sky-500/5" },
  { label: "B", badge: "bg-violet-500", ring: "ring-violet-500/60", tint: "bg-violet-500/5" },
] as const;

type Props = {
  index: 0 | 1;
  slot: Slot | null;
  /** Файл тащат где-то над страницей — подсвечиваем пустые слоты. */
  pageDragging: boolean;
  onFiles: (files: File[], index: 0 | 1) => void;
  onClear: (index: 0 | 1) => void;
};

/** Зона для одного файла: картинки или JSON. Принимает перетаскивание и клик. */
export function DropSlot({ index, slot, pageDragging, onFiles, onClear }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const side = SIDES[index];

  function drop(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    setOver(false);
    onFiles([...event.dataTransfer.files], index);
  }

  const highlight = over ? `ring-2 ${side.ring} ${side.tint}` : pageDragging && !slot ? `ring-2 ${side.ring}` : "";

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setOver(false);
      }}
      onDrop={drop}
      className={`relative flex h-[23rem] flex-col rounded-2xl border bg-white p-4 shadow-sm transition dark:bg-zinc-900 ${
        slot ? "border-zinc-200 dark:border-zinc-800" : "border-dashed border-zinc-300 dark:border-zinc-700"
      } ${highlight}`}
    >
      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        multiple
        hidden
        onChange={(e) => {
          onFiles([...(e.target.files ?? [])], index);
          e.target.value = "";
        }}
      />

      <div className="mb-3 flex items-center gap-2.5">
        <span className={`grid size-6 shrink-0 place-items-center rounded-md text-xs font-bold text-white ${side.badge}`}>
          {side.label}
        </span>
        {slot ? (
          <>
            {slot.type === "image" ? (
              <ImageIcon className="size-4 shrink-0 text-zinc-400" />
            ) : (
              <BracesIcon className="size-4 shrink-0 text-zinc-400" />
            )}
            <span className="min-w-0 flex-1 truncate text-sm font-medium" title={slot.name}>
              {slot.name}
            </span>
            <IconButton label="Заменить" onClick={() => input.current?.click()}>
              <ReplaceIcon className="size-4" />
            </IconButton>
            <IconButton label="Убрать" onClick={() => onClear(index)}>
              <CloseIcon className="size-4" />
            </IconButton>
          </>
        ) : (
          <span className="text-sm text-zinc-500">Пусто</span>
        )}
      </div>

      {slot ? (
        <div className="min-h-0 flex-1">
          <SlotPreview slot={slot} />
        </div>
      ) : (
        <button
          type="button"
          onClick={() => input.current?.click()}
          className="group flex flex-1 flex-col items-center justify-center gap-3 rounded-xl text-center outline-none focus-visible:ring-2 focus-visible:ring-zinc-400"
        >
          <span className="grid size-14 place-items-center rounded-2xl bg-zinc-100 text-zinc-500 transition group-hover:scale-105 group-hover:text-zinc-900 dark:bg-zinc-800 dark:group-hover:text-zinc-100">
            <UploadIcon className="size-6" />
          </span>
          <span className="font-medium">{over ? "Отпустите файл" : "Перетащите картинку или JSON"}</span>
          <span className="text-sm text-zinc-500">или нажмите, чтобы выбрать</span>
          <span className="mt-1 font-mono text-xs text-zinc-400">png · jpg · webp · gif · .json</span>
        </button>
      )}
    </div>
  );
}

function IconButton({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      className="grid size-7 shrink-0 place-items-center rounded-md text-zinc-400 transition hover:bg-zinc-100 hover:text-zinc-900 dark:hover:bg-zinc-800 dark:hover:text-zinc-100"
    >
      {children}
    </button>
  );
}
