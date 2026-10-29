/* eslint-disable @next/next/no-img-element -- превью из data: URI и чужих ссылок, next/image тут не нужен */

import { diffFields, type FieldRow, type FieldStatus } from "@/lib/diff";

const STATUS: Record<FieldStatus, { label: string; className: string }> = {
  same: { label: "совпадает", className: "text-emerald-600 dark:text-emerald-400" },
  diff: { label: "отличается", className: "text-amber-600 dark:text-amber-400" },
  onlyA: { label: "только в A", className: "text-sky-600 dark:text-sky-400" },
  onlyB: { label: "только в B", className: "text-violet-600 dark:text-violet-400" },
};

/** Таблица полей двух JSON: что совпало, что отличается, что Python не учитывает. */
export function FieldDiff({ a, b }: { a: Record<string, unknown>; b: Record<string, unknown> }) {
  const rows = diffFields(a, b);
  const same = rows.filter((r) => r.status === "same").length;

  return (
    <section className="rounded-2xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <div className="flex flex-wrap items-baseline justify-between gap-2 p-5 pb-3 sm:px-6">
        <h2 className="text-sm font-semibold">Поля JSON</h2>
        <span className="text-sm text-zinc-500">
          совпадает {same} из {rows.length}
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[36rem] text-sm">
          <thead>
            <tr className="border-y border-zinc-100 text-left text-xs text-zinc-500 dark:border-zinc-800">
              <th className="px-5 py-2 font-medium sm:pl-6">Поле</th>
              <th className="px-3 py-2 font-medium">
                <span className="mr-1.5 inline-block size-2 rounded-full bg-sky-500" />A
              </th>
              <th className="px-3 py-2 font-medium">
                <span className="mr-1.5 inline-block size-2 rounded-full bg-violet-500" />B
              </th>
              <th className="px-5 py-2 text-right font-medium sm:pr-6">Статус</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <Row key={row.path} row={row} />
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Row({ row }: { row: FieldRow }) {
  const status = STATUS[row.status];
  return (
    <tr
      className={`border-b border-zinc-100 align-top last:border-0 dark:border-zinc-800 ${
        row.status === "diff" ? "bg-amber-500/5" : ""
      } ${row.known ? "" : "opacity-60"}`}
    >
      <td className="px-5 py-2.5 sm:pl-6">
        <div className="font-mono text-xs">{row.path}</div>
        {!row.known && <div className="mt-0.5 text-[11px] text-zinc-500">Python не учитывает</div>}
      </td>
      <td className="max-w-64 px-3 py-2.5">
        <Value present={row.status !== "onlyB"} value={row.a} />
      </td>
      <td className="max-w-64 px-3 py-2.5">
        <Value present={row.status !== "onlyA"} value={row.b} />
      </td>
      <td className={`px-5 py-2.5 text-right text-xs whitespace-nowrap sm:pr-6 ${status.className}`}>{status.label}</td>
    </tr>
  );
}

function Value({ present, value }: { present: boolean; value: unknown }) {
  if (!present) return <span className="text-zinc-300 dark:text-zinc-700">—</span>;
  if (typeof value === "string" && /^(https?:|data:image\/)/.test(value)) {
    const isData = value.startsWith("data:");
    return (
      <div className="flex items-center gap-2">
        <span className="checker grid size-9 shrink-0 place-items-center overflow-hidden rounded-md">
          <img src={value} alt="" className="max-h-full max-w-full object-contain" />
        </span>
        <span className="truncate font-mono text-xs text-zinc-500" title={isData ? undefined : value}>
          {isData ? value.slice(5, value.indexOf(";")) : value}
        </span>
      </div>
    );
  }
  const text = typeof value === "string" ? value : (JSON.stringify(value) ?? "");
  return (
    <span className="line-clamp-3 break-words" title={text.length > 120 ? text : undefined}>
      {text}
    </span>
  );
}
