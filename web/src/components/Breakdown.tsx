import type { CompareResult, Kind } from "@/lib/api/client";
import { formatPercent, scoreColor } from "@/lib/format";

// Record<Kind, …>: если в Python появится новый Kind, после `npm run api:types` здесь упадёт проверка типов.
const KIND_LABEL: Record<Kind, { title: string; hint: string }> = {
  image_image: { title: "Картинка ↔ картинка", hint: "насколько похожи изображения" },
  text_text: { title: "Текст ↔ текст", hint: "название, атрибуты и описание" },
  image_text: {
    title: "Картинка ↔ текст",
    hint: "общих частей нет — картинка одного сравнивается с текстом другого",
  },
};

/** Из чего сложился итоговый процент: оценка, косинус и вес каждой части. */
export function Breakdown({ result }: { result: CompareResult }) {
  const formula = result.components
    .map((c) => `${formatPercent(c.weight)} × ${c.similarity.toFixed(1)}`)
    .join(" + ");

  return (
    <section className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm sm:p-6 dark:border-zinc-800 dark:bg-zinc-900">
      <h2 className="text-sm font-semibold">Из чего сложился результат</h2>

      <ul className="mt-5 space-y-5">
        {result.components.map((c) => (
          <li key={c.kind}>
            <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
              <div>
                <span className="font-medium">{KIND_LABEL[c.kind].title}</span>
                <span className="ml-2 text-sm text-zinc-500">{KIND_LABEL[c.kind].hint}</span>
              </div>
              <span className="font-semibold tabular-nums">{c.similarity.toFixed(1)}%</span>
            </div>
            <div className="mt-2 h-2 overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
              <div
                className="h-full rounded-full transition-[width] duration-700 ease-out motion-reduce:transition-none"
                style={{ width: `${c.similarity}%`, background: scoreColor(c.similarity) }}
              />
            </div>
            <div className="mt-1.5 font-mono text-xs text-zinc-500">
              косинус {c.cosine.toFixed(4)} · вес {formatPercent(c.weight)}
            </div>
          </li>
        ))}
      </ul>

      <p className="mt-6 border-t border-zinc-100 pt-4 font-mono text-xs text-zinc-500 dark:border-zinc-800">
        итог = {formula} = <span className="font-semibold text-zinc-900 dark:text-zinc-100">{result.similarity.toFixed(1)}%</span>
      </p>
    </section>
  );
}
