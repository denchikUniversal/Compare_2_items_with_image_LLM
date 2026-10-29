"use client";

import { useEffect, useRef, useState } from "react";

import type { CompareResult } from "@/lib/api/client";
import { scoreColor } from "@/lib/format";

export type Run =
  | { status: "idle" }
  | { status: "loading"; prev?: CompareResult }
  | { status: "done"; result: CompareResult }
  | { status: "error"; message: string };

const SIZE = 208;
const STROKE = 14;
const R = (SIZE - STROKE) / 2;
const CIRCUMFERENCE = 2 * Math.PI * R;

/** Кольцо с итоговым процентом и вердиктом «один товар / разные». */
export function ScoreRing({ run, filled }: { run: Run; filled: number }) {
  const result = run.status === "done" ? run.result : run.status === "loading" ? run.prev : undefined;
  const target = result?.similarity ?? 0;
  const value = useAnimatedNumber(target);
  const color = result ? scoreColor(target) : undefined;

  return (
    <div className="flex flex-col items-center gap-4" aria-live="polite">
      <div className="relative" style={{ width: SIZE, height: SIZE }}>
        <svg width={SIZE} height={SIZE} className="-rotate-90">
          <circle
            cx={SIZE / 2}
            cy={SIZE / 2}
            r={R}
            fill="none"
            strokeWidth={STROKE}
            className={run.status === "error" ? "stroke-rose-500/20" : "stroke-zinc-200 dark:stroke-zinc-800"}
          />
          {result && (
            <circle
              cx={SIZE / 2}
              cy={SIZE / 2}
              r={R}
              fill="none"
              stroke={color}
              strokeWidth={STROKE}
              strokeLinecap="round"
              strokeDasharray={CIRCUMFERENCE}
              strokeDashoffset={CIRCUMFERENCE * (1 - target / 100)}
              className={`transition-[stroke-dashoffset,stroke,opacity] duration-700 ease-out motion-reduce:transition-none ${run.status === "loading" ? "opacity-30" : ""}`}
            />
          )}
          {run.status === "loading" && (
            <circle
              cx={SIZE / 2}
              cy={SIZE / 2}
              r={R}
              fill="none"
              strokeWidth={STROKE}
              strokeLinecap="round"
              strokeDasharray={`${CIRCUMFERENCE * 0.22} ${CIRCUMFERENCE}`}
              className="origin-center animate-spin stroke-zinc-900 [animation-duration:1.1s] [transform-box:fill-box] dark:stroke-zinc-100"
            />
          )}
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
          {result ? (
            <>
              <div className="flex items-baseline font-semibold tracking-tight tabular-nums">
                <span className="text-5xl">{value.toFixed(1)}</span>
                <span className="ml-0.5 text-2xl text-zinc-400">%</span>
              </div>
              <div className="mt-1 text-xs tracking-wide text-zinc-500 uppercase">похожесть</div>
            </>
          ) : (
            <div className="px-8 text-sm text-zinc-500">
              {run.status === "loading" && "Сравниваю…"}
              {run.status === "error" && <span className="font-medium text-rose-600 dark:text-rose-400">Ошибка</span>}
              {run.status === "idle" && (filled === 0 ? "Добавьте два файла" : "Добавьте второй файл")}
            </div>
          )}
        </div>
      </div>

      {run.status === "done" && (
        <div
          className={`rounded-full px-4 py-1.5 text-sm font-medium ${
            run.result.is_same
              ? "bg-emerald-500/10 text-emerald-700 ring-1 ring-emerald-500/30 dark:text-emerald-300"
              : "bg-zinc-500/10 text-zinc-700 ring-1 ring-zinc-500/25 dark:text-zinc-300"
          }`}
        >
          {run.result.is_same ? "Один и тот же товар" : "Разные товары"}
        </div>
      )}
    </div>
  );
}

/** Плавно докручивает число до нового значения. */
function useAnimatedNumber(target: number, duration = 700): number {
  const [value, setValue] = useState(0);
  const current = useRef(0);

  useEffect(() => {
    const from = current.current;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const start = performance.now();
    let frame = 0;

    const tick = (now: number) => {
      const t = reduceMotion ? 1 : Math.min(1, (now - start) / duration);
      current.current = from + (target - from) * (1 - (1 - t) ** 3);
      setValue(current.current);
      if (t < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [target, duration]);

  return value;
}
