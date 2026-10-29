"use client";

import { useEffect, useState } from "react";

import { api } from "@/lib/api/client";

type Health = { state: "checking" } | { state: "up"; model: string; device: string } | { state: "down" };

/** Индикатор Python-API: какая модель загружена и на каком устройстве. */
export function ApiStatus() {
  const [health, setHealth] = useState<Health>({ state: "checking" });

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    let cancelled = false;

    async function check() {
      let next: Health = { state: "down" };
      try {
        const { data } = await api.GET("/health");
        if (data) next = { state: "up", model: data.model.split("/").pop() ?? data.model, device: data.device };
      } catch {}
      if (cancelled) return;
      setHealth(next);
      // Пока API не поднялся (модель грузится десяток секунд), проверяем чаще.
      timer = setTimeout(check, next.state === "up" ? 15_000 : 3_000);
    }

    check();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  const dot = {
    checking: "bg-zinc-400 animate-pulse",
    up: "bg-emerald-500",
    down: "bg-rose-500 animate-pulse",
  }[health.state];

  return (
    <div
      className="flex items-center gap-2 rounded-full border border-zinc-200 bg-white/70 px-3 py-1.5 text-xs text-zinc-600 backdrop-blur dark:border-zinc-800 dark:bg-zinc-900/70 dark:text-zinc-400"
      title={health.state === "down" ? "Запустите Python API: uv run item-compare-api" : undefined}
    >
      <span className={`size-2 rounded-full ${dot}`} />
      {health.state === "checking" && "Проверяю API…"}
      {health.state === "down" && "Python API не запущен"}
      {health.state === "up" && (
        <span>
          <span className="font-mono text-zinc-900 dark:text-zinc-100">{health.model}</span>
          <span className="mx-1.5 text-zinc-300 dark:text-zinc-700">·</span>
          {health.device}
        </span>
      )}
    </div>
  );
}
