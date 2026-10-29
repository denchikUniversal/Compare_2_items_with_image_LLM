import { ApiStatus } from "@/components/ApiStatus";
import { Comparer } from "@/components/Comparer";

export default function Home() {
  return (
    <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col px-4 sm:px-6">
      <header className="flex flex-wrap items-center justify-between gap-4 py-6 sm:py-8">
        <div className="flex items-center gap-3">
          <Logo />
          <div>
            <h1 className="text-lg font-semibold tracking-tight">Item Compare</h1>
            <p className="text-sm text-zinc-500">Похожесть двух картинок или товаров в JSON</p>
          </div>
        </div>
        <ApiStatus />
      </header>

      <main className="flex flex-1 flex-col pb-10">
        <Comparer />
      </main>

      <footer className="border-t border-zinc-200 py-5 text-xs text-zinc-500 dark:border-zinc-800">
        Next.js проксирует <code className="font-mono">/api/*</code> в Python FastAPI · типы сгенерированы из его OpenAPI
        (<code className="font-mono">npm run api:types</code>)
      </footer>
    </div>
  );
}

function Logo() {
  return (
    <svg viewBox="0 0 40 40" className="size-10" aria-hidden>
      <rect width="40" height="40" rx="10" className="fill-zinc-900 dark:fill-zinc-800" />
      <circle cx="16" cy="20" r="8" className="fill-sky-400/90 mix-blend-screen" />
      <circle cx="24" cy="20" r="8" className="fill-violet-400/90 mix-blend-screen" />
    </svg>
  );
}
