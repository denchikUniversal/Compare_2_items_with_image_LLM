import type { NextRequest } from "next/server";

// Адрес Python-API. По умолчанию собирается из тех же переменных, что читает
// item_compare.config.Settings, поэтому ITEM_COMPARE_PORT=9000 меняет порт сразу для обоих.
const API_URL =
  process.env.ITEM_COMPARE_API_URL ??
  `http://${process.env.ITEM_COMPARE_HOST ?? "127.0.0.1"}:${process.env.ITEM_COMPARE_PORT ?? "8000"}`;

/** Прокидывает /api/* в Python-API как есть, а если он не запущен — отвечает понятной ошибкой. */
async function forward(request: NextRequest, ctx: RouteContext<"/api/[...path]">) {
  const { path } = await ctx.params;
  const url = `${API_URL}/${path.map(encodeURIComponent).join("/")}${request.nextUrl.search}`;

  let upstream: Response;
  try {
    upstream = await fetch(url, {
      method: request.method,
      headers: { "content-type": request.headers.get("content-type") ?? "application/json" },
      body: request.method === "GET" ? undefined : await request.arrayBuffer(),
      cache: "no-store",
    });
  } catch {
    return Response.json(
      { detail: `Python API не отвечает на ${API_URL}. Запустите его: uv run item-compare-api` },
      { status: 502 },
    );
  }

  return new Response(upstream.body, {
    status: upstream.status,
    headers: { "content-type": upstream.headers.get("content-type") ?? "application/json" },
  });
}

export { forward as GET, forward as POST };
