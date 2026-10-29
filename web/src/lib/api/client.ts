import createClient from "openapi-fetch";

import openapi from "../../../openapi.json";
import type { components, paths } from "./schema";

// schema.ts и openapi.json генерируются из Pydantic-моделей Python (npm run api:types),
// поэтому формат запросов и ответов здесь всегда совпадает с item_compare.schemas.
export type Item = components["schemas"]["Item"];
export type CompareResult = components["schemas"]["CompareResult"];
export type ComponentScore = components["schemas"]["ComponentScore"];
export type Kind = components["schemas"]["Kind"];
export type ValidationError = components["schemas"]["ValidationError"];

/** Поля товара, которые знает Python-API. Остальные он игнорирует (extra="ignore"). */
export const ITEM_FIELDS: ReadonlySet<string> = new Set(Object.keys(openapi.components.schemas.Item.properties));

// Запросы идут в Next.js (/api/...), а он проксирует их в Python — см. app/api/[...path]/route.ts.
export const api = createClient<paths>({ baseUrl: "/api" });
