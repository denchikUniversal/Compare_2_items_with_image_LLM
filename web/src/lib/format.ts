export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} Б`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} КБ`;
  return `${(bytes / 1024 / 1024).toFixed(1)} МБ`;
}

export function formatPercent(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
}

/** Цвет оценки: от красного (0%) через жёлтый к зелёному (100%). */
export function scoreColor(similarity: number): string {
  return `oklch(0.7 0.16 ${20 + Math.min(100, Math.max(0, similarity)) * 1.25})`;
}
