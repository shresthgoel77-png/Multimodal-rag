export function scorePct(score: number) {
  return `${Math.round(Math.max(0, Math.min(1, score)) * 100)}%`;
}

export function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes)) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

export function formatLatency(value: number | undefined | null) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
}

export function formatScore(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toFixed(2);
}
