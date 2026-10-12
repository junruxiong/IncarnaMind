/** How long a server asks a client to wait, from `retry-after-ms` or `retry-after` (seconds or a date); null when it doesn't say. */
export function retryAfterMs(headers: Headers, now = Date.now()): number | null {
  const ms = Number(headers.get("retry-after-ms") ?? Number.NaN);
  if (Number.isFinite(ms) && ms >= 0) return ms;
  const value = headers.get("retry-after");
  if (value === null) return null;
  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) return seconds * 1000;
  const date = Date.parse(value);
  return Number.isNaN(date) ? null : Math.max(0, date - now);
}
