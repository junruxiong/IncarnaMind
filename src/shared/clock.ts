/** A Meeting's or a Transcript's time as a clock reads it: "4:07", "24:16" or "1:02:05". */
export function clockTime(ms: number): string {
  const seconds = Math.floor(Math.max(0, ms) / 1000);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = String(seconds % 60).padStart(2, "0");
  return hours > 0 ? `${hours}:${String(minutes).padStart(2, "0")}:${rest}` : `${minutes}:${rest}`;
}

/** A clock time read back ("14:26", "1:02:05"), in ms; null if it isn't one. */
export function parseClockTime(text: string): number | null {
  const match = /^(?:(\d{1,2}):)?(\d{1,3}):([0-5]\d)$/.exec(text.trim());
  if (!match) return null;
  const minutes = Number(match[2]);
  if (match[1] !== undefined && minutes > 59) return null;
  return ((Number(match[1] ?? 0) * 60 + minutes) * 60 + Number(match[3])) * 1000;
}
