import type { MessageKey, MessageParams } from "../../shared/i18n";

/** A Meeting's time as a clock reads it: "4:07", "24:16" or "1:02:05". */
export function clockTime(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = String(seconds % 60).padStart(2, "0");
  return hours > 0 ? `${hours}:${String(minutes).padStart(2, "0")}:${rest}` : `${minutes}:${rest}`;
}

/** How long, roughly, in words: "45 s", "3 min", "2 h 5 min". */
export function durationLabel(
  ms: number,
  t: (key: MessageKey, params?: MessageParams) => string,
): string {
  const seconds = Math.round(ms / 1000);
  if (seconds < 60) return t("meetings.duration.seconds", { count: seconds });
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return t("meetings.duration.minutes", { count: minutes });
  return t("meetings.duration.hours", {
    hours: Math.floor(minutes / 60),
    minutes: minutes % 60,
  });
}
