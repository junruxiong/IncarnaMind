import type { MessageKey, MessageParams } from "../../shared/i18n";

export { clockTime } from "../../shared/clock";

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
