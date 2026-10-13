import type { Language } from "../../core/language";
import type { MessageKey, MessageParams } from "../../shared/i18n";

const DAY = 24 * 60 * 60 * 1000;

const startOfDay = (date: Date) => new Date(date.getFullYear(), date.getMonth(), date.getDate());

/** When something happened: "Today, 10:42", "Yesterday", or the date, in the interface language. */
export function dayOf(
  at: Date,
  now: Date,
  language: Language,
  say: (key: MessageKey, params?: MessageParams) => string,
): string {
  const days = Math.round((startOfDay(now).getTime() - startOfDay(at).getTime()) / DAY);
  if (days <= 0) {
    const time = new Intl.DateTimeFormat(language, { hour: "2-digit", minute: "2-digit" }).format(
      at,
    );
    return say("date.today", { time });
  }
  if (days === 1) return say("date.yesterday");
  return new Intl.DateTimeFormat(language, {
    day: "numeric",
    month: "short",
    ...(at.getFullYear() === now.getFullYear() ? {} : { year: "numeric" }),
  }).format(at);
}

/** A day and time, "2 Oct, 15:20" (10月2日 15:20), with the year when it isn't this one. */
export function dayAndTime(at: Date, now: Date, language: Language): string {
  return new Intl.DateTimeFormat(language === "zh-CN" ? "zh-CN" : "en-GB", {
    day: "numeric",
    month: "short",
    ...(at.getFullYear() === now.getFullYear() ? {} : { year: "numeric" }),
    hour: "2-digit",
    minute: "2-digit",
  }).format(at);
}
