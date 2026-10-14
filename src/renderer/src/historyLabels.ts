import type { Checkpoint, CheckpointReason } from "../../core/api";
import type { Language } from "../../core/language";
import type { MessageKey, MessageParams } from "../../shared/i18n";

/** What History says of a Checkpoint: pure, so a test can pin the clock and the language. */

type Say = (key: MessageKey, params?: MessageParams) => string;

/** A Checkpoint saved just before the AI changed the Mind, marked with the spark: not a person's changes. */
export const beforeAiChange = (checkpoint: Pick<Checkpoint, "reason">): boolean =>
  checkpoint.reason !== "idle" && checkpoint.reason !== "restore" && checkpoint.reason !== "others";

const REASONS: Readonly<Record<Exclude<CheckpointReason, "idle">, MessageKey>> = {
  accept: "history.reason.accept",
  regenerate: "history.reason.regenerate",
  enhance: "history.reason.enhance",
  compare: "history.reason.compare",
  others: "history.reason.others",
  restore: "history.reason.restore",
};

/** Why it was saved, "before AI edits"; null for an idle one, which shows only its time. */
export const reasonLabel = (checkpoint: Pick<Checkpoint, "reason">, say: Say): string | null =>
  checkpoint.reason === "idle" ? null : say(REASONS[checkpoint.reason]);

/** The day it was saved, as its group in the picker is headed: "Friday 9 Oct", with the year when it isn't this one. */
export function dayLabel(iso: string, now: Date, language: Language): string {
  const at = new Date(iso);
  return new Intl.DateTimeFormat(language === "zh-CN" ? "zh-CN" : "en-GB", {
    weekday: "long",
    day: "numeric",
    month: "short",
    ...(at.getFullYear() === now.getFullYear() ? {} : { year: "numeric" }),
  }).format(at);
}

/** The time it was saved, "10:42". */
export const timeLabel = (iso: string, language: Language): string =>
  new Intl.DateTimeFormat(language === "zh-CN" ? "zh-CN" : "en-GB", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(iso));

/** The version it is, "Friday 9 Oct, 10:42". */
export const versionLabel = (
  checkpoint: Pick<Checkpoint, "createdAt">,
  now: Date,
  language: Language,
  say: Say,
): string =>
  say("history.when", {
    day: dayLabel(checkpoint.createdAt, now, language),
    time: timeLabel(checkpoint.createdAt, language),
  });

/** Checkpoints, newest first, in groups by the day they were saved. */
export function byDay(
  checkpoints: readonly Checkpoint[],
  now: Date,
  language: Language,
): { day: string; checkpoints: Checkpoint[] }[] {
  const groups: { day: string; checkpoints: Checkpoint[] }[] = [];
  for (const checkpoint of checkpoints) {
    const day = dayLabel(checkpoint.createdAt, now, language);
    const last = groups.at(-1);
    if (last?.day === day) last.checkpoints.push(checkpoint);
    else groups.push({ day, checkpoints: [checkpoint] });
  }
  return groups;
}
