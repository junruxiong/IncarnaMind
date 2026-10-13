import { type DeletedItem, type DeletedKind, RECENTLY_DELETED_DAYS } from "../../core/api";
import type { Language } from "../../core/language";
import type { MessageKey, MessageParams } from "../../shared/i18n";
import { dayOf } from "./dates";

/**
 * What Recently deleted shows of an item and says before it is lost. Pure, so a
 * test can pin the clock.
 */

/** The Library's view of Recently deleted, where a Folder's id or "all" would go. */
export const LIBRARY_DELETED = "deleted";

const DAY = 24 * 60 * 60 * 1000;

/** Whole days left before it is deleted permanently, at least 1 while it is still listed. */
export function daysLeft(item: Pick<DeletedItem, "deletedAt">, now: Date): number {
  const ends = Date.parse(item.deletedAt) + RECENTLY_DELETED_DAYS * DAY;
  return Math.max(1, Math.ceil((ends - now.getTime()) / DAY));
}

/** When it was deleted: "Today, 10:42", "Yesterday", or the date, in the interface language. */
export function whenDeleted(
  item: Pick<DeletedItem, "deletedAt">,
  now: Date,
  language: Language,
  say: (key: MessageKey, params?: MessageParams) => string,
): string {
  return dayOf(new Date(item.deletedAt), now, language, say);
}

/** The kinds in the order a confirmation lists what is lost. */
const KIND_ORDER: readonly DeletedKind[] = ["document", "mind", "conversation", "folder", "tag"];

/** A confirmation's sentences: each kind present says what is lost, the item named when it is the only one of its kind. */
export function lostSentences(
  items: readonly Pick<DeletedItem, "kind" | "name">[],
  nameOf: (item: Pick<DeletedItem, "kind" | "name">) => string,
): { key: MessageKey; params: MessageParams }[] {
  return KIND_ORDER.flatMap((kind) => {
    const of = items.filter((item) => item.kind === kind);
    const [first] = of;
    if (!first) return [];
    const sentence: { key: MessageKey; params: MessageParams } =
      of.length === 1
        ? { key: `deleted.lost.${kind}.one`, params: { name: nameOf(first) } }
        : { key: `deleted.lost.${kind}`, params: { count: of.length } };
    return [sentence];
  });
}
