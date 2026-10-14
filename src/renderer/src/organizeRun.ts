import type { DocumentGroupAssignment } from "../../core/api";
import { core } from "./core";
import { translateNow as t } from "./i18n";
import { useAppStore } from "./store";
import { useUndo } from "./undo";

const working = (item: DocumentGroupAssignment) =>
  item.status === "pending" || item.status === "classifying";

/** The Folder an assignment puts a Document in, and whether it only waits to be accepted. */
const placeOf = (item: DocumentGroupAssignment | undefined) =>
  item ? `${item.groupId ?? ""}|${item.suggested}` : "";

/**
 * Says what Organize did once it has gone through `ids` (#763): how many
 * Documents got a Suggested Folder, with Undo, which puts each back as it was
 * (Tags it suggested stay, to accept or reject). Said nothing when it
 * suggested none.
 */
export function sayWhenOrganized(
  ids: readonly string[],
  before: ReadonlyMap<string, DocumentGroupAssignment>,
): void {
  const mine = new Set(ids);
  let sawWorking = false;
  let stop = () => {};
  const finish = () => {
    stop();
    const changed = (useAppStore.getState().library?.assignments ?? []).filter(
      (item) =>
        mine.has(item.documentId) &&
        item.suggested &&
        placeOf(item) !== placeOf(before.get(item.documentId)),
    );
    if (changed.length === 0) return;
    const put = (item: DocumentGroupAssignment) => {
      const previous = before.get(item.documentId);
      if (!previous) return core.restoreAssignment(item.documentId, null);
      const { documentId: _id, ...rest } = previous;
      return core.restoreAssignment(item.documentId, rest);
    };
    useUndo.getState().push({
      message: t(changed.length === 1 ? "library.organize.done.one" : "library.organize.done", {
        count: changed.length,
      }),
      undo: async () => {
        await Promise.all(changed.map(put));
        await useAppStore.getState().refreshLibrary();
      },
    });
  };
  const unsubscribe = useAppStore.subscribe((state) => {
    const items = (state.library?.assignments ?? []).filter((item) => mine.has(item.documentId));
    if (items.some(working)) sawWorking = true;
    else if (sawWorking) finish();
  });
  // A run that never shows work (nothing to organize) is let go after a while.
  const giveUp = setTimeout(() => stop(), 30 * 60_000);
  stop = () => {
    clearTimeout(giveUp);
    unsubscribe();
  };
}
