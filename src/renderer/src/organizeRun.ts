import { create } from "zustand";
import type { DocumentGroupAssignment } from "../../core/api";
import { core } from "./core";
import { useAppStore } from "./store";
import { useUndo } from "./undo";

const working = (item: DocumentGroupAssignment) =>
  item.status === "pending" || item.status === "classifying";

/** The Folder an assignment puts a Document in, and whether it only waits to be accepted. */
const placeOf = (item: DocumentGroupAssignment | undefined) =>
  item ? `${item.groupId ?? ""}|${item.suggested}` : "";

/**
 * What Undo puts a Document back to: as it was before Organize, except that one Organize hadn't
 * finished with (waiting for a model, or queued) goes back with no assignment at all. Put back as
 * waiting it would show "Needs attention" with a model chosen, and be organized again at the next
 * start (#1391).
 */
const wasBefore = (
  before: ReadonlyMap<string, DocumentGroupAssignment>,
  id: string,
): DocumentGroupAssignment | undefined => {
  const item = before.get(id);
  return item && (working(item) || item.status === "waiting") ? undefined : item;
};

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
  const itemsNow = () =>
    (useAppStore.getState().library?.assignments ?? []).filter((item) => mine.has(item.documentId));
  let unsubscribe = () => {};
  const finish = () => {
    unsubscribe();
    const changed = itemsNow().filter(
      (item) => item.suggested && placeOf(item) !== placeOf(wasBefore(before, item.documentId)),
    );
    if (changed.length === 0) return;
    const stays = (now: ReadonlyMap<string, DocumentGroupAssignment>, wanted: "undone" | "done") =>
      changed.filter((item) => {
        const then = wanted === "undone" ? wasBefore(before, item.documentId) : item;
        return placeOf(now.get(item.documentId)) === placeOf(then);
      });
    const assignmentsNow = () =>
      new Map(
        (useAppStore.getState().library?.assignments ?? []).map((item) => [item.documentId, item]),
      );
    const put = (item: DocumentGroupAssignment) => {
      const previous = wasBefore(before, item.documentId);
      if (!previous) return core.restoreAssignment(item.documentId, null);
      const { documentId: _id, ...rest } = previous;
      return core.restoreAssignment(item.documentId, rest);
    };
    useUndo.getState().push({
      said: (t) =>
        t(changed.length === 1 ? "library.organize.done.one" : "library.organize.done", {
          count: changed.length,
        }),
      name: (t) => t("undo.name.organize"),
      undo: async () => {
        // A Document the User has moved since is theirs now: it stays where they put it.
        await Promise.all(stays(assignmentsNow(), "done").map(put));
        await useAppStore.getState().refreshLibrary();
      },
      redo: async () => {
        // The same again, in one step: the Documents still as Undo left them get their Suggested Folder back.
        await Promise.all(
          stays(assignmentsNow(), "undone").map(({ documentId, ...suggestion }) =>
            core.restoreAssignment(documentId, suggestion),
          ),
        );
        await useAppStore.getState().refreshLibrary();
      },
    });
  };
  // Organize has queued its Documents by now: with nothing working, there is nothing to wait for.
  let sawWorking = itemsNow().some(working);
  if (!sawWorking) return;
  unsubscribe = useAppStore.subscribe(() => {
    if (itemsNow().some(working)) sawWorking = true;
    else if (sawWorking) finish();
  });
}

/** The Documents Organize was asked for with no model chosen: what the row under the head says it needs. */
export const useOrganizeAsk = create<{
  ids: readonly string[] | null;
  ask(ids: readonly string[]): void;
  close(): void;
}>()((set) => ({
  ids: null,
  ask: (ids) => set({ ids }),
  close: () => set({ ids: null }),
}));
