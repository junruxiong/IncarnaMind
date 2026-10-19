import { type Dropped, type DropTarget, runDrop, useDragState } from "./dragDrop";
import { type MoveItems, moveLabel } from "./moves";
import type { ObjectRef } from "./objects";
import { useSelection } from "./selection";
import { useAppStore } from "./store";
import { linksIn, useWebPages } from "./webPages";

/**
 * The drop targets that file Minds and Documents: a Folder, Not in a Folder
 * (null) and a Tag, in the sidebar and on the Library page. A Folder lights
 * only while the drop would move something there, as Move to… does; a Tag
 * only while every Document dragged is one and one of them would get it. A
 * link dragged from a browser onto a Folder is saved there as a web page
 * (#261).
 */

export const itemsOfRefs = (refs: readonly ObjectRef[]): MoveItems => ({
  mindIds: refs.flatMap((ref) => (ref.kind === "mind" ? [ref.id] : [])),
  documentIds: refs.flatMap((ref) => (ref.kind === "document" ? [ref.id] : [])),
});

const folderNow = (folderId: string | null) =>
  useAppStore.getState().library?.groups.find((group) => group.id === folderId)?.name ?? "";

function wouldMove(refs: readonly ObjectRef[], folderId: string | null): boolean {
  const { minds, library } = useAppStore.getState();
  const folderOf = new Map(library?.assignments.map((each) => [each.documentId, each.groupId]));
  return refs.some((ref) =>
    ref.kind === "mind"
      ? (minds.find((mind) => mind.id === ref.id)?.folderId ?? null) !== folderId
      : ref.kind === "document" && (folderOf.get(ref.id) ?? null) !== folderId,
  );
}

/** A selection of one is the row shown, and stays; several chosen to drag together are done with. */
function doneWithSelection(dropped: Dropped): void {
  if (dropped.kind !== "objects" || !dropped.from) return;
  if (useSelection.getState().items.length > 1) useSelection.getState().clear(dropped.from);
}

export function folderTarget(folderId: string | null): DropTarget {
  return {
    accepts(dragged) {
      if (dragged.kind === "links") return { kind: "add" };
      if (dragged.kind !== "objects" || !wouldMove(dragged.refs, folderId)) return null;
      const items = itemsOfRefs(dragged.refs);
      return {
        kind: "move",
        said: (t) => moveLabel(t, items, folderId ? folderNow(folderId) : t("library.unsorted")),
      };
    },
    drop(dropped) {
      if (dropped.kind === "links") {
        void useWebPages.getState().add(linksIn(dropped.data), folderId);
        return null;
      }
      if (dropped.kind !== "objects") return null;
      const moved = useAppStore.getState().moveToFolder(itemsOfRefs(dropped.refs), folderId);
      doneWithSelection(dropped);
      return moved;
    },
  };
}

/** Move to…: the same as a drop on the Folder (null: out of every Folder). */
export const moveIntoFolder = (items: MoveItems, folderId: string | null): Promise<void> =>
  runDrop(folderTarget(folderId), [
    ...(items.mindIds ?? []).map((id) => ({ kind: "mind" as const, id })),
    ...(items.documentIds ?? []).map((id) => ({ kind: "document" as const, id })),
  ]);

/** Whether something in a Folder is being dragged: Not in a Folder then shows, empty or not, to take it. */
export const useDraggedOutOfFolder = (): boolean =>
  useDragState((state) => state.source !== null && wouldMove(state.source.refs, null));

/** The Documents dragged that a Tag would change: none unless everything dragged is a Document. */
function taggable(refs: readonly ObjectRef[], tagId: string): string[] {
  if (refs.some((ref) => ref.kind !== "document")) return [];
  const { documents } = useAppStore.getState();
  return refs.flatMap((ref) => {
    const link = documents
      .find((document) => document.id === ref.id)
      ?.tags.find((each) => each.tagId === tagId);
    // An automatic Tag, or one awaiting review, becomes the User's own.
    return link?.source !== "user" || link.needsReview ? [ref.id] : [];
  });
}

export function tagTarget(tagId: string): DropTarget {
  const tagName = () => useAppStore.getState().tags.find((tag) => tag.id === tagId)?.name ?? "";
  return {
    accepts(dragged) {
      if (dragged.kind !== "objects") return null;
      const count = taggable(dragged.refs, tagId).length;
      if (count === 0) return null;
      return {
        kind: "add",
        said: (t) =>
          count === 1
            ? t("drop.addTag", { tag: tagName() })
            : t("drop.addTag.documents", { tag: tagName(), count }),
      };
    },
    drop(dropped) {
      if (dropped.kind !== "objects") return null;
      const ids = taggable(dropped.refs, tagId);
      if (ids.length === 0) return null;
      const tagged = useAppStore.getState().addTagToDocuments(ids, tagId);
      doneWithSelection(dropped);
      return tagged;
    },
  };
}
