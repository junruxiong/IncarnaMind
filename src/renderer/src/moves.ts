/**
 * Moving Minds and Documents between Folders, as the sidebar's drag and drop,
 * a Mind's "Move to…" and (#212) the Library's do it: one shape for what is
 * moved, the store as it looks at once (before the core confirms), the line
 * that says what happened, and the moves that take it back. Pure: no store,
 * no bridge, so the tests import it.
 */
import type {
  Document,
  DocumentGroupAssignment,
  FolderMove,
  LibrarySnapshot,
  Mind,
} from "../../core/api";
import type { MessageKey, MessageParams } from "../../shared/i18n";
import { applyAssignments } from "./documentUpdates";

/** What a move takes: Minds and Documents, by id, any number of each. */
export interface MoveItems {
  mindIds?: readonly string[];
  documentIds?: readonly string[];
}

/** The ids of each kind, each once. */
export function itemsOf(items: MoveItems): { mindIds: string[]; documentIds: string[] } {
  return {
    mindIds: [...new Set(items.mindIds ?? [])],
    documentIds: [...new Set(items.documentIds ?? [])],
  };
}

/** The Minds and the Library as they are once these have moved, shown before the core says so. */
export function movedLocally(
  state: { minds: readonly Mind[]; library: LibrarySnapshot | null },
  items: MoveItems,
  folderId: string | null,
): { minds: Mind[]; library: LibrarySnapshot | null } {
  const { mindIds, documentIds } = itemsOf(items);
  const moving = new Set(mindIds);
  const minds = state.minds.map((mind) => (moving.has(mind.id) ? { ...mind, folderId } : mind));
  if (!state.library || documentIds.length === 0) return { minds, library: state.library };
  const current = new Map(state.library.assignments.map((each) => [each.documentId, each]));
  const changed = new Map<string, DocumentGroupAssignment>(
    documentIds.map((documentId) => [
      documentId,
      {
        documentId,
        model: null,
        ...current.get(documentId),
        groupId: folderId,
        source: "user",
        status: "classified",
        error: null,
      },
    ]),
  );
  return { minds, library: applyAssignments(state.library, changed) };
}

/**
 * The moves that take a move back: each thing to the Folder it was in, in as
 * few moves as there were Folders it came from. Those that didn't move are left.
 */
export function undoMoves(move: FolderMove): { items: MoveItems; folderId: string | null }[] {
  const back = new Map<string | null, { mindIds: string[]; documentIds: string[] }>();
  const add = (from: string | null, kind: "mindIds" | "documentIds", id: string) => {
    if (from === move.folderId) return;
    const entry = back.get(from) ?? { mindIds: [], documentIds: [] };
    entry[kind].push(id);
    back.set(from, entry);
  };
  for (const item of move.minds) add(item.from, "mindIds", item.id);
  for (const item of move.documents) add(item.from, "documentIds", item.id);
  return [...back].map(([folderId, items]) => ({ folderId, items }));
}

/**
 * What the line says once something moved: "Moved “Shortlist” to Finance",
 * or for several, how many of what ("Moved 3 Documents to Finance").
 */
export function moveMessage(
  t: (key: MessageKey, params?: MessageParams) => string,
  items: MoveItems,
  folderName: string,
  names: { minds: readonly Mind[]; documents: readonly Document[] },
): string {
  const { mindIds, documentIds } = itemsOf(items);
  const count = mindIds.length + documentIds.length;
  if (count === 1) {
    const mind = names.minds.find((each) => each.id === mindIds[0]);
    const document = names.documents.find((each) => each.id === documentIds[0]);
    const name = mind ? mind.title || t("mind.untitled") : (document?.name ?? "");
    return t("move.done", { name, folder: folderName });
  }
  const key: MessageKey =
    documentIds.length === 0
      ? "move.done.minds"
      : mindIds.length === 0
        ? "move.done.documents"
        : "move.done.items";
  return t(key, { count, folder: folderName });
}
