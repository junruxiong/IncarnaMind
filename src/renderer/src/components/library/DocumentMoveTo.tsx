import { create } from "zustand";
import { displayName } from "../../../../shared/documentNames";
import { useT } from "../../i18n";
import { selectedOn } from "../../selection";
import { useAppStore } from "../../store";
import { MoveToPicker } from "../MoveToPicker";

interface MoveRequest {
  /** What the picker opens beside: the row's ⋯, or the selection bar's button. */
  anchor: HTMLElement;
  documentIds: readonly string[];
}

/** The Documents whose "Move to…" is open on the Library page, if any. */
export const useDocumentMove = create<{
  request: MoveRequest | null;
  show(request: MoveRequest): void;
  hide(): void;
}>()((set) => ({
  request: null,
  show: (request) => set({ request }),
  hide: () => set({ request: null }),
}));

/** What "Move to…" on a row moves: every Document selected, when the row is one of them. */
export function documentsToMove(documentId: string): string[] {
  const chosen = selectedOn("library").flatMap((ref) => (ref.kind === "document" ? [ref.id] : []));
  return chosen.includes(documentId) ? chosen : [documentId];
}

/** The focus follows a Document moved to its row in the section it went to, if that shows it. */
function focusRowOf(documentId: string | undefined): void {
  if (!documentId) return;
  document
    .querySelector<HTMLElement>(
      `[data-surface="library"] [data-document-id="${CSS.escape(documentId)}"] [data-row-main]`,
    )
    ?.focus();
}

/**
 * The Library page's "Move to…" for Documents: the shared Folder picker, beside
 * what asked for it. Choosing a Folder moves them all, with the undo line.
 */
export function DocumentMoveTo() {
  const t = useT();
  const request = useDocumentMove((state) => state.request);
  const ids = request?.documentIds ?? [];
  const documents = useAppStore((state) => state.documents);
  const assignments = useAppStore((state) => state.library?.assignments);
  const [only] = ids;
  const one = ids.length === 1 ? documents.find((each) => each.id === only) : undefined;
  const folders = new Set(
    ids.map((id) => assignments?.find((each) => each.documentId === id)?.groupId ?? null),
  );
  const [current] = folders;
  return (
    <MoveToPicker
      anchor={request?.anchor ?? null}
      name={one ? displayName(one) : t("library.count", { count: ids.length })}
      current={folders.size === 1 ? (current ?? null) : undefined}
      onChoose={(folderId) => {
        // From a row, the focus goes with what it moved; from the selection's bar, it stays there.
        const fromRow = request?.anchor.closest("[data-document-id]") != null;
        void useAppStore
          .getState()
          .moveToFolder({ documentIds: [...ids] }, folderId)
          .then(() => {
            if (fromRow) requestAnimationFrame(() => focusRowOf(only));
          });
      }}
      onClose={() => useDocumentMove.getState().hide()}
    />
  );
}
