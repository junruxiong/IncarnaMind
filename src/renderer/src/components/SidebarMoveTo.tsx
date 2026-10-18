import { itemsOf } from "../moves";
import { nameOfMoved, useSidebarMove } from "../sidebarMoves";
import { useAppStore } from "../store";
import { MoveToPicker } from "./MoveToPicker";

/**
 * The sidebar's "Move to…" (⇧⌘M on a row, or its menu): the shared Folder
 * picker beside the row, for the row or the Minds and Documents selected with
 * it. Choosing a Folder moves them all, with the undo line.
 */
export function SidebarMoveTo() {
  const request = useSidebarMove((state) => state.request);
  const minds = useAppStore((state) => state.minds);
  const assignments = useAppStore((state) => state.library?.assignments);
  const { mindIds, documentIds } = itemsOf(request?.items ?? {});
  const folders = new Set([
    ...mindIds.map((id) => minds.find((mind) => mind.id === id)?.folderId ?? null),
    ...documentIds.map(
      (id) => assignments?.find((each) => each.documentId === id)?.groupId ?? null,
    ),
  ]);
  const [current] = folders;
  return (
    <MoveToPicker
      anchor={request?.anchor ?? null}
      name={request ? nameOfMoved(request.items) : ""}
      current={folders.size === 1 ? (current ?? null) : undefined}
      onChoose={(folderId) => {
        if (request) void useAppStore.getState().moveToFolder(request.items, folderId);
      }}
      onClose={() => useSidebarMove.getState().hide()}
    />
  );
}
