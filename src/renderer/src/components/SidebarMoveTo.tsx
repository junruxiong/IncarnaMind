import { itemsOf } from "../moves";
import { nameOfMoved, useSidebarMove } from "../sidebarMoves";
import { useAppStore } from "../store";
import { MoveToPicker } from "./MoveToPicker";

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
  // Only while asked for: the Library has a picker of its own, and a page has one at a time.
  if (!request) return null;
  return (
    <MoveToPicker
      anchor={request.anchor}
      name={nameOfMoved(request.items)}
      current={folders.size === 1 ? (current ?? null) : undefined}
      onChoose={(folderId) => void useAppStore.getState().moveToFolder(request.items, folderId)}
      onClose={() => useSidebarMove.getState().hide()}
    />
  );
}
