import type { DragEvent } from "react";
import type { Folder } from "../../core/api";

/** A Folder in the tree the sidebar shows, with its sub-Folders. */
export interface FolderNode {
  folder: Folder;
  /** 0 at the top level. */
  depth: number;
  children: FolderNode[];
}

/**
 * Builds the tree from the core's flat list, keeping its order among siblings.
 * A Folder whose parent isn't in the list shows at the top level.
 */
export function buildFolderTree(folders: readonly Folder[]): FolderNode[] {
  const byId = new Map(folders.map((folder) => [folder.id, folder]));
  const children = new Map<string | null, Folder[]>();
  for (const folder of folders) {
    const parentId = folder.parentId !== null && byId.has(folder.parentId) ? folder.parentId : null;
    const siblings = children.get(parentId) ?? [];
    siblings.push(folder);
    children.set(parentId, siblings);
  }
  const build = (parentId: string | null, depth: number): FolderNode[] =>
    (children.get(parentId) ?? []).map((folder) => ({
      folder,
      depth,
      children: build(folder.id, depth + 1),
    }));
  return build(null, 0);
}

/** Every Folder in the tree, depth first: the order an indented list shows them in. */
export function flattenFolderTree(nodes: readonly FolderNode[]): FolderNode[] {
  return nodes.flatMap((node) => [node, ...flattenFolderTree(node.children)]);
}

/** Whether `folderId` is `ancestorId` itself or somewhere below it. */
export function isSameOrInside(
  folders: readonly Folder[],
  folderId: string,
  ancestorId: string,
): boolean {
  const parents = new Map(folders.map((folder) => [folder.id, folder.parentId]));
  const seen = new Set<string>();
  let current: string | null | undefined = folderId;
  while (current && !seen.has(current)) {
    if (current === ancestorId) return true;
    seen.add(current);
    current = parents.get(current);
  }
  return false;
}

/** A Document or a Folder being dragged within the sidebar, to drop on a Folder. */
export interface SidebarDrag {
  kind: "document" | "folder";
  id: string;
}

/** Not "Files", so the window's file drop ignores these drags. */
const SIDEBAR_DRAG_TYPE = "application/x-incarnamind-sidebar-item";

/** Drag events only reveal their data on drop, so the item being dragged is kept here too. */
let dragging: SidebarDrag | null = null;

export function startSidebarDrag(event: DragEvent, item: SidebarDrag): void {
  dragging = item;
  event.dataTransfer.setData(SIDEBAR_DRAG_TYPE, JSON.stringify(item));
  event.dataTransfer.effectAllowed = "move";
}

export function endSidebarDrag(): void {
  dragging = null;
}

/** The sidebar item an event is dragging, or null for anything else (e.g. files). */
export function sidebarDragOf(event: DragEvent): SidebarDrag | null {
  return event.dataTransfer.types.includes(SIDEBAR_DRAG_TYPE) ? dragging : null;
}
