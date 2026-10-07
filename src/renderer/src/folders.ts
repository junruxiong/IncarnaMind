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
