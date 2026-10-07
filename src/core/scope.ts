/**
 * Resolving a Question's Search scope (CONTEXT.md) to the Documents its
 * search covers, when the Question is asked: the union of the Documents in
 * its Folders (each with its sub-Folders, at any depth), those with its
 * Tags, and its individual Documents. Folders, Tags and Documents deleted
 * since are ignored: they add nothing.
 */
import { hasSearchScope } from "../shared/searchScope";
import type { SearchScope } from "./api";
import type { DocumentFilter } from "./documents";

export interface ScopeSources {
  /** A live Folder's id and those of its live sub-Folders, at any depth; none for a deleted Folder. */
  folderTree(folderId: string): string[];
  /** The ids of the live Documents a filter keeps (a deleted Tag keeps none). */
  documentIds(filter: DocumentFilter): string[];
}

/**
 * The ids of the live Documents a Search scope covers, or null when there is
 * no Search scope, which means every Document. A scope whose Folders, Tags
 * and Documents hold no live Document (e.g. all of them deleted) covers none:
 * an empty list, never every Document.
 */
export function resolveSearchScope(scope: SearchScope, sources: ScopeSources): string[] | null {
  if (!hasSearchScope(scope)) return null;
  const found = new Set<string>();
  const add = (ids: readonly string[]) => {
    for (const id of ids) found.add(id);
  };
  const folderIds = [...new Set(scope.folderIds.flatMap((id) => sources.folderTree(id)))];
  if (folderIds.length > 0) add(sources.documentIds({ folderIds }));
  for (const tagId of scope.tagIds) add(sources.documentIds({ tagId }));
  if (scope.documentIds.length > 0) add(sources.documentIds({ ids: scope.documentIds }));
  return [...found];
}
