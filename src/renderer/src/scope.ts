import type {
  Document,
  DocumentKind,
  Folder,
  LibraryGroup,
  SearchScope,
  Tag,
} from "../../core/api";
import { SCOPE_KINDS, type ScopeKind, scopeIds } from "../../shared/searchScope";
import { buildFolderTree, flattenFolderTree } from "./folders";

/** The Folders, Tags and Documents a Search scope can name, as the app store holds them. */
export interface ScopeLibrary {
  folders: readonly Folder[];
  groups?: readonly LibraryGroup[];
  /** In-app Folders deleted since: a scope naming one still shows its name. */
  deletedGroups?: readonly LibraryGroup[];
  tags: readonly Tag[];
  documents: readonly Document[];
}

/** A Folder, Tag or Document the "@" picker offers for a Question's Search scope. */
export interface ScopeChoice {
  kind: ScopeKind;
  id: string;
  name: string;
  /** A Folder's parent Folders' names, outermost first: where it is. Empty otherwise. */
  path: string[];
  sourceLocation?: boolean;
  /** A Document's kind, for its icon. */
  documentKind: DocumentKind | null;
}

/** A Search scope item as a chip on its Question. */
export interface ScopeChip {
  kind: ScopeKind;
  id: string;
  /**
   * Its name, or null once it is deleted and its name is gone with it. An
   * in-app Folder deleted keeps its name here.
   */
  name: string | null;
  /** Deleted since: the search ignores it. */
  deleted: boolean;
  documentKind: DocumentKind | null;
}

/** The most choices of one kind the picker lists at once; typing narrows them. */
const MAX_CHOICES_PER_KIND = 50;

/** Every Folder, Tag and Document, in the sidebar's order: Folders as a tree, depth first. */
function everything(library: ScopeLibrary): ScopeChoice[] {
  const names = new Map(library.folders.map((folder) => [folder.id, folder.name]));
  const parents = new Map(library.folders.map((folder) => [folder.id, folder.parentId]));
  const pathOf = (folder: Folder) => {
    const path: string[] = [];
    const seen = new Set<string>([folder.id]);
    let parent = folder.parentId;
    for (; parent && !seen.has(parent); parent = parents.get(parent) ?? null) {
      seen.add(parent);
      const name = names.get(parent);
      if (name !== undefined) path.unshift(name);
    }
    return path;
  };
  return [
    ...(library.groups ?? []).map(
      (group): ScopeChoice => ({
        kind: "folder",
        id: group.id,
        name: group.name,
        path: [],
        documentKind: null,
      }),
    ),
    ...flattenFolderTree(buildFolderTree(library.folders)).map(
      ({ folder }): ScopeChoice => ({
        kind: "folder",
        id: folder.id,
        name: folder.name,
        path: pathOf(folder),
        sourceLocation: (library.groups?.length ?? 0) > 0,
        documentKind: null,
      }),
    ),
    ...library.tags.map(
      (tag): ScopeChoice => ({
        kind: "tag",
        id: tag.id,
        name: tag.name,
        path: [],
        documentKind: null,
      }),
    ),
    ...library.documents.map(
      (document): ScopeChoice => ({
        kind: "document",
        id: document.id,
        name: document.name,
        path: [],
        documentKind: document.kind,
      }),
    ),
  ];
}

/**
 * What the picker offers for `query` (typed after the "@"): the Folders, Tags
 * and Documents whose names contain it, ignoring case, those starting with it
 * first, grouped in that order. Those already in the scope aren't offered again.
 */
export function scopeChoices(
  library: ScopeLibrary,
  scope: SearchScope,
  query: string,
): ScopeChoice[] {
  const wanted = query.trim().toLocaleLowerCase();
  const chosen = new Set(
    SCOPE_KINDS.flatMap((kind) => scopeIds(scope, kind).map((id) => `${kind}:${id}`)),
  );
  const all = everything(library).filter((choice) => !chosen.has(`${choice.kind}:${choice.id}`));
  return SCOPE_KINDS.flatMap((kind) => {
    const ofKind = all.filter((choice) => choice.kind === kind);
    const named = (choice: ScopeChoice) => choice.name.toLocaleLowerCase();
    const matching = wanted
      ? [
          ...ofKind.filter((choice) => named(choice).startsWith(wanted)),
          ...ofKind.filter(
            (choice) => !named(choice).startsWith(wanted) && named(choice).includes(wanted),
          ),
        ]
      : ofKind;
    return matching.slice(0, MAX_CHOICES_PER_KIND);
  });
}

/** A Question's Search scope as chips, Folders first, then Tags, then Documents, each in the order chosen. */
export function scopeChips(library: ScopeLibrary, scope: SearchScope): ScopeChip[] {
  const folders = new Map(
    [...library.folders, ...(library.groups ?? [])].map((folder) => [folder.id, folder]),
  );
  const deletedFolders = new Map((library.deletedGroups ?? []).map((group) => [group.id, group]));
  const tags = new Map(library.tags.map((tag) => [tag.id, tag]));
  const documents = new Map(library.documents.map((document) => [document.id, document]));
  return [
    ...scope.folderIds.map((id): ScopeChip => {
      const live = folders.get(id);
      return {
        kind: "folder",
        id,
        name: live?.name ?? deletedFolders.get(id)?.name ?? null,
        deleted: !live,
        documentKind: null,
      };
    }),
    ...scope.tagIds.map((id): ScopeChip => {
      const tag = tags.get(id);
      return { kind: "tag", id, name: tag?.name ?? null, deleted: !tag, documentKind: null };
    }),
    ...scope.documentIds.map((id): ScopeChip => {
      const document = documents.get(id);
      return {
        kind: "document",
        id,
        name: document?.name ?? null,
        deleted: !document,
        documentKind: document?.kind ?? null,
      };
    }),
  ];
}
