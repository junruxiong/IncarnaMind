/**
 * A Question's Search scope as its Block stores it (see `QuestionAttributes`):
 * read the same way by the core, which resolves it when the Question is asked,
 * and by the editor, which shows it as chips.
 */
import type { QuestionAttributes, SearchScope } from "../core/api";

/** What a Search scope can name. */
export type ScopeKind = "folder" | "tag" | "document";

export const SCOPE_KINDS: readonly ScopeKind[] = ["folder", "tag", "document"];

/** The Question attribute that holds each kind's ids. */
export const SCOPE_ATTRIBUTES = {
  folder: "scopeFolderIds",
  tag: "scopeTagIds",
  document: "scopeDocumentIds",
} as const satisfies Record<ScopeKind, keyof QuestionAttributes>;

/** The ids in an attribute, in order, once each; anything that isn't a list of ids counts as none. */
function idsOf(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  const ids = value.filter((id): id is string => typeof id === "string" && id !== "");
  return [...new Set(ids)];
}

/** A Question's Search scope, from its Block's attributes. */
export function searchScopeOf(attributes: Readonly<Record<string, unknown>>): SearchScope {
  return {
    folderIds: idsOf(attributes[SCOPE_ATTRIBUTES.folder]),
    tagIds: idsOf(attributes[SCOPE_ATTRIBUTES.tag]),
    documentIds: idsOf(attributes[SCOPE_ATTRIBUTES.document]),
  };
}

/** The ids of one kind in a Search scope. */
export function scopeIds(scope: SearchScope, kind: ScopeKind): string[] {
  return kind === "folder" ? scope.folderIds : kind === "tag" ? scope.tagIds : scope.documentIds;
}

/** Whether a Question has a Search scope at all. Without one, every Document is searched. */
export function hasSearchScope(scope: SearchScope): boolean {
  return scope.folderIds.length + scope.tagIds.length + scope.documentIds.length > 0;
}
