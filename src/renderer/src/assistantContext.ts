import type { Translate } from "./objects";
import { DELETED_NAMES, type ScopeChip } from "./scope";

/**
 * The 12px line under the assistant's header: what the next Question reads
 * and searches (#253, #258). With a Mind: "Reads “Chapter 3” · searches
 * “Thesis”", naming the first item of the composer's Search scope (its
 * Folder, by default); with none in view: "Searches All Documents". Items
 * deleted since are left out, as the search leaves them out.
 */
export function readsLine(
  t: Translate,
  mindName: string | null,
  scope: readonly ScopeChip[],
): string {
  if (mindName === null) return t("assistant.searchesAll");
  const named = scope.filter((chip) => !chip.deleted && chip.name);
  const first = named[0]?.name;
  if (!first) return t("assistant.reads.all", { mind: mindName });
  if (named.length === 1) return t("assistant.reads.one", { mind: mindName, scope: first });
  return t("assistant.reads.more", {
    mind: mindName,
    scope: first,
    count: named.length - 1,
    total: named.length,
  });
}

/**
 * What a Question in the assistant searched, on its Answer's meta line:
 * "“Thesis”", "“Thesis” and 2 more", or "All Documents". A Folder deleted
 * since is still named, it was searched then; one gone without its name (a
 * Linked folder's folder removed on disk) is a "Deleted Folder", never left
 * out: an Answer that searched nothing doesn't read as having searched everything.
 */
export function scopeLine(t: Translate, scope: readonly ScopeChip[]): string {
  const first = scope[0];
  if (!first) return t("assistant.scope.all");
  const name = first.name ?? t(DELETED_NAMES[first.kind]);
  if (scope.length === 1) return t("assistant.scope.one", { scope: name });
  return t("assistant.scope.more", { scope: name, count: scope.length - 1, total: scope.length });
}
