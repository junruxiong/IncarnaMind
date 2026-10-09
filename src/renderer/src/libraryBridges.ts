/**
 * Bridges from the Library into writing (#59): "Ask about this Folder" puts
 * a Question scoped to the Folder in the most recent Mind, and "Start a Mind
 * from this Folder" makes a Mind with that Question first. With the list
 * filtered, the Question searches exactly the Documents shown instead.
 * Folders are Search scopes already (the "@" picker); this only chooses one.
 * Pure: no store, no bridge, so the tests import it.
 */

import type { Mind, SearchScope } from "../../core/api";

/** What the Library's sheet shows: every Document, the Unsorted ones, or a Folder's. */
export type LibraryView =
  | { kind: "all"; name: string }
  | { kind: "unsorted"; name: string }
  | { kind: "folder"; id: string; name: string };

/** A Question from the Library: what it searches, and the title of a Mind started from it. */
export interface LibraryBridge {
  /** "folder": the Folder itself; "documents": the Documents shown, one by one. */
  kind: "folder" | "documents";
  scope: SearchScope;
  /** How many Documents are shown, and so searched (for a Folder, as it is now). */
  count: number;
  /** The sheet's title: the Folder's name, "Unsorted" or "All Documents". */
  title: string;
}

/**
 * What a Question asked from the sheet searches, given the ids of the
 * Documents it shows and whether a filter (or a search, or a Tag) narrows
 * them. A Folder as it is is searched as the Folder, so Documents filed in
 * it later count too. Narrowed, or outside any Folder, it is exactly the
 * Documents shown. Null when there is nothing to ask about: nothing shown,
 * or all Documents unfiltered, which any Question searches already.
 */
export function bridgeFrom(
  view: LibraryView,
  shownIds: readonly string[],
  narrowed: boolean,
): LibraryBridge | null {
  if (shownIds.length === 0) return null;
  if (view.kind === "all" && !narrowed) return null;
  const count = shownIds.length;
  if (view.kind === "folder" && !narrowed) {
    return {
      kind: "folder",
      title: view.name,
      count,
      scope: { folderIds: [view.id], tagIds: [], documentIds: [] },
    };
  }
  return {
    kind: "documents",
    title: view.name,
    count,
    scope: { folderIds: [], tagIds: [], documentIds: [...new Set(shownIds)] },
  };
}

/**
 * The most recent Mind, which "Ask about this Folder" writes in: the one
 * shown before the Library covered it, or else the one edited last (`minds`
 * lists them so). Never the example Mind. Null with none: a new one is made.
 */
export function mindToAskIn({
  minds,
  openMindId,
  exampleMindId,
}: {
  minds: readonly Mind[];
  openMindId: string | null;
  exampleMindId: string | null;
}): string | null {
  const own = minds.filter((mind) => mind.id !== exampleMindId);
  return (own.find((mind) => mind.id === openMindId) ?? own[0])?.id ?? null;
}
