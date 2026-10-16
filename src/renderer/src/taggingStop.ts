import type { Document, LibrarySnapshot } from "../../core/api";
import type { Language } from "../../core/language";
import type { MessageKey, MessageParams } from "../../shared/i18n";
import { core } from "./core";
import { languageNow, translateNow } from "./i18n";
import { useAppStore } from "./store";
import { useUndo } from "./undo";

/** Where Tagging with AI can be stopped (#895): a Folder's id, or null for Not in a Folder. */
export type TaggingPlace = string | null;

type Translate = (key: MessageKey, params?: MessageParams) => string;

// As the core places a Document: a Suggested Folder counts, a deleted Folder doesn't.
function placeOf(library: LibrarySnapshot | null, documentId: string): TaggingPlace {
  const groupId = library?.assignments.find((each) => each.documentId === documentId)?.groupId;
  return groupId && library?.groups.some((group) => group.id === groupId) ? groupId : null;
}

/** Where the Documents being tagged are: what Stop stops. */
export function taggingPlaces(
  documents: readonly Document[],
  library: LibrarySnapshot | null,
): TaggingPlace[] {
  const tagging = documents.filter(
    (each) => each.status === "ready" && (each.tagging === "pending" || each.tagging === "tagging"),
  );
  return [...new Set(tagging.map((each) => placeOf(library, each.id)))];
}

/** The places where Tagging is stopped that are still there. */
export const stoppedPlaces = (library: LibrarySnapshot | null): TaggingPlace[] =>
  (library?.settings.taggingStopped ?? []).filter(
    (place) => place === null || library?.groups.some((group) => group.id === place),
  );

/** The places in words, joined as the language joins a list: in “Contracts” and “Board”, and not in a Folder. */
export function placesText(
  places: readonly TaggingPlace[],
  library: LibrarySnapshot | null,
  t: Translate,
  language: Language,
): string {
  const list = (items: string[]) =>
    new Intl.ListFormat(language, { type: "conjunction" }).format(items);
  const names = places.flatMap((place) => {
    const name = place && library?.groups.find((group) => group.id === place)?.name;
    return name ? [`“${name}”`] : [];
  });
  return list([
    ...(names.length > 0 ? [t("tagging.place.folders", { names: list(names) })] : []),
    ...(places.includes(null) ? [t("tagging.place.notInFolder")] : []),
  ]);
}

const placesNow = (places: readonly TaggingPlace[]) =>
  placesText(places, useAppStore.getState().library, translateNow, languageNow());

const setTagging = (places: readonly TaggingPlace[], on: boolean) => async () => {
  for (const place of places) await core.setTaggingWithAi(place, on);
};

/** Stop in the sidebar's status: nothing more is sent from where these Documents are, with Undo. */
export async function stopTagging(): Promise<void> {
  const places = await core.stopTagging();
  if (places.length === 0) return;
  useUndo.getState().push({
    message: translateNow(
      places.includes(null) ? "tagging.stopped.notInFolder" : "tagging.stopped",
      {
        places: placesNow(places),
      },
    ),
    undo: setTagging(places, true),
    redo: setTagging(places, false),
  });
}

/** "Tag with AI": Tagging goes on where it was stopped, with Undo. */
export async function tagWithAi(places: readonly TaggingPlace[]): Promise<void> {
  await setTagging(places, true)();
  useUndo.getState().push({
    message: translateNow("tagging.restarted", { places: placesNow(places) }),
    undo: setTagging(places, false),
    redo: setTagging(places, true),
  });
}
