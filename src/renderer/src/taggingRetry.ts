import type { Document } from "../../core/api";
import { useAppStore } from "./store";

/** The Documents whose tagging failed because the AI model couldn't be reached. */
export const unreachedDocuments = (documents: readonly Document[]): string[] =>
  documents.flatMap((each) =>
    each.tagging === "failed" && each.taggingError?.kind === "network" ? [each.id] : [],
  );

/** Once the computer is back online, the Documents that couldn't reach the AI model are tagged again. */
export function retryTaggingWhenOnline(): void {
  window.addEventListener("online", () => {
    const ids = unreachedDocuments(useAppStore.getState().documents);
    if (ids.length > 0) void useAppStore.getState().retagDocuments(ids);
  });
}
