/**
 * The files IncarnaMind wrote, which checks leave out of their evidence
 * (#80): whether a Document on screen is left out, and Count it, which the
 * Document's menu, its viewer's line and a check's result line (#96) all
 * call, each with Undo.
 */
import type { Document } from "../../core/api";
import { displayName } from "../../shared/documentNames";
import { leftOutOfEvidence } from "../../shared/evidence";
import { core } from "./core";
import { translateNow } from "./i18n";
import type { Undoable } from "./objects";
import { useAppStore } from "./store";
import { useUndo } from "./undo";

/** Whether checks leave the Document out of their evidence now (see `leftOutOfEvidence`), kept up to date. */
export function useLeftOutOfEvidence(document: Document | undefined): boolean {
  return useAppStore((state) =>
    document ? leftOutOfEvidence(document, state.settings, state.library) : false,
  );
}

/**
 * Counts Documents IncarnaMind wrote as evidence from now on ("Count it"),
 * or leaves them out again, and offers Undo, which puts each back. Every
 * window hears of it from the core's "document.status".
 */
export async function countAsEvidence(
  documents: readonly Pick<Document, "id" | "name" | "title">[],
  counted: boolean,
): Promise<Undoable> {
  for (const document of documents) await core.setCountedAsEvidence(document.id, counted);
  const [only] = documents;
  const message =
    only && documents.length === 1
      ? translateNow(counted ? "evidence.counted" : "evidence.leftOut", {
          name: displayName(only),
        })
      : translateNow(counted ? "evidence.countedMany" : "evidence.leftOutMany", {
          count: documents.length,
        });
  const undoable: Undoable = {
    message,
    undo: async () => {
      for (const document of documents) await core.setCountedAsEvidence(document.id, !counted);
    },
  };
  useUndo.getState().offer(undoable);
  return undoable;
}
