/**
 * The files IncarnaMind wrote, which checks leave out of their evidence
 * (#80): whether a Document on screen is left out, and Count it, which the
 * Document's menu, its viewer's line and a check's result line (#96) all
 * call, each with Undo; and the Documents the User keeps as their own
 * workings, which Answers never cite.
 */
import type { Document } from "../../core/api";
import { displayName } from "../../shared/documentNames";
import { leftOutOfEvidence } from "../../shared/evidence";
import type { MessageKey } from "../../shared/i18n";
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

interface Choice {
  set(documentId: string, on: boolean): Promise<unknown>;
  one: Record<"on" | "off", MessageKey>;
  many: Record<"on" | "off", MessageKey>;
}

const COUNTING: Choice = {
  set: (documentId, on) => core.setCountedAsEvidence(documentId, on),
  one: { on: "evidence.counted", off: "evidence.leftOut" },
  many: { on: "evidence.countedMany", off: "evidence.leftOutMany" },
};

const NOT_CITED: Choice = {
  set: (documentId, on) => core.setNotCited(documentId, on),
  one: { on: "evidence.notCited.done", off: "evidence.citeAgain.done" },
  many: { on: "evidence.notCited.doneMany", off: "evidence.citeAgain.doneMany" },
};

async function choose(
  documents: readonly Pick<Document, "id" | "name" | "title">[],
  choice: Choice,
  on: boolean,
): Promise<Undoable> {
  for (const document of documents) await choice.set(document.id, on);
  const [only] = documents;
  const state = on ? "on" : "off";
  const message =
    only && documents.length === 1
      ? translateNow(choice.one[state], { name: displayName(only) })
      : translateNow(choice.many[state], { count: documents.length });
  const undoable: Undoable = {
    message,
    undo: async () => {
      for (const document of documents) await choice.set(document.id, !on);
    },
    redo: async () => {
      for (const document of documents) await choice.set(document.id, on);
    },
  };
  useUndo.getState().push(undoable);
  return undoable;
}

/**
 * Counts Documents IncarnaMind wrote as evidence from now on ("Count it"),
 * or leaves them out again, and offers Undo, which puts each back. Every
 * window hears of it from the core's "document.status".
 */
export const countAsEvidence = (
  documents: readonly Pick<Document, "id" | "name" | "title">[],
  counted: boolean,
) => choose(documents, COUNTING, counted);

export const markNotCited = (
  documents: readonly Pick<Document, "id" | "name" | "title">[],
  notCited: boolean,
) => choose(documents, NOT_CITED, notCited);
