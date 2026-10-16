/**
 * Documents added since a Mind's checks (#993): the checks found unsupported
 * or not checked whose scope has gained Documents since, as the core counts
 * them (`documentsAddedSinceChecks`), for the line under the Mind's title and
 * each check's card. Closing the line keeps it closed, in the Mind's
 * settings, for the Documents it counted: one added later shows it again.
 */
import type { Editor } from "@tiptap/core";
import { useEditorState } from "@tiptap/react";
import { useEffect, useState } from "react";
import type * as Y from "yjs";
import { create } from "zustand";
import { MIND_SETTINGS_FIELD } from "../../core/api";
import { addedCouldChange } from "../../shared/claimChecks";
import { core } from "./core";
import { shownClaims } from "./editor/claimChecks";
import { COUNT_IDLE_MS } from "./editor/WordCount";
import { useAppStore } from "./store";

const CLOSED_KEY = "addedDocumentsClosed";

interface AddedDocumentsState {
  /** By Mind, then by check: the Documents added to its scope since it was made. */
  byMind: Readonly<Record<string, Readonly<Record<string, readonly string[]>>>>;
}

export const useAddedDocuments = create<AddedDocumentsState>()(() => ({ byMind: {} }));

function setAdded(mindId: string, checks: Readonly<Record<string, readonly string[]>> | null) {
  useAddedDocuments.setState((state) => {
    const byMind = { ...state.byMind };
    if (checks === null) delete byMind[mindId];
    else byMind[mindId] = checks;
    return { byMind };
  });
}

/** Keeps the Mind's count up to date while it is open: as its checks change, and as Documents come. */
export function useWatchAddedDocuments(editor: Editor, mindId: string): void {
  const checks = useEditorState({
    editor,
    selector: ({ editor: current }) =>
      shownClaims(current.state)
        .filter((claim) => addedCouldChange(claim.check, claim.shown))
        .map((claim) => `${claim.check.id}@${claim.check.checkedAt}`)
        .join(" "),
  });
  const documents = useAppStore((state) =>
    state.status.kind === "ready" ? state.documents : null,
  );
  const library = useAppStore((state) => state.library);
  useEffect(() => {
    if (checks === "" || documents === null || library === null) {
      setAdded(mindId, null);
      return;
    }
    let live = true;
    const timer = setTimeout(() => {
      core
        .documentsAddedSinceChecks(mindId)
        .then((found) => {
          if (!live) return;
          setAdded(
            mindId,
            Object.fromEntries(found.map((each) => [each.checkId, each.documentIds])),
          );
        })
        .catch((error: unknown) => {
          console.warn(`Couldn't count the Documents added since Mind ${mindId}'s checks.`, error);
        });
    }, COUNT_IDLE_MS);
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [checks, documents, library, mindId]);
  useEffect(() => () => setAdded(mindId, null), [mindId]);
}

const closedOf = (doc: Y.Doc): ReadonlySet<string> => {
  const value = doc.getMap<unknown>(MIND_SETTINGS_FIELD).get(CLOSED_KEY);
  return new Set(
    Array.isArray(value) ? value.filter((each): each is string => typeof each === "string") : [],
  );
};

/** The Documents the User closed the line for, in this Mind. */
export function useClosedAdded(doc: Y.Doc): ReadonlySet<string> {
  const [closed, setClosed] = useState(() => closedOf(doc));
  useEffect(() => {
    const settings = doc.getMap<unknown>(MIND_SETTINGS_FIELD);
    const changed = () => setClosed(closedOf(doc));
    changed();
    settings.observe(changed);
    return () => settings.unobserve(changed);
  }, [doc]);
  return closed;
}

export function closeAdded(doc: Y.Doc, documentIds: readonly string[]): void {
  doc.getMap<unknown>(MIND_SETTINGS_FIELD).set(CLOSED_KEY, [...documentIds]);
}
