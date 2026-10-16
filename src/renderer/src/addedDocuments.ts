/**
 * Documents added since a Mind's checks (#993): the checks found unsupported
 * or not checked whose scope has gained Documents since, as the core counts
 * them (`documentsAddedSinceChecks`), for the line under the Mind's title and
 * each check's card. Closing the line keeps it closed, in the Mind's
 * settings, for the Documents it counted: one added later shows it again.
 */
import { useEffect, useState } from "react";
import type * as Y from "yjs";
import { create } from "zustand";
import { MIND_SETTINGS_FIELD } from "../../core/api";

const CLOSED_KEY = "addedDocumentsClosed";

interface AddedDocumentsState {
  /** By Mind, then by check: the Documents added to its scope since it was made. */
  byMind: Readonly<Record<string, Readonly<Record<string, readonly string[]>>>>;
}

export const useAddedDocuments = create<AddedDocumentsState>()(() => ({ byMind: {} }));

/** The Documents added since any of the Mind's checks, each once. */
export const documentsAdded = (
  checks: Readonly<Record<string, readonly string[]>> | undefined,
): string[] => [...new Set(Object.values(checks ?? {}).flat())];

export function setAdded(
  mindId: string,
  checks: Readonly<Record<string, readonly string[]>> | null,
) {
  useAddedDocuments.setState((state) => {
    const byMind = { ...state.byMind };
    if (checks === null) delete byMind[mindId];
    else byMind[mindId] = checks;
    return { byMind };
  });
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
