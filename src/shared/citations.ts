/**
 * How a Citation shows: its badge state and the words for it. Pure, shared by
 * the editor and the tests.
 *
 * The badge says whether the quote was found on the cited page, never that the
 * sentence is verified: "found" means the quote is there, not that it supports
 * the claim.
 */
import type { CitationAttributes, CitationCheck, CitationCheckReason, Document } from "../core/api";
import type { MessageKey } from "./i18n";

export interface CitationState {
  /** What the badge shows. */
  check: CitationCheck;
  /** Why it wasn't found, or can't be checked. */
  reason: CitationCheckReason | null;
  /**
   * The live Document the Citation opens: its own, or one with the same
   * content added again after it was deleted. Null once the Document has
   * been deleted.
   */
  documentId: string | null;
  /**
   * The Document changed after the Citation was made: its current version
   * isn't the one quoted. The check still stands for the version quoted,
   * whose text is kept; `recheckCitation` checks the current one, so the
   * card can say "the Document changed after this was cited" and offer it.
   */
  changedAfterCited: boolean;
}

/**
 * The badge state of a Citation, given the User's live Documents (null while
 * they are loading). The check result stored when the Answer finished stands,
 * unless the Document has been deleted since: then it "can't be checked".
 */
export function citationState(
  attributes: Partial<CitationAttributes>,
  documents: readonly Pick<Document, "id" | "contentHash">[] | null,
): CitationState {
  const check = attributes.check ?? "checking";
  const reason = attributes.checkReason ?? null;
  const own = attributes.documentId ?? null;
  if (documents === null) return { check, reason, documentId: own, changedAfterCited: false };
  const live =
    documents.find((document) => document.id === own) ??
    (attributes.contentHash
      ? documents.find((document) => document.contentHash === attributes.contentHash)
      : undefined);
  const changedAfterCited =
    live !== undefined &&
    typeof attributes.contentHash === "string" &&
    attributes.contentHash !== "" &&
    live.contentHash !== attributes.contentHash;
  if (check === "checking") {
    return { check, reason: null, documentId: live?.id ?? own, changedAfterCited: false };
  }
  if (!live) {
    return { check: "cant-check", reason: "document-removed", documentId: null, changedAfterCited };
  }
  return { check, reason, documentId: live.id, changedAfterCited };
}

/** The cited pages, e.g. "3" or "3–4"; null for a Document without pages. */
export function citedPages(attributes: Partial<CitationAttributes>): string | null {
  const { pageFrom, pageTo } = attributes;
  if (typeof pageFrom !== "number") return null;
  return typeof pageTo === "number" && pageTo !== pageFrom
    ? `${pageFrom}–${pageTo}`
    : `${pageFrom}`;
}

/** A Citation as plain text, e.g. "[Attention Is All You Need, p. 3]", for copying as text. */
export function citationReference(attributes: Partial<CitationAttributes>): string {
  const name = attributes.documentName;
  if (!name) return "";
  const pages = citedPages(attributes);
  return pages ? `[${name}, p. ${pages}]` : `[${name}]`;
}

/** The message key and parameters of a Citation's badge. */
export function badgeMessage(
  state: CitationState,
  attributes: Partial<CitationAttributes>,
): { key: MessageKey; params?: Record<string, string> } {
  const pages = citedPages(attributes);
  const document = attributes.documentName ?? "";
  switch (state.check) {
    case "checking":
      return { key: "citation.badge.checking" };
    case "found":
      return pages
        ? { key: "citation.badge.found.page", params: { pages } }
        : { key: "citation.badge.found.document", params: { document } };
    case "not-found":
      return pages
        ? { key: "citation.badge.notFound.page", params: { pages } }
        : { key: "citation.badge.notFound.document", params: { document } };
    case "cant-check":
      return { key: "citation.badge.cantCheck" };
  }
}
