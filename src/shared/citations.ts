/**
 * How a Citation shows: its badge state and the words for it. Pure, shared by
 * the editor and the tests.
 *
 * The badge says whether the quote was found on the cited page, never that the
 * sentence is verified: "found" means the quote is there, not that it supports
 * the claim.
 */
import type {
  CitationAttributes,
  CitationCheck,
  CitationCheckReason,
  Document,
  KeptCitationText,
} from "../core/api";
import type { MessageKey } from "./i18n";
import {
  citationLocation,
  englishLocation,
  formatLocation,
  readsIn,
  type Translate,
} from "./locations";

export interface CitationState {
  /** What the badge shows. */
  check: CitationCheck;
  /** Why it wasn't found, or can't be checked. */
  reason: CitationCheckReason | null;
  /**
   * The live Document the Citation opens: its own, or one with the same
   * content added again after it was deleted. Null once the Document has
   * been deleted, or unlinked with its Linked folder.
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
 * Whether the stored text a Citation's check read is kept, though its
 * Document was unlinked with its Linked folder: every Unit it cites, of the
 * version it quotes (or of any version, if it doesn't say). The viewer asks
 * too, to say why it can't show the Document.
 */
export function citedTextKept(
  attributes: Partial<
    Pick<CitationAttributes, "documentId" | "contentHash" | "pageFrom" | "pageTo">
  >,
  kept: readonly KeptCitationText[],
): boolean {
  const { documentId, contentHash, pageFrom, pageTo } = attributes;
  if (!documentId) return false;
  return kept.some((text) => {
    if (text.documentId !== documentId) return false;
    if (contentHash && text.contentHash !== contentHash) return false;
    // A whole TXT or Markdown file cited before Units: its one text.
    if (typeof pageFrom !== "number") return text.units.length > 0;
    const other = typeof pageTo === "number" ? pageTo : pageFrom;
    const [from, to] = [Math.min(pageFrom, other), Math.max(pageFrom, other)];
    // Units are numbered without gaps, and each is kept once.
    return text.units.filter((unit) => unit >= from && unit <= to).length === to - from + 1;
  });
}

type LiveDocument = Pick<Document, "id" | "contentHash">;

/** A list of Documents by ID and by content: the first in the list with each. */
interface DocumentIndex {
  byId: ReadonlyMap<string, LiveDocument>;
  byContent: ReadonlyMap<string, LiveDocument>;
}

/**
 * Each list of Documents indexed once, by the list: a Mind looks up a
 * Document for every Citation it shows, often, and a list is replaced, never
 * changed in place, when the Documents change.
 */
const indexes = new WeakMap<readonly LiveDocument[], DocumentIndex>();

function indexOf(documents: readonly LiveDocument[]): DocumentIndex {
  let index = indexes.get(documents);
  if (!index) {
    const byId = new Map<string, LiveDocument>();
    const byContent = new Map<string, LiveDocument>();
    for (const document of documents) {
      if (!byId.has(document.id)) byId.set(document.id, document);
      if (document.contentHash && !byContent.has(document.contentHash)) {
        byContent.set(document.contentHash, document);
      }
    }
    index = { byId, byContent };
    indexes.set(documents, index);
  }
  return index;
}

/**
 * The badge state of a Citation, given the User's live Documents (null while
 * they are loading) and the text kept of Documents unlinked with their Linked
 * folder. The check result stored when the Answer finished stands, unless the
 * Document has been deleted since: then it "can't be checked". A Document
 * unlinked with its folder keeps the text its Citations cite, so their check
 * stands too.
 */
export function citationState(
  attributes: Partial<CitationAttributes>,
  documents: readonly LiveDocument[] | null,
  kept: readonly KeptCitationText[] = [],
): CitationState {
  const check = attributes.check ?? "checking";
  const reason = attributes.checkReason ?? null;
  const own = attributes.documentId ?? null;
  if (documents === null) return { check, reason, documentId: own, changedAfterCited: false };
  const { byId, byContent } = indexOf(documents);
  const live =
    (own === null ? undefined : byId.get(own)) ??
    (attributes.contentHash ? byContent.get(attributes.contentHash) : undefined);
  const changedAfterCited =
    live !== undefined &&
    typeof attributes.contentHash === "string" &&
    attributes.contentHash !== "" &&
    live.contentHash !== attributes.contentHash;
  if (check === "checking") {
    return { check, reason: null, documentId: live?.id ?? own, changedAfterCited: false };
  }
  if (!live && citedTextKept(attributes, kept)) {
    return { check, reason, documentId: null, changedAfterCited: false };
  }
  if (!live) {
    return { check: "cant-check", reason: "document-removed", documentId: null, changedAfterCited };
  }
  return { check, reason, documentId: live.id, changedAfterCited };
}

/**
 * A Citation's short Location label in a language ("p. 3", "slide 4",
 * "Revenue, rows 12–14", "§ 2.1 Sensitivity", "lines 120–134"); null for a
 * whole TXT or Markdown file cited before Locations.
 */
export function citedLocation(
  attributes: Partial<CitationAttributes>,
  t: Translate,
): string | null {
  const location = citationLocation(attributes);
  return location ? formatLocation(location, t) : null;
}

/** A Citation as plain text, e.g. "[Attention Is All You Need, p. 3]", for copying as text. */
export function citationReference(attributes: Partial<CitationAttributes>): string {
  const name = attributes.documentName;
  if (!name) return "";
  const location = citationLocation(attributes);
  return location ? `[${name}, ${englishLocation(location)}]` : `[${name}]`;
}

/** The message key and parameters of a Citation's badge, its Location worded with `t`. */
export function badgeMessage(
  state: CitationState,
  attributes: Partial<CitationAttributes>,
  t: Translate,
): { key: MessageKey; params?: Record<string, string> } {
  const location = citationLocation(attributes);
  const document = attributes.documentName ?? "";
  const params: Record<string, string> = location
    ? { location: formatLocation(location, t) }
    : { document };
  const where = location === null ? "document" : readsIn(location) ? "in" : "on";
  switch (state.check) {
    case "checking":
      return { key: "citation.badge.checking" };
    case "found":
      return { key: `citation.badge.found.${where}`, params };
    case "not-found":
      return { key: `citation.badge.notFound.${where}`, params };
    case "cant-check":
      return { key: "citation.badge.cantCheck" };
  }
}
