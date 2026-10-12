import { useCallback } from "react";
import type { CitationAttributes, CitationCheck } from "../../../core/api";
import { type CitationState, citationState } from "../../../shared/citations";
import { citationLocation, formatLocation } from "../../../shared/locations";
import { useTabPlace } from "../components/tabs/tabItems";
import { useT } from "../i18n";
import { useAppStore } from "../store";

/** Whether opening the cited place washes the quote: it is there, unless the check could not find it. */
export const quoteSaysSo = (check: CitationCheck) =>
  check === "found" || check === "unsupported" || check === "contradicted";

/** A Citation's state now, as its badge shows it: with the User's live Documents and the text kept of others. */
export function useCitationState(attributes: CitationAttributes | null): CitationState | null {
  const documents = useAppStore((state) =>
    state.status.kind === "ready" ? state.documents : null,
  );
  const kept = useAppStore((state) => state.keptCitationTexts);
  return attributes ? citationState(attributes, documents, kept) : null;
}

/**
 * Opens a Citation's cited place (a page, slide, section, rows or lines), or
 * its Document, beside the Mind, with the quote to highlight if asked, and
 * the Citation's check, number and label: the viewer colours the quote by
 * the check (amber when opened anyway after "not found") and shows the check
 * mark, with where it points, beside it. A Citation in the text and a
 * chart's value (#436) open the same way.
 */
export function useOpenCited(): (
  attributes: CitationAttributes,
  state: CitationState,
  options: { withQuote: boolean; number?: number },
) => void {
  const t = useT();
  const openDocument = useAppStore((store) => store.openDocument);
  /** The Mind's group: the Document opens in the group to its right. */
  const place = useTabPlace();
  return useCallback(
    (attributes, state, { withQuote, number = 0 }) => {
      const documentId = state.documentId ?? attributes.documentId;
      if (!documentId) return;
      const location = citationLocation(attributes);
      const where = location ? formatLocation(location, t) : null;
      openDocument(
        {
          documentId,
          // A whole TXT or Markdown file cited before Locations, whose quote wasn't found, opens at the top.
          pageFrom: attributes.pageFrom ?? undefined,
          pageTo: attributes.pageTo ?? undefined,
          quote: withQuote && attributes.quote ? attributes.quote : undefined,
          citation: {
            check: state.check,
            ...(number > 0 ? { number } : {}),
            ...(where ? { label: where } : {}),
            ...(attributes.contentHash ? { contentHash: attributes.contentHash } : {}),
          },
        },
        { as: "citation", ...(place && { from: place.groupId }) },
      );
    },
    [t, openDocument, place],
  );
}
