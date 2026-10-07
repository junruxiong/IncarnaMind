/**
 * Names shared by the main process and the renderer for the Document viewer.
 *
 * The main process serves each live Document's stored file at
 * `incarnamind-document://<documentId>/`, streamed from the data folder, so
 * whole files never cross IPC.
 */

import type { CitationCheck } from "../core/api";

export const DOCUMENT_SCHEME = "incarnamind-document";

/** The URL the main process serves a Document's file at. */
export const documentFileUrl = (documentId: string): string =>
  `${DOCUMENT_SCHEME}://${encodeURIComponent(documentId)}/`;

/** The Document id a `documentFileUrl` names, or null if the URL isn't one. */
export function documentIdFromUrl(url: string): string | null {
  const parsed = URL.parse(url);
  if (parsed?.protocol !== `${DOCUMENT_SCHEME}:` || !parsed.hostname) return null;
  if ((parsed.pathname !== "/" && parsed.pathname !== "") || parsed.search || parsed.hash) {
    return null;
  }
  return decodeURIComponent(parsed.hostname);
}

/**
 * Where to open a Document in the viewer. Citations open it at their page range
 * and quote; a click in the sidebar opens it at the top.
 */
export interface DocumentLocation {
  documentId: string;
  /** PDFs: the first page to show, from 1. */
  pageFrom?: number;
  /** PDFs: the last page the quote may be on. Defaults to `pageFrom`. */
  pageTo?: number;
  /**
   * Text to highlight if it is found: on the pages from `pageFrom` to `pageTo`
   * for a PDF, anywhere for TXT and Markdown (which the viewer scrolls to it).
   */
  quote?: string;
  /**
   * The Citation this opens, when a Citation opens it. Its check colours the
   * quote's highlight (green when found, amber when opened anyway after "not
   * found"); with its number too, the viewer shows its check mark in the page's
   * margin, beside the quote, as the Mind does beside the Answer.
   */
  citation?: ViewerCitation;
}

/** What the viewer knows of the Citation it was opened from. */
export interface ViewerCitation {
  check: CitationCheck;
  /** The number the Citation shows in its Answer, from 1. */
  number?: number;
}
