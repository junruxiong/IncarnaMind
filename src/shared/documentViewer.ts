/**
 * Names shared by the main process and the renderer for the Document viewer.
 *
 * The main process serves each live Document's file at
 * `incarnamind-document://<documentId>/`, streamed from where the User keeps
 * it, so whole files never cross IPC. A Document whose file is missing or
 * can't be reached gets a 404; its kept text comes from `readDocumentText`.
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
  /**
   * The first Unit to show, from 1 (ADR-0011): a PDF's page, a deck's slide,
   * otherwise the number of a section, block of rows or block of lines.
   */
  pageFrom?: number;
  /** The last Unit the quote may be in. Defaults to `pageFrom`. */
  pageTo?: number;
  /**
   * Text to highlight if it is found: in the Units from `pageFrom` to
   * `pageTo` (the viewer scrolls to it), or without them, anywhere in a
   * TXT or Markdown file.
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
  /** Its Location's short label, e.g. "slide 4" or "Revenue, rows 12–14": the mark shows it. */
  label?: string;
  /**
   * The version of the Document it quotes: with the Document gone, the viewer
   * tells whether that text was kept because its Linked folder was unlinked.
   */
  contentHash?: string;
}
