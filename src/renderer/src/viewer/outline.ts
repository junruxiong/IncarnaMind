/**
 * A PDF's outline (its bookmarks), read with pdf.js's `getOutline`, with the
 * page each entry goes to worked out up front. No DOM and no React, so the
 * tests run it with pdf.js in Node.
 */

/** One entry of the outline. */
export interface OutlineItem {
  /** As the PDF gives it, with runs of white space made one space. May be empty. */
  title: string;
  /**
   * The page it goes to, from 1. Null when it goes nowhere in this PDF: a web
   * link, or a destination that doesn't resolve to one of its pages.
   */
  page: number | null;
  /** The PDF shows it expanded at first (a positive `/Count`). */
  open: boolean;
  items: OutlineItem[];
}

/** An entry as pdf.js's `getOutline` gives it: the parts used here. */
interface PdfOutlineNode {
  title: string;
  /** A named destination, an explicit one (`[page, {name: "XYZ"}, …]`), or null. */
  dest: string | readonly unknown[] | null;
  count?: number | undefined;
  items: readonly PdfOutlineNode[];
}

/** The parts of pdf.js's `PDFDocumentProxy` reading an outline needs. */
export interface OutlineSource {
  readonly numPages: number;
  getOutline(): Promise<readonly PdfOutlineNode[] | null>;
  getDestination(id: string): Promise<readonly unknown[] | null>;
  getPageIndex(ref: { num: number; gen: number }): Promise<number>;
}

/** Outlines deeper than this are cut there: a malformed PDF can nest without end. */
const MAX_DEPTH = 32;

const isRef = (value: unknown): value is { num: number; gen: number } =>
  typeof value === "object" &&
  value !== null &&
  Number.isInteger((value as { num?: unknown }).num) &&
  Number.isInteger((value as { gen?: unknown }).gen);

/**
 * The page a destination goes to, from 1, or null. A destination is named
 * (looked up in the PDF) or explicit; an explicit one starts with a reference
 * to its page or, in some PDFs, the page's index.
 */
export async function destinationPage(
  source: OutlineSource,
  dest: PdfOutlineNode["dest"],
): Promise<number | null> {
  try {
    const explicit = typeof dest === "string" ? await source.getDestination(dest) : dest;
    if (!Array.isArray(explicit) || explicit.length === 0) return null;
    const [target] = explicit;
    const index = isRef(target)
      ? await source.getPageIndex(target)
      : Number.isInteger(target)
        ? (target as number)
        : null;
    return index !== null && index >= 0 && index < source.numPages ? index + 1 : null;
  } catch {
    return null; // a destination pdf.js can't follow goes nowhere
  }
}

const toItems = (
  source: OutlineSource,
  nodes: readonly PdfOutlineNode[],
  depth: number,
): Promise<OutlineItem[]> =>
  Promise.all(
    nodes.map(async (node) => ({
      title: (node.title ?? "").replace(/\s+/g, " ").trim(),
      page: await destinationPage(source, node.dest),
      open: (node.count ?? 0) > 0,
      items: depth + 1 < MAX_DEPTH ? await toItems(source, node.items ?? [], depth + 1) : [],
    })),
  );

/** The PDF's outline, with each entry's page; empty if it has none. */
export async function loadOutline(source: OutlineSource): Promise<OutlineItem[]> {
  const nodes = await source.getOutline();
  return nodes ? toItems(source, nodes, 0) : [];
}
