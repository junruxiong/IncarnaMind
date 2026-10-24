/**
 * What the User selected in a viewer, as the core cites it (`citeSelection`,
 * #101). On a PDF, a part for each page: the words of its text layer as the
 * page's stored text has them (pdf.js's runs, a line break at each `<br>` it
 * puts after a line), and how many of the page's lines with text come before
 * and after them.
 */
import { createContext, useContext, useEffect } from "react";
import type { SelectedPart } from "../../../core/api";

/** How a view reads what is selected in it, part by part (`pdfSelection`), for its tab's selection menu. */
export type SelectionReader = (range: Range) => SelectedPart[];

export const SelectionReaderContext = createContext<(reader: SelectionReader | null) => void>(
  () => undefined,
);

/** Lends the view's reader to its tab's selection menu while the view is there. */
export function useSelectionReader(reader: SelectionReader): void {
  const lend = useContext(SelectionReaderContext);
  useEffect(() => {
    lend(reader);
    return () => lend(null);
  }, [lend, reader]);
}

const linesWithText = (lines: readonly string[]) =>
  lines.filter((line) => line.trim() !== "").length;

function pagePart(layer: Element, range: Range, unit: number): SelectedPart | null {
  let text = "";
  let start = -1;
  let end = -1;
  const walker = layer.ownerDocument.createTreeWalker(
    layer,
    NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT,
  );
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (node.nodeName === "BR") {
      text += "\n";
      continue;
    }
    if (node.nodeType !== Node.TEXT_NODE) continue;
    const data = (node as Text).data;
    if (range.intersectsNode(node)) {
      const from = node === range.startContainer ? range.startOffset : 0;
      const to = node === range.endContainer ? range.endOffset : data.length;
      if (from < to) {
        if (start < 0) start = text.length + from;
        end = text.length + to;
      }
    }
    text += data;
  }
  if (start < 0) return null;
  return {
    unit,
    text: text.slice(start, end),
    // The line the selection starts in, and the one it ends in, are its own.
    linesBefore: linesWithText(text.slice(0, start).split("\n").slice(0, -1)),
    linesAfter: linesWithText(text.slice(end).split("\n").slice(1)),
  };
}

/** The parts of `range` on a PDF's pages, drawn in `root`, in order; none if it holds no text there. */
export function pdfSelection(range: Range, root: Element): SelectedPart[] {
  const parts: SelectedPart[] = [];
  for (const page of root.querySelectorAll<HTMLElement>("section[data-page-number]")) {
    const layer = page.querySelector(".textLayer");
    if (!layer || !range.intersectsNode(layer)) continue;
    const part = pagePart(layer, range, Number(page.dataset.pageNumber));
    if (part && part.text.trim() !== "") parts.push(part);
  }
  return parts;
}

/** Where the selection ends on screen, for its menu: its last line's end. */
export function selectionEnd(range: Range): DOMRect {
  const lines = [...range.getClientRects()].filter((rect) => rect.width > 0 && rect.height > 0);
  const last = lines.at(-1) ?? range.getBoundingClientRect();
  return new DOMRect(last.right, last.top, 0, last.height);
}
