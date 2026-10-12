/**
 * A section of the United States Code as GovInfo's API gives it, in HTML,
 * as text by section: the section, headed as it is cited ("1983 Civil action
 * for deprivation of rights"), each subsection and paragraph it numbers as a
 * section of its own (./provisions), so a quote is checked against "§
 * 78j(b)", then its source credit and the editorial and statutory notes
 * under their own headings. Footnote marks and the page numbers kept in
 * comments are left out.
 */
import { parseHTML } from "linkedom";
import { SourceError } from "./http";
import { tidy } from "./providers/common";
import { leadingMarkers, type ProvisionBlock, provisionSections } from "./provisions";
import type { TextSection } from "./types";

/** As much of the DOM as this reads: core code doesn't build against the DOM's types. */
interface DomElement {
  readonly localName: string;
  readonly textContent: string | null;
  getAttribute(name: string): string | null;
  remove(): void;
  querySelectorAll(selector: string): ArrayLike<DomElement>;
}

export interface UsCodeText {
  /** "1983 Civil action for deprivation of rights". */
  heading: string;
  /** "1983". */
  section: string;
  sections: TextSection[];
  /** ISO 8601: the date the edition's text is current through, when it says. */
  currentThrough: string | null;
}

const text = (element: DomElement) => tidy(element.textContent ?? "");

/** "§1983. Civil action for deprivation of rights" → "1983", "Civil action for deprivation of rights". */
function sectionHead(head: string): { section: string; subject: string } {
  const match = /^§+\s*([0-9A-Za-z–-]+)\.?\s*(.*)$/s.exec(head);
  return { section: (match?.[1] ?? "").replace(/–/g, "-"), subject: match?.[2]?.trim() ?? head };
}

/** Reads a section's HTML. Throws `SourceError` when it has no section in it. */
export function readUsCode(html: string): UsCodeText {
  const { document } = parseHTML(html);
  const root = document as unknown as DomElement;
  // Footnote marks: "agreement 1 any manipulative" reads "agreement any manipulative".
  for (const mark of Array.from(root.querySelectorAll("sup"))) mark.remove();
  const nodes = Array.from(root.querySelectorAll("h3, h4, p, table"));
  const headElement = nodes.find((node) => node.getAttribute("class") === "section-head");
  if (!headElement) throw new SourceError("service-error", "GovInfo's text has no section in it.");
  const { section, subject } = sectionHead(text(headElement));
  const heading = tidy(`${section} ${subject}`);
  const blocks: ProvisionBlock[] = [];
  const notes: TextSection[] = [];
  for (const node of nodes.slice(nodes.indexOf(headElement) + 1)) {
    const kind = node.getAttribute("class") ?? "";
    const words = text(node);
    if (!words) continue;
    if (node.localName === "h4" && kind === "note-head") {
      // "Editorial Notes" and "Statutory Notes and Related Subsidiaries" hold the notes headed under them.
      const group = /^(?:editorial|statutory) notes/i.test(words);
      notes.push({ heading: words, level: group ? 1 : 2, paragraphs: [] });
    } else if (notes.length > 0) {
      notes.at(-1)?.paragraphs.push(words);
    } else if (kind === "source-credit") {
      blocks.push({ kind: "source", text: words, markers: [] });
    } else if (kind.startsWith("statutory-body-block")) {
      // "statutory-body-block-2em" is indented two levels; without one it sits at the section's margin.
      const depth = Number(/-(\d)em/.exec(kind)?.[1] ?? 0);
      blocks.push({ kind: "flush", text: words, markers: [], depth });
    } else if (kind.startsWith("statutory-body") || /-head$/.test(kind)) {
      const { markers } = leadingMarkers(words);
      blocks.push({ kind: "paragraph", text: words, markers });
    } else {
      blocks.push({ kind: "plain", text: words, markers: [] });
    }
  }
  const through = /currentthrough:(\d{4})(\d{2})(\d{2})/.exec(html);
  return {
    heading,
    section,
    sections: [...provisionSections(heading, section, 1, blocks), ...notes],
    currentThrough: through ? `${through[1]}-${through[2]}-${through[3]}` : null,
  };
}
