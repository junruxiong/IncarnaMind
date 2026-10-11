/**
 * A drug label in SPL, the XML DailyMed keeps it in, as text by section: each
 * section and subsection of the label under the heading it has ("5 Warnings
 * and precautions", "5.1 Lactic acidosis"), nested as the label nests them.
 * Labels write a section's number in its title, so the number is kept as
 * written; one with none ("Uses", in an over-the-counter label) has none.
 * The highlights, which repeat the sections' text, and the product listing
 * and package label, which have no running text, are left out.
 */
import {
  child,
  descendants,
  elements,
  is,
  parseXml,
  textOf,
  type XmlElement,
  type XmlNode,
} from "../documents/formats/xml";
import { SourceError } from "./http";
import { compactDate, sentenceCase, tidy } from "./providers/common";
import type { TextSection } from "./types";

/** The sections whose heading is the same on every label, by LOINC code; the label's own title is kept as text. */
const BOXED_WARNING = "34066-1";

/** Sections with no running text: the product listing and the package label. */
const SKIPPED = new Set(["48780-1", "51945-4"]);

export interface SplLabel {
  title: string;
  sections: TextSection[];
  setId: string;
  version: string;
  /** ISO 8601 date the label took effect, when it says. */
  effective: string | null;
}

/** An element's text, with a space at a line break so words don't run together. */
function inline(node: XmlNode): string {
  if (typeof node === "string") return node;
  if (is(node, "br")) return " ";
  return node.children.map(inline).join("");
}

/** The paragraphs of a section's `text`: paragraphs, list items and table rows, one each. */
function paragraphsOf(text: XmlElement | undefined): string[] {
  const paragraphs: string[] = [];
  const add = (value: string) => {
    const line = tidy(value);
    if (line) paragraphs.push(line);
  };
  for (const node of text?.children ?? []) {
    if (typeof node === "string") add(node);
    else if (is(node, "renderMultiMedia")) continue;
    else if (is(node, "list")) {
      for (const item of descendants(node, "item")) add(`- ${inline(item)}`);
    } else if (is(node, "table")) {
      for (const row of descendants(node, "tr")) {
        const cells = elements(row)
          .map((cell) => tidy(inline(cell)))
          .filter(Boolean);
        add(cells.join(" | "));
      }
    } else add(inline(node));
  }
  return paragraphs;
}

/** "5 WARNINGS AND PRECAUTIONS" as "5 Warnings and precautions"; a title in mixed case as written. */
function headingOf(raw: string): string {
  const match = /^((?:\d+(?:\.\d+)*\.?\s+)?)(.*)$/s.exec(tidy(raw));
  return `${match?.[1] ?? ""}${sentenceCase(match?.[2] ?? "")}`.trim();
}

function walk(section: XmlElement, level: number, parent: string, into: TextSection[]): void {
  const code = child(section, "code")?.attrs.code ?? "";
  if (SKIPPED.has(code)) return;
  const title = child(section, "title");
  const own = paragraphsOf(child(section, "text"));
  const boxed = code === BOXED_WARNING;
  // A section with no title of its own is cited under the one around it.
  const heading = boxed ? "Boxed warning" : title ? headingOf(textOf(title)) : parent;
  const paragraphs = boxed && title ? [tidy(inline(title)), ...own] : own;
  if (paragraphs.length > 0 || heading !== parent) {
    into.push({ heading: heading || "Other", level, paragraphs });
  }
  const nested = elements(section)
    .filter((each) => is(each, "component"))
    .flatMap((component) => elements(component).filter((each) => is(each, "section")));
  for (const each of nested) walk(each, level + 1, heading, into);
}

/**
 * Reads an SPL document. Throws `SourceError` when it isn't readable or has
 * no running text.
 */
export function readSpl(xml: string): SplLabel {
  let root: XmlElement;
  try {
    root = parseXml(xml);
  } catch {
    throw new SourceError("service-error", "DailyMed's label isn't readable.");
  }
  const body = descendants(root, "structuredBody")[0];
  const sections: TextSection[] = [];
  if (body) {
    for (const component of elements(body).filter((each) => is(each, "component"))) {
      for (const section of elements(component).filter((each) => is(each, "section"))) {
        walk(section, 1, "", sections);
      }
    }
  }
  if (sections.every((section) => section.paragraphs.length === 0)) {
    throw new SourceError("not-open", "DailyMed's label has no text.");
  }
  const product = descendants(root, "manufacturedProduct").find((each) =>
    elements(each).some((inner) => is(inner, "manufacturedProduct")),
  );
  const inner = product ? child(product, "manufacturedProduct") : undefined;
  const name = inner
    ? tidy(textOf(child(inner, "name") ?? { name: "", attrs: {}, children: [] }))
    : "";
  return {
    title: sentenceCase(name) || "Drug label",
    sections,
    setId: child(root, "setId")?.attrs.root?.toLowerCase() ?? "",
    version: child(root, "versionNumber")?.attrs.value ?? "",
    effective: compactDate(child(root, "effectiveTime")?.attrs.value ?? ""),
  };
}
