/**
 * The Code of Federal Regulations in XML, as text by section: the eCFR's
 * versioner gives a part (DIV5) or a section (DIV8), GovInfo's annual edition
 * a section granule (CFRGRANULE). Each section is headed as it is cited,
 * "240.10b-5 Employment of manipulative and deceptive devices", and each
 * paragraph it numbers is a section of its own (./provisions), so a quote is
 * checked against "§ 240.10b-5(b)". A part's authority and source come under
 * its own heading; a subpart's and a subject group's headings stay, without
 * text. Tables are kept a row to a line, as drug labels' are.
 */
import { elements, is, parseXml, type XmlElement, type XmlNode } from "../documents/formats/xml";
import { SourceError } from "./http";
import { tidy } from "./providers/common";
import {
  ITALIC_END,
  ITALIC_START,
  leadingMarkers,
  type ProvisionBlock,
  provisionSections,
} from "./provisions";
import type { TextSection } from "./types";

export interface CfrText {
  /** How the text names itself: "240.10b-5 Employment of manipulative and deceptive devices", "Part 243—Regulation FD". */
  heading: string;
  sections: TextSection[];
}

/** Notes and quoted text, read as written: their paragraphs number nothing in the provision. */
const QUOTED = new Set(["EXTRACT", "NOTE", "EDNOTE", "EFFDNOT", "CROSSREF", "FTNT"]);
/** Where a provision comes from: its authority and its history. */
const SOURCE_NOTES = new Set(["CITA", "SECAUTH", "AUTH", "SOURCE"]);
/** Pictures and page furniture, with no running text. */
const SKIPPED = new Set(["GPH", "GID", "IMG", "PRTPAGE", "STARS", "FDSYS"]);

/** Inline text, with the CFR's italics (<I>, <E T="03">) marked so an italic marker is known. */
function inline(node: XmlNode): string {
  if (typeof node === "string") return node;
  const inner = node.children.map(inline).join("");
  // "Authority:" runs into its text in the XML.
  if (is(node, "HED")) return `${inner} `;
  const italic = is(node, "I") || (is(node, "E") && node.attrs.T === "03");
  return italic ? `${ITALIC_START}${inner}${ITALIC_END}` : inner;
}

const plainText = (node: XmlNode) => tidy(inline(node).replace(/[]/g, ""));

/** "PART 243—REGULATION FD" as "Part 243—REGULATION FD": the CFR's capitals kept, the level named as cited. */
function divisionHeading(head: string): string {
  return tidy(head).replace(
    /^(PART|SUBPART|SUBCHAPTER|CHAPTER)\b/,
    (word) => `${word.slice(0, 1)}${word.slice(1).toLowerCase()}`,
  );
}

/** A table's rows, a line each, its cells separated by " | ". */
function tableRows(table: XmlElement): string[] {
  const rows: string[] = [];
  const walk = (element: XmlElement) => {
    for (const each of elements(element)) {
      if (is(each, "ROW") || is(each, "TR") || is(each, "BOXHD")) {
        const cells = elements(each).map(plainText).filter(Boolean);
        if (cells.length > 0) rows.push(cells.join(" | "));
      } else walk(each);
    }
  };
  walk(table);
  return rows;
}

function isTable(element: XmlElement): boolean {
  return is(element, "GPOTABLE") || is(element, "TABLE") || element.attrs.class === "gpotbl_div";
}

/** A section's blocks, in order: paragraphs with their markers, flush text, notes, tables. */
function blocksOf(
  element: XmlElement,
  into: ProvisionBlock[] = [],
  quoted = false,
): ProvisionBlock[] {
  for (const each of elements(element)) {
    const name = each.name.toUpperCase();
    if (SKIPPED.has(name) || name === "HEAD" || name === "SECTNO" || name === "SUBJECT") continue;
    if (isTable(each)) {
      for (const row of tableRows(each)) into.push({ kind: "plain", text: row, markers: [] });
    } else if (QUOTED.has(name) && elements(each).some((inner) => is(inner, "P"))) {
      blocksOf(each, into, true);
    } else if (name === "P" && !quoted) {
      const { markers, text } = leadingMarkers(tidy(inline(each)));
      into.push({ kind: "paragraph", text, markers });
    } else if (name.startsWith("FP")) {
      // "FP-2" is indented two levels; a plain FP sits at the section's margin.
      const depth = Number(/FP-?(\d)/.exec(each.attrs.SOURCE ?? "")?.[1] ?? 0);
      into.push({ kind: quoted ? "plain" : "flush", text: plainText(each), markers: [], depth });
    } else if (SOURCE_NOTES.has(name)) {
      into.push({ kind: "source", text: plainText(each), markers: [] });
    } else if (/^DIV\d$/.test(name)) {
      // A section's own appendix or division: read as written.
      blocksOf(each, into, true);
    } else {
      into.push({ kind: "plain", text: plainText(each), markers: [] });
    }
  }
  return into;
}

/** "§ 240.10b-5 Employment of manipulative and deceptive devices." → the number and the subject. */
function sectionParts(element: XmlElement): { cite: string; subject: string } {
  const number = elements(element).find((each) => is(each, "SECTNO"));
  if (number) {
    const subject = elements(element).find((each) => is(each, "SUBJECT"));
    return {
      cite: plainText(number).replace(/^§+\s*/, ""),
      subject: subject ? plainText(subject) : "",
    };
  }
  const head = plainText(elements(element).find((each) => is(each, "HEAD")) ?? "").replace(
    /^§+\s*/,
    "",
  );
  const cite = element.attrs.N ?? head.split(" ")[0] ?? "";
  return { cite, subject: head.startsWith(cite) ? head.slice(cite.length).trim() : head };
}

function readSection(element: XmlElement): TextSection[] {
  const { cite, subject } = sectionParts(element);
  const heading = tidy(`${cite} ${subject}`).replace(/\.$/, "");
  return provisionSections(heading, cite, 1, blocksOf(element));
}

/** An appendix or a division that isn't a section: its heading, and its text as written. */
function readPlain(heading: string, element: XmlElement): TextSection {
  return {
    heading,
    level: 1,
    paragraphs: blocksOf(element)
      .map((block) => block.text.replace(/\s+/g, " ").trim())
      .filter(Boolean),
  };
}

function walk(element: XmlElement, into: TextSection[]): void {
  const type = (element.attrs.TYPE ?? "").toUpperCase();
  if (is(element, "SECTION") || type === "SECTION") {
    into.push(...readSection(element));
    return;
  }
  const head = elements(element).find((each) => is(each, "HEAD"));
  if (type === "APPENDIX") {
    into.push(readPlain(plainText(head ?? ""), element));
    return;
  }
  if (/^DIV\d$/i.test(element.name) && head) {
    const own = elements(element).filter(
      (each) => !/^DIV\d$/i.test(each.name) && !is(each, "HEAD"),
    );
    into.push(readPlain(divisionHeading(plainText(head)), { ...element, children: own }));
  }
  for (const each of elements(element)) {
    if (/^DIV\d$/i.test(each.name) || is(each, "SECTION")) walk(each, into);
  }
}

/** Reads CFR XML. Throws `SourceError` when it can't be read or has no text. */
export function readCfr(xml: string): CfrText {
  let root: XmlElement;
  try {
    root = parseXml(xml);
  } catch {
    throw new SourceError("service-error", "The regulation's text isn't readable.");
  }
  const sections: TextSection[] = [];
  walk(root, sections);
  if (sections.every((section) => section.paragraphs.length === 0)) {
    throw new SourceError("not-open", "The regulation has no text.");
  }
  return { heading: sections[0]?.heading ?? "", sections };
}
