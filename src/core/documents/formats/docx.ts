/**
 * Word (.docx) → Units: one per heading section, labelled with the heading
 * path, in reading order (ADR-0011: Word is cited by section, not by page).
 * A section longer than `MAX_SECTION_CHARS` is split at paragraph ends into
 * parts. Footnotes and endnotes form one last Unit.
 *
 * Read: body paragraphs and tables, in order (a table row is one line, its
 * cells separated by tabs), runs, tabs, breaks, hyperlinks, insertions,
 * content controls, and one branch of `mc:AlternateContent`. Left out:
 * deleted text, field instructions, headers and footers, comments, and list
 * numbers ("1.", "a)"). Headings are found from styles.xml, by name ("heading
 * N") or outline level, so localised style ids work. Ported from the
 * office-formats spike.
 */
import { type TextUnit, UnitBuilder } from "../../../shared/units";
import { ExtractionError } from "./errors";
import { child, elements, local, parseXml, type XmlElement } from "./xml";
import { openPackage } from "./zip";

/** A section longer than this many characters is split into parts at paragraph ends. */
export const MAX_SECTION_CHARS = 6000;

/** Style id → heading level (from 1), from styles.xml: "heading N", or an outline level. */
function headingStyles(styles: XmlElement | undefined): Map<string, number> {
  const levels = new Map<string, number>();
  if (!styles) return levels;
  for (const style of elements(styles)) {
    if (style.name !== "w:style" || style.attrs["w:type"] !== "paragraph") continue;
    const id = style.attrs["w:styleId"];
    if (!id) continue;
    const name = (child(style, "w:name")?.attrs["w:val"] ?? "").toLowerCase();
    const heading = /^heading (\d)$/.exec(name);
    const outline = child(child(style, "w:pPr") ?? style, "w:outlineLvl")?.attrs["w:val"];
    if (heading) levels.set(id, Number(heading[1]));
    else if (name === "title") levels.set(id, 0);
    else if (outline !== undefined && Number(outline) < 9) levels.set(id, Number(outline) + 1);
  }
  return levels;
}

interface Paragraph {
  text: string;
  /** 0 for a Title, 1–9 for headings, null for body text. */
  level: number | null;
}

/** The text of run content. */
function runText(node: XmlElement): string {
  let text = "";
  for (const element of elements(node)) {
    switch (local(element.name)) {
      case "t":
        text += element.children.filter((part) => typeof part === "string").join("");
        break;
      case "tab":
        text += "\t";
        break;
      case "br":
      case "cr":
        text += "\n";
        break;
      case "noBreakHyphen":
        text += "-";
        break;
      case "AlternateContent": {
        // The same content twice (DrawingML and a VML fallback): read the first choice only.
        const choice = elements(element)[0];
        if (choice) text += runText(choice);
        break;
      }
      case "del":
      case "delText":
      case "instrText":
      case "pPr":
      case "rPr":
      case "footnoteReference":
      case "endnoteReference":
        break;
      default:
        // r, hyperlink, ins, smartTag, fldSimple, sdt, sdtContent, …
        text += runText(element);
    }
  }
  return text;
}

function readBlocks(container: XmlElement, styles: Map<string, number>, out: Paragraph[]): void {
  for (const element of elements(container)) {
    if (element.name === "w:p") {
      const pPr = child(element, "w:pPr");
      const styleId = pPr && child(pPr, "w:pStyle")?.attrs["w:val"];
      const outline = pPr && child(pPr, "w:outlineLvl")?.attrs["w:val"];
      const text = runText(element).trim();
      const level =
        styleId !== undefined && styles.has(styleId)
          ? (styles.get(styleId) as number)
          : outline !== undefined && Number(outline) < 9
            ? Number(outline) + 1
            : null;
      if (text) out.push({ text, level });
    } else if (element.name === "w:tbl") {
      for (const row of elements(element).filter((each) => each.name === "w:tr")) {
        const cells = elements(row)
          .filter((each) => each.name === "w:tc")
          .map((cell) => {
            const inner: Paragraph[] = [];
            readBlocks(cell, styles, inner);
            return inner.map((paragraph) => paragraph.text).join(" ");
          });
        const text = cells.join("\t").trim();
        if (text) out.push({ text, level: null });
      }
    } else if (element.name === "w:sdt") {
      const content = child(element, "w:sdtContent");
      if (content) readBlocks(content, styles, out);
    }
  }
}

/** A heading of the Document, for the viewer's outline. */
export interface DocxHeading {
  /** From 1. */
  level: number;
  text: string;
  /** The Unit the heading starts. */
  unit: number;
}

export interface DocxResult {
  units: TextUnit[];
  headings: DocxHeading[];
  /** The ids of the paragraph styles that are headings: the viewer finds headings by them. */
  headingStyles: string[];
}

export async function extractDocx(bytes: Uint8Array): Promise<DocxResult> {
  const zip = openPackage(bytes);
  const documentXml = await zip.readText("word/document.xml");
  if (documentXml === undefined) {
    throw new ExtractionError("unreadable", "Not a Word document: it has no word/document.xml.");
  }
  const stylesXml = await zip.readText("word/styles.xml");
  const styles = headingStyles(stylesXml ? parseXml(stylesXml) : undefined);
  const body = child(parseXml(documentXml), "w:body");
  const paragraphs: Paragraph[] = [];
  if (body) readBlocks(body, styles, paragraphs);

  const units: TextUnit[] = [];
  const headings: DocxHeading[] = [];
  const path: string[] = [];
  let builder = new UnitBuilder();
  let part = 1;
  const flush = () => {
    if (!builder.empty) {
      units.push({
        page: units.length + 1,
        kind: "section",
        label: { path: path.filter(Boolean), ...(part > 1 ? { part } : {}) },
        text: builder.text,
        anchors: builder.anchors,
      });
    }
    builder = new UnitBuilder();
  };

  paragraphs.forEach((paragraph, index) => {
    if (paragraph.level !== null && paragraph.level > 0) {
      flush();
      path.length = paragraph.level - 1;
      path[paragraph.level - 1] = paragraph.text;
      part = 1;
      headings.push({ level: paragraph.level, text: paragraph.text, unit: units.length + 1 });
    } else if (builder.text.length + paragraph.text.length > MAX_SECTION_CHARS && !builder.empty) {
      flush();
      part++;
    }
    builder.add(paragraph.text, `p${index + 1}`);
  });
  flush();

  // Footnotes and endnotes: one Unit at the end, each note anchored.
  const notes = new UnitBuilder();
  for (const [name, tag] of [
    ["word/footnotes.xml", "w:footnote"],
    ["word/endnotes.xml", "w:endnote"],
  ] as const) {
    const xml = await zip.readText(name);
    if (!xml) continue;
    for (const note of elements(parseXml(xml)).filter((each) => each.name === tag)) {
      if (note.attrs["w:type"]) continue; // separators
      const inner: Paragraph[] = [];
      readBlocks(note, styles, inner);
      notes.add(
        inner.map((paragraph) => paragraph.text).join("\n"),
        `${local(tag)}${note.attrs["w:id"]}`,
      );
    }
  }
  if (!notes.empty) {
    units.push({
      page: units.length + 1,
      kind: "section",
      label: { path: [], notes: true },
      text: notes.text,
      anchors: notes.anchors,
    });
  }
  return {
    units,
    headings,
    headingStyles: [...styles].filter(([, level]) => level > 0).map(([id]) => id),
  };
}
