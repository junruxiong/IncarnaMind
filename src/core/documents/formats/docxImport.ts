/**
 * A Word document (.docx) as a Mind's Blocks (#104), for importing it.
 *
 * What the editor can show and edit becomes Notes: headings, paragraphs with
 * bold, italic, underline, strikethrough and highlight, lists (numbered
 * through the paragraph or its style), tables with merged cells, inline
 * pictures and hyperlinks. Everything else is kept as it is, in place, as an
 * atom (`WORD_ATOM_NODE` in the text, `WORD_BLOCK_NODE` as a Block) that
 * names where its XML is in the package, so export can write the original
 * back (#105): fields, content controls, text boxes, shapes, SmartArt,
 * charts, equations, footnote marks, comments and other people's tracked
 * changes (read-only for now), and, invisible, bookmarks, ranges, section
 * breaks and page breaks. The first header and footer its sections name are
 * Blocks at the top and the foot. Typed numbering ("1.", "一、") and bold lines
 * stay text.
 *
 * Every node mapped from a paragraph or a table records where it came from
 * (`WordSource`, without its fingerprint, which the importer adds once the
 * node is in the Mind). Nothing is dropped without being counted
 * (`WordImportReport`): an element this reader doesn't know becomes an atom
 * of kind "other". Only proofing marks and Word's own layout hints
 * (`IGNORED`) are left out, as they hold no content.
 *
 * Untrusted input, as for reading Documents: the package's size limits and
 * the XML parser's refusal of a DOCTYPE apply (./zip, ./xml).
 */
import type { MarkJSON, NodeJSON } from "../../../shared/blockJson";
import {
  BLOCK_ID_ATTRIBUTE,
  IMAGE_NODE,
  WORD_ATOM_NODE,
  WORD_BLOCK_NODE,
  type WordAtomInfo,
  type WordAtomKind,
  type WordImportReport,
  type WordSource,
} from "../../api";
import { headingLevelOf, textIn } from "./docx";
import {
  NS_STRICT,
  NS_TRANSITIONAL,
  type Relationship,
  refusalOf,
  relationshipsOf,
  type WordPackage,
  WordRefusedError,
  type WordReview,
} from "./docxPackage";
import { dataUrlOf, imageTypeOf } from "./images";
import { child, descendants, elements, parseXml, walk, type XmlElement } from "./xml";

/** The largest picture made editable; a larger one is kept as is. */
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
/** The most picture bytes one import puts into a Mind; the rest are kept as is. */
const MAX_IMAGES_BYTES = 64 * 1024 * 1024;

/** How deep a style's `basedOn` chain is followed. */
const MAX_STYLE_DEPTH = 20;

/** A Word document read as a Mind's Blocks. */
export interface WordStructure {
  /** The top-level Blocks, in order, every node that takes one with its id. */
  blocks: NodeJSON[];
  /** Where each node mapped from the document came from, by its id (fingerprints aren't known yet). */
  sources: Map<string, Omit<WordSource, "f">>;
  report: WordImportReport;
}

/** Reads a Word package's main document as Blocks. Throws `WordRefusedError` for one that can't be read. */
export async function readWordStructure(
  pkg: WordPackage,
  review: WordReview,
): Promise<WordStructure> {
  try {
    const reader = await Reader.open(pkg, review);
    return reader.read();
  } catch (error) {
    throw refusalOf(error);
  }
}

// ---------------------------------------------------------------------------
// Parts the document refers to

interface ParagraphStyle {
  basedOn: string | null;
  level: number | null;
  numId: string | null;
  ilvl: number | null;
}

interface Styles {
  paragraph: Map<string, ParagraphStyle>;
  /** The style of a paragraph that names none. */
  defaultId: string | null;
}

function readStyles(xml: string | undefined): Styles {
  const paragraph = new Map<string, ParagraphStyle>();
  let defaultId: string | null = null;
  for (const style of xml ? elements(parseXml(xml)) : []) {
    if (style.name !== "w:style" || style.attrs["w:type"] !== "paragraph") continue;
    const id = style.attrs["w:styleId"];
    if (!id) continue;
    if (style.attrs["w:default"] === "1" || style.attrs["w:default"] === "true") defaultId = id;
    const numPr = child(child(style, "w:pPr") ?? style, "w:numPr");
    const ilvl = numPr && child(numPr, "w:ilvl")?.attrs["w:val"];
    paragraph.set(id, {
      basedOn: child(style, "w:basedOn")?.attrs["w:val"] ?? null,
      level: headingLevelOf(style),
      numId: (numPr && child(numPr, "w:numId")?.attrs["w:val"]) ?? null,
      ilvl: ilvl === undefined || ilvl === null ? null : Number(ilvl),
    });
  }
  return { paragraph, defaultId };
}

/** The styles a style is based on, itself first, as far as `MAX_STYLE_DEPTH`, never twice. */
function styleChain(styles: Styles, id: string | null): ParagraphStyle[] {
  const chain: ParagraphStyle[] = [];
  const seen = new Set<string>();
  for (let at = id; at && !seen.has(at) && chain.length < MAX_STYLE_DEPTH; ) {
    seen.add(at);
    const style = styles.paragraph.get(at);
    if (!style) break;
    chain.push(style);
    at = style.basedOn;
  }
  return chain;
}

interface ListLevel {
  format: string;
  start: number;
}

interface Numbering {
  /** abstractNumId → level → its format and start. */
  abstracts: Map<string, Map<number, ListLevel>>;
  /** numId → its abstract numbering and its overrides by level. */
  nums: Map<string, { abstract: string; overrides: Map<number, Partial<ListLevel>> }>;
}

function readLevel(lvl: XmlElement): Partial<ListLevel> {
  const format = child(lvl, "w:numFmt")?.attrs["w:val"];
  const start = child(lvl, "w:start")?.attrs["w:val"];
  return {
    ...(format !== undefined && { format }),
    ...(start !== undefined && Number.isFinite(Number(start)) && { start: Number(start) }),
  };
}

function readNumbering(xml: string | undefined, styles: Styles): Numbering {
  const abstracts = new Map<string, Map<number, ListLevel>>();
  const nums: Numbering["nums"] = new Map();
  const styleLinks = new Map<string, string>();
  for (const element of xml ? elements(parseXml(xml)) : []) {
    if (element.name === "w:abstractNum") {
      const id = element.attrs["w:abstractNumId"];
      if (id === undefined) continue;
      const levels = new Map<number, ListLevel>();
      for (const lvl of elements(element).filter((each) => each.name === "w:lvl")) {
        const level = readLevel(lvl);
        levels.set(Number(lvl.attrs["w:ilvl"] ?? 0), {
          format: level.format ?? "decimal",
          start: level.start ?? 1,
        });
      }
      abstracts.set(id, levels);
      const link = child(element, "w:numStyleLink")?.attrs["w:val"];
      if (link) styleLinks.set(id, link);
    } else if (element.name === "w:num") {
      const id = element.attrs["w:numId"];
      const abstract = child(element, "w:abstractNumId")?.attrs["w:val"];
      if (id === undefined || abstract === undefined) continue;
      const overrides = new Map<number, Partial<ListLevel>>();
      for (const override of elements(element).filter((each) => each.name === "w:lvlOverride")) {
        const lvl = child(override, "w:lvl");
        const start = child(override, "w:startOverride")?.attrs["w:val"];
        overrides.set(Number(override.attrs["w:ilvl"] ?? 0), {
          ...(lvl ? readLevel(lvl) : {}),
          ...(start !== undefined && Number.isFinite(Number(start)) && { start: Number(start) }),
        });
      }
      nums.set(id, { abstract, overrides });
    }
  }
  // A numbering that only links to a style's: its levels are the style's numbering's.
  for (const [abstract, styleId] of styleLinks) {
    const numId = styleChain(styles, styleId).find((style) => style.numId)?.numId;
    const linked = numId ? nums.get(numId)?.abstract : undefined;
    const levels = linked ? abstracts.get(linked) : undefined;
    if (levels && (abstracts.get(abstract)?.size ?? 0) === 0) abstracts.set(abstract, levels);
  }
  return { abstracts, nums };
}

/** A comment's author, date and text. */
interface Comment {
  author: string;
  date: string;
  text: string;
}

/** The text of a part, or of an element in it: its paragraphs, a line each; a table row's cells by tabs. */
function blockText(container: XmlElement): string {
  const lines: string[] = [];
  /** A cell's paragraphs, read with its row. */
  const read = new Set<XmlElement>();
  for (const element of walk(
    container,
    (each) => each.name !== "w:p" && each.name !== "mc:Fallback",
  )) {
    if (element.name === "w:p" && !read.has(element)) lines.push(textIn(element).trim());
    else if (element.name === "w:tr") {
      const cells = elements(element).filter((each) => each.name === "w:tc");
      const row = cells.map((cell) => {
        const paragraphs = elements(cell).filter((each) => each.name === "w:p");
        for (const paragraph of paragraphs) read.add(paragraph);
        return paragraphs.map((paragraph) => textIn(paragraph).trim()).join(" ");
      });
      lines.push(row.join("\t"));
    }
  }
  return lines.filter(Boolean).join("\n");
}

/** The text of an equation, as its runs spell it, without what a tracked change deleted. */
function mathText(node: XmlElement): string {
  let text = "";
  for (const element of walk(node, (each) => each.name !== "w:del")) {
    if (element.name === "m:t") {
      text += element.children.filter((part) => typeof part === "string").join("");
    }
  }
  return text;
}

/** Deleted text, which Word keeps in `w:delText`. */
const deletedText = (node: XmlElement) =>
  descendants(node, "w:delText")
    .map((each) => each.children.filter((part) => typeof part === "string").join(""))
    .join("");

/** An instruction's field type: its first word, or two for an add-in's ("ADDIN ZOTERO_ITEM"). */
function fieldType(instruction: string): string {
  const words = instruction.trim().split(/\s+/);
  const first = (words[0] ?? "").toUpperCase();
  return first === "ADDIN" && words[1] ? `ADDIN ${words[1].toUpperCase()}` : first;
}

/** Whether a toggle in run properties (`w:b`, `w:i`…) is on. */
const isOn = (element: XmlElement | undefined) =>
  element !== undefined && !["false", "0", "off"].includes(element.attrs["w:val"] ?? "true");

/** The editor's marks for a run's own formatting. */
function runMarks(rPr: XmlElement | undefined): MarkJSON[] {
  if (!rPr) return [];
  const marks: MarkJSON[] = [];
  if (isOn(child(rPr, "w:b"))) marks.push({ type: "bold" });
  if (isOn(child(rPr, "w:i"))) marks.push({ type: "italic" });
  const underline = child(rPr, "w:u")?.attrs["w:val"];
  if (child(rPr, "w:u") && underline !== "none") marks.push({ type: "underline" });
  if (isOn(child(rPr, "w:strike")) || isOn(child(rPr, "w:dstrike"))) {
    marks.push({ type: "strike" });
  }
  const highlight = child(rPr, "w:highlight")?.attrs["w:val"];
  if (highlight && highlight !== "none") marks.push({ type: "highlight" });
  return marks;
}

/** A link mark, with the attributes the editor stores for one. */
const linkMark = (href: string): MarkJSON => ({
  type: "link",
  attrs: { href, target: "_blank", rel: "noopener noreferrer nofollow", class: null, title: null },
});

/** Elements that hold no content: proofing marks, and where Word last laid out a page. */
const IGNORED = new Set(["w:proofErr", "w:lastRenderedPageBreak", "w:noProof"]);

/** Properties of what holds them (a cell, a row, a table, a control): not content. */
const PROPERTIES = new Set([
  "w:tcPr",
  "w:trPr",
  "w:tblPr",
  "w:tblGrid",
  "w:tblPrEx",
  "w:sdtPr",
  "w:sdtEndPr",
]);

/** Inline containers that are only wrappers: their content is read as if they weren't there. */
const TRANSPARENT = new Set(["w:smartTag", "w:customXml", "w:dir", "w:bdo"]);

/** Marks that keep a place for something that spans text: a comment's range, a permission's, a move's. */
const RANGES = new Set([
  "w:commentRangeStart",
  "w:commentRangeEnd",
  "w:permStart",
  "w:permEnd",
  "w:moveFromRangeStart",
  "w:moveFromRangeEnd",
  "w:moveToRangeStart",
  "w:moveToRangeEnd",
  "w:customXmlInsRangeStart",
  "w:customXmlInsRangeEnd",
  "w:customXmlDelRangeStart",
  "w:customXmlDelRangeEnd",
  "w:customXmlMoveFromRangeStart",
  "w:customXmlMoveFromRangeEnd",
  "w:customXmlMoveToRangeStart",
  "w:customXmlMoveToRangeEnd",
]);

/** Tracked changes, which are kept as is: their fields don't open or close one outside them. */
const CHANGES = new Set(["w:ins", "w:del", "w:moveFrom", "w:moveTo"]);

/** Date parts Word writes as elements of their own, in old documents. */
const DATE_PARTS = new Set([
  "w:dayShort",
  "w:dayLong",
  "w:monthShort",
  "w:monthLong",
  "w:yearShort",
  "w:yearLong",
]);

/** EMUs (DrawingML's unit) in a CSS pixel. */
const EMU_PER_PIXEL = 9525;

interface OpenField {
  start: XmlElement;
  instruction: string;
  result: string;
  separated: boolean;
}

interface ListInfo {
  abstract: string;
  numId: string;
  level: number;
  ordered: boolean;
  number: number;
}

/** A Block mapped from the document, with its list numbering if it is a list's item. */
interface Item {
  node: NodeJSON;
  list: ListInfo | null;
}

const newId = () => crypto.randomUUID();

const paragraphNode = (type: "paragraph" | "heading", attrs: Record<string, unknown>) => ({
  type,
  attrs: { [BLOCK_ID_ATTRIBUTE]: newId(), ...attrs },
  content: [] as NodeJSON[],
});

// ---------------------------------------------------------------------------
// The reader

class Reader {
  private readonly ordinals = new Map<XmlElement, number>();
  private readonly duplicateParaIds = new Set<string>();
  private readonly sources = new Map<string, Omit<WordSource, "f">>();
  private readonly report: WordImportReport;
  /** How deep in fields the text being read is, carried from one paragraph to the next. */
  private carry = 0;
  private fieldDepth = 0;
  private field: OpenField | null = null;
  /** Body-level marks (bookmarks, ranges) waiting for the next paragraph to hold them. */
  private pending: NodeJSON[] = [];
  /** The content of the paragraph read last, which takes marks no paragraph follows. */
  private lastContent: NodeJSON[] | null = null;
  private readonly noteNumbers = { footnote: 0, endnote: 0 };
  /** abstractNumId → the number each level has reached; numIds seen, for their start overrides. */
  private readonly counters = new Map<string, number[]>();
  private readonly numIdsSeen = new Set<string>();

  private constructor(
    private readonly document: XmlElement,
    private readonly rels: Map<string, Relationship>,
    private readonly styles: Styles,
    private readonly numbering: Numbering,
    private readonly comments: Map<string, Comment>,
    private readonly notes: { footnote: Map<string, string>; endnote: Map<string, string> },
    private readonly images: Map<string, string>,
    /** The text of each header and footer part. */
    private readonly partTexts: Map<string, string>,
    review: WordReview,
  ) {
    this.report = {
      paragraphs: 0,
      headings: 0,
      listItems: 0,
      tables: 0,
      images: 0,
      links: 0,
      kept: {},
      changes: review.changes,
      comments: review.comments,
      sections: 0,
    };
    this.numberElements(document);
  }

  static async open(pkg: WordPackage, review: WordReview): Promise<Reader> {
    const { zip, part } = pkg;
    const documentXml = await zip.readText(part);
    if (documentXml === undefined) throw new WordRefusedError("damaged", "No document part.");
    const document = parseXml(documentXml);
    const namespace = document.attrs["xmlns:w"];
    if (namespace === NS_STRICT) throw new WordRefusedError("strict", "Strict Open XML.");
    if (namespace !== undefined && namespace !== NS_TRANSITIONAL) {
      throw new WordRefusedError("damaged", "The document isn't WordprocessingML.");
    }
    const rels = await relationshipsOf(zip, part);
    const partOf = (suffix: string) =>
      [...rels.values()].find((rel) => !rel.external && rel.type.endsWith(suffix))?.target;
    const read = async (suffix: string) => {
      const target = partOf(suffix);
      return target && zip.has(target) ? zip.readText(target) : undefined;
    };
    const styles = readStyles(await read("/styles"));
    const numbering = readNumbering(await read("/numbering"), styles);
    const comments = new Map<string, Comment>();
    const commentsXml = await read("/comments");
    for (const comment of commentsXml ? elements(parseXml(commentsXml)) : []) {
      const id = comment.attrs["w:id"];
      if (comment.name !== "w:comment" || id === undefined) continue;
      comments.set(id, {
        author: comment.attrs["w:author"]?.trim() ?? "",
        date: comment.attrs["w:date"] ?? "",
        text: blockText(comment),
      });
    }
    const notes = { footnote: new Map<string, string>(), endnote: new Map<string, string>() };
    for (const kind of ["footnote", "endnote"] as const) {
      const xml = await read(`/${kind}s`);
      for (const note of xml ? elements(parseXml(xml)) : []) {
        const id = note.attrs["w:id"];
        if (note.name !== `w:${kind}` || id === undefined || note.attrs["w:type"]) continue;
        notes[kind].set(id, blockText(note));
      }
    }
    const images = new Map<string, string>();
    let budget = MAX_IMAGES_BYTES;
    for (const [id, rel] of rels) {
      const type = imageTypeOf(rel.target);
      if (rel.external || !rel.type.endsWith("/image") || !type || !zip.has(rel.target)) continue;
      const raw = zip.raw(rel.target);
      if (!raw || raw.size > Math.min(MAX_IMAGE_BYTES, budget)) continue;
      const data = await zip.read(rel.target, MAX_IMAGE_BYTES);
      if (!data) continue;
      budget -= data.length;
      images.set(id, dataUrlOf(data, type));
    }
    const partTexts = new Map<string, string>();
    for (const rel of rels.values()) {
      if (rel.external || !/\/(header|footer)$/.test(rel.type) || !zip.has(rel.target)) continue;
      const xml = await zip.readText(rel.target);
      partTexts.set(rel.target, xml ? blockText(parseXml(xml)) : "");
    }
    return new Reader(
      document,
      rels,
      styles,
      numbering,
      comments,
      notes,
      images,
      partTexts,
      review,
    );
  }

  /** Gives every element its place among the elements of its name, in document order. */
  private numberElements(root: XmlElement): void {
    const counts = new Map<string, number>();
    const paraIds = new Set<string>();
    for (const element of walk(root)) {
      const count = counts.get(element.name) ?? 0;
      counts.set(element.name, count + 1);
      this.ordinals.set(element, count);
      if (element.name === "w:p") {
        const paraId = element.attrs["w14:paraId"];
        if (paraId && paraIds.has(paraId)) this.duplicateParaIds.add(paraId);
        else if (paraId) paraIds.add(paraId);
      }
    }
  }

  private place(element: XmlElement): string {
    return `${element.name}#${this.ordinals.get(element) ?? 0}`;
  }

  private sourceOf(element: XmlElement): Omit<WordSource, "f"> {
    const paraId = element.attrs["w14:paraId"];
    return paraId && !this.duplicateParaIds.has(paraId)
      ? { n: this.place(element), p: paraId }
      : { n: this.place(element) };
  }

  private count(kind: WordAtomKind): void {
    this.report.kept[kind] = (this.report.kept[kind] ?? 0) + 1;
  }

  private atom(
    kind: WordAtomKind,
    source: string,
    { label = "", text = "", info }: { label?: string; text?: string; info?: WordAtomInfo },
  ): NodeJSON {
    this.count(kind);
    return {
      type: WORD_ATOM_NODE,
      attrs: { kind, label, text, source, ...(info && { info }) },
    };
  }

  private blockAtom(
    kind: WordAtomKind,
    source: string,
    { label = "", text = "", info }: { label?: string; text?: string; info?: WordAtomInfo },
  ): NodeJSON {
    this.count(kind);
    return {
      type: WORD_BLOCK_NODE,
      attrs: {
        [BLOCK_ID_ATTRIBUTE]: newId(),
        kind,
        label,
        text,
        source,
        ...(info && { info }),
      },
    };
  }

  read(): WordStructure {
    const body = child(this.document, "w:body");
    const blocks: NodeJSON[] = [];
    if (body) this.blocks(body, blocks, false);
    if (this.pending.length > 0) {
      if (this.lastContent) this.lastContent.push(...this.pending);
      else blocks.push({ ...paragraphNode("paragraph", {}), content: this.pending });
      this.pending = [];
    }
    const { header, footer } = this.pageFurniture(body);
    return {
      blocks: [...header, ...blocks, ...footer],
      sources: this.sources,
      report: this.report,
    };
  }

  // -------------------------------------------------------------------------
  // Blocks

  /** Reads a container's Blocks (the body's, a cell's) into `into`, grouping list items into lists. */
  private blocks(container: XmlElement, into: NodeJSON[], inCell: boolean): void {
    const items: Item[] = [];
    this.collect(container, items, inCell);
    into.push(...groupLists(items));
  }

  private collect(container: XmlElement, items: Item[], inCell: boolean): void {
    for (const element of elements(container)) {
      switch (element.name) {
        case "w:p":
          items.push(this.paragraph(element));
          break;
        case "w:tbl":
          items.push({ node: this.table(element, inCell), list: null });
          break;
        case "w:sdt":
          items.push({ node: this.blockControl(element), list: null });
          break;
        case "w:customXml":
          this.collect(element, items, inCell);
          break;
        case "w:bookmarkStart":
        case "w:bookmarkEnd":
          this.pending.push(this.bookmark(element));
          break;
        case "w:sectPr":
          // The last section's layout: kept with the body, counted (see `pageFurniture`).
          this.report.sections++;
          break;
        default:
          if (RANGES.has(element.name))
            this.pending.push(this.atom("range", this.place(element), {}));
          else if (!IGNORED.has(element.name) && !PROPERTIES.has(element.name)) {
            items.push({
              node: this.blockAtom("other", this.place(element), { text: blockText(element) }),
              list: null,
            });
          }
      }
    }
  }

  /** A block content control (a table of contents, a cover page, a form's section): kept as is. */
  private blockControl(sdt: XmlElement): NodeJSON {
    const gallery = descendants(child(sdt, "w:sdtPr") ?? sdt, "w:docPartGallery")[0]?.attrs[
      "w:val"
    ];
    const content = child(sdt, "w:sdtContent") ?? sdt;
    const toc = gallery === "Table of Contents";
    return this.blockAtom("control", this.place(sdt), {
      text: blockText(content),
      ...(toc && { info: { field: "TOC" } }),
    });
  }

  private headingLevel(styleId: string | null, pPr: XmlElement | undefined): number | null {
    for (const style of styleChain(this.styles, styleId)) {
      if (style.level !== null) return style.level;
    }
    const outline = pPr && child(pPr, "w:outlineLvl")?.attrs["w:val"];
    return outline !== undefined && Number(outline) < 9 ? Number(outline) + 1 : null;
  }

  /** The list a paragraph is an item of, through its own numbering or its style's; null for none. */
  private listOf(styleId: string | null, pPr: XmlElement | undefined): ListInfo | null {
    const own = pPr && child(pPr, "w:numPr");
    const chain = styleChain(this.styles, styleId);
    const numId =
      (own && child(own, "w:numId")?.attrs["w:val"]) ??
      chain.find((style) => style.numId !== null)?.numId ??
      null;
    if (numId === null || numId === "0") return null;
    const ownLevel = own && child(own, "w:ilvl")?.attrs["w:val"];
    const level = Math.min(
      8,
      Math.max(
        0,
        ownLevel !== undefined && ownLevel !== null
          ? Number(ownLevel)
          : (chain.find((style) => style.ilvl !== null)?.ilvl ?? 0),
      ),
    );
    const num = this.numbering.nums.get(numId);
    if (!num) return null;
    const base = this.numbering.abstracts.get(num.abstract)?.get(level);
    const override = num.overrides.get(level);
    const format = override?.format ?? base?.format ?? "decimal";
    if (format === "none") return null;

    const start = (at: number) =>
      num.overrides.get(at)?.start ??
      this.numbering.abstracts.get(num.abstract)?.get(at)?.start ??
      1;
    let counters = this.counters.get(num.abstract);
    if (!counters) {
      counters = Array.from({ length: 9 }, (_, at) => start(at) - 1);
      this.counters.set(num.abstract, counters);
    }
    if (!this.numIdsSeen.has(numId)) {
      this.numIdsSeen.add(numId);
      for (const [at, value] of num.overrides) {
        if (value.start !== undefined) counters[at] = value.start - 1;
      }
    }
    counters[level] = (counters[level] ?? 0) + 1;
    for (let deeper = level + 1; deeper < 9; deeper++) counters[deeper] = start(deeper) - 1;
    return {
      abstract: num.abstract,
      numId,
      level,
      ordered: format !== "bullet",
      number: counters[level] ?? 1,
    };
  }

  private paragraph(p: XmlElement): Item {
    const startDepth = this.carry;
    const endDepth = fieldBalance(p, startDepth);
    this.carry = endDepth;
    if (startDepth > 0 || endDepth > 0) {
      // Part of a field over several paragraphs, such as a table of contents not in a control.
      return {
        node: this.blockAtom("field", this.place(p), { text: textIn(p).trim() }),
        list: null,
      };
    }

    const pPr = child(p, "w:pPr");
    const styleId = (pPr && child(pPr, "w:pStyle")?.attrs["w:val"]) ?? this.styles.defaultId;
    const level = this.headingLevel(styleId, pPr);
    const list = level === null ? this.listOf(styleId, pPr) : null;
    const node =
      level === null
        ? paragraphNode("paragraph", {})
        : paragraphNode("heading", { level: Math.min(6, Math.max(1, level)) });
    node.content.push(...this.pending);
    this.pending = [];
    this.fieldDepth = 0;
    this.field = null;
    this.inline(p, [], node.content);
    const unfinished = this.takeField();
    if (unfinished) {
      node.content.push(
        this.atom("field", this.place(unfinished.start), {
          label: unfinished.result,
          info: { field: fieldType(unfinished.instruction) },
        }),
      );
    }
    const sectPr = pPr && child(pPr, "w:sectPr");
    if (sectPr) {
      this.report.sections++;
      node.content.push(this.atom("sectionBreak", this.place(sectPr), {}));
    }
    this.sources.set(node.attrs[BLOCK_ID_ATTRIBUTE] as string, this.sourceOf(p));
    this.lastContent = node.content;
    this.report.paragraphs++;
    if (level !== null) this.report.headings++;
    if (list) this.report.listItems++;
    return { node, list };
  }

  // -------------------------------------------------------------------------
  // Tables

  private table(tbl: XmlElement, nested: boolean): NodeJSON {
    const rows = rowsOf(tbl, (element) => this.marker(element));
    if (nested || rows.length === 0) {
      return this.blockAtom("table", this.place(tbl), { text: blockText(tbl) });
    }
    const gridColumns = descendants(child(tbl, "w:tblGrid") ?? tbl, "w:gridCol").length;

    interface Placed {
      cell: XmlElement;
      column: number;
      span: number;
      merge: "restart" | "continue" | null;
    }
    const placed: Placed[][] = rows.map((row) => {
      const trPr = child(row.element, "w:trPr");
      let column = Number(trPr && child(trPr, "w:gridBefore")?.attrs["w:val"]) || 0;
      return row.cells.map((cell) => {
        const tcPr = child(cell, "w:tcPr");
        const span = Math.max(1, Number(tcPr && child(tcPr, "w:gridSpan")?.attrs["w:val"]) || 1);
        const vMerge = tcPr && child(tcPr, "w:vMerge");
        const merge = vMerge
          ? vMerge.attrs["w:val"] === "restart"
            ? ("restart" as const)
            : ("continue" as const)
          : null;
        const at = { cell, column, span, merge };
        column += span;
        return at;
      });
    });
    const width = Math.max(
      gridColumns,
      ...placed.map((row) => row.reduce((end, at) => Math.max(end, at.column + at.span), 0)),
    );

    // Each cell that starts a vertical merge spans down over the "continue" cells below it.
    const rowspans = new Map<Placed, number>();
    const mergedInto = new Map<Placed, Placed>();
    placed.forEach((row, r) => {
      for (const at of row) {
        if (at.merge === "continue") continue;
        let span = 1;
        for (let below = r + 1; below < placed.length; below++) {
          const next = placed[below]?.find(
            (each) => each.column === at.column && each.merge === "continue",
          );
          if (!next) break;
          mergedInto.set(next, at);
          span++;
        }
        if (at.merge === "restart" || span > 1) rowspans.set(at, span);
      }
    });

    const occupied = placed.map(() => new Array<boolean>(width).fill(false));
    const content = new Map<Placed, NodeJSON[]>();
    placed.forEach((row) => {
      for (const at of row) {
        if (at.merge === "continue" && mergedInto.has(at)) continue;
        const blocks: NodeJSON[] = [];
        this.blocks(at.cell, blocks, true);
        content.set(at, blocks);
      }
    });
    // A merged cell's own content, which Word doesn't show, joins the cell it was merged into.
    for (const [cell, into] of mergedInto) {
      const empty =
        blockText(cell.cell).trim() === "" &&
        !["w:drawing", "w:pict", "w:object", "w:bookmarkStart"].some(
          (name) => descendants(cell.cell, name).length > 0,
        );
      if (!empty) this.blocks(cell.cell, content.get(into) ?? [], true);
    }

    const tableRows: NodeJSON[] = placed.map((row, r) => {
      const trPr = rows[r] && child(rows[r].element, "w:trPr");
      // A row Word repeats at the top of each page is the table's heading row.
      const cellType = isOn(trPr && child(trPr, "w:tblHeader")) ? "tableHeader" : "tableCell";
      const starts = new Map<number, Placed>();
      for (const at of row) {
        if (content.has(at)) starts.set(at.column, at);
      }
      const cells: NodeJSON[] = [];
      for (let column = 0; column < width; ) {
        const at = starts.get(column);
        if (at) {
          const rowspan = Math.max(1, Math.min(rowspans.get(at) ?? 1, placed.length - r));
          const colspan = Math.min(at.span, width - column);
          for (let down = r; down < r + rowspan; down++) {
            for (let across = column; across < column + colspan; across++) {
              const line = occupied[down];
              if (line) line[across] = true;
            }
          }
          const blocks = content.get(at) ?? [];
          cells.push({
            type: cellType,
            attrs: { colspan, rowspan },
            content: blocks.length > 0 ? blocks : [paragraphNode("paragraph", {})],
          });
          column += colspan;
        } else if (occupied[r]?.[column]) {
          column++;
        } else {
          // A place no cell fills (gridBefore, gridAfter, a short row): an empty cell keeps the table whole.
          let end = column + 1;
          while (end < width && !occupied[r]?.[end] && !starts.has(end)) end++;
          for (let across = column; across < end; across++) {
            const line = occupied[r];
            if (line) line[across] = true;
          }
          cells.push({
            type: cellType,
            attrs: { colspan: end - column, rowspan: 1 },
            content: [paragraphNode("paragraph", {})],
          });
          column = end;
        }
      }
      return { type: "tableRow", content: cells };
    });

    const id = newId();
    this.sources.set(id, { n: this.place(tbl) });
    this.report.tables++;
    return { type: "table", attrs: { [BLOCK_ID_ATTRIBUTE]: id }, content: tableRows };
  }

  // -------------------------------------------------------------------------
  // Text

  /** Reads inline content (a paragraph's, a hyperlink's) into `out`, each text with `marks`. */
  private inline(container: XmlElement, marks: MarkJSON[], out: NodeJSON[]): void {
    for (const element of elements(container)) {
      if (this.field) {
        this.inField(element, out);
        continue;
      }
      switch (element.name) {
        case "w:pPr":
          break;
        case "w:r":
          this.run(element, marks, out);
          break;
        case "w:hyperlink": {
          const href = this.hrefOf(element);
          if (href) this.report.links++;
          this.inline(element, href ? [...marks, linkMark(href)] : marks, out);
          break;
        }
        case "w:fldSimple": {
          const instruction = element.attrs["w:instr"] ?? "";
          out.push(
            this.atom("field", this.place(element), {
              label: textIn(element),
              info: { field: fieldType(instruction) },
            }),
          );
          break;
        }
        case "w:sdt":
          out.push(
            this.atom("control", this.place(element), {
              label: textIn(child(element, "w:sdtContent") ?? element),
            }),
          );
          break;
        case "w:ins":
        case "w:moveTo":
          out.push(this.change("insertion", element));
          break;
        case "w:del":
        case "w:moveFrom":
          out.push(this.change("deletion", element));
          break;
        case "w:bookmarkStart":
        case "w:bookmarkEnd":
          out.push(this.bookmark(element));
          break;
        case "m:oMath":
          out.push(this.atom("equation", this.place(element), { label: mathText(element) }));
          break;
        case "m:oMathPara":
          out.push(
            this.atom("equation", this.place(element), {
              label: mathText(element),
              info: { display: true },
            }),
          );
          break;
        default:
          if (TRANSPARENT.has(element.name)) this.inline(element, marks, out);
          else if (RANGES.has(element.name)) out.push(this.atom("range", this.place(element), {}));
          else if (!IGNORED.has(element.name)) {
            out.push(this.atom("other", this.place(element), { label: textIn(element) }));
          }
      }
    }
  }

  private hrefOf(hyperlink: XmlElement): string | null {
    const id = hyperlink.attrs["r:id"];
    const target = id ? this.rels.get(id) : undefined;
    if (target?.external) {
      const anchor = hyperlink.attrs["w:anchor"];
      return anchor ? `${target.target}#${anchor}` : target.target;
    }
    const anchor = hyperlink.attrs["w:anchor"];
    return anchor ? `#${anchor}` : null;
  }

  /** Something between a table's rows or cells: a bookmark or a range waits for the next paragraph. */
  private marker(element: XmlElement): void {
    if (PROPERTIES.has(element.name) || IGNORED.has(element.name)) return;
    if (element.name === "w:bookmarkStart" || element.name === "w:bookmarkEnd") {
      this.pending.push(this.bookmark(element));
    } else if (RANGES.has(element.name)) {
      this.pending.push(this.atom("range", this.place(element), {}));
    } else {
      this.pending.push(this.atom("other", this.place(element), { label: textIn(element) }));
    }
  }

  private bookmark(element: XmlElement): NodeJSON {
    const name = element.attrs["w:name"];
    return this.atom("bookmark", this.place(element), name ? { info: { name } } : {});
  }

  private change(kind: "insertion" | "deletion", element: XmlElement): NodeJSON {
    const author = element.attrs["w:author"]?.trim();
    const date = element.attrs["w:date"];
    return this.atom(kind, this.place(element), {
      // A move's old place keeps its text in w:t, a deletion in w:delText.
      label:
        kind === "deletion" && element.name === "w:del" ? deletedText(element) : textIn(element),
      info: { ...(author && { author }), ...(date && { date }) },
    });
  }

  private run(run: XmlElement, marks: MarkJSON[], out: NodeJSON[]): void {
    const own = [...marks, ...runMarks(child(run, "w:rPr"))];
    for (const element of elements(run)) {
      if (element.name === "w:rPr") continue;
      if (element.name === "w:fldChar") {
        this.fieldChar(element, run, out);
        continue;
      }
      if (this.field) {
        this.fieldContent(element);
        continue;
      }
      this.runContent(element, own, out);
    }
  }

  private text(text: string, marks: MarkJSON[], out: NodeJSON[]): void {
    if (text) out.push(marks.length > 0 ? { type: "text", text, marks } : { type: "text", text });
  }

  private runContent(element: XmlElement, marks: MarkJSON[], out: NodeJSON[]): void {
    switch (element.name) {
      case "w:t":
        this.text(element.children.filter((part) => typeof part === "string").join(""), marks, out);
        break;
      case "w:tab":
      case "w:ptab":
        this.text("\t", marks, out);
        break;
      case "w:noBreakHyphen":
        this.text("‑", marks, out);
        break;
      case "w:softHyphen":
        this.text("­", marks, out);
        break;
      case "w:cr":
        out.push({ type: "hardBreak" });
        break;
      case "w:br": {
        const type = element.attrs["w:type"];
        if (type === "page" || type === "column") {
          out.push(this.atom("pageBreak", this.place(element), {}));
        } else out.push({ type: "hardBreak" });
        break;
      }
      case "w:drawing":
        out.push(this.drawing(element, this.place(element)));
        break;
      case "w:pict":
        out.push(this.vml(element, this.place(element)));
        break;
      case "w:object":
        out.push(this.object(element, this.place(element)));
        break;
      case "mc:AlternateContent": {
        // The same content twice (DrawingML and a VML fallback): read the first choice only.
        const choice = elements(element)[0];
        const inner = choice && elements(choice)[0];
        const source = this.place(element);
        if (inner?.name === "w:drawing") out.push(this.drawing(inner, source));
        else if (inner?.name === "w:pict") out.push(this.vml(inner, source));
        else if (inner?.name === "w:object") out.push(this.object(inner, source));
        else out.push(this.atom("other", source, { label: choice ? textIn(choice) : "" }));
        break;
      }
      case "w:footnoteReference":
      case "w:endnoteReference": {
        const kind = element.name === "w:footnoteReference" ? "footnote" : "endnote";
        const number = ++this.noteNumbers[kind];
        out.push(
          this.atom(kind, this.place(element), {
            label: String(number),
            text: this.notes[kind].get(element.attrs["w:id"] ?? "") ?? "",
          }),
        );
        break;
      }
      case "w:commentReference": {
        const comment = this.comments.get(element.attrs["w:id"] ?? "");
        out.push(
          this.atom("comment", this.place(element), {
            text: comment?.text ?? "",
            info: {
              ...(comment?.author && { author: comment.author }),
              ...(comment?.date && { date: comment.date }),
            },
          }),
        );
        break;
      }
      case "w:sym":
        out.push(this.atom("symbol", this.place(element), {}));
        break;
      case "w:pgNum":
        out.push(this.atom("field", this.place(element), { info: { field: "PAGE" } }));
        break;
      case "w:annotationRef":
      case "w:footnoteRef":
      case "w:endnoteRef":
      case "w:separator":
      case "w:continuationSeparator":
      case "w:delText":
      case "w:instrText":
      case "w:delInstrText":
        break;
      default:
        if (DATE_PARTS.has(element.name)) {
          out.push(this.atom("field", this.place(element), { info: { field: "DATE" } }));
        } else if (!IGNORED.has(element.name)) {
          out.push(this.atom("other", this.place(element), { label: textIn(element) }));
        }
    }
  }

  // -------------------------------------------------------------------------
  // Fields

  private fieldChar(element: XmlElement, run: XmlElement, out: NodeJSON[]): void {
    const type = element.attrs["w:fldCharType"];
    if (type === "begin") {
      if (this.fieldDepth === 0) {
        this.field = { start: run, instruction: "", result: "", separated: false };
      }
      this.fieldDepth++;
    } else if (type === "separate") {
      if (this.fieldDepth === 1 && this.field) this.field.separated = true;
    } else if (type === "end" && this.fieldDepth > 0) {
      this.fieldDepth--;
      const field = this.field;
      if (this.fieldDepth === 0 && field) {
        this.field = null;
        out.push(
          this.atom("field", `${this.place(field.start)}..${this.place(run)}`, {
            label: field.result,
            info: { field: fieldType(field.instruction) },
          }),
        );
      }
    }
  }

  /** The field still open, which the paragraph ended before its end: it is closed. */
  private takeField(): OpenField | null {
    const field = this.field;
    this.field = null;
    this.fieldDepth = 0;
    return field;
  }

  /** Something inside a run of a field: its instruction, or its result's text. */
  private fieldContent(element: XmlElement): void {
    const field = this.field;
    if (!field) return;
    if (element.name === "w:instrText") {
      if (this.fieldDepth === 1 && !field.separated) {
        field.instruction += element.children.filter((part) => typeof part === "string").join("");
      }
    } else if (field.separated) {
      if (element.name === "w:t") {
        field.result += element.children.filter((part) => typeof part === "string").join("");
      } else if (element.name === "w:tab") field.result += "\t";
    }
  }

  /**
   * An element inside a field, outside a run: part of the field, read only for
   * where the field ends. What follows its end in a container is read as usual.
   */
  private inField(element: XmlElement, out: NodeJSON[]): void {
    if (element.name === "w:r") {
      this.run(element, [], out);
      return;
    }
    if (element.name !== "w:hyperlink" && !TRANSPARENT.has(element.name)) return;
    const inner = elements(element);
    for (let index = 0; index < inner.length; index++) {
      if (!this.field) {
        const href = element.name === "w:hyperlink" ? this.hrefOf(element) : null;
        this.inline(
          { ...element, children: inner.slice(index) },
          href ? [linkMark(href)] : [],
          out,
        );
        return;
      }
      this.inField(inner[index] as XmlElement, out);
    }
  }

  // -------------------------------------------------------------------------
  // Drawings

  private drawing(drawing: XmlElement, source: string): NodeJSON {
    const frame = elements(drawing)[0];
    if (!frame) return this.atom("other", source, {});
    const inline = frame.name === "wp:inline";
    const docPr = child(frame, "wp:docPr");
    const alt = (docPr?.attrs.descr || docPr?.attrs.title || "").trim();
    const data = descendants(frame, "a:graphicData")[0];
    const uri = data?.attrs.uri ?? "";
    const boxText = () =>
      descendants(frame, "w:txbxContent")
        .map((box) => blockText(box))
        .filter(Boolean)
        .join("\n");
    if (uri.endsWith("/picture")) {
      const blip = descendants(data ?? frame, "a:blip")[0];
      const embed = blip?.attrs["r:embed"];
      const src = embed && !blip?.attrs["r:link"] ? this.images.get(embed) : undefined;
      if (inline && src) {
        const extent = child(frame, "wp:extent");
        const width = Math.round(Number(extent?.attrs.cx) / EMU_PER_PIXEL);
        const height = Math.round(Number(extent?.attrs.cy) / EMU_PER_PIXEL);
        this.report.images++;
        return {
          type: IMAGE_NODE,
          attrs: {
            src,
            alt,
            ...(width > 0 && { width }),
            ...(height > 0 && { height }),
            source,
          },
        };
      }
      return this.atom("picture", source, {
        label: alt,
        ...(!inline && { info: { display: true } }),
      });
    }
    const display = { display: true };
    if (/\/chart(ex)?$/.test(uri) || uri.includes("chartex")) {
      return this.atom("chart", source, { label: alt, info: display });
    }
    if (uri.endsWith("/diagram"))
      return this.atom("smartArt", source, { label: alt, info: display });
    const text = boxText();
    if (uri.endsWith("/wordprocessingShape") && descendants(frame, "wps:txbx").length > 0) {
      return this.atom("textBox", source, { label: alt, text, info: display });
    }
    return this.atom("shape", source, { label: alt, text, info: display });
  }

  /** A VML picture or shape (`w:pict`), as old documents and fallbacks have them. */
  private vml(pict: XmlElement, source: string): NodeJSON {
    const box = descendants(pict, "w:txbxContent");
    if (box.length > 0) {
      return this.atom("textBox", source, {
        text: box.map((each) => blockText(each)).join("\n"),
        info: { display: true },
      });
    }
    const image = descendants(pict, "v:imagedata")[0];
    if (image) {
      const shape = descendants(pict, "v:shape")[0];
      const alt = (shape?.attrs.alt || image.attrs["o:title"] || "").trim();
      const src = this.images.get(image.attrs["r:id"] ?? "");
      const floating = /position\s*:\s*absolute/i.test(shape?.attrs.style ?? "");
      if (src && !floating) {
        this.report.images++;
        return { type: IMAGE_NODE, attrs: { src, alt, source } };
      }
      return this.atom("picture", source, { label: alt, info: { display: true } });
    }
    return this.atom("shape", source, { info: { display: true } });
  }

  /** An embedded object (OLE): an old equation (Equation 3.0, MathType) or another program's content. */
  private object(object: XmlElement, source: string): NodeJSON {
    const program = descendants(object, "o:OLEObject")[0]?.attrs.ProgID ?? "";
    return this.atom(/^Equation\.|MathType/i.test(program) ? "equation" : "object", source, {});
  }

  // -------------------------------------------------------------------------
  // Headers and footers

  /**
   * The first header and footer the sections name, as Blocks for the top and
   * the foot of the Mind. Every header and footer part the sections use is counted,
   * whether shown or not: each is kept in the package as it is.
   */
  private pageFurniture(body: XmlElement | undefined): { header: NodeJSON[]; footer: NodeJSON[] } {
    const sections = body
      ? [
          ...descendants(body, "w:pPr").flatMap((pPr) => {
            const sectPr = child(pPr, "w:sectPr");
            return sectPr ? [sectPr] : [];
          }),
          ...elements(body).filter((each) => each.name === "w:sectPr"),
        ]
      : [];
    const references = (sectPr: XmlElement, kind: "header" | "footer") =>
      elements(sectPr).filter((each) => each.name === `w:${kind}Reference`);
    const furniture = (kind: "header" | "footer"): NodeJSON[] => {
      const parts = new Set(
        sections.flatMap((sectPr) =>
          references(sectPr, kind).flatMap((each) => {
            const target = this.rels.get(each.attrs["r:id"] ?? "")?.target;
            return target ? [target] : [];
          }),
        ),
      );
      if (parts.size > 0) this.report.kept[kind] = parts.size;
      // A section with none of its own goes on with the one before: the first that names one shows.
      const first =
        sections.map((sectPr) => references(sectPr, kind)).find((each) => each.length > 0) ?? [];
      const reference =
        first.find((each) => (each.attrs["w:type"] ?? "default") === "default") ?? first[0];
      const target = reference && this.rels.get(reference.attrs["r:id"] ?? "")?.target;
      if (!target) return [];
      return [
        {
          type: WORD_BLOCK_NODE,
          attrs: {
            [BLOCK_ID_ATTRIBUTE]: newId(),
            kind,
            label: "",
            text: this.partTexts.get(target) ?? "",
            source: target,
          },
        },
      ];
    };
    return { header: furniture("header"), footer: furniture("footer") };
  }
}

/** What a paragraph's text is read without looking inside: atoms of their own, and fallbacks. */
const OPAQUE = new Set([
  "w:pPr",
  "w:sdt",
  "w:fldSimple",
  "w:drawing",
  "w:pict",
  "w:object",
  "mc:AlternateContent",
  "m:oMath",
  "m:oMathPara",
]);

/**
 * How deep in fields the end of a paragraph is, from how deep its start is:
 * each field's begin goes one deeper and its end one back. Tracked changes,
 * text boxes and fallbacks aren't counted, as they are read apart.
 */
function fieldBalance(paragraph: XmlElement, depth: number): number {
  let at = depth;
  const walk = (node: XmlElement) => {
    for (const element of elements(node)) {
      if (element.name === "w:fldChar") {
        const type = element.attrs["w:fldCharType"];
        if (type === "begin") at++;
        else if (type === "end" && at > 0) at--;
      } else if (!CHANGES.has(element.name) && !OPAQUE.has(element.name)) {
        walk(element);
      }
    }
  };
  walk(paragraph);
  return at;
}

/** A table's rows and each row's cells, past the controls and custom XML that may wrap them. */
function rowsOf(
  tbl: XmlElement,
  other: (element: XmlElement) => void,
): { element: XmlElement; cells: XmlElement[] }[] {
  const rows: { element: XmlElement; cells: XmlElement[] }[] = [];
  const unwrap = (node: XmlElement, name: string, into: XmlElement[]) => {
    for (const element of elements(node)) {
      if (element.name === name) into.push(element);
      else if (element.name === "w:sdt")
        unwrap(child(element, "w:sdtContent") ?? element, name, into);
      else if (element.name === "w:customXml") unwrap(element, name, into);
      else other(element);
    }
  };
  const trs: XmlElement[] = [];
  unwrap(tbl, "w:tr", trs);
  for (const tr of trs) {
    const cells: XmlElement[] = [];
    unwrap(tr, "w:tc", cells);
    if (cells.length > 0) rows.push({ element: tr, cells });
  }
  return rows;
}

/**
 * Groups list items into the editor's lists: consecutive items of one
 * numbering, nested by level (a level deeper than the one before goes one
 * deeper only), numbered from where Word's numbering has got to.
 */
function groupLists(items: readonly Item[]): NodeJSON[] {
  const out: NodeJSON[] = [];
  const open: { list: NodeJSON; ordered: boolean; numId: string }[] = [];
  for (const { node, list } of items) {
    if (!list) {
      open.length = 0;
      out.push(node);
      continue;
    }
    const depth = Math.min(list.level, open.length);
    open.length = Math.min(open.length, depth + 1);
    const top = open[depth];
    if (top && (top.ordered !== list.ordered || (depth === 0 && top.numId !== list.numId))) {
      open.length = depth;
    }
    if (open.length === depth) {
      const created: NodeJSON = list.ordered
        ? {
            type: "orderedList",
            attrs: { [BLOCK_ID_ATTRIBUTE]: newId(), start: list.number },
            content: [],
          }
        : { type: "bulletList", attrs: { [BLOCK_ID_ATTRIBUTE]: newId() }, content: [] };
      const parent = open[depth - 1];
      if (parent) parent.list.content?.at(-1)?.content?.push(created);
      else out.push(created);
      open.push({ list: created, ordered: list.ordered, numId: list.numId });
    }
    open[depth]?.list.content?.push({ type: "listItem", content: [node] });
  }
  return out;
}
