/**
 * A Mind as a Word document (.docx), the deliverable: Office Open XML
 * WordprocessingML, written here rather than with the `docx` package. The
 * export needs a fixed, small part of the format (styles, headings, lists,
 * tables, images and footnotes), and the package would bring five more
 * packages (jszip, xml-js, …) and about 17 MB for it. The ZIP container is
 * written with Node's zlib (./zip).
 *
 * Headings use Word's built-in heading styles, so they show in Word's
 * navigation pane; lists are Word lists; each Citation is a real Word
 * footnote; maths is Word's own equations (./math), or its LaTeX as text
 * where it can't be converted; code is monospaced paragraphs.
 */

import type { Block, Footnote, Image, Inline, Marks, TableCell } from "../mindText";
import { latexToOmml, MATH_NAMESPACE } from "./math";
import { zip } from "./zip";

export interface DocxOptions {
  /** The Mind's title, as the document's title. Empty: none. */
  title: string;
  labels: {
    /** Marks a Question, e.g. "Question:". */
    question: string;
    /** Follows an unverified Citation's source in its footnote, e.g. "[unverified]". */
    unverified: string;
  };
  /** The language Word checks the text in, e.g. "en-US". */
  language: string;
  paper: "letter" | "a4";
  /** When the document was made, ISO 8601. */
  created: string;
}

const NS = {
  w: "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
  r: "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
  wp: "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
  a: "http://schemas.openxmlformats.org/drawingml/2006/main",
  pic: "http://schemas.openxmlformats.org/drawingml/2006/picture",
  m: MATH_NAMESPACE,
  packageRelationships: "http://schemas.openxmlformats.org/package/2006/relationships",
  contentTypes: "http://schemas.openxmlformats.org/package/2006/content-types",
} as const;

const RELATIONSHIP = {
  officeDocument: `${NS.r}/officeDocument`,
  coreProperties:
    "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties",
  styles: `${NS.r}/styles`,
  settings: `${NS.r}/settings`,
  numbering: `${NS.r}/numbering`,
  footnotes: `${NS.r}/footnotes`,
  image: `${NS.r}/image`,
  hyperlink: `${NS.r}/hyperlink`,
} as const;

const MAIN_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml";

const XML_DECLARATION = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n';

/** Page sizes and margins, in twentieths of a point ("twips"): 1 inch margins. */
const PAPER = {
  letter: { width: 12240, height: 15840 },
  a4: { width: 11906, height: 16838 },
} as const;
const MARGIN = 1440;

/** One level of a list, or of a quote, indents this much (half an inch). */
const INDENT = 720;
/** A list item's number or bullet hangs this far left of its text. */
const HANGING = 360;
/** EMUs (DrawingML's unit) per twip, and per CSS pixel at 96 dpi. */
const EMU_PER_TWIP = 635;
const EMU_PER_PIXEL = 9525;

export function renderDocx(blocks: readonly Block[], options: DocxOptions): Uint8Array {
  const paper = PAPER[options.paper];
  const writer = new DocxWriter(options, paper.width - 2 * MARGIN);
  const title = options.title ? paragraph({ style: "Title" }, run(options.title)) : "";
  const body = writer.body(blocks);

  const document =
    `${XML_DECLARATION}<w:document xmlns:w="${NS.w}" xmlns:r="${NS.r}" xmlns:wp="${NS.wp}" xmlns:a="${NS.a}" xmlns:pic="${NS.pic}" xmlns:m="${NS.m}"><w:body>` +
    `${title}${body}` +
    `<w:sectPr><w:pgSz w:w="${paper.width}" w:h="${paper.height}"/><w:pgMar w:top="${MARGIN}" w:right="${MARGIN}" w:bottom="${MARGIN}" w:left="${MARGIN}" w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>` +
    "</w:body></w:document>";

  const relationships = [
    { id: "rIdStyles", type: RELATIONSHIP.styles, target: "styles.xml" },
    { id: "rIdSettings", type: RELATIONSHIP.settings, target: "settings.xml" },
    { id: "rIdNumbering", type: RELATIONSHIP.numbering, target: "numbering.xml" },
    { id: "rIdFootnotes", type: RELATIONSHIP.footnotes, target: "footnotes.xml" },
    ...writer.relationships,
  ];

  return zip([
    { name: "[Content_Types].xml", data: contentTypes() },
    {
      name: "_rels/.rels",
      data: relationshipsXml([
        { id: "rIdDocument", type: RELATIONSHIP.officeDocument, target: "word/document.xml" },
        { id: "rIdCore", type: RELATIONSHIP.coreProperties, target: "docProps/core.xml" },
      ]),
    },
    { name: "docProps/core.xml", data: coreProperties(options) },
    { name: "word/document.xml", data: document },
    { name: "word/_rels/document.xml.rels", data: relationshipsXml(relationships) },
    { name: "word/styles.xml", data: styles(options.language) },
    { name: "word/settings.xml", data: SETTINGS },
    { name: "word/numbering.xml", data: numbering(writer.lists) },
    { name: "word/footnotes.xml", data: footnotesXml(writer.footnotes) },
    ...writer.media.map((file) => ({ name: `word/media/${file.name}`, data: file.data })),
  ]);
}

interface Relationship {
  id: string;
  type: string;
  target: string;
  external?: boolean;
}

/** Where Blocks are being written. */
interface Context {
  /** The left indent of their text, in twips. */
  indent: number;
  /** The paragraph style of their text: quotes and list items have their own. */
  style: string | null;
  /** How deep in lists they are, from 0. */
  listDepth: number;
  /** All their text is bold, e.g. in a table's header row. */
  bold: boolean;
  /** The width available, in twips. */
  width: number;
}

interface List {
  ordered: boolean;
  level: number;
  start: number;
}

class DocxWriter {
  /** The `<w:footnote>` of each Citation, numbered from 1 in order. */
  readonly footnotes: string[] = [];
  readonly relationships: Relationship[] = [];
  readonly media: { name: string; data: Uint8Array }[] = [];
  /** Each list written: a Word numbering instance, numbered from 1. */
  readonly lists: List[] = [];
  private readonly links = new Map<string, string>();
  private drawings = 0;

  constructor(
    private readonly options: DocxOptions,
    private readonly textWidth: number,
  ) {}

  body(blocks: readonly Block[]): string {
    const xml = this.blocks(blocks, {
      indent: 0,
      style: null,
      listDepth: 0,
      bold: false,
      width: this.textWidth,
    });
    // The body's last element before its section properties must be a paragraph.
    return xml.endsWith("</w:p>") ? xml : `${xml}<w:p/>`;
  }

  private blocks(blocks: readonly Block[], context: Context): string {
    let xml = "";
    for (const block of blocks) {
      const next = this.block(block, context);
      // Two tables in a row would merge into one.
      if (xml.endsWith("</w:tbl>") && next.startsWith("<w:tbl>")) xml += "<w:p/>";
      xml += next;
    }
    return xml;
  }

  private block(block: Block, context: Context): string {
    const { indent } = context;
    switch (block.kind) {
      case "paragraph":
        return paragraph({ style: context.style, indent }, this.inline(block.content, context));
      case "heading":
        return paragraph(
          { style: `Heading${block.level}`, indent },
          this.inline(block.content, context),
        );
      case "question":
        return paragraph(
          { style: "Question", indent },
          run(`${this.options.labels.question} `, { bold: true }) +
            this.inline(block.content, context),
        );
      case "code": {
        const lines = block.code.split(/\r\n|\r|\n/);
        return lines
          .map((line, index) =>
            paragraph(
              { style: "Code", indent, spacingAfter: index === lines.length - 1 ? 160 : undefined },
              run(line),
            ),
          )
          .join("");
      }
      case "math": {
        // A display equation, or its LaTeX if it can't be one.
        const equation = latexToOmml(block.latex, true);
        return paragraph(
          { style: "Math", indent },
          equation ? `<m:oMathPara>${equation}</m:oMathPara>` : run(`$$${block.latex}$$`),
        );
      }
      case "list":
        return this.list(block, context);
      case "quote":
        return this.blocks(block.content, { ...context, indent: indent + INDENT, style: "Quote" });
      case "rule":
        return paragraph({ indent, bottomBorder: true }, "");
      case "table":
        return this.table(block.rows, context);
      case "image":
        return paragraph({ style: context.style, indent }, this.image(block.image, context));
    }
  }

  /** A Word list: the first paragraph of each item carries its number or bullet. */
  private list(block: Extract<Block, { kind: "list" }>, context: Context): string {
    const level = Math.min(context.listDepth, 8);
    this.lists.push({ ordered: block.ordered, level, start: block.start });
    const numbering = { id: this.lists.length, level };
    const textIndent = context.indent + INDENT * (level + 1);
    const numbered = (content: string) =>
      paragraph(
        { style: "ListParagraph", numbering, indent: textIndent, hanging: HANGING },
        content,
      );

    let xml = "";
    for (const item of block.items) {
      const [first, ...rest] = item;
      if (first?.kind === "paragraph") xml += numbered(this.inline(first.content, context));
      else xml += numbered("");
      for (const child of first?.kind === "paragraph" ? rest : item) {
        xml +=
          child.kind === "list"
            ? this.list(child, { ...context, listDepth: context.listDepth + 1 })
            : this.block(child, { ...context, indent: textIndent, style: "ListParagraph" });
      }
    }
    return xml;
  }

  private table(rows: readonly TableCell[][], context: Context): string {
    if (rows.length === 0) return "";
    const columns = Math.max(
      1,
      ...rows.map((row) => row.reduce((count, cell) => count + cell.colspan, 0)),
    );
    const column = Math.floor((context.width - context.indent) / columns);
    const cell = (content: string, span: number, header: boolean) =>
      `<w:tc><w:tcPr><w:tcW w:w="${column * span}" w:type="dxa"/>${span > 1 ? `<w:gridSpan w:val="${span}"/>` : ""}${header ? '<w:shd w:val="clear" w:color="auto" w:fill="F3F4F6"/>' : ""}</w:tcPr>${content}</w:tc>`;

    let xml = `<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="${column * columns}" w:type="dxa"/>${context.indent ? `<w:tblInd w:w="${context.indent}" w:type="dxa"/>` : ""}</w:tblPr>`;
    xml += `<w:tblGrid>${`<w:gridCol w:w="${column}"/>`.repeat(columns)}</w:tblGrid>`;
    for (const row of rows) {
      const header = row.every((each) => each.header);
      xml += `<w:tr>${header ? "<w:trPr><w:tblHeader/></w:trPr>" : ""}`;
      let filled = 0;
      for (const each of row) {
        const content = this.blocks(each.content, {
          indent: 0,
          style: null,
          listDepth: 0,
          bold: each.header,
          width: column * each.colspan,
        });
        // A cell ends with a paragraph.
        xml += cell(
          content.endsWith("</w:p>") ? content : `${content}<w:p/>`,
          each.colspan,
          each.header,
        );
        filled += each.colspan;
      }
      for (; filled < columns; filled++) xml += cell("<w:p/>", 1, false);
      xml += "</w:tr>";
    }
    return `${xml}</w:tbl>`;
  }

  private inline(content: readonly Inline[], context: Context): string {
    let xml = "";
    for (let index = 0; index < content.length; index++) {
      const item = content[index] as Inline;
      if (item.kind === "text" && item.marks.link && isWebLink(item.marks.link)) {
        // Consecutive text with the same link is one hyperlink.
        const href = item.marks.link;
        let runs = "";
        for (; index < content.length; index++) {
          const next = content[index] as Inline;
          if (next.kind !== "text" || next.marks.link !== href) break;
          runs += run(next.text, formatOf(next.marks, context, "Hyperlink"));
        }
        index--;
        xml += `<w:hyperlink r:id="${this.link(href)}" w:history="1">${runs}</w:hyperlink>`;
        continue;
      }
      switch (item.kind) {
        case "text":
          xml += run(item.text, formatOf(item.marks, context));
          break;
        case "math":
          xml += latexToOmml(item.latex, false) ?? run(`$${item.latex}$`, { bold: context.bold });
          break;
        case "break":
          xml += "<w:r><w:br/></w:r>";
          break;
        case "citation":
          xml += this.footnote(item.footnote);
          break;
        case "image":
          xml += this.image(item.image, context);
          break;
      }
    }
    return xml;
  }

  /** A Citation's footnote, and the reference to it in the text. */
  private footnote(footnote: Footnote): string {
    const id = this.footnotes.length + 1;
    const marker = footnote.unverified
      ? run(` ${this.options.labels.unverified}`, { bold: true })
      : "";
    this.footnotes.push(
      `<w:footnote w:id="${id}"><w:p><w:pPr><w:pStyle w:val="FootnoteText"/></w:pPr><w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr><w:footnoteRef/></w:r>${run(` ${footnote.source}`)}${marker}</w:p></w:footnote>`,
    );
    return `<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr><w:footnoteReference w:id="${id}"/></w:r>`;
  }

  private link(href: string): string {
    let id = this.links.get(href);
    if (!id) {
      id = `rIdLink${this.links.size + 1}`;
      this.links.set(href, id);
      this.relationships.push({ id, type: RELATIONSHIP.hyperlink, target: href, external: true });
    }
    return id;
  }

  /** An image embedded in the document, or its description if it can't be. */
  private image(image: Image, context: Context): string {
    const decoded = decodeImage(image.src);
    if (!decoded) return run(image.alt || image.src);
    const number = this.media.length + 1;
    const name = `image${number}.${decoded.extension}`;
    const id = `rIdImage${number}`;
    this.media.push({ name, data: decoded.data });
    this.relationships.push({ id, type: RELATIONSHIP.image, target: `media/${name}` });

    // Its size in pixels, shrunk to fit the width available.
    const maxWidth = Math.max(context.width - context.indent, INDENT) * EMU_PER_TWIP;
    let cx = decoded.width * EMU_PER_PIXEL;
    let cy = decoded.height * EMU_PER_PIXEL;
    if (cx > maxWidth) {
      cy = Math.round((cy * maxWidth) / cx);
      cx = maxWidth;
    }
    const drawing = ++this.drawings;
    const description = escapeXml(image.alt);
    return (
      `<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="${cx}" cy="${cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>` +
      `<wp:docPr id="${drawing}" name="Picture ${drawing}" descr="${description}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>` +
      `<a:graphic><a:graphicData uri="${NS.pic}"><pic:pic><pic:nvPicPr><pic:cNvPr id="${drawing}" name="${name}" descr="${description}"/><pic:cNvPicPr/></pic:nvPicPr>` +
      `<pic:blipFill><a:blip r:embed="${id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>` +
      `<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="${cx}" cy="${cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>`
    );
  }
}

const isWebLink = (href: string) => /^(https?:|mailto:)/i.test(href);

// ---------------------------------------------------------------------------
// Paragraphs and runs

interface ParagraphProperties {
  style?: string | null;
  numbering?: { id: number; level: number };
  bottomBorder?: boolean;
  spacingAfter?: number;
  /** The left indent, in twips. */
  indent?: number;
  /** How far the first line hangs left of `indent`, in twips. */
  hanging?: number;
}

function paragraph(properties: ParagraphProperties, content: string): string {
  const { style, numbering, bottomBorder, spacingAfter, indent, hanging } = properties;
  // In the order the schema requires.
  const pPr = [
    style ? `<w:pStyle w:val="${style}"/>` : "",
    numbering
      ? `<w:numPr><w:ilvl w:val="${numbering.level}"/><w:numId w:val="${numbering.id}"/></w:numPr>`
      : "",
    bottomBorder
      ? '<w:pBdr><w:bottom w:val="single" w:sz="6" w:space="1" w:color="auto"/></w:pBdr>'
      : "",
    spacingAfter !== undefined ? `<w:spacing w:after="${spacingAfter}"/>` : "",
    indent ? `<w:ind w:left="${indent}"${hanging ? ` w:hanging="${hanging}"` : ""}/>` : "",
  ].join("");
  return `<w:p>${pPr ? `<w:pPr>${pPr}</w:pPr>` : ""}${content}</w:p>`;
}

interface RunFormat {
  /** A character style, e.g. "Hyperlink". */
  style?: string;
  bold?: boolean;
  italic?: boolean;
  strike?: boolean;
  /** Word's yellow text highlight, as the editor's. */
  highlight?: boolean;
  underline?: boolean;
  /** Monospaced, for code inside a link (code elsewhere uses its character style). */
  monospace?: boolean;
}

function formatOf(marks: Marks, context: Context, style?: string): RunFormat {
  return {
    style: style ?? (marks.code ? "CodeChar" : undefined),
    bold: marks.bold || context.bold,
    italic: marks.italic,
    strike: marks.strike,
    highlight: marks.highlight,
    underline: marks.underline,
    monospace: marks.code && style !== undefined,
  };
}

function run(text: string, format: RunFormat = {}): string {
  if (!text) return "";
  // In the order the schema requires.
  const rPr = [
    format.style ? `<w:rStyle w:val="${format.style}"/>` : "",
    format.monospace ? '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/>' : "",
    format.bold ? "<w:b/>" : "",
    format.italic ? "<w:i/>" : "",
    format.strike ? "<w:strike/>" : "",
    format.highlight ? '<w:highlight w:val="yellow"/>' : "",
    format.underline ? '<w:u w:val="single"/>' : "",
  ].join("");
  let content = "";
  for (const part of text.split(/(\r\n|\r|\n|\t)/)) {
    if (part === "\t") content += "<w:tab/>";
    else if (part === "\n" || part === "\r\n" || part === "\r") content += "<w:br/>";
    else if (part) content += `<w:t xml:space="preserve">${escapeXml(part)}</w:t>`;
  }
  return `<w:r>${rPr ? `<w:rPr>${rPr}</w:rPr>` : ""}${content}</w:r>`;
}

/** Characters XML can't hold at all, even escaped. */
// biome-ignore lint/suspicious/noControlCharactersInRegex: these are the characters it removes.
const NOT_XML = /[\u0000-\u0008\u000B\u000C\u000E-\u001F￾￿]/g;

function escapeXml(text: string): string {
  return text
    .replace(NOT_XML, "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ---------------------------------------------------------------------------
// Images

interface DecodedImage {
  data: Uint8Array;
  extension: "png" | "jpeg" | "gif";
  /** In pixels. */
  width: number;
  height: number;
}

/**
 * A PNG, JPEG or GIF given as a data: URL, with its size. Images elsewhere
 * (on the web) aren't fetched: the export never goes online.
 */
function decodeImage(src: string): DecodedImage | null {
  const match = /^data:image\/[\w.+-]+;base64,([\s\S]*)$/i.exec(src);
  if (!match?.[1]) return null;
  const data = Buffer.from(match[1], "base64");
  const size = imageSize(data);
  if (!size || size.width <= 0 || size.height <= 0) return null;
  return { data: new Uint8Array(data), ...size };
}

function imageSize(data: Buffer): Omit<DecodedImage, "data"> | null {
  // PNG: the IHDR chunk comes first.
  if (data.length >= 24 && data.readUInt32BE(0) === 0x89504e47) {
    return { extension: "png", width: data.readUInt32BE(16), height: data.readUInt32BE(20) };
  }
  if (data.length >= 10 && data.toString("latin1", 0, 4) === "GIF8") {
    return { extension: "gif", width: data.readUInt16LE(6), height: data.readUInt16LE(8) };
  }
  // JPEG: the size is in the first start-of-frame segment.
  if (data.length >= 4 && data[0] === 0xff && data[1] === 0xd8) {
    let offset = 2;
    while (offset + 9 <= data.length && data[offset] === 0xff) {
      const marker = data[offset + 1] as number;
      const isFrame = marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker);
      if (isFrame) {
        return {
          extension: "jpeg",
          height: data.readUInt16BE(offset + 5),
          width: data.readUInt16BE(offset + 7),
        };
      }
      offset += 2 + data.readUInt16BE(offset + 2);
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// The package's other parts

function contentTypes(): string {
  const types = [
    `<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>`,
    `<Default Extension="xml" ContentType="application/xml"/>`,
    `<Default Extension="png" ContentType="image/png"/>`,
    `<Default Extension="jpeg" ContentType="image/jpeg"/>`,
    `<Default Extension="gif" ContentType="image/gif"/>`,
    `<Override PartName="/word/document.xml" ContentType="${MAIN_CONTENT_TYPE}.document.main+xml"/>`,
    `<Override PartName="/word/styles.xml" ContentType="${MAIN_CONTENT_TYPE}.styles+xml"/>`,
    `<Override PartName="/word/settings.xml" ContentType="${MAIN_CONTENT_TYPE}.settings+xml"/>`,
    `<Override PartName="/word/numbering.xml" ContentType="${MAIN_CONTENT_TYPE}.numbering+xml"/>`,
    `<Override PartName="/word/footnotes.xml" ContentType="${MAIN_CONTENT_TYPE}.footnotes+xml"/>`,
    `<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>`,
  ];
  return `${XML_DECLARATION}<Types xmlns="${NS.contentTypes}">${types.join("")}</Types>`;
}

function relationshipsXml(relationships: readonly Relationship[]): string {
  const items = relationships.map(
    ({ id, type, target, external }) =>
      `<Relationship Id="${id}" Type="${type}" Target="${escapeXml(target)}"${external ? ' TargetMode="External"' : ""}/>`,
  );
  return `${XML_DECLARATION}<Relationships xmlns="${NS.packageRelationships}">${items.join("")}</Relationships>`;
}

function coreProperties({ title, created }: DocxOptions): string {
  const date = `${created.slice(0, 19)}Z`;
  return (
    `${XML_DECLARATION}<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">` +
    (title ? `<dc:title>${escapeXml(title)}</dc:title>` : "") +
    `<dcterms:created xsi:type="dcterms:W3CDTF">${date}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">${date}</dcterms:modified>` +
    "</cp:coreProperties>"
  );
}

/** Footnotes 1 and up are the Citations'; -1 and 0 are the separator lines Word draws above them. */
function footnotesXml(footnotes: readonly string[]): string {
  const separator = (type: string, id: number, mark: string) =>
    `<w:footnote w:type="${type}" w:id="${id}"><w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><w:r><${mark}/></w:r></w:p></w:footnote>`;
  return (
    `${XML_DECLARATION}<w:footnotes xmlns:w="${NS.w}" xmlns:r="${NS.r}">` +
    separator("separator", -1, "w:separator") +
    separator("continuationSeparator", 0, "w:continuationSeparator") +
    footnotes.join("") +
    "</w:footnotes>"
  );
}

const SETTINGS =
  `${XML_DECLARATION}<w:settings xmlns:w="${NS.w}">` +
  '<w:defaultTabStop w:val="720"/><w:characterSpacingControl w:val="doNotCompress"/>' +
  '<w:footnotePr><w:footnote w:id="-1"/><w:footnote w:id="0"/></w:footnotePr>' +
  '<w:compat><w:compatSetting w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat>' +
  "</w:settings>";

const BULLETS = ["•", "◦", "▪"];
const NUMBER_FORMATS = ["decimal", "lowerLetter", "lowerRoman"];

/** Two list definitions, bullets (0) and numbers (1), and an instance of one for each list. */
function numbering(lists: readonly List[]): string {
  const definition = (id: number, ordered: boolean) => {
    let levels = "";
    for (let level = 0; level < 9; level++) {
      const format = ordered ? NUMBER_FORMATS[level % 3] : "bullet";
      const text = ordered ? `%${level + 1}.` : BULLETS[level % 3];
      levels += `<w:lvl w:ilvl="${level}"><w:start w:val="1"/><w:numFmt w:val="${format}"/><w:lvlText w:val="${text}"/><w:lvlJc w:val="left"/><w:pPr><w:ind w:left="${INDENT * (level + 1)}" w:hanging="${HANGING}"/></w:pPr></w:lvl>`;
    }
    return `<w:abstractNum w:abstractNumId="${id}"><w:multiLevelType w:val="hybridMultilevel"/>${levels}</w:abstractNum>`;
  };
  // Each ordered list starts at its own number, rather than going on from the last.
  const instances = lists.map(
    (list, index) =>
      `<w:num w:numId="${index + 1}"><w:abstractNumId w:val="${list.ordered ? 1 : 0}"/>${list.ordered ? `<w:lvlOverride w:ilvl="${list.level}"><w:startOverride w:val="${list.start}"/></w:lvlOverride>` : ""}</w:num>`,
  );
  return `${XML_DECLARATION}<w:numbering xmlns:w="${NS.w}">${definition(0, false)}${definition(1, true)}${instances.join("")}</w:numbering>`;
}

const CODE_FONT = '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/>';

function styles(language: string): string {
  const paragraphStyle = (
    id: string,
    name: string,
    { pPr = "", rPr = "", next = false }: { pPr?: string; rPr?: string; next?: boolean },
  ) =>
    `<w:style w:type="paragraph" w:styleId="${id}"><w:name w:val="${name}"/><w:basedOn w:val="Normal"/>${next ? '<w:next w:val="Normal"/>' : ""}<w:qFormat/>${pPr ? `<w:pPr>${pPr}</w:pPr>` : ""}${rPr ? `<w:rPr>${rPr}</w:rPr>` : ""}</w:style>`;
  const headingSizes = [32, 28, 26, 24, 22, 22];
  const headings = headingSizes
    .map((size, index) =>
      paragraphStyle(`Heading${index + 1}`, `heading ${index + 1}`, {
        next: true,
        pPr: `<w:keepNext/><w:keepLines/><w:spacing w:before="${index === 0 ? 360 : 240}" w:after="80"/><w:outlineLvl w:val="${index}"/>`,
        rPr: `<w:b/>${index >= 3 ? "<w:i/>" : ""}<w:sz w:val="${size}"/><w:szCs w:val="${size}"/>`,
      }),
    )
    .join("");
  const border = (color: string) =>
    `<w:pBdr><w:left w:val="single" w:sz="12" w:space="8" w:color="${color}"/></w:pBdr>`;

  return (
    `${XML_DECLARATION}<w:styles xmlns:w="${NS.w}">` +
    `<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/><w:sz w:val="22"/><w:szCs w:val="22"/><w:lang w:val="${language}"/></w:rPr></w:rPrDefault>` +
    '<w:pPrDefault><w:pPr><w:spacing w:after="160" w:line="259" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>' +
    '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>' +
    '<w:style w:type="character" w:default="1" w:styleId="DefaultParagraphFont"><w:name w:val="Default Paragraph Font"/><w:uiPriority w:val="1"/><w:semiHidden/><w:unhideWhenUsed/></w:style>' +
    '<w:style w:type="table" w:default="1" w:styleId="TableNormal"><w:name w:val="Normal Table"/><w:uiPriority w:val="99"/><w:semiHidden/><w:unhideWhenUsed/><w:tblPr><w:tblInd w:w="0" w:type="dxa"/><w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="108" w:type="dxa"/><w:bottom w:w="0" w:type="dxa"/><w:right w:w="108" w:type="dxa"/></w:tblCellMar></w:tblPr></w:style>' +
    '<w:style w:type="numbering" w:default="1" w:styleId="NoList"><w:name w:val="No List"/><w:uiPriority w:val="99"/><w:semiHidden/><w:unhideWhenUsed/></w:style>' +
    paragraphStyle("Title", "Title", {
      next: true,
      pPr: '<w:spacing w:after="240"/>',
      rPr: '<w:sz w:val="48"/><w:szCs w:val="48"/>',
    }) +
    headings +
    paragraphStyle("Quote", "Quote", {
      pPr: `${border("D1D5DB")}<w:ind w:left="${INDENT}"/>`,
      rPr: '<w:i/><w:color w:val="4B5563"/>',
    }) +
    paragraphStyle("Question", "Question", {
      next: true,
      pPr: border("0EA5E9"),
      rPr: '<w:color w:val="374151"/>',
    }) +
    paragraphStyle("ListParagraph", "List Paragraph", {
      pPr: `<w:ind w:left="${INDENT}"/><w:contextualSpacing/>`,
    }) +
    paragraphStyle("Code", "Code", {
      pPr: '<w:shd w:val="clear" w:color="auto" w:fill="F3F4F6"/><w:spacing w:after="0" w:line="240" w:lineRule="auto"/>',
      rPr: `${CODE_FONT}<w:sz w:val="20"/><w:szCs w:val="20"/>`,
    }) +
    paragraphStyle("Math", "Math", { pPr: '<w:jc w:val="center"/>' }) +
    '<w:style w:type="paragraph" w:styleId="FootnoteText"><w:name w:val="footnote text"/><w:basedOn w:val="Normal"/><w:uiPriority w:val="99"/><w:unhideWhenUsed/><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:style>' +
    '<w:style w:type="character" w:styleId="FootnoteReference"><w:name w:val="footnote reference"/><w:basedOn w:val="DefaultParagraphFont"/><w:uiPriority w:val="99"/><w:unhideWhenUsed/><w:rPr><w:vertAlign w:val="superscript"/></w:rPr></w:style>' +
    '<w:style w:type="character" w:styleId="Hyperlink"><w:name w:val="Hyperlink"/><w:basedOn w:val="DefaultParagraphFont"/><w:uiPriority w:val="99"/><w:unhideWhenUsed/><w:rPr><w:color w:val="0563C1"/><w:u w:val="single"/></w:rPr></w:style>' +
    `<w:style w:type="character" w:styleId="CodeChar"><w:name w:val="Inline Code"/><w:basedOn w:val="DefaultParagraphFont"/><w:rPr>${CODE_FONT}<w:shd w:val="clear" w:color="auto" w:fill="F3F4F6"/></w:rPr></w:style>` +
    '<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:basedOn w:val="TableNormal"/><w:uiPriority w:val="39"/><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/></w:tblBorders></w:tblPr></w:style>' +
    "</w:styles>"
  );
}
