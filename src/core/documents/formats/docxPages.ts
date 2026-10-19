/**
 * A Word file's pages as Page view lays them out (#803): each section's size,
 * orientation and margins, its headers and footers with their PAGE, NUMPAGES
 * and SECTIONPAGES fields, whether the first page and the odd and even pages
 * differ, and where its page numbers start. Read from the package kept at
 * import, in the document's own units (twips). A header or footer a section
 * doesn't name is the one before's, as Word has it, so each section comes
 * with the ones it shows.
 */
import type {
  PageFurniture,
  PageFurnitureParagraph,
  PageFurnitureRun,
  PageFurnitureSet,
  PageNumberFormat,
  PageSection,
  PageSetup,
} from "../../api";
import { type Relationship, relationshipsOf, type WordPackage } from "./docxPackage";
import { child, elements, onOff, parseXml, textOf, walk, type XmlElement } from "./xml";

/** Word's own defaults for a section with no `w:pgSz` or `w:pgMar`: US Letter and one inch. */
const LETTER = { width: 12240, height: 15840 };
const INCH = 1440;
const HEADER_AND_FOOTER_DISTANCE = 720;

const FORMATS: Readonly<Record<string, PageNumberFormat>> = {
  decimal: "decimal",
  lowerRoman: "lowerRoman",
  upperRoman: "upperRoman",
  lowerLetter: "lowerLetter",
  upperLetter: "upperLetter",
};

/** A field's `\* roman` switch: Word's names for the same formats. */
const SWITCHES: Readonly<Record<string, PageNumberFormat>> = {
  arabic: "decimal",
  roman: "lowerRoman",
  ROMAN: "upperRoman",
  alphabetic: "lowerLetter",
  ALPHABETIC: "upperLetter",
};

const START_TYPES = {
  nextPage: "nextPage",
  continuous: "continuous",
  evenPage: "evenPage",
  oddPage: "oddPage",
} as const;

const FIELDS = new Set(["PAGE", "NUMPAGES", "SECTIONPAGES"]);

const twips = (value: string | undefined, fallback: number): number => {
  const parsed = Number(value);
  return value !== undefined && Number.isFinite(parsed) ? Math.round(parsed) : fallback;
};

const ALIGNS = {
  left: "left",
  start: "left",
  center: "center",
  right: "right",
  end: "right",
  both: "justify",
  distribute: "justify",
} as const;

/** The paper and margins a section has when nothing says; the caller gives the paper for the region. */
export interface PageDefaults {
  width: number;
  height: number;
}

/** What a header or footer paragraph holds, one run at a time. */
class RunReader {
  readonly runs: PageFurnitureRun[] = [];
  private readonly fields: {
    instruction: string;
    inResult: boolean;
    bold: boolean;
    italic: boolean;
    result: PageFurnitureRun[];
  }[] = [];

  private add(run: PageFurnitureRun): void {
    const open = this.fields.at(-1);
    const target = open?.inResult ? open.result : open ? null : this.runs;
    if (!target) return;
    const last = target.at(-1);
    if (
      run.kind === "text" &&
      last?.kind === "text" &&
      last.bold === run.bold &&
      last.italic === run.italic
    ) {
      target[target.length - 1] = { ...last, text: last.text + run.text };
    } else {
      target.push(run);
    }
  }

  read(node: XmlElement): void {
    for (const element of elements(node)) {
      switch (element.name) {
        case "w:r":
          this.run(element);
          break;
        case "w:fldSimple":
          this.simpleField(element);
          break;
        case "w:hyperlink":
        case "w:ins":
        case "w:smartTag":
        case "w:sdt":
        case "w:sdtContent":
        case "w:customXml":
          this.read(element);
          break;
        default:
      }
    }
  }

  private formatting(run: XmlElement): { bold?: true; italic?: true } {
    const props = child(run, "w:rPr");
    return {
      ...(props && onOff(child(props, "w:b")) && { bold: true as const }),
      ...(props && onOff(child(props, "w:i")) && { italic: true as const }),
    };
  }

  private run(run: XmlElement): void {
    const format = this.formatting(run);
    for (const part of elements(run)) {
      if (part.name === "w:fldChar") this.fieldChar(part, format);
      else if (part.name === "w:instrText") {
        const open = this.fields.at(-1);
        if (open && !open.inResult) open.instruction += textOf(part);
      } else if (part.name === "w:t") this.add({ kind: "text", text: textOf(part), ...format });
      else if (part.name === "w:tab") this.add({ kind: "text", text: "\t", ...format });
      else if (part.name === "w:br" || part.name === "w:cr") {
        this.add({ kind: "text", text: "\n", ...format });
      }
    }
  }

  private fieldChar(mark: XmlElement, format: { bold?: true; italic?: true }): void {
    const type = mark.attrs["w:fldCharType"];
    if (type === "begin") {
      this.fields.push({
        instruction: "",
        inResult: false,
        bold: format.bold === true,
        italic: format.italic === true,
        result: [],
      });
    } else if (type === "separate") {
      const open = this.fields.at(-1);
      if (open) open.inResult = true;
    } else if (type === "end") {
      const closed = this.fields.pop();
      if (closed) this.finish(closed.instruction, closed.result, closed);
    }
  }

  private simpleField(field: XmlElement): void {
    const inside = new RunReader();
    inside.read(field);
    this.finish(field.attrs["w:instr"] ?? "", inside.runs, { bold: false, italic: false });
  }

  /** A field IncarnaMind draws becomes a field run; any other is its last result as text. */
  private finish(
    instruction: string,
    result: readonly PageFurnitureRun[],
    format: { bold: boolean; italic: boolean },
  ): void {
    const words = instruction.trim().split(/\s+/);
    const name = (words[0] ?? "").toUpperCase();
    if (FIELDS.has(name)) {
      const switched = words.findIndex((word) => word === "\\*");
      const given = switched >= 0 ? words[switched + 1] : undefined;
      const field = name as "PAGE" | "NUMPAGES" | "SECTIONPAGES";
      this.add({
        kind: "field",
        field,
        format: (given !== undefined && SWITCHES[given]) || null,
        ...(format.bold && { bold: true as const }),
        ...(format.italic && { italic: true as const }),
      });
      return;
    }
    for (const run of result) this.add(run);
  }
}

/** A text box's other half, the VML a file keeps for older readers, says the same again. */
const notFallback = (element: XmlElement) => element.name !== "mc:Fallback";

function furnitureOf(root: XmlElement): PageFurniture {
  // A page number is often a text box in the footer: its paragraphs follow the one it is anchored in.
  const boxed = new Set<XmlElement>();
  for (const box of walk(root, notFallback)) {
    if (box.name !== "w:txbxContent") continue;
    for (const inside of walk(box)) if (inside.name === "w:p") boxed.add(inside);
  }
  const paragraphs: PageFurnitureParagraph[] = [];
  for (const element of walk(root, notFallback)) {
    if (element.name !== "w:p") continue;
    const reader = new RunReader();
    reader.read(element);
    const jc = child(child(element, "w:pPr") ?? element, "w:jc")?.attrs["w:val"] ?? "left";
    paragraphs.push({
      align: ALIGNS[jc as keyof typeof ALIGNS] ?? "left",
      runs: reader.runs,
      ...(boxed.has(element) && { inTextBox: true as const }),
    });
  }
  return { paragraphs };
}

const furnitureKinds = ["default", "first", "even"] as const;

class Furniture {
  private readonly parts = new Map<string, Promise<PageFurniture | null>>();

  constructor(
    private readonly pkg: WordPackage,
    private readonly relationships: ReadonlyMap<string, Relationship>,
  ) {}

  /** What a section's references name, by kind; a kind it doesn't name is left out. */
  async named(
    sectPr: XmlElement | undefined,
    reference: "w:headerReference" | "w:footerReference",
  ): Promise<Partial<Record<(typeof furnitureKinds)[number], PageFurniture | null>>> {
    const found: Partial<Record<(typeof furnitureKinds)[number], PageFurniture | null>> = {};
    for (const each of sectPr ? elements(sectPr) : []) {
      if (each.name !== reference) continue;
      const type = each.attrs["w:type"] ?? "default";
      if (!furnitureKinds.some((kind) => kind === type)) continue;
      const rel = this.relationships.get(each.attrs["r:id"] ?? "");
      found[type as (typeof furnitureKinds)[number]] =
        rel && !rel.external ? await this.part(rel.target) : null;
    }
    return found;
  }

  private part(name: string): Promise<PageFurniture | null> {
    let read = this.parts.get(name);
    if (!read) {
      read = (async () => {
        if (!this.pkg.zip.has(name)) return null;
        const xml = await this.pkg.zip.readText(name);
        return xml ? furnitureOf(parseXml(xml)) : null;
      })();
      this.parts.set(name, read);
    }
    return read;
  }
}

/** Each `w:sectPr` of the document in order, with the place its break has in the Mind. */
function sectionsOf(
  document: XmlElement,
): { sectPr: XmlElement | undefined; breakPlace: string | null }[] {
  const body = child(document, "w:body");
  if (!body) return [];
  const places = new Map<XmlElement, string>();
  let count = 0;
  for (const element of walk(document)) {
    if (element.name === "w:sectPr") places.set(element, `w:sectPr#${count++}`);
  }
  const found: { sectPr: XmlElement | undefined; breakPlace: string | null }[] = [];
  for (const block of elements(body)) {
    const sectPr = block.name === "w:p" ? child(child(block, "w:pPr") ?? block, "w:sectPr") : null;
    if (sectPr) found.push({ sectPr, breakPlace: places.get(sectPr) ?? null });
  }
  // The last section's properties are the body's own; a file with none has the defaults.
  found.push({ sectPr: child(body, "w:sectPr"), breakPlace: null });
  return found;
}

/** The pages of a Word package, as its sections set them up. */
export async function readPageSetup(
  pkg: WordPackage,
  defaults: PageDefaults = LETTER,
): Promise<PageSetup> {
  const xml = await pkg.zip.readText(pkg.part);
  if (!xml) return { source: "word", oddAndEven: false, sections: [] };
  const document = parseXml(xml);
  const relationships = await relationshipsOf(pkg.zip, pkg.part);
  const furniture = new Furniture(pkg, relationships);

  const settings = [...relationships.values()].find((rel) => rel.type.endsWith("/settings"));
  const settingsXml =
    settings && pkg.zip.has(settings.target) ? await pkg.zip.readText(settings.target) : undefined;
  const oddAndEven = settingsXml
    ? onOff(child(parseXml(settingsXml), "w:evenAndOddHeaders")) === true
    : false;

  const empty: PageFurnitureSet = { default: null, first: null, even: null };
  let headers = empty;
  let footers = empty;
  const sections: PageSection[] = [];
  for (const { sectPr, breakPlace } of sectionsOf(document)) {
    const size = sectPr && child(sectPr, "w:pgSz");
    const margins = sectPr && child(sectPr, "w:pgMar");
    const numbering = sectPr && child(sectPr, "w:pgNumType");
    const width = twips(size?.attrs["w:w"], defaults.width);
    const height = twips(size?.attrs["w:h"], defaults.height);
    headers = { ...headers, ...(await furniture.named(sectPr, "w:headerReference")) };
    footers = { ...footers, ...(await furniture.named(sectPr, "w:footerReference")) };
    const type = (sectPr && child(sectPr, "w:type")?.attrs["w:val"]) ?? "nextPage";
    const start = numbering?.attrs["w:start"];
    sections.push({
      breakPlace,
      start: START_TYPES[type as keyof typeof START_TYPES] ?? "nextPage",
      width,
      height,
      orientation:
        size?.attrs["w:orient"] === "landscape" || (!size?.attrs["w:orient"] && width > height)
          ? "landscape"
          : "portrait",
      margins: {
        top: twips(margins?.attrs["w:top"], INCH),
        right: twips(margins?.attrs["w:right"], INCH),
        bottom: twips(margins?.attrs["w:bottom"], INCH),
        left: twips(margins?.attrs["w:left"], INCH),
        gutter: twips(margins?.attrs["w:gutter"], 0),
        header: twips(margins?.attrs["w:header"], HEADER_AND_FOOTER_DISTANCE),
        footer: twips(margins?.attrs["w:footer"], HEADER_AND_FOOTER_DISTANCE),
      },
      differentFirstPage: sectPr ? onOff(child(sectPr, "w:titlePg")) === true : false,
      pageNumbers: {
        start: start !== undefined && Number.isFinite(Number(start)) ? Number(start) : null,
        format: FORMATS[numbering?.attrs["w:fmt"] ?? ""] ?? "decimal",
      },
      headers,
      footers,
    });
  }
  return { source: "word", oddAndEven, sections };
}
