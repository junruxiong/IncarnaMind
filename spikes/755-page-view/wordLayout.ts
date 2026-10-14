/**
 * Spike for #755, not for merging: the layout properties a Page view needs from a .docx, read in
 * Node by the spike's e2e spec and handed to the renderer. A crude stand-in for #746's Word look:
 * fonts, sizes, line and paragraph spacing, alignment, indents, keep rules, page breaks before,
 * widow control, and each section's page size, margins, document grid, headers and footers.
 */
import { openWordPackage, relationshipsOf } from "../../src/core/documents/formats/docxPackage";
import { child, elements, parseXml, walk, type XmlElement } from "../../src/core/documents/formats/xml";

export interface SpikeSection {
  width: number;
  height: number;
  top: number;
  right: number;
  bottom: number;
  left: number;
  header: number;
  footer: number;
  titlePg: boolean;
  type: string;
  /** The document grid's line pitch in px, when the section snaps lines to it. */
  grid: number | null;
  headers: Partial<Record<"default" | "first" | "even", string>>;
  footers: Partial<Record<"default" | "first" | "even", string>>;
}

export interface SpikeRunLook {
  family: string;
  size: number;
}

export interface SpikeBlockLook {
  family: string;
  /** In px. */
  size: number;
  bold: boolean;
  align: string | null;
  before: number;
  after: number;
  /** A multiple of the font's single line ("auto"), or null when exact/atLeast says. */
  lineMultiple: number | null;
  lineExact: number | null;
  lineAtLeast: number | null;
  indentLeft: number;
  indentRight: number;
  firstLine: number;
  keepNext: boolean;
  keepLines: boolean;
  pageBreakBefore: boolean;
  widowControl: boolean;
  snapToGrid: boolean;
  section: number;
}

export interface SpikeLook {
  blocks: Record<string, SpikeBlockLook>;
  sections: SpikeSection[];
  footnote: SpikeRunLook & { lineMultiple: number };
}

const TWIP_PX = 1 / 15;
const num = (value: string | undefined) => (value === undefined ? undefined : Number(value));
const isOn = (element: XmlElement | undefined) =>
  element !== undefined && !["false", "0", "off"].includes(element.attrs["w:val"] ?? "true");

interface Props {
  family?: string;
  size?: number;
  bold?: boolean;
  align?: string;
  before?: number;
  after?: number;
  line?: number;
  lineRule?: string;
  indentLeft?: number;
  indentRight?: number;
  firstLine?: number;
  keepNext?: boolean;
  keepLines?: boolean;
  pageBreakBefore?: boolean;
  widowControl?: boolean;
  snapToGrid?: boolean;
}

function readRun(rPr: XmlElement | undefined, theme: Record<string, string>, into: Props): void {
  if (!rPr) return;
  const fonts = child(rPr, "w:rFonts");
  if (fonts) {
    const east = fonts.attrs["w:eastAsia"];
    const ascii =
      fonts.attrs["w:ascii"] ??
      (fonts.attrs["w:asciiTheme"] ? theme[fonts.attrs["w:asciiTheme"]] : undefined);
    const eastTheme = fonts.attrs["w:eastAsiaTheme"]
      ? theme[fonts.attrs["w:eastAsiaTheme"]]
      : undefined;
    const family = [ascii, east ?? eastTheme].filter(Boolean).join("|");
    if (family) into.family = family;
  }
  const sz = num(child(rPr, "w:sz")?.attrs["w:val"]);
  if (sz) into.size = (sz / 2) * (96 / 72);
  const b = child(rPr, "w:b");
  if (b) into.bold = isOn(b);
}

function readParagraph(pPr: XmlElement | undefined, into: Props): void {
  if (!pPr) return;
  const spacing = child(pPr, "w:spacing");
  if (spacing) {
    const before = num(spacing.attrs["w:before"]);
    const after = num(spacing.attrs["w:after"]);
    const line = num(spacing.attrs["w:line"]);
    if (before !== undefined && spacing.attrs["w:beforeAutospacing"] !== "1") into.before = before;
    if (after !== undefined && spacing.attrs["w:afterAutospacing"] !== "1") into.after = after;
    if (line !== undefined) {
      into.line = line;
      into.lineRule = spacing.attrs["w:lineRule"] ?? "auto";
    }
  }
  const jc = child(pPr, "w:jc")?.attrs["w:val"];
  if (jc) into.align = jc;
  const ind = child(pPr, "w:ind");
  if (ind) {
    const left = num(ind.attrs["w:left"] ?? ind.attrs["w:start"]);
    const right = num(ind.attrs["w:right"] ?? ind.attrs["w:end"]);
    const first = num(ind.attrs["w:firstLine"]);
    const hanging = num(ind.attrs["w:hanging"]);
    if (left !== undefined) into.indentLeft = left;
    if (right !== undefined) into.indentRight = right;
    if (first !== undefined) into.firstLine = first;
    if (hanging !== undefined) into.firstLine = -hanging;
  }
  for (const name of ["keepNext", "keepLines", "pageBreakBefore", "widowControl"] as const) {
    const element = child(pPr, `w:${name}`);
    if (element) into[name] = isOn(element);
  }
  const snap = child(pPr, "w:snapToGrid");
  if (snap) into.snapToGrid = isOn(snap);
}

interface Style {
  basedOn: string | null;
  props: Props;
}

function readStyles(xml: string | undefined, theme: Record<string, string>) {
  const styles = new Map<string, Style>();
  const defaults: Props = {};
  let defaultParagraph: string | null = null;
  let footnoteStyle: string | null = null;
  if (!xml) return { styles, defaults, defaultParagraph, footnoteStyle };
  const root = parseXml(xml);
  const docDefaults = child(root, "w:docDefaults");
  if (docDefaults) {
    readRun(child(child(docDefaults, "w:rPrDefault") ?? docDefaults, "w:rPr"), theme, defaults);
    readParagraph(child(child(docDefaults, "w:pPrDefault") ?? docDefaults, "w:pPr"), defaults);
  }
  for (const style of elements(root)) {
    if (style.name !== "w:style" || style.attrs["w:type"] !== "paragraph") continue;
    const id = style.attrs["w:styleId"];
    if (!id) continue;
    if (style.attrs["w:default"] === "1" || style.attrs["w:default"] === "true") {
      defaultParagraph = id;
    }
    const name = child(style, "w:name")?.attrs["w:val"]?.toLowerCase();
    if (name === "footnote text") footnoteStyle = id;
    const props: Props = {};
    readParagraph(child(style, "w:pPr"), props);
    readRun(child(style, "w:rPr"), theme, props);
    styles.set(id, { basedOn: child(style, "w:basedOn")?.attrs["w:val"] ?? null, props });
  }
  return { styles, defaults, defaultParagraph, footnoteStyle };
}

function styleProps(styles: Map<string, Style>, id: string | null): Props {
  const chain: Props[] = [];
  const seen = new Set<string>();
  for (let at = id; at && !seen.has(at) && chain.length < 20; ) {
    seen.add(at);
    const style = styles.get(at);
    if (!style) break;
    chain.unshift(style.props);
    at = style.basedOn;
  }
  return Object.assign({}, ...chain);
}

function readTheme(xml: string | undefined): Record<string, string> {
  if (!xml) return {};
  const root = parseXml(xml);
  const out: Record<string, string> = {};
  for (const element of walk(root)) {
    if (element.name === "a:majorFont" || element.name === "a:minorFont") {
      const kind = element.name === "a:majorFont" ? "major" : "minor";
      const latin = child(element, "a:latin")?.attrs.typeface;
      const ea = child(element, "a:ea")?.attrs.typeface;
      if (latin) out[`${kind}HAnsi`] = out[`${kind}Ascii`] = latin;
      if (ea) out[`${kind}EastAsia`] = ea;
    }
  }
  return out;
}

/** A header or footer part's text, its PAGE and NUMPAGES fields as {page} and {pages}. */
function furnitureText(xml: string): string {
  const root = parseXml(xml);
  const lines: string[] = [];
  for (const p of walk(root)) {
    if (p.name !== "w:p") continue;
    let text = "";
    let field: { instr: string; separated: boolean } | null = null;
    for (const element of walk(p)) {
      if (element.name === "w:fldSimple") {
        const instr = (element.attrs["w:instr"] ?? "").trim().split(/\s+/)[0]?.toUpperCase();
        text += instr === "PAGE" ? "{page}" : instr === "NUMPAGES" ? "{pages}" : "";
      } else if (element.name === "w:fldChar") {
        const type = element.attrs["w:fldCharType"];
        if (type === "begin") field = { instr: "", separated: false };
        else if (type === "separate" && field) field.separated = true;
        else if (type === "end" && field) {
          const instr = field.instr.trim().split(/\s+/)[0]?.toUpperCase();
          text += instr === "PAGE" ? "{page}" : instr === "NUMPAGES" ? "{pages}" : "";
          field = null;
        }
      } else if (element.name === "w:instrText" && field) {
        field.instr += element.children.filter((part) => typeof part === "string").join("");
      } else if (element.name === "w:t" && !field) {
        text += element.children.filter((part) => typeof part === "string").join("");
      } else if (element.name === "w:tab" && !field && !element.attrs["w:val"]) {
        text += "\t";
      }
    }
    lines.push(text);
  }
  return lines.join("\n").trim();
}

export async function readSpikeLook(bytes: Uint8Array): Promise<SpikeLook> {
  const pkg = await openWordPackage(bytes);
  const { zip, part } = pkg;
  const rels = await relationshipsOf(zip, part);
  const target = (suffix: string) =>
    [...rels.values()].find((rel) => !rel.external && rel.type.endsWith(suffix))?.target;
  const text = async (name: string | undefined) =>
    name && zip.has(name) ? zip.readText(name) : undefined;
  const theme = readTheme(await text(target("/theme")));
  const { styles, defaults, defaultParagraph, footnoteStyle } = readStyles(
    await text(target("/styles")),
    theme,
  );
  const documentXml = await zip.readText(part);
  if (!documentXml) throw new Error("No document part.");
  const root = parseXml(documentXml);
  const body = child(root, "w:body");
  if (!body) throw new Error("No body.");

  const sectPrs: XmlElement[] = [];
  const sectionEnds = new Set<XmlElement>();
  for (const element of walk(body)) {
    if (element.name === "w:p") {
      const sectPr = child(child(element, "w:pPr") ?? element, "w:sectPr");
      if (sectPr) {
        sectPrs.push(sectPr);
        sectionEnds.add(element);
      }
    }
  }
  const last = elements(body).find((each) => each.name === "w:sectPr");
  if (last) sectPrs.push(last);
  const sections: SpikeSection[] = [];
  for (const sectPr of sectPrs) {
    const pgSz = child(sectPr, "w:pgSz");
    const pgMar = child(sectPr, "w:pgMar");
    const grid = child(sectPr, "w:docGrid");
    const twips = (element: XmlElement | undefined, name: string, fallback: number) =>
      (num(element?.attrs[name]) ?? fallback) * TWIP_PX;
    const furniture = async (kind: "header" | "footer") => {
      const out: SpikeSection["headers"] = {};
      for (const reference of elements(sectPr).filter((each) => each.name === `w:${kind}Reference`)) {
        const type = (reference.attrs["w:type"] ?? "default") as "default" | "first" | "even";
        const rel = rels.get(reference.attrs["r:id"] ?? "");
        const xml = rel ? await text(rel.target) : undefined;
        if (xml) out[type] = furnitureText(xml);
      }
      return out;
    };
    const previous = sections.at(-1);
    const headers = await furniture("header");
    const footers = await furniture("footer");
    const gridType = grid?.attrs["w:type"];
    sections.push({
      width: twips(pgSz, "w:w", 11906),
      height: twips(pgSz, "w:h", 16838),
      top: Math.abs(twips(pgMar, "w:top", 1440)),
      right: twips(pgMar, "w:right", 1440),
      bottom: Math.abs(twips(pgMar, "w:bottom", 1440)),
      left: twips(pgMar, "w:left", 1440),
      header: twips(pgMar, "w:header", 720),
      footer: twips(pgMar, "w:footer", 720),
      titlePg: isOn(child(sectPr, "w:titlePg")),
      type: child(sectPr, "w:type")?.attrs["w:val"] ?? "nextPage",
      grid:
        (gridType === "lines" || gridType === "linesAndChars") && grid?.attrs["w:linePitch"]
          ? Number(grid.attrs["w:linePitch"]) * TWIP_PX
          : null,
      // A section without its own goes on with the one before's.
      headers: { ...(previous?.headers ?? {}), ...headers },
      footers: { ...(previous?.footers ?? {}), ...footers },
    });
  }
  if (sections.length === 0) {
    sections.push({
      width: 793.7,
      height: 1122.5,
      top: 96,
      right: 96,
      bottom: 96,
      left: 96,
      header: 48,
      footer: 48,
      titlePg: false,
      type: "nextPage",
      grid: null,
      headers: {},
      footers: {},
    });
  }

  const blocks: Record<string, SpikeBlockLook> = {};
  let ordinal = 0;
  let section = 0;
  for (const element of walk(root)) {
    if (element.name !== "w:p") continue;
    const place = `w:p#${ordinal++}`;
    const pPr = child(element, "w:pPr");
    const styleId = (pPr && child(pPr, "w:pStyle")?.attrs["w:val"]) ?? defaultParagraph;
    const props: Props = { ...defaults, ...styleProps(styles, styleId) };
    readParagraph(pPr, props);
    const firstRun = elements(element).find((each) => each.name === "w:r");
    const runProps: Props = {};
    readRun(child(firstRun ?? element, "w:rPr"), theme, runProps);
    const own = { ...props, ...runProps };
    const line = own.line ?? 240;
    const rule = own.lineRule ?? "auto";
    blocks[place] = {
      family: own.family ?? "Calibri",
      size: own.size ?? (11 * 96) / 72,
      bold: own.bold ?? false,
      align: own.align ?? null,
      before: (own.before ?? 0) * TWIP_PX,
      after: (own.after ?? 0) * TWIP_PX,
      lineMultiple: rule === "auto" ? line / 240 : null,
      lineExact: rule === "exact" ? line * TWIP_PX : null,
      lineAtLeast: rule === "atLeast" ? line * TWIP_PX : null,
      indentLeft: (own.indentLeft ?? 0) * TWIP_PX,
      indentRight: (own.indentRight ?? 0) * TWIP_PX,
      firstLine: (own.firstLine ?? 0) * TWIP_PX,
      keepNext: own.keepNext ?? false,
      keepLines: own.keepLines ?? false,
      pageBreakBefore: own.pageBreakBefore ?? false,
      widowControl: own.widowControl ?? false,
      snapToGrid: own.snapToGrid ?? true,
      section,
    };
    if (sectionEnds.has(element)) section++;
  }
  const footnoteProps: Props = { ...defaults, ...styleProps(styles, footnoteStyle) };
  return {
    blocks,
    sections,
    footnote: {
      family: footnoteProps.family ?? "Calibri",
      size: footnoteProps.size ?? (10 * 96) / 72,
      lineMultiple: footnoteProps.lineRule === "auto" ? (footnoteProps.line ?? 240) / 240 : 1,
    },
  };
}
