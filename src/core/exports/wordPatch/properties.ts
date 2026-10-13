/**
 * Paragraph and run properties (`w:pPr`, `w:rPr`) changed in place (#105):
 * the children a change doesn't touch keep their bytes and their order, and
 * a child added goes where WordprocessingML's schema puts it, as Word
 * repairs a file whose children are out of order.
 */
import { runMarks } from "../../documents/formats/docxImport";
import { decodeEntities, elements, parseXml } from "../../documents/formats/xml";
import { escapeXml } from "../wordml";
import { marksOf, type TextMarks } from "./mind";
import { sliceOf } from "./text";

/** CT_PPr's children, in order. */
const PARAGRAPH_ORDER = [
  "w:pStyle",
  "w:keepNext",
  "w:keepLines",
  "w:pageBreakBefore",
  "w:framePr",
  "w:widowControl",
  "w:numPr",
  "w:suppressLineNumbers",
  "w:pBdr",
  "w:shd",
  "w:tabs",
  "w:suppressAutoHyphens",
  "w:kinsoku",
  "w:wordWrap",
  "w:overflowPunct",
  "w:topLinePunct",
  "w:autoSpaceDE",
  "w:autoSpaceDN",
  "w:bidi",
  "w:adjustRightInd",
  "w:snapToGrid",
  "w:spacing",
  "w:ind",
  "w:contextualSpacing",
  "w:mirrorIndents",
  "w:suppressOverlap",
  "w:jc",
  "w:textDirection",
  "w:textAlignment",
  "w:textboxTightWrap",
  "w:outlineLvl",
  "w:divId",
  "w:cnfStyle",
  "w:rPr",
  "w:sectPr",
  "w:pPrChange",
];

/** CT_RPr's children, in order. */
const RUN_ORDER = [
  "w:rStyle",
  "w:rFonts",
  "w:b",
  "w:bCs",
  "w:i",
  "w:iCs",
  "w:caps",
  "w:smallCaps",
  "w:strike",
  "w:dstrike",
  "w:outline",
  "w:shadow",
  "w:emboss",
  "w:imprint",
  "w:noProof",
  "w:snapToGrid",
  "w:vanish",
  "w:webHidden",
  "w:color",
  "w:spacing",
  "w:w",
  "w:kern",
  "w:position",
  "w:sz",
  "w:szCs",
  "w:highlight",
  "w:u",
  "w:effect",
  "w:bdr",
  "w:shd",
  "w:fitText",
  "w:vertAlign",
  "w:rtl",
  "w:cs",
  "w:em",
  "w:lang",
  "w:eastAsianLayout",
  "w:specVanish",
  "w:oMath",
  "w:rPrChange",
];

interface Child {
  name: string;
  xml: string;
}

/** Properties as a list of children, each as written, that changes are made to. */
export class Properties {
  private constructor(
    private readonly tag: string,
    private readonly children: Child[],
    private readonly order: readonly string[],
  ) {}

  /** `xml` is the whole `w:pPr` or `w:rPr` element as written, or "" for none. */
  static paragraph(xml: string): Properties {
    return Properties.parse("w:pPr", xml, PARAGRAPH_ORDER);
  }

  static run(xml: string): Properties {
    return Properties.parse("w:rPr", xml, RUN_ORDER);
  }

  private static parse(tag: string, xml: string, order: readonly string[]): Properties {
    if (!xml) return new Properties(tag, [], order);
    const root = parseXml(xml);
    return new Properties(
      tag,
      elements(root).map((each) => ({ name: each.name, xml: sliceOf(xml, each) })),
      order,
    );
  }

  has(name: string): boolean {
    return this.children.some((each) => each.name === name);
  }

  get(name: string): string | null {
    return this.children.find((each) => each.name === name)?.xml ?? null;
  }

  /** Sets a child, replacing the one there, or adding it in the schema's order. */
  set(name: string, xml: string): this {
    const at = this.children.findIndex((each) => each.name === name);
    if (at >= 0) {
      this.children[at] = { name, xml };
      return this;
    }
    const rank = this.order.indexOf(name);
    const before = this.children.findIndex((each) => {
      const other = this.order.indexOf(each.name);
      // A child the schema doesn't list (an extension's) comes after those it does.
      return other < 0 || other > rank;
    });
    this.children.splice(before < 0 ? this.children.length : before, 0, { name, xml });
    return this;
  }

  remove(...names: string[]): this {
    for (let at = this.children.length - 1; at >= 0; at--) {
      if (names.includes(this.children[at]?.name ?? "")) this.children.splice(at, 1);
    }
    return this;
  }

  /** The element, or "" when it has no children left. */
  toXml(): string {
    if (this.children.length === 0) return "";
    return `<${this.tag}>${this.children.map((each) => each.xml).join("")}</${this.tag}>`;
  }
}

/** The editor's marks that a run's own properties give its text, as the import reads them. */
export const marksOfRun = (rPr: string): TextMarks =>
  marksOf(rPr ? runMarks(parseXml(rPr)) : []);

/** The value a child's `w:val` has, as written in its XML. */
const valueOf = (xml: string | null) => {
  const value = xml && /\bw:val\s*=\s*"([^"]*)"/.exec(xml)?.[1];
  return value ? decodeEntities(value) : null;
};

const CODE_FONT = '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/>';

/**
 * Run properties for text marked `to`, from properties written for text
 * marked as they say: only the toggles whose mark changed are set or
 * removed, so a font, a size or a colour the editor doesn't show stays.
 */
export function runPropertiesFor(rPr: string, to: TextMarks, linkStyle: string | null): string {
  const from = marksOfRun(rPr);
  const properties = Properties.run(rPr);
  const toggle = (on: boolean, was: boolean, names: string[], xml: string[]) => {
    if (on === was) return;
    if (on) names.forEach((name, index) => properties.set(name, xml[index] as string));
    else properties.remove(...names);
  };
  toggle(to.bold, from.bold, ["w:b", "w:bCs"], ["<w:b/>", "<w:bCs/>"]);
  toggle(to.italic, from.italic, ["w:i", "w:iCs"], ["<w:i/>", "<w:iCs/>"]);
  toggle(to.underline, from.underline, ["w:u"], ['<w:u w:val="single"/>']);
  if (to.strike !== from.strike) {
    if (to.strike) properties.set("w:strike", "<w:strike/>");
    else properties.remove("w:strike", "w:dstrike");
  }
  toggle(to.highlight, from.highlight, ["w:highlight"], ['<w:highlight w:val="yellow"/>']);
  if (to.code) properties.set("w:rFonts", CODE_FONT);
  const style = valueOf(properties.get("w:rStyle"));
  if (to.link && linkStyle && !style) {
    properties.set("w:rStyle", `<w:rStyle w:val="${escapeXml(linkStyle)}"/>`);
  } else if (!to.link && linkStyle && style === linkStyle) {
    properties.remove("w:rStyle");
  }
  return properties.toXml();
}
