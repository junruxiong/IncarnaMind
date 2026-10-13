/**
 * The main document part written back into (#105). Its paragraphs, tables
 * and kept-as-is Blocks are matched with the Mind's by where each came from;
 * those the Mind kept in order stay where they are, byte for byte unless
 * edited, those it deleted go, and what is new or moved is written after the
 * one before it. A table keeps its rows and cells and is written into cell by
 * cell, unless rows or columns were added, removed or merged: then it is
 * written again, each row and cell from the one it was, with its properties.
 */
import type { NodeJSON } from "../../../shared/blockJson";
import { readWordAtom, wordAtomText } from "../../../shared/wordFiles";
import type { WordAtomKind, WordExportLoss } from "../../api";
import type { WordDocument } from "../../documents/formats/docxImport";
import type { Relationship } from "../../documents/formats/docxPackage";
import { child, elements, walk, type XmlElement } from "../../documents/formats/xml";
import { type Block, type TableCell, tableLayout } from "../../mindText";
import { type DocxWriter, WRITER_NAMESPACES } from "../docx";
import { escapeXml } from "../wordml";
import type { AtomPlan } from "./atoms";
import { type ListPlace, NO_MARKS, type Unit, unitsOf } from "./mind";
import {
  baseTokens,
  type ParagraphContext,
  sourceOf,
  type Token,
  writeParagraph,
} from "./paragraph";
import type { DocumentStyles, PatchTarget } from "./parts";
import { Properties } from "./properties";
import {
  appendInside,
  applySplices,
  closeTag,
  endOf,
  type Namespaces,
  openTag,
  parentsOf,
  rescoped,
  type Splice,
  scopeOf,
  selfClosing,
  sliceOf,
  startOf,
} from "./text";

export interface DocumentPatchInput {
  base: WordDocument;
  /** The Mind's Blocks, as JSON. */
  current: readonly NodeJSON[];
  /** Where a node of the Mind came from in the document, if it did. */
  currentKey(node: NodeJSON): string | null;
  /** Whether a node imported from the document was edited since. */
  edited(node: NodeJSON): boolean;
  /** A paragraph's text now, Citations written in the Mind's style. */
  tokensOf(node: NodeJSON): Token[];
  /** What the native writer writes for a Block the document didn't have. */
  blocksOf(node: NodeJSON): Block[];
  writer: DocxWriter;
  target: PatchTarget;
  styles: DocumentStyles;
  /** The main part's relationships. */
  relationships: ReadonlyMap<string, Relationship>;
  plan: AtomPlan;
  /** The Bibliography a citation style adds, at the end of the document. */
  bibliography: readonly Block[];
}

/** A unit as imported, and the element its XML is. */
interface Based {
  unit: Unit;
  element: XmlElement;
}

/** Where new paragraphs take their properties and their text's formatting from. */
interface Reference {
  pPr: string;
  rPr: string;
}

/** CT_TcPr's children, in order. */
const CELL_ORDER = [
  "w:cnfStyle",
  "w:tcW",
  "w:gridSpan",
  "w:hMerge",
  "w:vMerge",
  "w:tcBorders",
  "w:shd",
  "w:noWrap",
  "w:tcMar",
  "w:textDirection",
  "w:tcFitText",
  "w:vAlign",
  "w:hideMark",
  "w:headers",
  "w:cellIns",
  "w:cellDel",
  "w:cellMerge",
  "w:tcPrChange",
];

const isTable = (element: XmlElement | null | undefined) => element?.name === "w:tbl";

export class DocumentPatch {
  private readonly text: string;
  private readonly parents: Map<XmlElement, XmlElement>;
  private readonly based = new Map<string, Based>();
  private readonly baseKey: (node: NodeJSON) => string | null;
  readonly losses: WordExportLoss[] = [];
  /** The numId each list of the Mind is written with, once it has one. */
  private readonly lists = new Map<NodeJSON, number>();
  /** Each list of the Mind's items, in order. */
  private readonly listItems = new Map<NodeJSON, Unit[]>();
  private readonly context: ParagraphContext;
  /** Paragraphs and tables holding something kept as is that is no longer written there. */
  private readonly forced = new Set<XmlElement>();

  constructor(private readonly input: DocumentPatchInput) {
    const { base } = input;
    this.text = base.text;
    this.parents = parentsOf(base.root);
    const ids = new Map<string, string>();
    for (const [id, source] of base.structure.sources) ids.set(id, source.n);
    this.baseKey = (node) => {
      const id = node.attrs?.id;
      return typeof id === "string" ? (ids.get(id) ?? null) : null;
    };
    this.collect(unitsOf(base.structure.blocks, this.baseKey), this.baseKey, (unit) => {
      const element = unit.key ? base.places.get(unit.key) : undefined;
      if (unit.key && element) this.based.set(unit.key, { unit, element });
    });
    this.collect(unitsOf(input.current, input.currentKey), input.currentKey, (unit) => {
      if (!unit.list) return;
      this.listItems.set(unit.list.list, [...(this.listItems.get(unit.list.list) ?? []), unit]);
    });
    for (const { unit, element } of this.based.values()) {
      if (unit.kind !== "paragraph") continue;
      const unwritten = (unit.node.content ?? []).some((node) => {
        const source = sourceOf(node);
        const at = source ? base.places.get(source) : undefined;
        return (
          source !== null &&
          at !== undefined &&
          this.contains(element, at) &&
          !input.plan.written(source)
        );
      });
      if (!unwritten) continue;
      for (let at: XmlElement | undefined = element; at; at = this.parents.get(at))
        this.forced.add(at);
    }
    const { writer, styles } = input;
    this.context = {
      text: this.text,
      parents: this.parents,
      places: base.places,
      special: (inline) => writer.inlineXml([inline]),
      hrefOf: (hyperlink) => this.hrefOf(hyperlink),
      newLink: (href) =>
        href.startsWith("#")
          ? `<w:hyperlink w:anchor="${escapeXml(href.slice(1))}" w:history="1">`
          : `<w:hyperlink r:id="${input.target.link(href, false)}" w:history="1">`,
      linkStyle: styles.find("character", "Hyperlink"),
    };
  }

  /** Every unit, those in tables' cells too. */
  private collect(
    units: readonly Unit[],
    keyOf: (node: NodeJSON) => string | null,
    visit: (unit: Unit) => void,
  ): void {
    for (const unit of units) {
      visit(unit);
      if (unit.kind !== "table") continue;
      for (const row of unit.node.content ?? []) {
        for (const cell of row.content ?? [])
          this.collect(unitsOf(cell.content ?? [], keyOf), keyOf, visit);
      }
    }
  }

  private hrefOf(hyperlink: XmlElement): string | null {
    const anchor = hyperlink.attrs["w:anchor"];
    const target = this.input.relationships.get(hyperlink.attrs["r:id"] ?? "");
    if (target?.external) return anchor ? `${target.target}#${anchor}` : target.target;
    return anchor ? `#${anchor}` : null;
  }

  /** The main part's text with every change made. */
  write(): string {
    const body = child(this.input.base.root, "w:body");
    if (!body) return this.text;
    const splices = this.container({
      element: body,
      contentStart: body.openEnd ?? startOf(body),
      base: unitsOf(this.input.base.structure.blocks, this.baseKey),
      current: unitsOf(this.input.current, this.input.currentKey),
      extra: [],
    });
    if (this.input.bibliography.length > 0) {
      const xml = this.newXml(this.input.writer.render(this.input.bibliography));
      const last = elements(body).at(-1);
      splices.push(
        last?.name === "w:sectPr"
          ? { from: startOf(last), to: startOf(last), text: xml }
          : appendInside(this.text, body, xml),
      );
    }
    if (selfClosing(body)) {
      // An empty body: what goes into it, in order, inside it.
      const added = splices.map((splice) => splice.text).join("");
      return applySplices(this.text, [appendInside(this.text, body, added)]);
    }
    splices.push(...this.unwrittenOutside(splices));
    return applySplices(this.text, splices);
  }

  /** XML the native writer wrote, for where it goes. */
  private newXml(xml: string, scope?: Namespaces): string {
    return rescoped(xml, WRITER_NAMESPACES, scope ?? scopeOf(this.input.base.root, this.parents));
  }

  // -------------------------------------------------------------------------
  // Containers

  /**
   * The changes to a container's units: the body's, or a cell's. Units in
   * `extra` (cells merged into this one) are kept or removed where they are,
   * but nothing new goes into them: they aren't shown.
   */
  private container(spec: {
    element: XmlElement;
    contentStart: number;
    base: readonly Unit[];
    current: readonly Unit[];
    extra: readonly XmlElement[];
  }): Splice[] {
    const { element } = spec;
    const scope = scopeOf(element, this.parents);
    const baseKeys = spec.base.flatMap((unit) =>
      unit.key && this.based.has(unit.key) ? [unit.key] : [],
    );
    const baseIndex = new Map(baseKeys.map((key, index) => [key, index]));
    const seen = new Set<string>();
    const matched = spec.current.map((unit) => {
      if (!unit.key || seen.has(unit.key) || !baseIndex.has(unit.key)) return -1;
      seen.add(unit.key);
      return baseIndex.get(unit.key) as number;
    });
    const kept = longestIncreasing(matched);
    const keptKeys = new Set([...kept].map((index) => baseKeys[index] as string));

    const splices: Splice[] = [];
    for (const key of baseKeys) {
      if (keptKeys.has(key)) continue;
      const based = this.based.get(key) as Based;
      splices.push({ from: startOf(based.element), to: endOf(based.element), text: "" });
    }

    const inElement = (node: XmlElement) => {
      for (let at = this.parents.get(node); at; at = this.parents.get(at)) {
        if (at === element) return true;
        if (at.name === "w:tc" || at.name === "w:body") return false;
      }
      return false;
    };
    let anchor = spec.contentStart;
    /** The unit written last: whether it is a table, and its place among the base's if it stayed. */
    let previous: { table: boolean; index: number | null } | null = null;
    spec.current.forEach((unit, index) => {
      const keptHere = matched[index] !== -1 && kept.has(matched[index] as number);
      const based = unit.key ? this.based.get(unit.key) : undefined;
      const table = unit.kind === "table" || (unit.kind === "kept" && isTable(based?.element));
      // Two tables next to each other would run together into one.
      const apart =
        table &&
        previous?.table === true &&
        !(keptHere && previous.index !== null && previous.index === (matched[index] as number) - 1);
      if (keptHere && based) {
        if (apart)
          splices.push({
            from: startOf(based.element),
            to: startOf(based.element),
            text: "<w:p/>",
          });
        splices.push(
          ...this.rewriteInPlace(unit, based, () => this.referenceAround(spec.current, index)),
        );
        if (inElement(based.element)) anchor = endOf(based.element);
      } else {
        const xml = this.unitXml(unit, scope, () => this.referenceAround(spec.current, index));
        splices.push({ from: anchor, to: anchor, text: apart ? `<w:p/>${xml}` : xml });
      }
      previous = { table, index: keptHere ? (matched[index] as number) : null };
    });

    // A cell ends with a paragraph, and so does the body before its last section's properties.
    if (splices.length > 0) {
      const last = spec.current.at(-1);
      const lastBased = last?.key ? this.based.get(last.key) : undefined;
      const endsInTable =
        last === undefined || last.kind === "table" || isTable(lastBased?.element);
      const stillLast =
        lastBased !== undefined &&
        last?.key === baseKeys.at(-1) &&
        keptKeys.has(lastBased.unit.key ?? "");
      if (endsInTable && !(stillLast && element.name === "w:body")) {
        splices.push({ from: anchor, to: anchor, text: "<w:p/>" });
      }
    }
    // A merged cell whose every paragraph went still needs one.
    for (const cell of spec.extra) {
      const units = baseKeys.filter(
        (key) => this.parents.get((this.based.get(key) as Based).element) === cell,
      );
      if (units.length > 0 && units.every((key) => !keptKeys.has(key))) {
        splices.push(appendInside(this.text, cell, "<w:p/>"));
      }
    }
    return splices;
  }

  /** The changes to a unit the Mind kept in its place: none unless it changed. */
  private rewriteInPlace(unit: Unit, based: Based, references: () => Reference[]): Splice[] {
    const { element } = based;
    const replace = (xml: string): Splice[] =>
      xml === sliceOf(this.text, element)
        ? []
        : [{ from: startOf(element), to: endOf(element), text: xml }];
    if (unit.kind === "paragraph") {
      if (!this.needsWriting(unit, based)) return [];
      return replace(this.paragraphXml(unit, based, scopeOf(element, this.parents), references));
    }
    if (unit.kind === "table") {
      if (!this.input.edited(unit.node) && !this.forced.has(element)) return [];
      return this.table(unit, based, true);
    }
    if (unit.kind === "kept") {
      const atom = readWordAtom(unit.node.attrs ?? {});
      if (atom.source && this.input.plan.writes(atom.source, unit.node, atom.kind)) return [];
      this.input.plan.copied(unit.node);
      return replace(this.newXml(this.input.writer.render(textBlocks(unit.node))));
    }
    return [];
  }

  /** A unit's XML, written where it wasn't before. */
  private unitXml(unit: Unit, scope: Namespaces, references: () => Reference[]): string {
    const based = unit.key ? this.based.get(unit.key) : undefined;
    const moved = (element: XmlElement) =>
      rescoped(
        sliceOf(this.text, element),
        scopeOf(this.parents.get(element) ?? element, this.parents),
        scope,
      );
    if (unit.kind === "paragraph") {
      if (based && !this.needsWriting(unit, based)) return moved(based.element);
      return this.paragraphXml(unit, based ?? null, scope, references);
    }
    if (unit.kind === "table" && based) {
      if (!this.input.edited(unit.node) && !this.forced.has(based.element))
        return moved(based.element);
      const [splice] = this.table(unit, based, false);
      const from = scopeOf(this.parents.get(based.element) ?? based.element, this.parents);
      return rescoped(splice?.text ?? "", from, scope);
    }
    if (unit.kind === "kept" && based) {
      const atom = readWordAtom(unit.node.attrs ?? {});
      if (atom.source && this.input.plan.writes(atom.source, unit.node, atom.kind)) {
        return moved(based.element);
      }
      this.input.plan.copied(unit.node);
      return this.newXml(this.input.writer.render(textBlocks(unit.node)), scope);
    }
    if (unit.kind === "kept")
      return this.newXml(this.input.writer.render(textBlocks(unit.node)), scope);
    return this.newXml(this.input.writer.render(this.input.blocksOf(unit.node)), scope);
  }

  /** Whether an imported paragraph must be written again: edited, moved in or out of a list, or holding what goes. */
  private needsWriting(unit: Unit, based: Based): boolean {
    return (
      this.input.edited(unit.node) ||
      this.forced.has(based.element) ||
      !sameList(unit.list, based.unit.list) ||
      this.sectionMoved(unit, based)
    );
  }

  private sectionMoved(unit: Unit, based: Based): boolean {
    const own = child(child(based.element, "w:pPr") ?? based.element, "w:sectPr");
    const ownSource = own ? this.input.base.places.placeOf(own) : null;
    const now = this.sectionSources(unit);
    return (
      (ownSource !== null && !now.includes(ownSource)) || now.some((each) => each !== ownSource)
    );
  }

  private contains(outer: XmlElement, inner: XmlElement): boolean {
    for (let at = this.parents.get(inner); at; at = this.parents.get(at))
      if (at === outer) return true;
    return false;
  }

  // -------------------------------------------------------------------------
  // Paragraphs

  /** The section breaks a paragraph holds now, by source, that it writes. */
  private sectionSources(unit: Unit): string[] {
    const sources: string[] = [];
    for (const node of unit.node.content ?? []) {
      const source = sourceOf(node);
      const element = source ? this.input.base.places.get(source) : undefined;
      if (
        source &&
        element?.name === "w:sectPr" &&
        this.input.plan.writes(source, unit.node, "sectionBreak")
      ) {
        sources.push(source);
      }
    }
    return sources;
  }

  private paragraphXml(
    unit: Unit,
    based: Based | null,
    scope: Namespaces,
    references: () => Reference[],
  ): string {
    const { plan } = this.input;
    const tokens = this.input.tokensOf(unit.node).flatMap((token): Token[] => {
      if (token.kind !== "atom" || !token.source) return [token];
      const node = (unit.node.content ?? []).find((each) => sourceOf(each) === token.source);
      const kind = node ? readWordAtom(node.attrs ?? {}).kind : ("other" as WordAtomKind);
      if (plan.writes(token.source, unit.node, node?.type === "image" ? "picture" : kind))
        return [token];
      if (!node || !plan.written(token.source)) return [];
      plan.copied(node);
      const text = node.type === "image" ? "" : wordAtomText(readWordAtom(node.attrs ?? {}));
      return Array.from(text, (each) => ({
        ...token,
        kind: "char" as const,
        key: each,
        char: each,
        marks: NO_MARKS,
        source: null,
      }));
    });
    const basePPr = based ? this.pPrOf(based.element) : "";
    const { pPr, rPr } = this.paragraphProperties(unit, based, references, basePPr);
    const { xml, lost } = writeParagraph(
      {
        base: based
          ? {
              element: based.element,
              tokens: baseTokens(based.unit.node.content ?? [], this.input.base.origins),
            }
          : null,
        current: tokens,
        pPr,
        pPrChanged: pPr !== basePPr,
        fallbackRun: rPr,
        scope,
      },
      {
        ...this.context,
        special: (inline) => this.newXml(this.input.writer.inlineXml([inline]), scope),
        newLink: (href) => this.newXml(this.context.newLink(href), scope),
      },
    );
    if (lost > 0) {
      const text = (unit.node.content ?? []).map((each) => each.text ?? "").join("");
      this.losses.push({
        key: `edit:${unit.key ?? text}`,
        kind: "edit",
        author: null,
        text: text.slice(0, 120),
      });
    }
    return xml;
  }

  private pPrOf(element: XmlElement): string {
    const pPr = child(element, "w:pPr");
    return pPr ? sliceOf(this.text, pPr) : "";
  }

  /**
   * A paragraph's properties as they are to be written, and the run
   * properties its new text starts from: an imported paragraph's own, changed
   * only where the Mind changed what it is (a heading's level, a list it is
   * in, a section break it ends); a new one's from the paragraph it follows,
   * as pressing Enter there would make it, or the list item beside it.
   */
  private paragraphProperties(
    unit: Unit,
    based: Based | null,
    references: () => Reference[],
    basePPr: string,
  ): { pPr: string; rPr: string } {
    const { styles } = this.input;
    const level =
      unit.node.type === "heading"
        ? Math.min(Math.max(Number(unit.node.attrs?.level) || 1, 1), 6)
        : null;
    let properties: Properties;
    let rPr = "";
    if (based) {
      properties = Properties.paragraph(basePPr);
      rPr = runOfMark(properties.get("w:rPr"));
      const baseLevel =
        based.unit.node.type === "heading" ? Number(based.unit.node.attrs?.level) || 1 : null;
      if (level !== baseLevel) {
        const direct = properties.has("w:outlineLvl");
        if (level !== null && direct && baseLevel !== null) {
          properties.set("w:outlineLvl", `<w:outlineLvl w:val="${level - 1}"/>`);
        } else if (level !== null) {
          properties.remove("w:outlineLvl");
          properties.set(
            "w:pStyle",
            `<w:pStyle w:val="${escapeXml(styles.id(`heading${level}` as "heading1"))}"/>`,
          );
        } else {
          properties.remove("w:outlineLvl");
          if (!direct) properties.remove("w:pStyle");
        }
      }
      if (!sameList(unit.list, based.unit.list)) this.setList(properties, unit.list, based);
    } else if (level !== null) {
      properties = Properties.paragraph("");
      properties.set(
        "w:pStyle",
        `<w:pStyle w:val="${escapeXml(styles.id(`heading${level}` as "heading1"))}"/>`,
      );
    } else if (unit.list) {
      const sibling = this.siblingOf(unit);
      properties = Properties.paragraph(sibling ? cleaned(this.pPrOf(sibling.element)) : "");
      rPr = sibling ? this.lastRunOf(sibling) : "";
      if (!sibling) {
        properties.set("w:pStyle", `<w:pStyle w:val="${escapeXml(styles.id("listParagraph"))}"/>`);
      }
      this.setList(properties, unit.list, null);
    } else {
      const reference = references()[0];
      properties = Properties.paragraph(reference ? reference.pPr : "");
      rPr = reference?.rPr ?? "";
    }

    // A section break goes with the paragraph that holds it now.
    const sections = this.sectionSources(unit);
    const sectPr = properties.get("w:sectPr");
    const ownSection = based
      ? child(child(based.element, "w:pPr") ?? based.element, "w:sectPr")
      : undefined;
    const ownSource = ownSection ? this.input.base.places.placeOf(ownSection) : null;
    const wanted = sections[0] ?? null;
    if (sectPr && wanted !== ownSource) properties.remove("w:sectPr");
    if (wanted && wanted !== ownSource) {
      const element = this.input.base.places.get(wanted);
      if (element) properties.set("w:sectPr", sliceOf(this.text, element));
    }
    return { pPr: !based || properties.changed ? properties.toXml() : basePPr, rPr };
  }

  /** Puts a paragraph in the list it is in now, or takes it out of any. */
  private setList(properties: Properties, list: ListPlace | null, based: Based | null): void {
    if (!list) {
      const own = properties.has("w:numPr");
      properties.remove("w:numPr");
      // Numbered by its style: only a list of none turns that off.
      if (based?.unit.list && !own) {
        properties.set("w:numPr", '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr>');
      }
      return;
    }
    const numId = this.numIdOf(list);
    properties.set(
      "w:numPr",
      `<w:numPr><w:ilvl w:val="${Math.min(list.depth, 8)}"/><w:numId w:val="${numId}"/></w:numPr>`,
    );
  }

  /** The numbering a list of the Mind is written with: that of its imported items, else a new one. */
  private numIdOf(list: ListPlace): number {
    const known = this.lists.get(list.list);
    if (known !== undefined) return known;
    for (const item of this.listItems.get(list.list) ?? []) {
      const based = item.key ? this.based.get(item.key) : undefined;
      if (!based?.unit.list || based.unit.list.ordered !== list.ordered) continue;
      const numId = numIdIn(this.pPrOf(based.element));
      if (numId !== null && numId !== "0") {
        this.lists.set(list.list, Number(numId));
        return Number(numId);
      }
    }
    const start = Math.max(Math.trunc(Number(list.list.attrs?.start)) || 1, 0);
    const numId = this.input.target.list(list.ordered, Math.min(list.depth, 8), start);
    this.lists.set(list.list, numId);
    return numId;
  }

  /** An imported item of the same list, the nearest before this one, else after: a new item takes its look. */
  private siblingOf(unit: Unit): Based | null {
    if (!unit.list) return null;
    const items = this.listItems.get(unit.list.list) ?? [];
    const at = items.indexOf(unit);
    const order = [...items.slice(0, at).reverse(), ...items.slice(at + 1)];
    for (const item of order) {
      const based = item.key ? this.based.get(item.key) : undefined;
      if (based && based.element.name === "w:p") return based;
    }
    return null;
  }

  /** The run properties of a paragraph's last text: what text typed after it would take. */
  private lastRunOf(based: Based): string {
    const nodes = based.unit.node.content ?? [];
    for (let index = nodes.length - 1; index >= 0; index--) {
      const node = nodes[index] as NodeJSON;
      if (node.type !== "text") continue;
      const run = this.input.base.origins.get(node);
      const rPr = run ? child(run, "w:rPr") : undefined;
      if (!run || this.inHyperlink(run)) continue;
      return rPr ? sliceOf(this.text, rPr) : "";
    }
    return runOfMark(Properties.paragraph(this.pPrOf(based.element)).get("w:rPr"));
  }

  private inHyperlink(element: XmlElement): boolean {
    for (let at = this.parents.get(element); at && at.name !== "w:p"; at = this.parents.get(at)) {
      if (at.name === "w:hyperlink") return true;
    }
    return false;
  }

  /**
   * What new paragraphs among a container's units take their look from: the
   * nearest plain imported paragraph before, else after.
   */
  private referenceAround(units: readonly Unit[], index: number): Reference[] {
    const plain = (unit: Unit | undefined) => {
      const based = unit?.key ? this.based.get(unit.key) : undefined;
      return based &&
        based.unit.node.type === "paragraph" &&
        !based.unit.list &&
        based.element.name === "w:p"
        ? based
        : null;
    };
    for (const order of [units.slice(0, index).reverse(), units.slice(index + 1)]) {
      for (const unit of order) {
        const based = plain(unit);
        if (based) return [{ pPr: cleaned(this.pPrOf(based.element)), rPr: this.lastRunOf(based) }];
      }
    }
    return [];
  }

  // -------------------------------------------------------------------------
  // Tables

  /** A table's changes: cell by cell when its shape is the same, else the table written again. */
  private table(unit: Unit, based: Based, inPlace: boolean): Splice[] {
    const { element } = based;
    const baseRows = based.unit.node.content ?? [];
    const rows = unit.node.content ?? [];
    const span = (cell: NodeJSON, name: string) =>
      Math.max(Math.trunc(Number(cell.attrs?.[name])) || 1, 1);
    const sameShape =
      rows.length === baseRows.length &&
      rows.every((row, r) => {
        const baseCells = baseRows[r]?.content ?? [];
        const cells = row.content ?? [];
        return (
          cells.length === baseCells.length &&
          cells.every((cell, c) => {
            const baseCell = baseCells[c] as NodeJSON;
            return (
              span(cell, "colspan") === span(baseCell, "colspan") &&
              span(cell, "rowspan") === span(baseCell, "rowspan")
            );
          })
        );
      });
    if (!sameShape) {
      const xml = this.rebuiltTable(unit, based);
      return [{ from: startOf(element), to: endOf(element), text: xml }];
    }
    const splices: Splice[] = [];
    rows.forEach((row, r) => {
      const baseRow = baseRows[r] as NodeJSON;
      const tr = this.input.base.origins.get(baseRow);
      const header = isHeaderRow(row);
      if (tr && header !== isHeaderRow(baseRow)) splices.push(...this.headerSplices(tr, header));
      (row.content ?? []).forEach((cell, c) => {
        const baseCell = (baseRow.content ?? [])[c] as NodeJSON;
        const tc = this.input.base.origins.get(baseCell);
        if (!tc) {
          this.paddingLost(cell);
          return;
        }
        splices.push(...this.cell(tc, baseCell, cell));
      });
    });
    if (inPlace) return splices;
    const start = startOf(element);
    const xml = applySplices(
      sliceOf(this.text, element),
      splices.map((splice) => ({ ...splice, from: splice.from - start, to: splice.to - start })),
    );
    return [{ from: start, to: endOf(element), text: xml }];
  }

  /** Text typed into a place no cell of the document fills, which can't be written. */
  private paddingLost(cell: NodeJSON): void {
    const text = [...flatText(cell)].join("").trim();
    if (text)
      this.losses.push({
        key: `edit:cell:${text.slice(0, 40)}`,
        kind: "edit",
        author: null,
        text: text.slice(0, 120),
      });
  }

  private cell(tc: XmlElement, baseCell: NodeJSON, cell: NodeJSON): Splice[] {
    const tcPr = child(tc, "w:tcPr");
    return this.container({
      element: tc,
      contentStart: tcPr ? endOf(tcPr) : (tc.openEnd ?? startOf(tc)),
      base: unitsOf(baseCell.content ?? [], this.baseKey),
      current: unitsOf(cell.content ?? [], this.input.currentKey),
      extra: this.input.base.merged.get(baseCell) ?? [],
    });
  }

  private headerSplices(tr: XmlElement, header: boolean): Splice[] {
    const trPr = child(tr, "w:trPr");
    if (trPr) {
      const properties = elements(trPr).filter((each) => each.name !== "w:tblHeader");
      const inner =
        properties.map((each) => sliceOf(this.text, each)).join("") +
        (header ? "<w:tblHeader/>" : "");
      return [
        { from: startOf(trPr), to: endOf(trPr), text: inner ? `<w:trPr>${inner}</w:trPr>` : "" },
      ];
    }
    if (!header) return [];
    const before = child(tr, "w:tblPrEx");
    const at = before ? endOf(before) : (tr.openEnd ?? startOf(tr));
    return [{ from: at, to: at, text: "<w:trPr><w:tblHeader/></w:trPr>" }];
  }

  /**
   * A table whose rows or columns changed, written again: its properties as
   * they were, its grid as it was if it is as wide, each row and cell from the
   * one the Mind's came from (properties and content), new ones like those
   * beside them, and merged cells as Word writes them.
   */
  private rebuiltTable(unit: Unit, based: Based): string {
    const { element } = based;
    const text = this.text;
    const tblPr = child(element, "w:tblPr");
    const grid = child(element, "w:tblGrid");
    const widths = grid
      ? elements(grid)
          .filter((each) => each.name === "w:gridCol")
          .map((each) => Number(each.attrs["w:w"]) || 0)
      : [];

    const rows = unit.node.content ?? [];
    const shapes: TableCell[][] = rows.map((row) =>
      (row.content ?? []).map((cell) => ({
        header: cell.type === "tableHeader",
        colspan: Math.max(Math.trunc(Number(cell.attrs?.colspan)) || 1, 1),
        rowspan: Math.max(Math.trunc(Number(cell.attrs?.rowspan)) || 1, 1),
        align: null,
        content: [],
      })),
    );
    const nodeOf = new Map<TableCell, NodeJSON>();
    for (const [r, cells] of shapes.entries()) {
      for (const [c, cell] of cells.entries())
        nodeOf.set(cell, (rows[r]?.content ?? [])[c] as NodeJSON);
    }
    const layout = tableLayout(shapes);
    const total = widths.reduce((sum, width) => sum + width, 0) || 9000;
    const columnWidths =
      widths.length === layout.columns && widths.every((width) => width > 0)
        ? widths
        : Array.from({ length: layout.columns }, () => Math.floor(total / layout.columns));
    const gridXml =
      grid && widths.length === layout.columns
        ? sliceOf(text, grid)
        : `<w:tblGrid>${columnWidths.map((width) => `<w:gridCol w:w="${width}"/>`).join("")}</w:tblGrid>`;

    // Where each of the Mind's cells came from: the cell whose paragraphs it holds.
    const baseCells = new Map<string, { tc: XmlElement; tr: XmlElement; node: NodeJSON }>();
    for (const baseRow of based.unit.node.content ?? []) {
      const tr = this.input.base.origins.get(baseRow);
      for (const baseCell of baseRow.content ?? []) {
        const tc = this.input.base.origins.get(baseCell);
        if (!tr || !tc) continue;
        for (const inner of unitsOf(baseCell.content ?? [], this.baseKey)) {
          if (inner.key) baseCells.set(inner.key, { tc, tr, node: baseCell });
        }
      }
    }
    const originOf = (cell: NodeJSON) => {
      for (const inner of unitsOf(cell.content ?? [], this.input.currentKey)) {
        const found = inner.key ? baseCells.get(inner.key) : undefined;
        if (found) return found;
      }
      return null;
    };
    const used = new Set<XmlElement>();
    const chainOf = (from: XmlElement, upTo: XmlElement) => {
      const chain: XmlElement[] = [];
      for (let at = this.parents.get(from); at && at !== upTo; at = this.parents.get(at))
        chain.unshift(at);
      return chain;
    };
    const rowWrappers = new Wrapping(text);
    let rowChain: XmlElement[] = [];

    let xml = openTag(text, element) + (tblPr ? sliceOf(text, tblPr) : "") + gridXml;
    xml += this.keptBetween(element);
    layout.rows.forEach((slots, r) => {
      const row = rows[r] as NodeJSON;
      const origins = (row.content ?? []).map(originOf);
      const found = origins.find((each) => each !== null)?.tr ?? null;
      const tr = found && !used.has(found) ? found : null;
      if (tr) used.add(tr);
      // A new row goes where the row before it is: inside a repeating section, say.
      if (found) rowChain = chainOf(found, element);
      xml += rowWrappers.to(rowChain);
      const header = isHeaderRow(row);
      let rowXml = tr ? openTag(text, tr) : "<w:tr>";
      const tblPrEx = tr ? child(tr, "w:tblPrEx") : undefined;
      if (tblPrEx) rowXml += sliceOf(text, tblPrEx);
      const trPr = tr ? child(tr, "w:trPr") : undefined;
      const rowProperties = trPr
        ? elements(trPr)
            .filter((each) => each.name !== "w:tblHeader")
            .map((each) => sliceOf(text, each))
            .join("")
        : "";
      const rowPr = rowProperties + (header ? "<w:tblHeader/>" : "");
      if (rowPr) rowXml += `<w:trPr>${rowPr}</w:trPr>`;
      const cellWrappers = new Wrapping(text);
      let cellChain: XmlElement[] = [];
      let column = 0;
      for (const slot of slots) {
        const colspan = slot.cell?.colspan ?? 1;
        const width = columnWidths
          .slice(column, column + colspan)
          .reduce((sum, each) => sum + each, 0);
        column += colspan;
        if (!slot.cell || slot.continued) {
          rowXml += `${cellWrappers.to(cellChain)}<w:tc><w:tcPr><w:tcW w:w="${width}" w:type="dxa"/>${colspan > 1 ? `<w:gridSpan w:val="${colspan}"/>` : ""}${slot.continued ? "<w:vMerge/>" : ""}</w:tcPr><w:p/></w:tc>`;
          continue;
        }
        const node = nodeOf.get(slot.cell) as NodeJSON;
        const origin = originOf(node);
        const fresh = origin && !used.has(origin.tc) ? origin : null;
        if (fresh) used.add(fresh.tc);
        if (origin) cellChain = chainOf(origin.tc, origin.tr);
        rowXml += cellWrappers.to(cellChain) + this.rebuiltCell(node, fresh, slot.cell, width);
      }
      xml += `${rowXml}${cellWrappers.to([])}</w:tr>`;
    });
    return `${xml}${rowWrappers.to([])}${closeTag(element)}`;
  }

  private rebuiltCell(
    node: NodeJSON,
    origin: { tc: XmlElement; node: NodeJSON } | null,
    shape: TableCell,
    width: number,
  ): string {
    const text = this.text;
    const tcPr = origin ? child(origin.tc, "w:tcPr") : undefined;
    const properties = new CellProperties(tcPr ? sliceOf(text, tcPr) : "");
    properties.set("w:tcW", `<w:tcW w:w="${width}" w:type="dxa"/>`);
    properties.remove("w:hMerge", "w:vMerge", "w:gridSpan");
    if (shape.colspan > 1) properties.set("w:gridSpan", `<w:gridSpan w:val="${shape.colspan}"/>`);
    if (shape.rowspan > 1) properties.set("w:vMerge", '<w:vMerge w:val="restart"/>');
    let content: string;
    if (origin) {
      const { tc } = origin;
      const splices = this.cell(tc, origin.node, node);
      const start = startOf(tc);
      const patched = applySplices(
        sliceOf(text, tc),
        splices.map((splice) => ({ ...splice, from: splice.from - start, to: splice.to - start })),
      );
      const own = child(tc, "w:tcPr");
      const from = (own ? endOf(own) : (tc.openEnd ?? start)) - start;
      content = selfClosing(tc) ? "" : patched.slice(from, patched.length - closeTag(tc).length);
    } else {
      content = unitsOf(node.content ?? [], this.input.currentKey)
        .map((unit) => this.unitXml(unit, scopeOf(this.input.base.root, this.parents), () => []))
        .join("");
    }
    if (!content.trimEnd().endsWith("</w:p>") && !content.trimEnd().endsWith("<w:p/>"))
      content += "<w:p/>";
    const open = origin ? openTag(text, origin.tc) : "<w:tc>";
    return `${open}${properties.toXml()}${content}</w:tc>`;
  }

  /** Marks between a table's rows and cells (a bookmark) that are still written, kept at its start. */
  private keptBetween(table: XmlElement): string {
    let xml = "";
    for (const each of walk(table, (element) => element.name !== "w:tc")) {
      const place = this.input.base.places.placeOf(each);
      if (place && each.name !== "w:tc" && this.input.plan.written(place))
        xml += sliceOf(this.text, each);
    }
    return xml;
  }

  // -------------------------------------------------------------------------

  /**
   * Kept-as-is marks between paragraphs (a bookmark the Mind holds in the
   * paragraph after it) that nothing writes any more: removed where they are,
   * unless what holds them is written again anyway.
   */
  private unwrittenOutside(splices: readonly Splice[]): Splice[] {
    const unitElements = new Set([...this.based.values()].map((each) => each.element));
    const covered = (element: XmlElement) =>
      splices.some(
        (splice) =>
          splice.from < splice.to && splice.from <= startOf(element) && endOf(element) <= splice.to,
      );
    const removed: Splice[] = [];
    for (const based of this.based.values()) {
      if (based.unit.kind !== "paragraph") continue;
      for (const node of based.unit.node.content ?? []) {
        const source = sourceOf(node);
        const element = source ? this.input.base.places.get(source) : undefined;
        if (!source || !element || this.input.plan.written(source)) continue;
        let inside = false;
        for (let at = this.parents.get(element); at; at = this.parents.get(at)) {
          if (unitElements.has(at)) inside = true;
        }
        if (!inside && !covered(element))
          removed.push({ from: startOf(element), to: endOf(element), text: "" });
      }
    }
    return removed;
  }
}

/** The properties of a wrapper (a content control's, custom XML's), written when it is opened. */
const WRAPPER_PROPERTIES = new Set(["w:sdtPr", "w:sdtEndPr", "w:customXmlPr"]);

/**
 * Wrappers (content controls, custom XML) around rows or cells as a table is
 * written again: opened and closed so that exactly those a row or cell was in
 * are open. One closed once isn't opened again, so its id isn't written twice.
 */
class Wrapping {
  private open: XmlElement[] = [];
  private readonly closed = new Set<XmlElement>();

  constructor(private readonly text: string) {}

  to(chain: readonly XmlElement[]): string {
    const wanted = chain.filter((each) => this.open.includes(each) || !this.closed.has(each));
    let same = 0;
    while (same < this.open.length && same < wanted.length && this.open[same] === wanted[same])
      same++;
    let xml = "";
    for (let at = this.open.length - 1; at >= same; at--) {
      const element = this.open[at] as XmlElement;
      xml += closeTag(element);
      this.closed.add(element);
    }
    for (const element of wanted.slice(same)) {
      xml += openTag(this.text, element);
      for (const each of elements(element)) {
        if (WRAPPER_PROPERTIES.has(each.name)) xml += sliceOf(this.text, each);
      }
    }
    this.open = wanted;
    return xml;
  }
}

/** A cell's properties, changed in place. */
class CellProperties {
  private readonly properties: Properties;
  constructor(xml: string) {
    this.properties = Properties.ofOrder("w:tcPr", xml, CELL_ORDER);
  }
  set(name: string, xml: string): void {
    this.properties.set(name, xml);
  }
  remove(...names: string[]): void {
    this.properties.remove(...names);
  }
  toXml(): string {
    return this.properties.toXml() || "<w:tcPr/>";
  }
}

const sameList = (a: ListPlace | null, b: ListPlace | null): boolean =>
  (a === null && b === null) ||
  (a !== null && b !== null && a.ordered === b.ordered && a.depth === b.depth);

const isHeaderRow = (row: NodeJSON) =>
  (row.content ?? []).length > 0 &&
  (row.content ?? []).every((cell) => cell.type === "tableHeader");

/** The numId a paragraph's own numbering names, if it has any. */
function numIdIn(pPr: string): string | null {
  return /<w:numId\b[^>]*\bw:val="([^"]*)"/.exec(pPr)?.[1] ?? null;
}

/**
 * Properties to start a new paragraph from: what makes a paragraph the one it
 * is (its section, a frame, a page break before it, a tracked change to it)
 * left out.
 */
function cleaned(pPr: string): string {
  const properties = Properties.paragraph(pPr).remove(
    "w:sectPr",
    "w:pPrChange",
    "w:framePr",
    "w:pageBreakBefore",
    "w:numPr",
    "w:outlineLvl",
  );
  const rPr = properties.get("w:rPr");
  if (rPr) {
    const run = Properties.run(rPr).remove(
      "w:ins",
      "w:del",
      "w:moveFrom",
      "w:moveTo",
      "w:rPrChange",
    );
    const xml = run.toXml();
    if (xml) properties.set("w:rPr", xml);
    else properties.remove("w:rPr");
  }
  return properties.toXml();
}

/** A paragraph mark's run properties as a run's: without its own tracked changes. */
function runOfMark(rPr: string | null): string {
  if (!rPr) return "";
  return Properties.run(rPr)
    .remove("w:ins", "w:del", "w:moveFrom", "w:moveTo", "w:rPrChange")
    .toXml();
}

/** A kept-as-is Block's text, a paragraph a line, for writing a copy. */
function textBlocks(node: NodeJSON): Block[] {
  return wordAtomText(readWordAtom(node.attrs ?? {}))
    .split("\n")
    .filter((line) => line.trim() !== "")
    .map((line) => ({
      kind: "paragraph",
      content: [
        {
          kind: "text",
          text: line,
          marks: {
            bold: false,
            italic: false,
            strike: false,
            underline: false,
            highlight: false,
            code: false,
            link: null,
          },
        },
      ],
    }));
}

function* flatText(node: NodeJSON): Generator<string> {
  if (node.text) yield node.text;
  for (const each of node.content ?? []) yield* flatText(each);
}

/** The indices in the longest increasing run of `values` (ignoring -1s): what stays in its place. */
function longestIncreasing(values: readonly number[]): Set<number> {
  const tails: number[] = [];
  const tailIndex: number[] = [];
  const previous = new Array<number>(values.length).fill(-1);
  values.forEach((value, index) => {
    if (value < 0) return;
    let low = 0;
    let high = tails.length;
    while (low < high) {
      const middle = (low + high) >> 1;
      if ((tails[middle] as number) < value) low = middle + 1;
      else high = middle;
    }
    tails[low] = value;
    tailIndex[low] = index;
    previous[index] = low > 0 ? (tailIndex[low - 1] as number) : -1;
  });
  const kept = new Set<number>();
  for (let at = tailIndex[tails.length - 1] ?? -1; at >= 0; at = previous[at] as number) {
    kept.add(values[at] as number);
  }
  return kept;
}
