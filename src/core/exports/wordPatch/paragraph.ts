/**
 * A paragraph written back into its own XML (#105). Its text as imported and
 * as it is now are compared character by character (Myers' diff), and only
 * what changed is written again: each child of the `w:p` whose text was all
 * kept, with its marks, is copied as it was; a child with an edit is written
 * again run by run, each character in the run it came from, with that run's
 * properties, so formatting the editor doesn't show (a font, a size, a
 * colour) stays. Typed text takes the formatting of the character before it,
 * as Word's typing does; the editor's marks are then set on top. What the
 * import kept as is (a field, a footnote mark, a bookmark…) goes back as its
 * original XML. A new paragraph is written the same way, with nothing to keep.
 */
import type { MarkJSON, NodeJSON } from "../../../shared/blockJson";
import { readWordAtom } from "../../../shared/wordFiles";
import { IMAGE_NODE, WORD_ATOM_NODE } from "../../api";
import type { Places } from "../../documents/formats/docxImport";
import { elements, type XmlElement } from "../../documents/formats/xml";
import type { Inline } from "../../mindText";
import { diffSteps, splitWords } from "../../proposals/diff";
import { escapeXml } from "../wordml";
import { marksOf, NO_MARKS, sameMarks, type TextMarks } from "./mind";
import { runPropertiesFor } from "./properties";
import {
  closeTag,
  endOf,
  type Namespaces,
  openTag,
  rescoped,
  scopeOf,
  sliceOf,
  startOf,
} from "./text";

/** One thing in a paragraph's text: a character, a line break, something kept as is, or new. */
export interface Token {
  kind: "char" | "break" | "atom" | "special";
  /** What compares equal: the character, a line break, the atom's source. A special never does. */
  key: string;
  char: string;
  marks: TextMarks;
  /** As imported: the run its text was read from. */
  run: XmlElement | null;
  /** An atom's source: "w:bookmarkStart#3", "w:r#40..w:r#44". */
  source: string | null;
  /** A special's content, for the native writer: a Citation's footnote, maths, a new picture. */
  inline: Inline | null;
}

const BREAK = "\u0001break";
let specials = 0;

const char = (value: string, marks: TextMarks, run: XmlElement | null): Token => ({
  kind: "char",
  key: value,
  char: value,
  marks,
  run,
  source: null,
  inline: null,
});

const atom = (source: string): Token => ({
  kind: "atom",
  key: `\u0001${source}`,
  char: "",
  marks: NO_MARKS,
  run: null,
  source,
  inline: null,
});

/** A special: never equal to anything, so it is always written new. */
export const special = (inline: Inline): Token => ({
  kind: "special",
  key: `\u0002${specials++}`,
  char: "",
  marks: NO_MARKS,
  run: null,
  source: null,
  inline,
});

const textTokens = (text: string, marks: TextMarks, run: XmlElement | null) =>
  Array.from(text, (each) => char(each, marks, run));

/** A paragraph's text as the import read it, each character with the run it came from. */
export function baseTokens(
  content: readonly NodeJSON[],
  origins: ReadonlyMap<NodeJSON, XmlElement>,
): Token[] {
  const tokens: Token[] = [];
  for (const node of content) {
    const run = origins.get(node) ?? null;
    if (node.type === "text") tokens.push(...textTokens(node.text ?? "", marksOf(node.marks), run));
    else if (node.type === "hardBreak")
      tokens.push({ ...char("\n", NO_MARKS, run), kind: "break", key: BREAK });
    else {
      const source = sourceOf(node);
      if (source) tokens.push(atom(source));
    }
  }
  return tokens;
}

/** Where a node kept from the document came from: an atom's or an imported picture's source. */
export function sourceOf(node: NodeJSON): string | null {
  if (node.type === WORD_ATOM_NODE) return readWordAtom(node.attrs ?? {}).source;
  if (node.type === IMAGE_NODE) {
    const source = node.attrs?.source;
    return typeof source === "string" && source !== "" ? source : null;
  }
  return null;
}

/**
 * A paragraph's text as it is now. `specialOf` reads what the native writer
 * writes (Citations, footnotes, maths, a new picture); other inline nodes
 * read as their text.
 */
export function currentTokens(
  content: readonly NodeJSON[],
  specialOf: (node: NodeJSON) => Inline[],
  textOf: (node: NodeJSON) => string,
): Token[] {
  const tokens: Token[] = [];
  for (const node of content) {
    if (node.type === "text") {
      tokens.push(...textTokens(node.text ?? "", marksOf(node.marks as MarkJSON[]), null));
    } else if (node.type === "hardBreak") {
      tokens.push({ ...char("\n", NO_MARKS, null), kind: "break", key: BREAK });
    } else {
      const source = sourceOf(node);
      if (source) tokens.push(atom(source));
      else {
        const inlines = specialOf(node);
        if (inlines.length > 0) tokens.push(...inlines.map(special));
        else tokens.push(...textTokens(textOf(node), NO_MARKS, null));
      }
    }
  }
  return tokens;
}

/** What writing a paragraph needs of the part it is in. */
export interface ParagraphContext {
  text: string;
  parents: ReadonlyMap<XmlElement, XmlElement>;
  places: Places;
  /** A special's XML, from the native writer. */
  special(inline: Inline): string;
  /** Where an original hyperlink leads, as the import reads it. */
  hrefOf(hyperlink: XmlElement): string | null;
  /** A new hyperlink's start tag. */
  newLink(href: string): string;
  /** The id of the character style links are in, if the document has one. */
  linkStyle: string | null;
  /** The id of that style, added to the document if it hasn't one: for a new link. */
  addLinkStyle(): string;
}

export interface ParagraphInput {
  /** The paragraph as imported, and its text; none for a new paragraph. */
  base: { element: XmlElement; tokens: Token[] } | null;
  current: Token[];
  /** The paragraph's properties as they are to be written (a whole `w:pPr`, or ""). */
  pPr: string;
  /** Whether `pPr` differs from what the paragraph has. */
  pPrChanged: boolean;
  /** Run properties for text with no character before or after it to take its formatting from. */
  fallbackRun: string;
  /** Where the paragraph is written: the namespaces in scope there. */
  scope: Namespaces;
}

/** A child of the paragraph, or several a field spans, written as one. */
interface Piece {
  from: number;
  to: number;
  /** Indices of the base tokens it holds. */
  tokens: number[];
  /** Its field's runs aren't siblings: it can only be written as it was. */
  frozen: boolean;
  dirty: boolean;
  written: boolean;
}

/** A paragraph's XML; `lost` counts edits next to a field that couldn't be written. */
export function writeParagraph(
  input: ParagraphInput,
  context: ParagraphContext,
): { xml: string; lost: number } {
  const { base, current } = input;
  const { text, parents, places } = context;
  const element = base?.element ?? null;
  const children = element ? elements(element).filter((each) => each.name !== "w:pPr") : [];
  const childIndex = new Map(children.map((each, index) => [each, index]));
  const scope = element ? scopeOf(element, parents) : input.scope;

  /** The index of the paragraph's child an element is in; -1 outside the paragraph. */
  const segmentOf = (node: XmlElement | undefined): number => {
    for (let at = node; at; at = parents.get(at)) {
      const index = childIndex.get(at);
      if (index !== undefined && parents.get(at) === element) return index;
      if (at === element) return -1;
    }
    return -1;
  };
  /** The elements an atom's source names: one, or a field's first and last run. */
  const elementsOf = (source: string): XmlElement[] =>
    source
      .split("..")
      .map((each) => places.get(each))
      .filter((each): each is XmlElement => each !== undefined);

  const baseList = base?.tokens ?? [];
  const spans = baseList.map((token): [number, number] => {
    if (token.kind === "atom" && token.source) {
      const [first, last = first] = elementsOf(token.source);
      return [segmentOf(first), segmentOf(last)];
    }
    const at = segmentOf(token.run ?? undefined);
    return [at, at];
  });

  // Pieces: the children, those a field spans joined.
  const pieceOf = new Array<Piece | null>(children.length).fill(null);
  const pieces: Piece[] = [];
  for (let index = 0; index < children.length; ) {
    let to = index;
    for (const [from, last] of spans) if (from >= 0 && from <= to && last > to) to = last;
    const piece: Piece = {
      from: index,
      to,
      tokens: [],
      frozen: false,
      dirty: false,
      written: false,
    };
    for (let at = index; at <= to; at++) pieceOf[at] = piece;
    pieces.push(piece);
    index = to + 1;
  }
  baseList.forEach((token, index) => {
    const [from] = spans[index] as [number, number];
    const piece = from >= 0 ? pieceOf[from] : null;
    if (!piece) return;
    piece.tokens.push(index);
    if (token.kind === "atom" && token.source?.includes("..")) {
      const [first, last] = elementsOf(token.source);
      if (first && last && parents.get(first) !== parents.get(last)) piece.frozen = true;
    }
  });
  const pieceOfToken = (index: number) => {
    const from = spans[index]?.[0] ?? -1;
    return from >= 0 ? pieceOf[from] : null;
  };

  // What changed, word by word, as Proposals compare text.
  const writable = current;
  const steps = diffTokens(baseList, writable);
  /** For each current token, the base token it is (equal), or -1 (new). */
  const baseOf = new Array<number>(writable.length).fill(-1);
  /** Each new token's child: that of the base token before it in the comparison, else after. */
  const homeOf = new Array<Piece | null>(writable.length).fill(null);
  /** Each character's formatting: its own run's; typed text takes the run of what it replaces, else of the character before it. */
  const formatRun = new Array<XmlElement | null>(writable.length).fill(null);
  let x = 0;
  let y = 0;
  let lastPiece: Piece | null = null;
  let previousRun: XmlElement | null = null;
  let replacing: XmlElement | null = null;
  const waiting: number[] = [];
  const isText = (token: Token | undefined) => token?.kind === "char" || token?.kind === "break";
  for (const step of steps) {
    if (step === "insert") {
      homeOf[y] = lastPiece;
      if (!lastPiece) waiting.push(y);
      if (isText(writable[y])) {
        formatRun[y] = replacing ?? previousRun;
        previousRun = formatRun[y] ?? previousRun;
      }
      y++;
      continue;
    }
    const piece = pieceOfToken(x);
    const was = baseList[x] as Token;
    if (step === "equal") {
      baseOf[y] = x;
      const now = writable[y] as Token;
      if (piece && isText(was) && !sameMarks(was.marks, now.marks)) piece.dirty = true;
      if (isText(was)) {
        formatRun[y] = was.run;
        previousRun = was.run ?? previousRun;
      }
      replacing = null;
      y++;
    } else {
      if (piece) piece.dirty = true;
      if (isText(was) && replacing === null) replacing = was.run;
    }
    if (piece) {
      for (const at of waiting) homeOf[at] = piece;
      waiting.length = 0;
      lastPiece = piece;
    }
    x++;
  }
  for (const [index, home] of homeOf.entries()) {
    if (baseOf[index] !== -1 || !home) continue;
    if (home.frozen) homeOf[index] = null;
    else home.dirty = true;
  }
  let following: XmlElement | null = null;
  for (let index = writable.length - 1; index >= 0; index--) {
    if (!isText(writable[index])) continue;
    if (baseOf[index] !== -1) following = formatRun[index] ?? following;
    formatRun[index] ??= following;
  }

  if (
    base &&
    !input.pPrChanged &&
    steps.every((step) => step === "equal") &&
    pieces.every((piece) => !piece.dirty)
  ) {
    return { xml: sliceOf(text, base.element), lost: 0 };
  }

  const writer = new RunWriter(input, context, scope, element);
  let out = "";
  let lost = 0;
  /** The last child written as it was, so the space between it and the next is kept. */
  let lastVerbatim = -1;
  const verbatim = (piece: Piece) => {
    const from = children[piece.from] as XmlElement;
    const to = children[piece.to] as XmlElement;
    out += writer.flush();
    const gapStart =
      lastVerbatim >= 0 && lastVerbatim === piece.from - 1
        ? endOf(children[lastVerbatim] as XmlElement)
        : startOf(from);
    out += text.slice(gapStart, endOf(to));
    lastVerbatim = piece.to;
    piece.written = true;
  };
  /** Children with nothing the Mind shows (a proofing mark), kept where they were. */
  const passiveBefore = (index: number) => {
    for (const piece of pieces) {
      if (piece.to >= index) break;
      if (!piece.written && piece.tokens.length === 0) verbatim(piece);
    }
  };

  writable.forEach((token, index) => {
    const from = baseOf[index] as number;
    const piece = from >= 0 ? pieceOfToken(from) : homeOf[index];
    if (from >= 0 && !piece) return; // kept outside the paragraph, where it is
    if (piece && !piece.dirty) {
      if (!piece.written) {
        passiveBefore(piece.from);
        verbatim(piece);
      }
      return;
    }
    if (piece?.frozen) {
      if (!piece.written) {
        passiveBefore(piece.from);
        verbatim(piece);
      }
      if (from < 0 || !sameMarks((baseList[from] as Token).marks, token.marks)) lost++;
      return;
    }
    if (piece) passiveBefore(piece.from);
    lastVerbatim = -1;
    writer.add(token, formatRun[index] ?? null);
  });
  for (const piece of pieces) {
    if (piece.frozen && piece.dirty && !piece.written) {
      // A field joined to text that was all deleted: the field and its text stay.
      verbatim(piece);
      lost += piece.tokens.length;
    }
  }
  out += writer.flush();
  passiveBefore(Number.POSITIVE_INFINITY);
  out += writer.flush();

  const open = element ? openTag(text, element) : "<w:p>";
  return { xml: `${open}${input.pPr}${out}</w:p>`, lost };
}

/**
 * The steps from one paragraph's tokens to another's, comparing text by words
 * (`splitWords`), so a word replaced is replaced whole rather than kept letter
 * by letter where the two happen to share some.
 */
function diffTokens(a: readonly Token[], b: readonly Token[]): ReturnType<typeof diffSteps> {
  const unitsOf = (tokens: readonly Token[]) => {
    const units: { key: string; length: number }[] = [];
    for (let index = 0; index < tokens.length; ) {
      const token = tokens[index] as Token;
      if (token.kind !== "char") {
        units.push({ key: token.key, length: 1 });
        index++;
        continue;
      }
      let end = index;
      let text = "";
      for (; tokens[end]?.kind === "char"; end++) text += (tokens[end] as Token).char;
      for (const word of splitWords(text))
        units.push({ key: word, length: Array.from(word).length });
      index = end;
    }
    return units;
  };
  const left = unitsOf(a);
  const right = unitsOf(b);
  const steps: ReturnType<typeof diffSteps> = [];
  let x = 0;
  let y = 0;
  for (const step of diffSteps(
    left.map((unit) => unit.key),
    right.map((unit) => unit.key),
  )) {
    if (step === "equal") {
      for (let at = 0; at < (left[x] as { length: number }).length; at++) steps.push("equal");
      x++;
      y++;
    } else if (step === "delete") {
      for (let at = 0; at < (left[x] as { length: number }).length; at++) steps.push("delete");
      x++;
    } else {
      for (let at = 0; at < (right[y] as { length: number }).length; at++) steps.push("insert");
      y++;
    }
  }
  return steps;
}

/** A wrapper open around runs: an original element (a hyperlink, a smart tag), or a new link. */
interface Wrapper {
  key: XmlElement | string;
  open: string;
  close: string;
}

/** Writes tokens as runs, inside the wrappers each needs, joining those that can share a run. */
class RunWriter {
  private out = "";
  private open: Wrapper[] = [];
  private run: {
    key: string;
    rPr: string;
    startTag: string;
    content: string;
    text: string;
  } | null = null;

  constructor(
    private readonly input: ParagraphInput,
    private readonly context: ParagraphContext,
    private readonly scope: Namespaces,
    private readonly paragraph: XmlElement | null,
  ) {}

  /** The wrappers around an element, outermost first, up to the paragraph. */
  private wrappersOf(element: XmlElement | null): XmlElement[] {
    const chain: XmlElement[] = [];
    if (!this.paragraph || !element) return chain;
    for (
      let at = this.context.parents.get(element);
      at && at !== this.paragraph;
      at = this.context.parents.get(at)
    ) {
      chain.unshift(at);
    }
    return this.inside(element) ? chain : [];
  }

  private inside(element: XmlElement): boolean {
    for (let at: XmlElement | undefined = element; at; at = this.context.parents.get(at)) {
      if (at === this.paragraph) return true;
    }
    return false;
  }

  private wrapper(element: XmlElement): Wrapper {
    const { text } = this.context;
    const properties = elements(element)
      .filter((each) => /Pr$/.test(each.name) && each.name !== "w:rPr")
      .map((each) => sliceOf(text, each))
      .join("");
    return { key: element, open: openTag(text, element) + properties, close: closeTag(element) };
  }

  /** Opens and closes wrappers until exactly `wanted` are open. */
  private enter(wanted: Wrapper[]): void {
    let same = 0;
    while (
      same < this.open.length &&
      same < wanted.length &&
      this.open[same]?.key === wanted[same]?.key
    )
      same++;
    if (same === this.open.length && same === wanted.length) return;
    this.endRun();
    for (let at = this.open.length - 1; at >= same; at--) this.out += this.open[at]?.close ?? "";
    for (const wrapper of wanted.slice(same)) this.out += wrapper.open;
    this.open = wanted;
  }

  private endRun(): void {
    if (!this.run) return;
    this.out += `${this.run.startTag}${this.run.rPr}${this.run.content}${textElement(this.run.text)}</w:r>`;
    this.run = null;
  }

  add(token: Token, formatRun: XmlElement | null): void {
    if (token.kind === "char" || token.kind === "break") this.addText(token, formatRun);
    else if (token.kind === "atom" && token.source) this.addAtom(token.source);
    else if (token.inline) {
      this.enter([]);
      this.endRun();
      this.out += this.context.special(token.inline);
    }
  }

  private addText(token: Token, formatRun: XmlElement | null): void {
    const { text, hrefOf, linkStyle, addLinkStyle, newLink } = this.context;
    const original = this.wrappersOf(formatRun);
    const inheritedLink =
      original
        .filter((each) => each.name === "w:hyperlink")
        .map(hrefOf)
        .at(-1) ?? null;
    const kept = original.filter(
      (each) => each.name !== "w:hyperlink" || hrefOf(each) === token.marks.link,
    );
    const wanted = kept.map((each) => this.wrapper(each));
    const linked = kept.some((each) => each.name === "w:hyperlink");
    if (token.marks.link && !linked) {
      wanted.push({
        key: `link:${token.marks.link}`,
        open: newLink(token.marks.link),
        close: "</w:hyperlink>",
      });
    }
    this.enter(wanted);

    const rPrElement = formatRun
      ? elements(formatRun).find((each) => each.name === "w:rPr")
      : undefined;
    const baseRPr = formatRun
      ? rPrElement
        ? sliceOf(text, rPrElement)
        : ""
      : this.input.fallbackRun;
    const linkChanged = token.marks.link !== inheritedLink;
    const style = !linkChanged ? null : token.marks.link ? addLinkStyle() : linkStyle;
    const rPr = runPropertiesFor(baseRPr, token.marks, style);
    const startTag = formatRun ? openTag(text, formatRun) : "<w:r>";
    const key = `${startTag}\u0000${rPr}`;
    if (this.run?.key !== key) {
      this.endRun();
      this.run = { key, rPr, startTag, content: "", text: "" };
    }
    const element = elementOf(token);
    if (element === null) {
      this.run.text += token.char;
    } else {
      this.run.content += textElement(this.run.text) + element;
      this.run.text = "";
    }
  }

  private addAtom(source: string): void {
    const { text, parents, places } = this.context;
    const [first, last] = source.split("..").map((each) => places.get(each));
    if (!first) return;
    if (first.name === "w:sectPr") return; // A section break is written in the paragraph's properties.
    const parent = parents.get(first);
    let xml: string;
    let around: XmlElement | null = first;
    if (last && last !== first) {
      xml = text.slice(startOf(first), endOf(last));
    } else if (parent?.name === "w:r") {
      const rPr = elements(parent).find((each) => each.name === "w:rPr");
      xml = `${openTag(text, parent)}${rPr ? sliceOf(text, rPr) : ""}${sliceOf(text, first)}</w:r>`;
      around = parent;
    } else {
      xml = sliceOf(text, first);
    }
    this.enter(this.wrappersOf(around).map((each) => this.wrapper(each)));
    this.endRun();
    this.out += this.inside(first)
      ? xml
      : rescoped(xml, scopeOf(parent ?? first, parents), this.scope);
  }

  /** What was written since the last flush, every wrapper closed. */
  flush(): string {
    this.enter([]);
    this.endRun();
    const out = this.out;
    this.out = "";
    return out;
  }
}

/** Text as a run's `w:t`; none for no text. */
const textElement = (text: string) =>
  text ? `<w:t xml:space="preserve">${escapeXml(text)}</w:t>` : "";

/** The element a character is written as, if not as text: a tab, a line break, Word's hyphens. */
function elementOf(token: Token): string | null {
  if (token.kind === "break") return "<w:br/>";
  switch (token.char) {
    case "\t":
      return "<w:tab/>";
    case "‑":
      return "<w:noBreakHyphen/>";
    case "­":
      return "<w:softHyphen/>";
    case "\n":
      return "<w:br/>";
    default:
      return null;
  }
}
