/**
 * LaTeX as a Word equation, for the .docx export: Office Math (OMML), which
 * Word shows and edits as its own equations.
 *
 * KaTeX, which the editor already uses to show maths, reads the LaTeX and
 * writes it as MathML; this turns KaTeX's MathML into OMML. That keeps one
 * reading of LaTeX (macros, environments, \left…\right) for the editor and
 * the export, with no new dependency: the MathML KaTeX writes is a small,
 * regular part of MathML. LaTeX KaTeX can't read, or MathML this doesn't
 * know, gives null, and the export keeps the LaTeX as text.
 */
import katex from "katex";

/** Office Math's namespace, for the `m:` prefix. */
export const MATH_NAMESPACE = "http://schemas.openxmlformats.org/officeDocument/2006/math";

/**
 * KaTeX writes what it won't do (e.g. a link, which isn't trusted) as red
 * LaTeX: in this colour, so such maths is told apart and kept as LaTeX.
 */
const UNSUPPORTED_COLOR = "#c0ffee";

/** MathML this conversion doesn't know. */
class Unsupported extends Error {}

/**
 * LaTeX as an `<m:oMath>` element, inline or (`display`) a display equation,
 * which the caller puts in an `<m:oMathPara>`. Null if it can't be converted.
 */
export function latexToOmml(latex: string, display: boolean): string | null {
  if (latex.trim() === "") return null;
  try {
    const mathml = katex.renderToString(latex, {
      output: "mathml",
      displayMode: display,
      throwOnError: true,
      strict: "ignore",
      trust: false,
      errorColor: UNSUPPORTED_COLOR,
    });
    const math = findElement(parseXml(mathml), "math");
    if (!math) return null;
    const content = new Converter().sequence(math.children);
    return `<m:oMath>${content}</m:oMath>`;
  } catch {
    // Unreadable LaTeX, MathML this doesn't know, or anything else: the LaTeX stays as text.
    return null;
  }
}

// ---------------------------------------------------------------------------
// KaTeX's MathML, read

interface XmlElement {
  kind: "element";
  name: string;
  attributes: Readonly<Record<string, string>>;
  children: XmlNode[];
}

interface XmlText {
  kind: "text";
  text: string;
}

type XmlNode = XmlElement | XmlText;

/** A closing tag, an opening or empty tag with its attributes, or text. */
const TOKEN = /<\/([\w:.-]+)\s*>|<([\w:.-]+)((?:\s+[\w:.-]+="[^"]*")*)\s*(\/?)>|([^<]+)/g;
const ATTRIBUTE = /([\w:.-]+)="([^"]*)"/g;

const ENTITIES: Readonly<Record<string, string>> = {
  amp: "&",
  lt: "<",
  gt: ">",
  quot: '"',
  apos: "'",
};

const decode = (text: string) =>
  text.replace(/&(#x[\da-f]+|#\d+|\w+);/gi, (entity, name: string) => {
    if (name.startsWith("#x") || name.startsWith("#X")) {
      return String.fromCodePoint(Number.parseInt(name.slice(2), 16));
    }
    if (name.startsWith("#")) return String.fromCodePoint(Number.parseInt(name.slice(1), 10));
    return ENTITIES[name] ?? entity;
  });

/** Parses the markup KaTeX writes: elements, attributes in double quotes, text and entities. */
function parseXml(xml: string): XmlElement {
  const root: XmlElement = { kind: "element", name: "", attributes: {}, children: [] };
  const open: XmlElement[] = [root];
  let at = 0;
  for (const match of xml.matchAll(TOKEN)) {
    if (match.index !== at) throw new Unsupported("Not markup KaTeX writes.");
    at = match.index + match[0].length;
    const parent = open.at(-1) as XmlElement;
    const [, closing, name, attributes = "", empty, text] = match;
    if (closing) {
      if (parent.name !== closing || open.length === 1) throw new Unsupported("Unbalanced tags.");
      open.pop();
    } else if (name) {
      const element: XmlElement = {
        kind: "element",
        name,
        attributes: Object.fromEntries(
          [...attributes.matchAll(ATTRIBUTE)].map(([, key, value]) => [key, decode(value ?? "")]),
        ),
        children: [],
      };
      parent.children.push(element);
      if (!empty) open.push(element);
    } else if (text !== undefined) {
      parent.children.push({ kind: "text", text: decode(text) });
    }
  }
  if (at !== xml.length || open.length !== 1) throw new Unsupported("Unbalanced tags.");
  return root;
}

function findElement(node: XmlElement, name: string): XmlElement | null {
  for (const child of node.children) {
    if (child.kind !== "element") continue;
    if (child.name === name) return child;
    const found = findElement(child, name);
    if (found) return found;
  }
  return null;
}

const elementsOf = (element: XmlElement) =>
  element.children.filter((child): child is XmlElement => child.kind === "element");

const textOf = (node: XmlNode): string =>
  node.kind === "text" ? node.text : node.children.map(textOf).join("");

/** An operator `<mo>` with only text in it, as a character or word: null otherwise. */
const operatorText = (node: XmlElement | undefined): string | null =>
  node?.name === "mo" && node.children.every((child) => child.kind === "text")
    ? textOf(node).trim()
    : null;

const isFence = (node: XmlElement | undefined) =>
  node?.name === "mo" && node.attributes.fence === "true";

// ---------------------------------------------------------------------------
// OMML, written

/** Characters XML can't hold at all, even escaped. */
// biome-ignore lint/suspicious/noControlCharactersInRegex: these are the characters it removes.
const NOT_XML = /[\u0000-\u0008\u000B\u000C\u000E-\u001F￾￿]/g;
/** Function application, invisible times, separator and plus: MathML's, never shown. */
const INVISIBLE = /[⁡-⁤]/g;

const escapeXml = (text: string) =>
  text
    .replace(NOT_XML, "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

/** Word's maths font, which Word writes on every run of an equation. */
const MATH_FONT = '<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr>';

/** How a run's letters are set (`m:rPr`): plain, bold or italic, in a script, or as text. */
interface RunStyle {
  /** Text, as in `\text{…}`: the document's font, spaces and all. */
  text?: boolean;
  script?: string;
  sty?: "p" | "b" | "i" | "bi";
}

/** MathML's `mathvariant`, as OMML sets it. */
const VARIANTS: Readonly<Record<string, RunStyle>> = {
  normal: { sty: "p" },
  bold: { sty: "b" },
  italic: { sty: "i" },
  "bold-italic": { sty: "bi" },
  "double-struck": { script: "double-struck", sty: "p" },
  script: { script: "script", sty: "p" },
  "bold-script": { script: "script", sty: "b" },
  fraktur: { script: "fraktur", sty: "p" },
  "bold-fraktur": { script: "fraktur", sty: "b" },
  "sans-serif": { script: "sans-serif", sty: "p" },
  "bold-sans-serif": { script: "sans-serif", sty: "b" },
  "sans-serif-italic": { script: "sans-serif", sty: "i" },
  "sans-serif-bold-italic": { script: "sans-serif", sty: "bi" },
  monospace: { script: "monospace", sty: "p" },
};

function run(text: string, style: RunStyle = {}): string {
  const shown = text.replace(INVISIBLE, "");
  if (shown === "") return "";
  const properties = style.text
    ? "<m:nor/>"
    : `${style.script ? `<m:scr m:val="${style.script}"/>` : ""}${style.sty ? `<m:sty m:val="${style.sty}"/>` : ""}`;
  return (
    `<m:r>${properties ? `<m:rPr>${properties}</m:rPr>` : ""}${style.text ? "" : MATH_FONT}` +
    `<m:t xml:space="preserve">${escapeXml(shown)}</m:t></m:r>`
  );
}

const on = (name: string) => `<m:${name} m:val="1"/>`;
const value = (name: string, val: string) => `<m:${name} m:val="${escapeXml(val)}"/>`;

/** Large operators: with limits and an operand they are Word's n-ary operators. */
const NARY = new Set([..."∑∏∐∫∬∭∮∯∰∱∲∳⋀⋁⋂⋃⨀⨁⨂⨄⨆"]);

/** Accents as KaTeX writes them, and the combining marks Word puts over a letter. */
const ACCENTS: Readonly<Record<string, string>> = {
  "^": "̂",
  ˆ: "̂",
  "~": "̃",
  "˜": "̃",
  ˉ: "̄",
  "¯": "̄",
  "˘": "̆",
  "˙": "̇",
  "¨": "̈",
  "˚": "̊",
  ˇ: "̌",
  ˊ: "́",
  "´": "́",
  ˋ: "̀",
  "`": "̀",
  "→": "⃗",
  "←": "⃖",
  "↔": "⃡",
};

/** Lines over or under, as `\overline` and `\underline` draw them. */
const BARS = new Set(["‾", "¯", "_", "̲"]);
/** Braces and brackets over or under, as `\overbrace` and `\underbrace` draw them. */
const OVER_GROUPS = new Set(["⏞", "⏜", "⎴"]);
const UNDER_GROUPS = new Set(["⏟", "⏝", "⎵"]);

/** An em space for each em of a MathML space, or a narrower space for less. */
function space(width: string | undefined): string {
  const ems = Number.parseFloat(width ?? "");
  if (!Number.isFinite(ems) || ems <= 0 || !/em\s*$/.test(width ?? "")) return "";
  if (ems >= 1) return " ".repeat(Math.round(ems));
  if (ems >= 0.5) return " ";
  if (ems >= 0.3) return " ";
  if (ems >= 0.22) return " ";
  return " ";
}

class Converter {
  /** A sequence of MathML nodes: a large operator with limits takes the next one as its operand. */
  sequence(nodes: readonly XmlNode[]): string {
    let xml = "";
    for (let index = 0; index < nodes.length; index++) {
      const node = nodes[index] as XmlNode;
      if (node.kind === "text") {
        xml += node.text.trim() === "" ? "" : run(node.text);
        continue;
      }
      const next = nodes.slice(index + 1).find((each) => each.kind === "element");
      const nary = next && this.nary(node, next);
      if (nary) {
        xml += nary;
        index = nodes.indexOf(next);
        continue;
      }
      xml += this.element(node);
    }
    return xml;
  }

  private element(node: XmlElement): string {
    const children = elementsOf(node);
    const [first, second, third] = children;
    switch (node.name) {
      case "math":
      case "mstyle":
      case "mpadded":
        if (node.attributes.mathcolor === UNSUPPORTED_COLOR) {
          throw new Unsupported("KaTeX left something out.");
        }
        return this.sequence(node.children);
      case "semantics":
      case "maction":
        // The maths, without its annotations (the LaTeX), or an action's first choice.
        return first ? this.element(first) : "";
      case "annotation":
      case "annotation-xml":
        return "";
      case "mrow":
        return this.row(children);
      case "mi": {
        const text = textOf(node);
        const variant = node.attributes.mathvariant;
        // A one-letter identifier is italic; a name such as "sin" is upright.
        const style = variant ? VARIANTS[variant] : [...text].length > 1 ? { sty: "p" } : {};
        return run(text, style as RunStyle);
      }
      case "mn":
        return run(textOf(node), VARIANTS[node.attributes.mathvariant ?? ""] ?? {});
      case "mo":
        // An operator made of other parts, e.g. `\mathop{\mathrm{Res}}`.
        if (children.length > 0) return this.sequence(node.children);
        return run(textOf(node));
      case "mtext":
        return run(textOf(node), { text: true });
      case "ms":
        return run(`"${textOf(node)}"`, { text: true });
      case "mspace":
        return run(space(node.attributes.width), { sty: "p" });
      case "msup":
        return `<m:sSup><m:e>${this.part(first)}</m:e><m:sup>${this.part(second)}</m:sup></m:sSup>`;
      case "msub":
        return `<m:sSub><m:e>${this.part(first)}</m:e><m:sub>${this.part(second)}</m:sub></m:sSub>`;
      case "msubsup":
        return `<m:sSubSup><m:e>${this.part(first)}</m:e><m:sub>${this.part(second)}</m:sub><m:sup>${this.part(third)}</m:sup></m:sSubSup>`;
      case "mfrac": {
        const noBar = /^0(\.0*)?([a-z]+)?$/i.test(node.attributes.linethickness ?? "");
        return `<m:f>${noBar ? `<m:fPr>${value("type", "noBar")}</m:fPr>` : ""}<m:num>${this.part(first)}</m:num><m:den>${this.part(second)}</m:den></m:f>`;
      }
      case "msqrt":
        return `<m:rad><m:radPr>${on("degHide")}</m:radPr><m:deg></m:deg><m:e>${this.sequence(node.children)}</m:e></m:rad>`;
      case "mroot":
        return `<m:rad><m:deg>${this.part(second)}</m:deg><m:e>${this.part(first)}</m:e></m:rad>`;
      case "mover":
        return this.over(node, first, second);
      case "munder":
        return this.under(node, first, second);
      case "munderover":
        return this.limits(this.limits(this.part(first), "limLow", second), "limUpp", third);
      case "mtable":
        return this.table(node, children);
      case "menclose":
        return this.enclosed(node);
      case "mphantom":
        return `<m:phant><m:phantPr>${value("show", "0")}</m:phantPr><m:e>${this.sequence(node.children)}</m:e></m:phant>`;
      default:
        throw new Unsupported(`<${node.name}> isn't converted.`);
    }
  }

  /** An argument of a structure, e.g. a fraction's numerator. */
  private part(node: XmlElement | undefined): string {
    return node ? this.element(node) : "";
  }

  /**
   * A row; `\left…\right` brackets around it make it Word's delimiter, which
   * grows with what it holds, and `\middle` ones separate its parts.
   */
  private row(children: readonly XmlElement[]): string {
    const first = children[0];
    const last = children.at(-1);
    const opens = isFence(first);
    const closes = children.length > 1 && isFence(last);
    if (!opens && !closes) return this.sequence(children);
    const inner = children.slice(opens ? 1 : 0, closes ? -1 : undefined);
    const parts: XmlElement[][] = [[]];
    let separator: string | null = null;
    for (const child of inner) {
      if (isFence(child)) {
        separator ??= textOf(child).trim();
        parts.push([]);
      } else (parts.at(-1) as XmlElement[]).push(child);
    }
    const properties =
      value("begChr", opens && first ? textOf(first).trim() : "") +
      (separator !== null ? value("sepChr", separator) : "") +
      value("endChr", closes && last ? textOf(last).trim() : "");
    const elements = parts.map((part) => `<m:e>${this.sequence(part)}</m:e>`).join("");
    return `<m:d><m:dPr>${properties}</m:dPr>${elements}</m:d>`;
  }

  /**
   * A large operator with limits, e.g. `\sum_{i=1}^n`, over its operand
   * (`next`): Word's n-ary operator. Null for anything else.
   */
  private nary(node: XmlElement, next: XmlElement): string | null {
    const [base, first, second] = elementsOf(node);
    const chr = operatorText(base);
    if (!chr || !NARY.has(chr)) return null;
    let sub: XmlElement | undefined;
    let sup: XmlElement | undefined;
    switch (node.name) {
      case "msub":
      case "munder":
        sub = first;
        break;
      case "msup":
      case "mover":
        sup = first;
        break;
      case "msubsup":
      case "munderover":
        sub = first;
        sup = second;
        break;
      default:
        return null;
    }
    const below = node.name.startsWith("mu") || node.name === "mover";
    const properties =
      value("chr", chr) +
      value("limLoc", below ? "undOvr" : "subSup") +
      (sub ? "" : on("subHide")) +
      (sup ? "" : on("supHide"));
    return `<m:nary><m:naryPr>${properties}</m:naryPr><m:sub>${this.part(sub)}</m:sub><m:sup>${this.part(sup)}</m:sup><m:e>${this.element(next)}</m:e></m:nary>`;
  }

  /** Something over a base: an accent, a line, a brace, or else a limit. */
  private over(node: XmlElement, base?: XmlElement, over?: XmlElement): string {
    const mark = operatorText(over);
    if (mark !== null && BARS.has(mark)) {
      return `<m:bar><m:barPr>${value("pos", "top")}</m:barPr><m:e>${this.part(base)}</m:e></m:bar>`;
    }
    if (mark !== null && OVER_GROUPS.has(mark)) {
      return this.group(mark, "top", base);
    }
    if (mark !== null && node.attributes.accent === "true") {
      const chr = ACCENTS[mark] ?? mark;
      return `<m:acc><m:accPr>${value("chr", chr)}</m:accPr><m:e>${this.part(base)}</m:e></m:acc>`;
    }
    return this.limits(this.part(base), "limUpp", over);
  }

  /** Something under a base: a line, a brace, or else a limit. */
  private under(node: XmlElement, base?: XmlElement, under?: XmlElement): string {
    const mark = operatorText(under);
    if (mark !== null && BARS.has(mark)) {
      return `<m:bar><m:barPr>${value("pos", "bot")}</m:barPr><m:e>${this.part(base)}</m:e></m:bar>`;
    }
    if (mark !== null && (UNDER_GROUPS.has(mark) || node.attributes.accentunder === "true")) {
      return this.group(mark, "bot", base);
    }
    return this.limits(this.part(base), "limLow", under);
  }

  private group(chr: string, pos: "top" | "bot", base?: XmlElement): string {
    const properties =
      value("chr", chr) + value("pos", pos) + value("vertJc", pos === "top" ? "bot" : "top");
    return `<m:groupChr><m:groupChrPr>${properties}</m:groupChrPr><m:e>${this.part(base)}</m:e></m:groupChr>`;
  }

  private limits(base: string, kind: "limLow" | "limUpp", limit?: XmlElement): string {
    return `<m:${kind}><m:e>${base}</m:e><m:lim>${this.part(limit)}</m:lim></m:${kind}>`;
  }

  /** A matrix or an array of aligned equations: Word's matrix, its columns aligned as given. */
  private table(node: XmlElement, rows: readonly XmlElement[]): string {
    // `\tag`: the equation, then its tag, as KaTeX lays them out in a full-width table.
    if (node.attributes.width === "100%") {
      const cells = rows.length === 1 && rows[0] ? elementsOf(rows[0]) : [];
      const [, equation, , tag] = cells;
      if (cells.length !== 4 || !equation || !tag) throw new Unsupported("An unusual tag.");
      return `${this.sequence(equation.children)}${run("  ", { sty: "p" })}${this.sequence(tag.children)}`;
    }
    const cellsOf = rows.map((row) => {
      if (row.name !== "mtr") throw new Unsupported(`<${row.name}> isn't converted.`);
      return elementsOf(row);
    });
    const columns = Math.max(0, ...cellsOf.map((cells) => cells.length));
    if (columns === 0) return "";
    const aligns = (node.attributes.columnalign ?? "").split(/\s+/);
    let alignments = "";
    for (let column = 0; column < columns; column++) {
      const align = aligns[Math.min(column, aligns.length - 1)];
      const jc = align === "left" || align === "right" ? align : "center";
      alignments += `<m:mc><m:mcPr>${value("count", "1")}${value("mcJc", jc)}</m:mcPr></m:mc>`;
    }
    const matrixRows = cellsOf
      .map((cells) => {
        let xml = "";
        for (let column = 0; column < columns; column++) {
          const cell = cells[column];
          xml += `<m:e>${cell ? this.sequence(cell.children) : ""}</m:e>`;
        }
        return `<m:mr>${xml}</m:mr>`;
      })
      .join("");
    return `<m:m><m:mPr>${on("plcHide")}<m:mcs>${alignments}</m:mcs></m:mPr>${matrixRows}</m:m>`;
  }

  /** `\boxed`, `\cancel` and the like: Word's border box, its sides and strikes as asked. */
  private enclosed(node: XmlElement): string {
    const notation = new Set((node.attributes.notation ?? "box").split(/\s+/));
    const box = ["box", "roundedbox", "circle"].some((each) => notation.has(each));
    const side = (name: string, hidden: string) => (box || notation.has(name) ? "" : on(hidden));
    // In the order OMML's schema has them.
    const properties =
      side("top", "hideTop") +
      side("bottom", "hideBot") +
      side("left", "hideLeft") +
      side("right", "hideRight") +
      (notation.has("horizontalstrike") ? on("strikeH") : "") +
      (notation.has("verticalstrike") ? on("strikeV") : "") +
      (notation.has("updiagonalstrike") ? on("strikeBLTR") : "") +
      (notation.has("downdiagonalstrike") ? on("strikeTLBR") : "");
    return `<m:borderBox>${properties ? `<m:borderBoxPr>${properties}</m:borderBoxPr>` : ""}<m:e>${this.sequence(node.children)}</m:e></m:borderBox>`;
  }
}
