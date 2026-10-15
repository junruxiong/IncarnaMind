import type { WordBorder, WordLook, WordParagraphLook, WordRunLook } from "../../../core/api";
import { WORD_LOOK_DOM_ATTRIBUTE, WORD_RUN_DOM_ATTRIBUTE } from "./wordSchema";

/** Word's single line spacing for common fonts; measured per font later in #746. */
const SINGLE_LINE = 1.15;
/** Word draws super- and subscript at about two thirds of the size. */
const RAISED_SIZE = 0.66;
const DEFAULT_BODY = 11;

const BORDER_STYLES: Record<string, string> = {
  double: "double",
  dotted: "dotted",
  dashed: "dashed",
  dashSmallGap: "dashed",
  dotDash: "dashed",
  dotDotDash: "dotted",
};

const HIGHLIGHTS: Record<string, string> = {
  yellow: "#FFFF00",
  green: "#00FF00",
  cyan: "#00FFFF",
  magenta: "#FF00FF",
  blue: "#0000FF",
  red: "#FF0000",
  darkBlue: "#000080",
  darkCyan: "#008080",
  darkGreen: "#008000",
  darkMagenta: "#800080",
  darkRed: "#800000",
  darkYellow: "#808000",
  darkGray: "#808080",
  lightGray: "#C0C0C0",
  black: "#000000",
  white: "#FFFFFF",
};

/** The scope's class: every rule is under it, so two Minds side by side never share one. */
export const wordLookScope = (mindId: string) => `word-look-${mindId}`;

const pt = (value: number) => `calc(var(--word-pt) * ${value})`;
const quoted = (font: string) => `"${font.replace(/["\\]/g, "")}"`;
const attribute = (name: string, key: string) => `[${name}="${key.replace(/["\\]/g, "")}"]`;

function border(value: WordBorder): string {
  const width = Math.max(value.width, 0.25);
  return `${pt(width)} ${BORDER_STYLES[value.style] ?? "solid"} ${value.color ? `#${value.color}` : "currentColor"}`;
}

function fontFamily(look: WordRunLook): string | null {
  const fonts = [...new Set([look.font, look.eastAsia])].filter((each): each is string =>
    Boolean(each),
  );
  return fonts.length > 0 ? `${fonts.map(quoted).join(", ")}, var(--font-serif)` : null;
}

/** What a run's look sets: every property, so a run never keeps its paragraph's. */
function runDeclarations(look: WordRunLook): string[] {
  const size = look.vertical && look.size !== undefined ? look.size * RAISED_SIZE : look.size;
  const family = fontFamily(look);
  return [
    family ? `font-family: ${family}` : "",
    size !== undefined ? `font-size: ${pt(size)}` : "",
    `font-weight: ${look.bold ? 700 : 400}`,
    `font-style: ${look.italic ? "italic" : "normal"}`,
    `text-transform: ${look.caps ? "uppercase" : "none"}`,
    `font-variant-caps: ${look.smallCaps ? "small-caps" : "normal"}`,
    `letter-spacing: ${look.letterSpacing ? pt(look.letterSpacing) : "normal"}`,
    look.color ? `color: #${look.color}` : "",
  ].filter(Boolean);
}

/** What only an inline element can draw: lines through or under its text, its background, raising. */
function inlineDeclarations(look: WordRunLook): string[] {
  const decoration = [
    look.underline && "underline",
    (look.strike || look.doubleStrike) && "line-through",
  ].filter(Boolean);
  const background = look.highlight ? HIGHLIGHTS[look.highlight] : look.shade && `#${look.shade}`;
  return [
    decoration.length > 0 ? `text-decoration-line: ${decoration.join(" ")}` : "",
    look.underline === "double" || look.doubleStrike ? "text-decoration-style: double" : "",
    look.underline === "dotted" ? "text-decoration-style: dotted" : "",
    background ? `background-color: ${background}` : "",
    look.vertical ? `vertical-align: ${look.vertical === "superscript" ? "super" : "sub"}` : "",
  ].filter(Boolean);
}

function lineHeight(line: WordParagraphLook["line"]): string {
  if (!line) return String(SINGLE_LINE);
  if (line.rule === "auto") return String(Math.round(SINGLE_LINE * line.value * 1000) / 1000);
  if (line.rule === "exact") return pt(line.value);
  return `max(${SINGLE_LINE}em, ${pt(line.value)})`;
}

function paragraphDeclarations(look: WordParagraphLook): string[] {
  return [
    `text-align: ${look.align === "distribute" ? "justify" : (look.align ?? "left")}`,
    look.align === "distribute" ? "text-align-last: justify" : "",
    `margin-left: ${look.left ? pt(look.left) : "0"}`,
    `margin-right: ${look.right ? pt(look.right) : "0"}`,
    `text-indent: ${look.firstLine ? pt(look.firstLine) : "0"}`,
    `margin-top: ${look.before ? pt(look.before) : "0"}`,
    `margin-bottom: ${look.after ? pt(look.after) : "0"}`,
    `line-height: ${lineHeight(look.line)}`,
    look.shade ? `background-color: #${look.shade}` : "",
    ...(["top", "bottom", "left", "right"] as const).flatMap((side) => {
      const value = look.borders?.[side];
      return value
        ? [`border-${side}: ${border(value)}`, `padding-${side}: ${pt(value.space)}`]
        : [];
    }),
  ].filter(Boolean);
}

const rule = (selector: string, declarations: string[]) =>
  declarations.length > 0 ? `${selector} { ${declarations.join("; ")}; }` : "";

/** Sizes and lengths are relative to the file's body size, which takes the Mind's writing size. */
export function wordLookCss(mindId: string, look: WordLook): string {
  const scope = `.${wordLookScope(mindId)}`;
  const rules = [
    `${scope} { --word-pt: calc(var(--text-body) / ${look.body ?? DEFAULT_BODY}); }`,
    look.width ? `${scope} { max-width: ${pt(look.width)}; }` : "",
    // A list draws its own indents; the file's are read for lists in a later part of #746.
    `${scope} li > [${WORD_LOOK_DOM_ATTRIBUTE}] { margin-left: 0; margin-right: 0; text-indent: 0; }`,
  ];
  for (const [key, paragraph] of Object.entries(look.paragraphs)) {
    const run = paragraph.run ? look.runs[paragraph.run] : undefined;
    const selector = `${scope} ${attribute(WORD_LOOK_DOM_ATTRIBUTE, key)}`;
    rules.push(
      rule(selector, [...paragraphDeclarations(paragraph), ...(run ? runDeclarations(run) : [])]),
    );
    if (paragraph.contextual) {
      rules.push(
        rule(`${selector} + ${attribute(WORD_LOOK_DOM_ATTRIBUTE, key)}`, ["margin-top: 0"]),
      );
      rules.push(
        rule(`${selector}:has(+ ${attribute(WORD_LOOK_DOM_ATTRIBUTE, key)})`, ["margin-bottom: 0"]),
      );
    }
  }
  for (const [key, run] of Object.entries(look.runs)) {
    rules.push(
      rule(`${scope} span${attribute(WORD_RUN_DOM_ATTRIBUTE, key)}`, [
        ...runDeclarations(run),
        ...inlineDeclarations(run),
      ]),
    );
  }
  return rules.filter(Boolean).join("\n");
}
