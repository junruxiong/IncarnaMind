/**
 * Text a model writes where it should have called a Tool: a `<cite>` block of
 * records, a `<tool_call>` of JSON, as tags or escaped ("&lt;cite&gt;"). The
 * Answer never shows it. The records of a `<cite>` block go back to be
 * recorded as `cite` would have, so the Citations they name are not lost;
 * any other block just goes, and the sentences it was for read as uncited.
 *
 * A block is only what is shaped like a call: a tag of a Tool, then JSON. The
 * same tag around words ("the <cite> element"), or inside code or maths, is
 * the text's own.
 */
import { shownRanges } from "../../shared/answerBlocks";

const CITE = "cite";

/** The tags that are tool-call text whether or not the Answer was offered such a Tool. */
export const TOOL_TEXT_TAGS: readonly string[] = [
  CITE,
  "calculate",
  "search_documents",
  "tool_call",
  "tool_calls",
  "tool_use",
  "function_call",
  "function_calls",
  "invoke",
];

const OPEN = "(?:<|&lt;)";
const CLOSE = "(?:>|&gt;)";
const ATTRIBUTES = "(?:[ \\t][^<>\\n]*)?";
/** What JSON of a call starts with: an array of objects, or an object. */
const JSON_START = '(?:\\[\\s*[{\\]]|\\{\\s*(?:"|&quot;|\\}))';

const escapeRegExp = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const alternatives = (names: readonly string[]) => names.map(escapeRegExp).join("|");

/** A block: its opening tag, JSON, and its closing tag, or the end of the text when it was cut off. */
const blockPattern = (names: readonly string[]) =>
  new RegExp(
    `${OPEN}(${alternatives(names)})${ATTRIBUTES}${CLOSE}\\s*(${JSON_START}[\\s\\S]*?)(?:\\s*${OPEN}/\\1\\s*${CLOSE}|$)`,
    "gi",
  );

interface Block {
  from: number;
  to: number;
  tag: string;
  content: string;
}

function blocksIn(text: string, names: readonly string[]): Block[] {
  const found = [...text.matchAll(blockPattern(names))].map((each) => ({
    from: each.index,
    to: each.index + each[0].length,
    tag: (each[1] ?? "").toLowerCase(),
    content: each[0].startsWith("&") ? decoded(each[2] ?? "") : (each[2] ?? ""),
  }));
  const shown = shownRanges(
    text,
    found.map((each) => ({ start: each.from, end: each.to })),
  );
  return found.filter((_, index) => shown[index]);
}

const decoded = (text: string) =>
  text
    .replace(/&quot;/g, '"')
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&");

/** The first JSON value in a block's words, which a marker or other words may follow. */
function firstJson(content: string): unknown {
  let depth = 0;
  let inString = false;
  for (let at = 0; at < content.length; at++) {
    const char = content[at];
    if (inString) {
      if (char === "\\") at++;
      else if (char === '"') inString = false;
    } else if (char === '"') inString = true;
    else if (char === "[" || char === "{") depth++;
    else if ((char === "]" || char === "}") && --depth === 0) {
      try {
        return JSON.parse(content.slice(0, at + 1));
      } catch {
        return undefined;
      }
    }
  }
  return undefined;
}

/** The tool-call text in `text`, each block as written. */
export const toolTextIn = (text: string, names: readonly string[] = TOOL_TEXT_TAGS): string[] =>
  blocksIn(text, names).map((block) => text.slice(block.from, block.to));

/**
 * `text` without its tool-call text, and the records a `<cite>` block gave, as
 * the model wrote them. A block alone on its lines takes them with it.
 */
export function takeToolText(
  text: string,
  names: readonly string[] = TOOL_TEXT_TAGS,
): { text: string; records: unknown[] } {
  const blocks = blocksIn(text, names);
  if (blocks.length === 0) return { text, records: [] };
  const records: unknown[] = [];
  let out = "";
  let at = 0;
  for (const block of blocks) {
    if (block.tag === CITE) {
      const value = firstJson(block.content);
      if (value !== undefined) records.push(...(Array.isArray(value) ? value : [value]));
    }
    const before = text.slice(at, block.from).replace(/[ \t]+$/, "");
    const alone =
      (before === "" || before.endsWith("\n")) && /^[ \t]*(?:\n|$)/.test(text.slice(block.to));
    out += before;
    at = block.to;
    if (alone) at += /^[ \t]*\n?/.exec(text.slice(block.to))?.[0].length ?? 0;
  }
  const rest = text.slice(at);
  out += out.endsWith("\n\n") ? rest.replace(/^\n+/, "") : rest;
  return { text: at >= text.length ? out.trimEnd() : out, records };
}

/**
 * Where the Answer's text is held back while it streams, so tool-call text
 * never shows even for a moment: from the first opening tag that has JSON
 * after it or nothing yet, or from an opening tag still being typed.
 */
export function heldBackFrom(text: string, names: readonly string[] = TOOL_TEXT_TAGS): number {
  const opened = new RegExp(
    `${OPEN}(?:${alternatives(names)})${ATTRIBUTES}${CLOSE}\\s*(?=$|\\[\\s*(?:[{\\]]|$)|\\{\\s*(?:"|&quot;|\\}|$))`,
    "i",
  ).exec(text)?.index;
  const typing = /(?:<|&(?:l(?:t;?)?)?)([a-z_]*)$/i.exec(text);
  const partial =
    typing && names.some((name) => name.startsWith((typing[1] ?? "").toLowerCase()))
      ? typing.index
      : undefined;
  return Math.min(opened ?? text.length, partial ?? text.length);
}
