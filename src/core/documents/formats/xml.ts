/**
 * A small, non-validating XML parser for the parts of Office packages:
 * elements, attributes, text and CDATA. Enough to read text out of a
 * package; not a general XML library. Ported from the office-formats spike.
 *
 * Untrusted input: a DOCTYPE is refused (Office Open XML never needs one), so
 * there are no external or custom entities to expand (no XXE, no "billion
 * laughs"). Only the five predefined entities and numeric character
 * references are decoded.
 */
import { ExtractionError } from "./errors";

export interface XmlElement {
  /** The qualified name as written, e.g. "w:p". */
  name: string;
  attrs: Record<string, string>;
  children: XmlNode[];
}

export type XmlNode = XmlElement | string;

const ENTITY = /&(lt|gt|amp|quot|apos|#\d+|#x[0-9a-fA-F]+);/g;
const NAMED: Record<string, string> = { lt: "<", gt: ">", amp: "&", quot: '"', apos: "'" };
const ATTRIBUTE = /([^\s=/>]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g;

export function decodeEntities(text: string): string {
  if (!text.includes("&")) return text;
  return text.replace(ENTITY, (_, entity: string) => {
    if (entity[0] !== "#") return NAMED[entity] as string;
    const code =
      entity[1] === "x"
        ? Number.parseInt(entity.slice(2), 16)
        : Number.parseInt(entity.slice(1), 10);
    return code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : "�";
  });
}

const broken = (message: string) => new ExtractionError("unreadable", `Broken XML: ${message}`);

/** Parses one XML document (or a well-formed fragment with one root element). */
export function parseXml(xml: string): XmlElement {
  const root: XmlElement = { name: "#document", attrs: {}, children: [] };
  const stack: XmlElement[] = [root];
  let at = 0;
  const top = () => stack[stack.length - 1] as XmlElement;
  const addText = (text: string) => {
    // Indentation between elements carries no content; a space inside a run does.
    if (text.trim() === "" && text.includes("\n")) return;
    top().children.push(decodeEntities(text));
  };

  while (at < xml.length) {
    const lt = xml.indexOf("<", at);
    if (lt < 0) {
      addText(xml.slice(at));
      break;
    }
    if (lt > at) addText(xml.slice(at, lt));
    if (xml.startsWith("<!--", lt)) {
      const end = xml.indexOf("-->", lt + 4);
      if (end < 0) throw broken("an unclosed comment.");
      at = end + 3;
    } else if (xml.startsWith("<![CDATA[", lt)) {
      const end = xml.indexOf("]]>", lt + 9);
      if (end < 0) throw broken("an unclosed CDATA section.");
      top().children.push(xml.slice(lt + 9, end));
      at = end + 3;
    } else if (xml.startsWith("<!", lt)) {
      throw broken("a DOCTYPE or other declaration, which is refused.");
    } else if (xml.startsWith("<?", lt)) {
      const end = xml.indexOf("?>", lt + 2);
      if (end < 0) throw broken("an unclosed processing instruction.");
      at = end + 2;
    } else {
      const gt = xml.indexOf(">", lt + 1);
      if (gt < 0) throw broken("an unclosed tag.");
      const tag = xml.slice(lt + 1, gt);
      at = gt + 1;
      if (tag[0] === "/") {
        const name = tag.slice(1).trim();
        if (top().name !== name) throw broken(`a mismatched </${name}>.`);
        stack.pop();
        continue;
      }
      const selfClosing = tag.endsWith("/");
      const body = selfClosing ? tag.slice(0, -1) : tag;
      const space = body.search(/\s/);
      const name = space < 0 ? body : body.slice(0, space);
      const attrs: Record<string, string> = {};
      if (space >= 0) {
        for (const match of body.slice(space).matchAll(ATTRIBUTE)) {
          attrs[match[1] as string] = decodeEntities(match[2] ?? match[3] ?? "");
        }
      }
      const element: XmlElement = { name, attrs, children: [] };
      top().children.push(element);
      if (!selfClosing) stack.push(element);
    }
  }
  if (stack.length !== 1) throw broken(`an unclosed <${top().name}>.`);
  const document = root.children.find((node): node is XmlElement => typeof node !== "string");
  if (!document) throw broken("no root element.");
  return document;
}

/** The part of a name after its prefix: "w:p" → "p". */
export const local = (name: string): string => name.slice(name.indexOf(":") + 1);

export const elements = (node: XmlElement): XmlElement[] =>
  node.children.filter((child): child is XmlElement => typeof child !== "string");

/**
 * Whether an element has this name. A name without a prefix matches any
 * prefix: SpreadsheetML is usually unprefixed, but some writers use "x:".
 */
export const is = (element: XmlElement, name: string): boolean =>
  element.name === name || (!name.includes(":") && local(element.name) === name);

/** The first child element with this name. */
export const child = (node: XmlElement, name: string): XmlElement | undefined =>
  elements(node).find((element) => is(element, name));

/** Every descendant element with this name, in document order. */
export function descendants(node: XmlElement, name: string, into: XmlElement[] = []): XmlElement[] {
  for (const element of elements(node)) {
    if (is(element, name)) into.push(element);
    descendants(element, name, into);
  }
  return into;
}

/** All the text under a node, joined. */
export function textOf(node: XmlNode): string {
  if (typeof node === "string") return node;
  return node.children.map(textOf).join("");
}

/**
 * Reads the elements with these names (any prefix) out of XML that arrives in
 * chunks, one complete element at a time, parsed: for parts too large to
 * parse whole, such as a sheet's rows. What comes between them is skipped,
 * and they mustn't nest in each other. Stop early by breaking out of the loop.
 */
export async function* streamElements(
  chunks: AsyncIterable<string>,
  names: readonly string[],
): AsyncGenerator<XmlElement> {
  const open = new RegExp(`<(?:[\\w.-]+:)?(?:${names.join("|")})(?=[\\s/>])`, "g");
  let buffer = "";
  for await (const chunk of chunks) {
    buffer += chunk;
    let consumed = 0;
    for (;;) {
      open.lastIndex = consumed;
      const start = open.exec(buffer);
      if (!start) {
        // Keep a tail that could be the start of a tag cut by the chunk boundary.
        consumed = Math.max(consumed, buffer.length - 64);
        break;
      }
      const tagEnd = buffer.indexOf(">", start.index);
      if (tagEnd < 0) {
        consumed = start.index;
        break;
      }
      let end: number;
      if (buffer[tagEnd - 1] === "/") {
        end = tagEnd + 1;
      } else {
        const qualified = start[0].slice(1);
        const close = buffer.indexOf(`</${qualified}>`, tagEnd);
        if (close < 0) {
          consumed = start.index;
          break;
        }
        end = close + qualified.length + 3;
      }
      yield parseXml(buffer.slice(start.index, end));
      consumed = end;
    }
    buffer = buffer.slice(consumed);
  }
}
