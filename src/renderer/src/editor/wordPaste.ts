import { Extension } from "@tiptap/core";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import { groupWordLists, type WordListPlace } from "../../../shared/wordLists";
import { headingLevelOfStyleName, mindHeadingLevel } from "../../../shared/wordStyles";

const FROM_WORD = /urn:schemas-microsoft-com:office:word|class=["']?Mso|mso-list:/i;

const LIST = /mso-list:\s*(l\d+)\s+level(\d+)\s+(lfo\d+)/i;

// Word writes a list's marker as text; a bullet's is a symbol-font character such as "·" or "§".
const NUMBERED = /^\(?([0-9]+|[a-z]|[ivxlcdm]+|[一二三四五六七八九十]+)[.)、]$/i;

const ROMAN: Readonly<Record<string, number>> = {
  i: 1,
  v: 5,
  x: 10,
  l: 50,
  c: 100,
  d: 500,
  m: 1000,
};

function numberOf(marker: string): number {
  if (/^\d+$/.test(marker)) return Number(marker);
  const lower = marker.toLowerCase();
  // A list that starts at "i." counts in Roman numerals, not from the ninth letter.
  if (/^[ivxlcdm]+$/.test(lower) && (lower.length > 1 || lower === "i")) {
    let total = 0;
    for (const [index, letter] of [...lower].entries()) {
      const value = ROMAN[letter] ?? 0;
      const next = ROMAN[lower[index + 1] ?? ""] ?? 0;
      total += value < next ? -value : value;
    }
    return total;
  }
  if (/^[a-z]$/.test(lower)) return lower.charCodeAt(0) - 96;
  return 1;
}

function listPlace(paragraph: Element): WordListPlace | null {
  const found = LIST.exec(paragraph.getAttribute("style") ?? "");
  if (!found) return null;
  const marker = paragraph.querySelector('[style*="mso-list:Ignore" i]');
  const shown = (marker?.textContent ?? "").replace(/\s/g, "");
  marker?.remove();
  const numbered = NUMBERED.exec(shown);
  return {
    listId: `${found[1]} ${found[3]}`,
    level: Number(found[2]) - 1,
    ordered: numbered !== null,
    number: numbered ? numberOf(numbered[1] ?? "") : 1,
  };
}

// Word copies a comment as an anchor around its words, a "[JM1]" mark after them and its text at the end.
function dropComments(body: HTMLElement): void {
  for (const anchor of body.querySelectorAll('a[style*="mso-comment-reference" i]')) {
    anchor.replaceWith(...anchor.childNodes);
  }
  for (const each of body.querySelectorAll(
    '.MsoCommentReference, a.msocomanchor, a.msocomoff, [style*="mso-element:comment-list" i]',
  )) {
    each.remove();
  }
}

function headingOf(paragraph: Element, document: Document): Element | null {
  const style = /^Mso(.+?)(?:CxSp\w+)?$/.exec(paragraph.className)?.[1];
  const level = style === undefined ? null : headingLevelOfStyleName(style);
  if (level === null) return null;
  const heading = document.createElement(`h${mindHeadingLevel(level)}`);
  heading.append(...paragraph.childNodes);
  return heading;
}

/** HTML copied from Word with its lists, Title and comments as the Word import reads them; other HTML as it is. */
export function fromWord(html: string): string {
  if (!FROM_WORD.test(html)) return html;
  const document = new DOMParser().parseFromString(html, "text/html");
  const { body } = document;
  dropComments(body);
  const lists = (container: Element) => {
    const children = [...container.children];
    if (!children.some((child) => LIST.test(child.getAttribute("style") ?? ""))) return;
    const blocks = children.map((child) => {
      const list = child.tagName === "P" ? listPlace(child) : null;
      return { node: list ? paragraphOf(child, document) : child, list };
    });
    container.replaceChildren(
      ...groupWordLists<Element>(blocks, {
        list: (place) => {
          const list = document.createElement(place.ordered ? "ol" : "ul");
          if (place.ordered && place.number !== 1) list.setAttribute("start", String(place.number));
          return list;
        },
        item: (list, block) => {
          const item = document.createElement("li");
          item.append(block);
          list.append(item);
        },
        nest: (parent, list) => {
          parent.lastElementChild?.append(list);
        },
      }),
    );
  };
  lists(body);
  for (const cell of body.querySelectorAll("td, th, div")) lists(cell);
  for (const paragraph of body.querySelectorAll('p[class^="Mso"]')) {
    const heading = headingOf(paragraph, document);
    if (heading) paragraph.replaceWith(heading);
  }
  return body.innerHTML;
}

function paragraphOf(paragraph: Element, document: Document): Element {
  const plain = document.createElement("p");
  plain.append(...paragraph.childNodes);
  return plain;
}

export const WordPaste = Extension.create({
  name: "wordPaste",

  addProseMirrorPlugins() {
    return [
      new Plugin({
        key: new PluginKey("wordPaste"),
        props: { transformPastedHTML: fromWord },
      }),
    ];
  },
});
