import { Extension } from "@tiptap/core";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import { drawnMarker, groupWordLists, type WordListPlace } from "../../../shared/wordLists";
import { headingLevelOfStyleName, mindHeadingLevel } from "../../../shared/wordStyles";

const FROM_WORD = /urn:schemas-microsoft-com:office:word|class=["']?Mso|mso-list:/i;

const LIST = /mso-list:\s*(l\d+)\s+level(\d+)\s+(lfo\d+)/i;

interface ListParagraph {
  element: Element;
  /** The number or bullet Word typed before the words, in the span it styles `mso-list:Ignore`. */
  marker: Element | null;
  shown: string;
  listId: string;
  level: number;
}

function listParagraphOf(element: Element): ListParagraph | null {
  const found = element.tagName === "P" ? LIST.exec(element.getAttribute("style") ?? "") : null;
  if (!found) return null;
  const marker = element.querySelector('[style*="mso-list:Ignore" i]');
  return {
    element,
    marker,
    shown: (marker?.textContent ?? "").replace(/\s/g, ""),
    listId: `${found[1]} ${found[3]}`,
    level: Number(found[2]) - 1,
  };
}

/**
 * Each run of list paragraphs as the editor's lists when it can draw every marker in the run; a
 * run with any other numbering style stays paragraphs, each with its number typed before its words.
 */
function listBlocks(children: readonly Element[], document: Document) {
  const found = children.map(listParagraphOf);
  const blocks: { node: Element; list: WordListPlace | null }[] = [];
  for (let at = 0; at < children.length; ) {
    let end = at;
    while (end < children.length && found[end]) end++;
    const run = found.slice(at, end).filter((each) => each !== null);
    if (run.length === 0) {
      const child = children[at];
      if (child) blocks.push({ node: child, list: null });
      at++;
      continue;
    }
    const markers = run.map((each) => drawnMarker(each.shown));
    const drawn = markers.every((marker) => marker !== null);
    for (const [index, each] of run.entries()) {
      const marker = drawn ? markers[index] : null;
      if (marker) each.marker?.remove();
      else each.marker?.replaceWith(`${each.shown} `);
      blocks.push({
        node: paragraphOf(each.element, document),
        list: marker ? { listId: each.listId, level: each.level, ...marker } : null,
      });
    }
    at = end;
  }
  return blocks;
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

function paragraphOf(paragraph: Element, document: Document): Element {
  const plain = document.createElement("p");
  plain.append(...paragraph.childNodes);
  return plain;
}

/** HTML copied from Word with its lists, Title and comments as the Word import reads them; other HTML as it is. */
export function fromWord(html: string): string {
  if (!FROM_WORD.test(html)) return html;
  const document = new DOMParser().parseFromString(html, "text/html");
  const { body } = document;
  dropComments(body);
  for (const empty of [...body.getElementsByTagName("o:p")]) {
    if (empty.textContent === "") empty.remove();
  }
  const lists = (container: Element) => {
    const children = [...container.children];
    if (!children.some((child) => LIST.test(child.getAttribute("style") ?? ""))) return;
    container.replaceChildren(
      ...groupWordLists<Element>(listBlocks(children, document), {
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
