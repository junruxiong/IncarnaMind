import { Extension } from "@tiptap/core";
import { Plugin, PluginKey } from "@tiptap/pm/state";

/** Word's list paragraphs: `mso-list:l0 level2 lfo1` says which list, and how deep. */
const LIST_STYLE = /mso-list:\s*l\d+\s+level(\d+)/i;
/** The one numbered style the editor draws: "1." (the others keep their numbers as typed text). */
const DECIMAL = /^(\d+)\.$/;
/** What Word writes before a bulleted item (its Symbol and Wingdings bullets come out as these). */
const BULLET = /^[·•o§▪■◦●○□➢➤✓\-–—*]$|^[-]$/u;

interface Item {
  element: Element;
  level: number;
  words: string;
  /** A bullet, or the number a "1." marker says; null for any other marker. */
  kind: "bullet" | number | null;
}

function itemOf(paragraph: Element): Item | null {
  const level = LIST_STYLE.exec(paragraph.getAttribute("style") ?? "")?.[1];
  if (paragraph.tagName !== "P" || level === undefined) return null;
  const copy = paragraph.cloneNode(true) as Element;
  let marker = "";
  for (const span of copy.querySelectorAll("span")) {
    if (!/mso-list:\s*ignore/i.test(span.getAttribute("style") ?? "")) continue;
    marker = (span.textContent ?? "").replace(/\s+/g, " ").trim();
    span.remove();
  }
  for (const each of [...copy.querySelectorAll("*")]) {
    if (each.tagName.toLowerCase() === "o:p") each.remove();
  }
  const decimal = DECIMAL.exec(marker)?.[1];
  return {
    element: paragraph,
    level: Math.max(1, Number(level)),
    words: copy.innerHTML,
    kind: BULLET.test(marker) ? "bullet" : decimal !== undefined ? Number(decimal) : null,
  };
}

/** One list built from `items`, each level nested in the item above it. */
function listFrom(items: readonly Item[], document: Document): Element {
  const make = (item: Item) => {
    const list = document.createElement(item.kind === "bullet" ? "ul" : "ol");
    if (item.kind !== "bullet" && item.kind !== null && item.kind !== 1) {
      list.setAttribute("start", String(item.kind));
    }
    return list;
  };
  const stack: { list: Element; item: Element | null }[] = [];
  for (const item of items) {
    if (stack.length === 0) stack.push({ list: make(item), item: null });
    while (stack.length > item.level) stack.pop();
    while (stack.length < item.level) {
      const owner = stack[stack.length - 1];
      const holder = owner?.item ?? owner?.list.appendChild(document.createElement("li"));
      if (!owner || !holder) break;
      owner.item = holder;
      const inner = make(item);
      holder.appendChild(inner);
      stack.push({ list: inner, item: null });
    }
    const current = stack[stack.length - 1];
    if (!current) continue;
    const entry = document.createElement("li");
    const text = document.createElement("p");
    text.innerHTML = item.words;
    entry.appendChild(text);
    current.list.appendChild(entry);
    current.item = entry;
  }
  return stack[0]?.list ?? document.createElement("ul");
}

/**
 * Pasted from Word, a list is paragraphs with a level in their style, not list markup. A run of
 * them whose markers are bullets or "1." comes back as nested lists (#1341). Any other marker
 * ("1、", "（1）", "一、", "a)", "1.1.") is a style the editor doesn't draw, so that run stays
 * paragraphs with its markers typed. HTML with no Word list in it comes back as it was.
 */
export function listsFromWord(html: string, document: Document): string {
  if (!/mso-list:/i.test(html)) return html;
  const root = document.createElement("div");
  root.innerHTML = html;
  const parents = new Set<Element>(
    [...root.querySelectorAll("p")]
      .filter((each) => itemOf(each) !== null)
      .flatMap((each) => (each.parentElement ? [each.parentElement] : [])),
  );
  for (const parent of parents) {
    let run: Item[] = [];
    const flush = () => {
      const [first] = run;
      if (first && run.every((each) => each.kind !== null)) {
        first.element.before(listFrom(run, document));
        for (const each of run) each.element.remove();
      }
      run = [];
    };
    for (const child of [...parent.children]) {
      const item = itemOf(child);
      if (item) run.push(item);
      else flush();
    }
    flush();
  }
  return root.innerHTML;
}

export const WordListPaste = Extension.create({
  name: "wordListPaste",

  addProseMirrorPlugins() {
    return [
      new Plugin({
        key: new PluginKey("wordListPaste"),
        props: { transformPastedHTML: (html) => listsFromWord(html, document) },
      }),
    ];
  },
});
