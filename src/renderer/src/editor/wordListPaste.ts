import { Extension } from "@tiptap/core";
import { Plugin, PluginKey } from "@tiptap/pm/state";

/** Word's list paragraphs: `mso-list:l0 level2 lfo1` says which list, and how deep. */
const LIST_STYLE = /mso-list:\s*l\d+\s+level(\d+)/i;
/** What Word writes before a numbered item's words: "1.", "a)", "(iv)". */
const NUMBERED = /^\(?(?:\d+|[a-z]+)[.)]$/i;

function wordsOf(paragraph: Element): { html: string; ordered: boolean } {
  const copy = paragraph.cloneNode(true) as Element;
  let ordered = false;
  for (const span of copy.querySelectorAll("span")) {
    if (!/mso-list:\s*ignore/i.test(span.getAttribute("style") ?? "")) continue;
    ordered = NUMBERED.test((span.textContent ?? "").replace(/\s+/g, " ").trim());
    span.remove();
  }
  for (const each of [...copy.querySelectorAll("*")]) {
    if (each.tagName.toLowerCase() === "o:p") each.remove();
  }
  return { html: copy.innerHTML, ordered };
}

/**
 * Pasted from Word, a list is paragraphs with a level in their style, not list markup. This puts
 * the markup back, nested by level, so a pasted list keeps its levels (#1341). HTML with no
 * Word list in it comes back as it was.
 */
export function listsFromWord(html: string, document: Document): string {
  if (!/mso-list:/i.test(html)) return html;
  const root = document.createElement("div");
  root.innerHTML = html;
  const levelOf = (element: Element): number | null => {
    if (element.tagName !== "P") return null;
    const level = LIST_STYLE.exec(element.getAttribute("style") ?? "")?.[1];
    return level === undefined ? null : Math.max(1, Number(level));
  };
  const parents = new Set<Element>(
    [...root.querySelectorAll("p")]
      .filter((each) => levelOf(each) !== null)
      .flatMap((each) => (each.parentElement ? [each.parentElement] : [])),
  );
  for (const parent of parents) {
    let top: Element | null = null;
    const stack: { list: Element; item: Element | null }[] = [];
    for (const child of [...parent.children]) {
      const level = levelOf(child);
      if (level === null) {
        top = null;
        stack.length = 0;
        continue;
      }
      const { html: words, ordered } = wordsOf(child);
      if (top === null) {
        top = document.createElement(ordered ? "ol" : "ul");
        child.before(top);
        stack.push({ list: top, item: null });
      }
      while (stack.length > level) stack.pop();
      while (stack.length < level) {
        const owner = stack[stack.length - 1];
        const holder = owner?.item ?? owner?.list.appendChild(document.createElement("li"));
        if (!owner || !holder) break;
        owner.item = holder;
        const inner = document.createElement(ordered ? "ol" : "ul");
        holder.appendChild(inner);
        stack.push({ list: inner, item: null });
      }
      const current = stack[stack.length - 1];
      if (!current) continue;
      const item = document.createElement("li");
      const text = document.createElement("p");
      text.innerHTML = words;
      item.appendChild(text);
      current.list.appendChild(item);
      current.item = item;
      child.remove();
    }
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
