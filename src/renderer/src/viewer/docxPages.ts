/**
 * Finishing what docx-preview draws of a Word Document, so its pages look as
 * Word shows them: list bullets and symbols set in Word's symbol fonts, and
 * comments as notes in a margin beside the pages, as Word's markup area
 * shows them, rather than docx-preview's hover pop-ups. DOM only.
 */

/**
 * Word sets its bullets and symbols in Symbol and Wingdings, at code points
 * those fonts keep in the private-use area, which other fonts don't have:
 * these are the characters they stand for.
 */
const SYMBOLS: Readonly<Record<string, string>> = {
  "": "•", // Symbol: Word's first-level bullet
  "": "▪", // Wingdings: its third-level bullet
  "": "➢",
  "": "❖",
  "": "✓",
  "": "✗",
  "": "■",
  "": "□",
  "": "❑",
  "": "◻",
  "": "➔",
  "": "→",
  "": "▸",
  "": "–",
};
const PRIVATE_SYMBOL = new RegExp(`[${Object.keys(SYMBOLS).join("")}]`, "g");

const unprivate = (text: string) => text.replace(PRIVATE_SYMBOL, (char) => SYMBOLS[char] ?? char);

/** Swaps Word's symbol-font characters for the ones they stand for, in the styles and the pages. */
export function fixSymbols(styles: HTMLElement, body: HTMLElement): void {
  for (const style of styles.querySelectorAll("style")) {
    const text = style.textContent ?? "";
    if (PRIVATE_SYMBOL.test(text)) style.textContent = unprivate(text);
    PRIVATE_SYMBOL.lastIndex = 0;
  }
  const walker = body.ownerDocument.createTreeWalker(body, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const text = node.nodeValue ?? "";
    if (PRIVATE_SYMBOL.test(text)) node.nodeValue = unprivate(text);
    PRIVATE_SYMBOL.lastIndex = 0;
  }
}

/** A comment taken out of the pages, to be shown in the margin. */
export interface DocxComment {
  /** An empty mark where its reference was in the text, to place the note level with. */
  anchor: HTMLElement;
  /** The note shown in the margin. */
  note: HTMLElement;
}

/**
 * Takes docx-preview's comments (a 💬 mark and a pop-up, in the text) out of
 * the pages, leaving an empty anchor where each was, and makes each a note
 * for the margin: its author, date and text, in the comment's own formatting.
 * The commented text stays marked by docx-preview's highlight.
 */
export function takeComments(
  body: HTMLElement,
  className: string,
  label: (author: string) => string,
): DocxComment[] {
  const comments: DocxComment[] = [];
  for (const reference of body.querySelectorAll<HTMLElement>(`.${className}-comment-ref`)) {
    const popover = reference.nextElementSibling;
    if (
      !(popover instanceof HTMLElement) ||
      !popover.classList.contains(`${className}-comment-popover`)
    ) {
      continue;
    }
    const document = body.ownerDocument;
    const anchor = document.createElement("span");
    anchor.dataset.commentAnchor = String(comments.length);
    reference.replaceWith(anchor);
    popover.remove();
    const author = popover.querySelector(`.${className}-comment-author`)?.textContent ?? "";
    const date = popover.querySelector(`.${className}-comment-date`)?.textContent ?? "";
    popover.querySelector(`.${className}-comment-author`)?.remove();
    popover.querySelector(`.${className}-comment-date`)?.remove();
    const note = document.createElement("aside");
    note.className = "docx-note";
    note.dataset.testid = "viewer-docx-comment";
    note.setAttribute("aria-label", label(author));
    const header = document.createElement("div");
    header.className = "docx-note-header";
    const name = document.createElement("span");
    name.className = "docx-note-author";
    name.textContent = author;
    const when = document.createElement("span");
    when.className = "docx-note-date";
    when.textContent = date;
    header.append(name, when);
    const content = document.createElement("div");
    content.className = "docx-note-body";
    content.append(...popover.childNodes);
    note.append(header, content);
    comments.push({ anchor, note });
  }
  return comments;
}

/** The margin notes' width, and the gap between a page and them, in CSS pixels. */
export const NOTE_WIDTH = 220;
export const NOTE_GAP = 16;

/**
 * Lays the notes out in the margin to the right of the pages, each level with
 * its reference's line, pushed down below the one before it. `zoom` is the
 * pages' zoom, which positions inside them are scaled by.
 */
export function placeComments(
  wrapper: HTMLElement,
  comments: readonly DocxComment[],
  zoom: number,
): void {
  let layer = wrapper.querySelector<HTMLElement>(":scope > .docx-notes");
  if (!layer) {
    layer = wrapper.ownerDocument.createElement("div");
    layer.className = "docx-notes";
    // Not part of the text a quote is looked for in.
    layer.dataset.quoteSkip = "";
    wrapper.append(layer);
  }
  const origin = wrapper.getBoundingClientRect();
  let below = 0;
  const placed = comments
    .map((comment) => {
      const line = comment.anchor.getBoundingClientRect();
      const page = comment.anchor.closest("section")?.getBoundingClientRect() ?? line;
      return {
        comment,
        top: (line.top - origin.top) / zoom,
        left: (page.right - origin.left) / zoom,
      };
    })
    .sort((a, b) => a.top - b.top);
  for (const { comment, top, left } of placed) {
    const note = comment.note;
    if (note.parentElement !== layer) layer.append(note);
    const at = Math.max(top, below);
    note.style.top = `${at}px`;
    note.style.left = `${left + NOTE_GAP}px`;
    note.style.width = `${NOTE_WIDTH}px`;
    below = at + note.getBoundingClientRect().height / zoom + 8;
  }
}
