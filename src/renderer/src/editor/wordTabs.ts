import { Mark } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { type EditorState, Plugin } from "@tiptap/pm/state";
import type { EditorView } from "@tiptap/pm/view";
import type * as Y from "yjs";
import {
  WORD_LOOK_ATTRIBUTE,
  WORD_TAB_MARK,
  type WordLook,
  type WordParagraphLook,
} from "../../../core/api";
import { observeWordLook, wordLookOf } from "../../../shared/wordLook";
import { editorDoc } from "./proposalReview";
import { tabStopAfter, tabWidth } from "./wordTabStops";

export const WORD_TAB_DOM_ATTRIBUTE = "data-word-tab";
const TAB_SELECTOR = `[${WORD_TAB_DOM_ATTRIBUTE}]`;

/** A long Mind's first pass goes a frame at a time, so typing never waits for it. */
const FRAME_BUDGET_MS = 8;
/** Up to this many paragraphs are measured at once, before the browser paints. */
const MEASURED_AT_ONCE = 40;

interface Pending {
  node: ProseMirrorNode;
  pos: number;
}

interface Measuring {
  paragraph: HTMLElement;
  spans: HTMLElement[];
  look: WordParagraphLook | undefined;
  origin: number;
  scale: number;
  next: number;
}

const holdsTab = (node: ProseMirrorNode) => {
  let found = false;
  node.forEach((child) => {
    if (!found && child.marks.some((mark) => mark.type.name === WORD_TAB_MARK)) found = true;
  });
  return found;
};

/** What comes after a tab up to the next one (or the paragraph's end), and up to its first decimal point. */
function following(span: HTMLElement, next: HTMLElement | undefined, paragraph: HTMLElement) {
  const range = document.createRange();
  range.setStartAfter(span);
  if (next) range.setEndBefore(next);
  else range.setEnd(paragraph, paragraph.childNodes.length);
  const width = range.getBoundingClientRect().width;
  const walker = document.createTreeWalker(range.commonAncestorContainer, NodeFilter.SHOW_TEXT);
  for (let text = walker.nextNode(); text; text = walker.nextNode()) {
    if (!range.intersectsNode(text)) continue;
    const point = text.textContent?.indexOf(".") ?? -1;
    if (point < 0) continue;
    const before = document.createRange();
    before.setStartAfter(span);
    before.setEnd(text, point);
    return { width, beforeDecimal: before.getBoundingClientRect().width };
  }
  return { width, beforeDecimal: width };
}

/**
 * Draws each tab to its stop: measures a paragraph's tabs once each time the
 * paragraph changes, with no decorations, and sets the widths on the tabs'
 * own elements. Only while the Mind shows the Word file's look.
 */
class TabMeasurer {
  private measured = new WeakSet<ProseMirrorNode>();
  private frame = 0;
  private doc: Y.Doc | null = null;
  private look: WordLook | null = null;
  private stopObserving: (() => void) | null = null;
  private readonly again = () => {
    this.measured = new WeakSet();
    this.later();
  };
  private width = 0;
  // A new width can wrap a tab onto another line, where it reaches another stop.
  private readonly resized = new ResizeObserver(() => {
    const width = this.view.dom.clientWidth;
    if (width === this.width) {
      this.later();
      return;
    }
    this.width = width;
    this.again();
  });

  constructor(
    private readonly view: EditorView,
    private readonly docOf: () => Y.Doc | null,
  ) {
    document.fonts?.addEventListener("loadingdone", this.again);
    this.width = view.dom.clientWidth;
    this.resized.observe(view.dom);
    this.update(view);
  }

  update(view: EditorView, previous?: EditorState): void {
    this.bind();
    if (previous && previous.doc === view.state.doc) return;
    if (!this.shown()) return;
    const pending = this.pending();
    if (pending.length === 0) return;
    this.measure(pending.slice(0, MEASURED_AT_ONCE));
    if (pending.length > MEASURED_AT_ONCE) this.later();
  }

  destroy(): void {
    cancelAnimationFrame(this.frame);
    document.fonts?.removeEventListener("loadingdone", this.again);
    this.resized.disconnect();
    this.stopObserving?.();
  }

  private shown(): boolean {
    return this.look !== null && this.view.dom.closest("[class*='word-look-']") !== null;
  }

  private bind(): void {
    const doc = this.docOf();
    if (doc === this.doc) return;
    this.stopObserving?.();
    this.doc = doc;
    this.look = doc ? wordLookOf(doc) : null;
    this.stopObserving = doc
      ? observeWordLook(doc, () => {
          this.look = wordLookOf(doc);
          this.again();
        })
      : null;
  }

  private later(): void {
    cancelAnimationFrame(this.frame);
    this.frame = requestAnimationFrame(() => {
      const start = performance.now();
      if (!this.shown()) return;
      for (let pending = this.pending(); pending.length > 0; pending = this.pending()) {
        if (performance.now() - start > FRAME_BUDGET_MS) {
          this.later();
          return;
        }
        // Nothing measured: the Mind isn't laid out (a tab in the background); its resize brings it back.
        if (this.measure(pending.slice(0, MEASURED_AT_ONCE)) === 0) return;
      }
    });
  }

  private pending(): Pending[] {
    const found: Pending[] = [];
    this.view.state.doc.descendants((node, pos) => {
      if (!node.isTextblock) return true;
      if (!this.measured.has(node)) {
        if (holdsTab(node)) found.push({ node, pos });
        else this.measured.add(node);
      }
      return false;
    });
    return found;
  }

  /** How many paragraphs it measured; none while the Mind isn't laid out. */
  private measure(pending: Pending[]): number {
    const look = this.look;
    if (!look) return 0;
    const ptToPx = Number.parseFloat(getComputedStyle(this.view.dom).fontSize) / (look.body ?? 11);
    let measuring: Measuring[] = [];
    for (const { node, pos } of pending) {
      const paragraph = this.view.nodeDOM(pos);
      if (!(paragraph instanceof HTMLElement) || paragraph.offsetWidth === 0) continue;
      this.measured.add(node);
      const spans = [...paragraph.querySelectorAll<HTMLElement>(TAB_SELECTOR)];
      if (spans.length === 0) continue;
      const box = paragraph.getBoundingClientRect();
      const scale = box.width / paragraph.offsetWidth;
      const key = node.attrs[WORD_LOOK_ATTRIBUTE];
      measuring.push({
        paragraph,
        spans,
        look: typeof key === "string" ? look.paragraphs[key] : undefined,
        origin: box.left - Number.parseFloat(getComputedStyle(paragraph).marginLeft) * scale,
        scale,
        next: 0,
      });
    }
    const count = measuring.length;
    // A round reads every paragraph's next tab, then sets them all, so the page lays out once a round.
    while (measuring.length > 0) {
      const widths = measuring.flatMap((item) => {
        const span = item.spans[item.next];
        if (!span) return [];
        const x = (span.getBoundingClientRect().left - item.origin) / item.scale;
        const stop = tabStopAfter(x / ptToPx, item.look);
        const after = following(span, item.spans[item.next + 1], item.paragraph);
        const width = tabWidth(stop, {
          x,
          stopX: stop.at * ptToPx,
          following: after.width / item.scale,
          beforeDecimal: after.beforeDecimal / item.scale,
        });
        return [{ span, stop, width }];
      });
      for (const { span, stop, width } of widths) {
        span.style.width = `${width}px`;
        if (stop.leader === "none") delete span.dataset.leader;
        else span.dataset.leader = stop.leader;
      }
      measuring = measuring.filter((item) => ++item.next < item.spans.length);
    }
    return count;
  }
}

/**
 * A tab character of a Word file's text, drawn to its tab stop. Not inclusive:
 * what is typed beside a tab is text.
 */
export const WordTab = Mark.create({
  name: WORD_TAB_MARK,
  inclusive: false,

  parseHTML: () => [{ tag: `span${TAB_SELECTOR}` }],
  renderHTML: () => ["span", { [WORD_TAB_DOM_ATTRIBUTE]: "" }, 0],

  addMarkView() {
    return () => {
      const dom = document.createElement("span");
      dom.setAttribute(WORD_TAB_DOM_ATTRIBUTE, "");
      // The measuring sets the width and leader; the editor mustn't read them back as an edit.
      return { dom, contentDOM: dom, ignoreMutation: (mutation) => mutation.type === "attributes" };
    };
  },

  addProseMirrorPlugins() {
    const editor = this.editor;
    return [new Plugin({ view: (view) => new TabMeasurer(view, () => editorDoc(editor)) })];
  },
});
