import { Mark } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { type EditorState, Plugin, TextSelection } from "@tiptap/pm/state";
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

  private measure(pending: Pending[]): number {
    const look = this.look;
    if (!look) return 0;
    // The content's point, with the look's zoom to fit the frame in it.
    const ptToPx = Number.parseFloat(getComputedStyle(this.view.dom).getPropertyValue("--word-pt"));
    if (!(ptToPx > 0)) return 0;
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
        // Capped at the right margin, so a far stop never runs the line off the column.
        const margin = look.width ?? Number.POSITIVE_INFINITY;
        // Tabs side by side share one element: each goes on from the stop the one before reached.
        const tabs = Math.max(1, (span.textContent ?? "").split("\t").length - 1);
        let stop = tabStopAfter(x / ptToPx, item.look, look.defaultTab ?? undefined);
        for (let more = 1; more < tabs; more++) {
          stop = tabStopAfter(Math.min(stop.at, margin), item.look, look.defaultTab ?? undefined);
        }
        const after = following(span, item.spans[item.next + 1], item.paragraph);
        const stopAt = Math.min(stop.at, margin);
        const width = tabWidth(stop, {
          x,
          stopX: stopAt * ptToPx,
          following: after.width / item.scale,
          beforeDecimal: after.beforeDecimal / item.scale,
        });
        return [{ span, stop, width, tabs }];
      });
      for (const { span, stop, width, tabs } of widths) {
        // The tab characters span their box, so the caret and a selection of them fall where the text does.
        span.style.width = `${width}px`;
        span.style.tabSize = `${width / tabs}px`;
        if (stop.leader === "none") delete span.dataset.leader;
        else span.dataset.leader = stop.leader;
      }
      measuring = measuring.filter((item) => ++item.next < item.spans.length);
    }
    return count;
  }
}

function isTab(doc: ProseMirrorNode, at: number): boolean {
  if (at < 0 || at >= doc.content.size || doc.textBetween(at, at + 1) !== "\t") return false;
  const mark = doc.type.schema.marks[WORD_TAB_MARK];
  return mark !== undefined && doc.rangeHasMark(at, at + 1, mark);
}

function overTab(view: EditorView, head: number, forward: boolean): number | null {
  const tab = forward ? head : head - 1;
  if (!isTab(view.state.doc, tab)) return null;
  const paragraph = view.nodeDOM(view.state.doc.resolve(head).before());
  if (!(paragraph instanceof HTMLElement) || getComputedStyle(paragraph).direction === "rtl") {
    return null;
  }
  return forward ? tab + 1 : tab;
}

function lineEdge(view: EditorView, head: number, end: boolean): number | null {
  if (!isTab(view.state.doc, head - 1)) return null;
  const $head = view.state.doc.resolve(head);
  const paragraph = view.nodeDOM($head.before());
  if (!(paragraph instanceof HTMLElement)) return null;
  // A line of stops never wraps: its edges are the paragraph's.
  if (getComputedStyle(paragraph).whiteSpace === "pre") return end ? $head.end() : $head.start();
  // The paragraph's own edges, its first line's hanging indent out to the left included.
  const box = paragraph.getBoundingClientRect();
  const style = getComputedStyle(paragraph);
  const scale = box.width / paragraph.offsetWidth;
  const px = (value: string) => Number.parseFloat(value) * scale;
  const left = box.left + px(style.paddingLeft) + Math.min(0, px(style.textIndent));
  const right = box.right - px(style.paddingRight);
  const caret = view.coordsAtPos(head);
  const found = view.posAtCoords({
    left: end ? right - 1 : left + 1,
    top: (caret.top + caret.bottom) / 2,
  });
  if (!found) return null;
  return Math.min(Math.max(found.pos, $head.start()), $head.end());
}

/**
 * The caret keys around a tab: at an inline-block's edge Chrome can stop on
 * the edge itself, which ProseMirror reads as no move, and from inside the
 * box Home and End reach only the box's own line.
 */
const tabKeys = new Plugin({
  props: {
    handleKeyDown(view, event) {
      if (event.altKey || event.ctrlKey) return false;
      const arrow = event.key === "ArrowLeft" || event.key === "ArrowRight";
      const edge = event.key === "Home" || event.key === "End" || (event.metaKey && arrow);
      if (!edge && (!arrow || event.metaKey)) return false;
      const { doc, selection } = view.state;
      if (!(selection instanceof TextSelection) || (!event.shiftKey && !selection.empty)) {
        return false;
      }
      const forward = event.key === "ArrowRight" || event.key === "End";
      const head = edge
        ? lineEdge(view, selection.head, forward)
        : overTab(view, selection.head, forward);
      if (head === null) return false;
      const anchor = event.shiftKey ? selection.anchor : head;
      view.dispatch(view.state.tr.setSelection(TextSelection.create(doc, anchor, head)));
      return true;
    },
  },
});

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
    return [
      tabKeys,
      new Plugin({ view: (view) => new TabMeasurer(view, () => editorDoc(editor)) }),
    ];
  },
});
