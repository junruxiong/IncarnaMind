/**
 * The lines over an edit the User asked for (#880; DESIGN.md, AI edits),
 * drawn in the text above the Block they are about. ProposalBars handles
 * their buttons, by `data-applied-action`. They go once the User edits.
 */
import { type Editor, Extension } from "@tiptap/core";
import type { Fragment, Node as ProseMirrorNode } from "@tiptap/pm/model";
import { type EditorState, Plugin, PluginKey, TextSelection } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";
import { relativePositionToAbsolutePosition, ySyncPluginKey } from "@tiptap/y-tiptap";
import type { Language } from "../../../core/language";
import type { MessageKey, MessageParams } from "../../../shared/i18n";
import { type ChangeCounts, changeFact } from "../changeFacts";
import { SPARK_PATH } from "../components/iconPaths";
import {
  type ChangedBlock,
  LINE_WIDGET,
  originalShown,
  rewrittenSections,
  sectionKey,
  sectionLine,
  toggled,
  wasLine,
} from "./changeOriginals";
import { fromDocument, proposalsIn } from "./proposalReview";
import { blockAt } from "./writingPlace";

export const APPLIED_ACTIONS = [
  "undo",
  "review",
  "undo-all",
  "done",
  "apply",
  "discard",
  "remove-kept",
] as const;
export type AppliedAction = (typeof APPLIED_ACTIONS)[number];

export const isAppliedAction = (value: string): value is AppliedAction =>
  (APPLIED_ACTIONS as readonly string[]).includes(value);

export interface KeptLine {
  blockId: string;
  count: number;
  /** Of more than one kind: named Blocks, not paragraphs. */
  mixed: boolean;
}

/** One change an applied edit made: where its new words are, as Yjs positions that hold through edits, and the old ones it took out. */
export interface AppliedChange {
  from: unknown;
  to: unknown;
  removed: Fragment;
  /** Whole Blocks, added or taken out; `to` of added ones is just inside the last one's end. */
  block: "added" | "removed" | null;
  /**
   * The paragraph it wrote anew, by Block id (#1028): its place is all that
   * paragraph's text, as the positions before its old words went with them.
   */
  paragraph?: string;
}

export type AppliedLine = (
  | {
      kind: "applied";
      /** Null at the Mind's end. */
      blockId: string | null;
      /** The paragraphs and headings it changed, added or removed. */
      paragraphs: number;
      /** What the assistant was asked for, as its Answer counts it (#880): "Wrote 3 paragraphs". */
      facts?: ChangeCounts;
      reason: string | null;
      changes: readonly AppliedChange[];
      kept: KeptLine | null;
    }
  | {
      kind: "review";
      blockId: string | null;
      changes: readonly AppliedChange[];
      byAssistant?: true;
      undo?: { available(): boolean; mind: string };
      /** Changes, and sections, whose old text shows, struck (`changeOriginals`). */
      shown?: readonly string[];
    }
  | {
      kind: "held";
      blockId: string | null;
      request: string;
      /** Asked of the assistant: all of its requests in the Mind, held as they would rewrite most of the User's text. */
      asked?: { mindId: string; requests: readonly string[]; facts: ChangeCounts };
    }
) & {
  /** The language it was last drawn in, once that changed. */
  language?: Language;
};

const LINE_KINDS: ReadonlySet<unknown> = new Set<AppliedLine["kind"]>([
  "applied",
  "review",
  "held",
]);

const isAppliedLine = (value: unknown): value is AppliedLine =>
  typeof value === "object" && value !== null && "kind" in value && LINE_KINDS.has(value.kind);

interface LineState {
  line: AppliedLine | null;
  marks: DecorationSet;
}

type Translate = (key: MessageKey, params?: MessageParams) => string;

interface Options {
  translate: Translate;
  undoKeys(): string;
}

export const appliedLineKey = new PluginKey<LineState>("appliedEditLine");

function button(action: AppliedAction, label: string): HTMLButtonElement {
  const each = document.createElement("button");
  each.type = "button";
  each.className = "proposal-action";
  each.dataset.appliedAction = action;
  each.dataset.testid = `applied-${action}`;
  each.textContent = label;
  return each;
}

function dot(): HTMLSpanElement {
  const each = document.createElement("span");
  each.className = "proposal-dot";
  each.setAttribute("aria-hidden", "true");
  each.textContent = "·";
  return each;
}

function lineElement(testId: string, className: string, parts: (Node | string)[]): HTMLElement {
  const line = document.createElement("div");
  line.contentEditable = "false";
  line.className = className;
  line.dataset.testid = testId;
  line.setAttribute("role", "group");
  line.append(...parts);
  return line;
}

function sparkle(): SVGSVGElement {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  for (const [name, value] of Object.entries({
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    "stroke-width": "2",
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
    "aria-hidden": "true",
    class: "proposals-bar-icon",
  })) {
    svg.setAttribute(name, value);
  }
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", SPARK_PATH);
  svg.append(path);
  return svg;
}

function keys(label: string): HTMLElement {
  const kbd = document.createElement("kbd");
  kbd.className = "applied-keys";
  kbd.textContent = label;
  return kbd;
}

function text(className: string, words: string, testId?: string): HTMLSpanElement {
  const span = document.createElement("span");
  span.className = className;
  span.textContent = words;
  if (testId) span.dataset.testid = testId;
  return span;
}

const waitingOf = (doc: ProseMirrorNode, request: string) =>
  proposalsIn(doc).filter((each) => each.request === request).length;

/** Where a change is in the editor now; null once its words are gone. */
export function placeOfChange(
  state: EditorState,
  change: AppliedChange,
): { from: number; to: number } | null {
  if (change.paragraph) {
    const found = blockAt(state.doc, change.paragraph);
    return found ? { from: found.pos + 1, to: found.pos + found.node.nodeSize - 1 } : null;
  }
  const binding = ySyncPluginKey.getState(state)?.binding;
  if (!binding) return null;
  const at = (relative: unknown): number | null =>
    relativePositionToAbsolutePosition(binding.doc, binding.type, relative, binding.mapping);
  const from = at(change.from);
  const inside = at(change.to);
  const end = inside === null ? null : state.doc.resolve(inside);
  const to = change.block !== "added" ? inside : end && end.depth > 0 ? end.after(1) : null;
  return from === null || to === null || to < from ? null : { from, to };
}

const textOf = (removed: Fragment) => removed.textBetween(0, removed.size, "\n", "￼");

const COUNTED: ReadonlySet<string> = new Set(["paragraph", "heading"]);

/** How many paragraphs and headings the changes shown again are in: what Review changes counts. */
export function paragraphsOfChanges(state: EditorState, changes: readonly AppliedChange[]): number {
  const counted = new Set<number>();
  let removed = 0;
  for (const change of changes) {
    const place = placeOfChange(state, change);
    if (!place) continue;
    if (change.block === "removed") {
      let blocks = 0;
      change.removed.descendants((node) => {
        if (COUNTED.has(node.type.name)) blocks++;
        return !node.isTextblock;
      });
      removed += Math.max(blocks, 1);
      continue;
    }
    state.doc.nodesBetween(place.from, Math.max(place.to, place.from + 1), (node, pos) => {
      if (COUNTED.has(node.type.name)) counted.add(pos);
      return !node.isTextblock;
    });
  }
  return counted.size + removed;
}

function mainLine(line: AppliedLine, options: Options, paragraphs: number): HTMLElement {
  const t = options.translate;
  if (line.kind === "applied") {
    const what = text(
      "min-w-0",
      line.facts
        ? changeFact(t, line.facts, null)
        : line.paragraphs === 1
          ? t("edit.applied.one")
          : t("edit.applied", { count: line.paragraphs }),
      "applied-count",
    );
    const parts: (Node | string)[] = [
      sparkle(),
      what,
      dot(),
      button("undo", t("edit.applied.undo")),
      keys(options.undoKeys()),
    ];
    if (line.changes.length > 0) parts.push(dot(), button("review", t("edit.applied.review")));
    const element = lineElement("applied-line", "applied-line", parts);
    if (line.reason) element.title = t("proposal.why", { reason: line.reason });
    return element;
  }
  if (line.kind === "review") {
    const count = paragraphs;
    const undo = button("undo-all", t("edit.review.undoAll"));
    if (line.undo && !line.undo.available()) {
      undo.setAttribute("aria-disabled", "true");
      undo.title = t("mindChanges.undo.changedSince", { mind: line.undo.mind });
    }
    return lineElement("applied-line", "applied-line", [
      sparkle(),
      text(
        "min-w-0",
        line.byAssistant
          ? count === 1
            ? t("edit.review.assistant.one")
            : t("edit.review.assistant", { count })
          : count === 1
            ? t("edit.review.one")
            : t("edit.review", { count }),
        "applied-count",
      ),
      dot(),
      undo,
      dot(),
      button("done", t("edit.review.done")),
    ]);
  }
  return lineElement("applied-line", "applied-line", [
    sparkle(),
    text("min-w-0", t(line.asked ? "edit.held.own" : "edit.held"), "applied-count"),
    dot(),
    button("apply", t("edit.held.apply")),
    dot(),
    button("discard", t("edit.held.discard")),
  ]);
}

function keptLine(kept: KeptLine, t: Translate): HTMLElement {
  const words =
    kept.count === 1
      ? t(kept.mixed ? "edit.kept.blocks.one" : "edit.kept.paragraphs.one")
      : t(kept.mixed ? "edit.kept.blocks" : "edit.kept.paragraphs", { count: kept.count });
  return lineElement("applied-kept", "applied-kept-line", [
    text("min-w-0", words),
    dot(),
    button("remove-kept", kept.count === 1 ? t("edit.kept.remove.one") : t("edit.kept.remove")),
  ]);
}

function placeOf(doc: ProseMirrorNode, blockId: string | null): number | null {
  if (blockId === null) return doc.content.size;
  return blockAt(doc, blockId)?.pos ?? null;
}

const lineText = (text: string) =>
  text
    .replace(/\uFFFC/g, "")
    .replace(/\s+/g, " ")
    .trim();

function changedBlockOf(
  state: EditorState,
  change: AppliedChange,
  index: number,
  place: { from: number; to: number },
): ChangedBlock {
  const key = `applied:${index}`;
  if (change.block) {
    return { key, pos: place.from, how: change.block === "added" ? "added" : "removed" };
  }
  const $from = state.doc.resolve(place.from);
  const pos = $from.depth > 0 ? $from.before() : place.from;
  return change.paragraph
    ? {
        key,
        pos,
        how: "rewritten",
        old: lineText(textOf(change.removed)),
        title: lineText(state.doc.textBetween(place.from, place.to, " ", "\uFFFC")),
      }
    : { key, pos, how: "words" };
}

function reviewedBlocks(state: EditorState, changes: readonly AppliedChange[]) {
  const placed = changes.flatMap((change, index) => {
    const place = placeOfChange(state, change);
    return place
      ? [{ change, index, place, block: changedBlockOf(state, change, index, place) }]
      : [];
  });
  const sections = rewrittenSections(
    state.doc,
    placed.map(({ block }) => block),
    false,
  );
  return { placed, sections };
}

/**
 * What shows the old text of the change shown again that the cursor is in,
 * its section's when it is in one, and whether it shows; null for one not
 * written anew.
 */
export function appliedOriginalAt(state: EditorState): { key: string; shown: boolean } | null {
  const line = appliedLineIn(state);
  if (line?.kind !== "review") return null;
  const { placed, sections } = reviewedBlocks(state, line.changes);
  const { from } = state.selection;
  const here = placed.find(({ place }) => place.from <= from && from <= place.to);
  if (!here) return null;
  const shownKeys = new Set(line.shown);
  const { inSection, shown } = originalShown(here.block, sections, shownKeys);
  if (inSection) return { key: sectionKey(inSection), shown };
  return here.block.how === "rewritten" ? { key: here.block.key, shown } : null;
}

/**
 * An applied edit's changes shown again (Review changes): old words struck,
 * new ones underlined, the doc left as it is; a paragraph written anew, and
 * a section, read as `changeOriginals` says.
 */
function changeMarks(
  state: EditorState,
  changes: readonly AppliedChange[],
  shownKeys: ReadonlySet<string>,
  t: Translate,
): Decoration[] {
  const { placed, sections } = reviewedBlocks(state, changes);
  const lines: Decoration[] = sections.map((section) => {
    const open = shownKeys.has(sectionKey(section));
    return Decoration.widget(section.heading.pos, () => sectionLine(t, section, open), {
      ...LINE_WIDGET,
      key: `rewrote:${section.heading.key}:${open}:${section.heading.title}:${t("change.showOriginal")}`,
    });
  });
  return lines.concat(
    placed.flatMap(({ change, index, place, block }) => {
      const { inSection, shown } = originalShown(block, sections, shownKeys);
      const marks: Decoration[] = [];
      if (block.how === "rewritten" && !inSection) {
        marks.push(
          Decoration.widget(block.pos, () => wasLine(t, block, shown), {
            ...LINE_WIDGET,
            key: `was:${block.key}:${shown}:${block.old}:${t("change.showOriginal")}`,
          }),
        );
      }
      const oldShows =
        block.how === "rewritten" || (block.how === "removed" && inSection) ? shown : true;
      if (change.removed.size > 0 && oldShows) {
        const old = () => {
          const element = document.createElement(change.block ? "div" : "del");
          element.className = change.block ? "applied-old applied-old--block" : "applied-old";
          element.dataset.appliedChange = String(index);
          element.textContent = textOf(change.removed);
          return element;
        };
        marks.push(
          Decoration.widget(place.from, old, {
            side: -1,
            ignoreSelection: true,
            key: `applied-old:${index}:${textOf(change.removed)}`,
          }),
        );
      }
      if (place.to > place.from) {
        const attributes = { "data-applied-change": String(index) };
        marks.push(
          change.block
            ? Decoration.node(place.from, place.to, {
                ...attributes,
                class: "applied-new applied-new--block",
              })
            : Decoration.inline(place.from, place.to, { ...attributes, class: "applied-new" }),
        );
      }
      return marks;
    }),
  );
}

function decorations(state: EditorState, line: AppliedLine, options: Options): DecorationSet {
  const { doc } = state;
  const marks: Decoration[] =
    line.kind === "review"
      ? changeMarks(state, line.changes, new Set(line.shown), options.translate)
      : [];
  const at = placeOf(doc, line.blockId);
  const shared = { side: -1, ignoreSelection: true, stopEvent: () => true };
  if (at !== null) {
    const count =
      line.kind === "applied"
        ? line.paragraphs
        : line.kind === "review"
          ? paragraphsOfChanges(state, line.changes)
          : 0;
    const canUndo = line.kind !== "review" || !line.undo || line.undo.available();
    marks.push(
      Decoration.widget(at, () => mainLine(line, options, count), {
        ...shared,
        // Drawn again in the interface language once it changes.
        key: `applied:${line.kind}:${count}:${line.blockId}:${canUndo}:${options.translate("edit.applied.undo")}`,
      }),
    );
  }
  const kept = line.kind === "applied" ? line.kept : null;
  const keptAt = kept ? placeOf(doc, kept.blockId) : null;
  if (kept && keptAt !== null) {
    marks.push(
      Decoration.widget(keptAt, () => keptLine(kept, options.translate), {
        ...shared,
        side: -2,
        key: `applied-kept:${kept.blockId}:${kept.count}:${options.translate("edit.kept.remove")}`,
      }),
    );
  }
  return DecorationSet.create(doc, marks);
}

export const AppliedEditLine = Extension.create<Options>({
  name: "appliedEditLine",

  addOptions() {
    return { translate: (key) => key, undoKeys: () => "" };
  },

  addProseMirrorPlugins() {
    const options = this.options;
    return [
      new Plugin<LineState>({
        key: appliedLineKey,
        state: {
          init: () => ({ line: null, marks: DecorationSet.empty }),
          apply: (tr, value, _before, state) => {
            const meta: unknown = tr.getMeta(appliedLineKey);
            // Typing ends the line, but a held edit waits for its Apply or Discard.
            const typedOver = tr.docChanged && !fromDocument(tr) && value.line?.kind !== "held";
            const shown =
              meta === null || isAppliedLine(meta) ? meta : typedOver ? null : value.line;
            // Applied or discarded one by one, a held edit's changes are all gone; restored, a reviewed one's.
            const gone =
              shown !== null &&
              (shown.kind === "held"
                ? (shown !== value.line || tr.docChanged) &&
                  (shown.asked?.requests ?? [shown.request]).every(
                    (request) => waitingOf(state.doc, request) === 0,
                  )
                : shown.kind === "review" && shown.changes.length === 0);
            const line = gone ? null : shown;
            if (!line) return { line: null, marks: DecorationSet.empty };
            if (line !== value.line || (tr.docChanged && fromDocument(tr))) {
              return { line, marks: decorations(state, line, options) };
            }
            return tr.docChanged ? { line, marks: value.marks.map(tr.mapping, tr.doc) } : value;
          },
        },
        props: {
          decorations: (state) => appliedLineKey.getState(state)?.marks ?? null,
          handleDOMEvents: {
            // A click on a change shown again puts the cursor in it, for ⌘⌫ and its bar.
            mousedown(view, event) {
              const target = event.target instanceof Element ? event.target : null;
              const index = Number(
                target?.closest("[data-applied-change]")?.getAttribute("data-applied-change"),
              );
              const line = appliedLineKey.getState(view.state)?.line;
              const change = line?.kind === "review" ? line.changes[index] : undefined;
              const place = change ? placeOfChange(view.state, change) : null;
              if (!place) return false;
              event.preventDefault();
              view.focus();
              view.dispatch(
                view.state.tr.setSelection(TextSelection.create(view.state.doc, place.from)),
              );
              return true;
            },
          },
        },
      }),
    ];
  },
});

/** Shows an applied change's old text, or its section's, struck in place; or hides it again. */
export function toggleAppliedOriginal(editor: Editor, key: string): void {
  const line = appliedLineIn(editor.state);
  if (line?.kind !== "review") return;
  setAppliedLine(editor, { ...line, shown: [...toggled(new Set(line.shown), key)] });
}

export const appliedLineIn = (state: EditorState): AppliedLine | null =>
  appliedLineKey.getState(state)?.line ?? null;

export function setAppliedLine(editor: Editor, line: AppliedLine | null): void {
  if (editor.isDestroyed) return;
  editor.view.dispatch(editor.state.tr.setMeta(appliedLineKey, line));
}

export function redrawAppliedLine(editor: Editor, language: Language): void {
  const line = appliedLineIn(editor.state);
  if (line && line.language !== language) setAppliedLine(editor, { ...line, language });
}
