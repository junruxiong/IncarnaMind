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
import { SPARK_PATH } from "../components/iconPaths";
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
}

export type AppliedLine = (
  | {
      kind: "applied";
      /** Null at the Mind's end. */
      blockId: string | null;
      places: number;
      reason: string | null;
      changes: readonly AppliedChange[];
      kept: KeptLine | null;
    }
  | { kind: "review"; blockId: string | null; changes: readonly AppliedChange[] }
  | { kind: "held"; blockId: string | null; request: string }
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
  const binding = ySyncPluginKey.getState(state)?.binding;
  if (!binding) return null;
  const at = (relative: unknown): number | null =>
    relativePositionToAbsolutePosition(binding.doc, binding.type, relative, binding.mapping);
  const from = at(change.from);
  const inside = at(change.to);
  const to = inside === null || change.block !== "added" ? inside : inside + 1;
  return from === null || to === null || to < from ? null : { from, to };
}

const textOf = (removed: Fragment) => removed.textBetween(0, removed.size, "\n", "￼");

function mainLine(line: AppliedLine, options: Options): HTMLElement {
  const t = options.translate;
  if (line.kind === "applied") {
    const what = text(
      "min-w-0",
      line.places === 1 ? t("edit.applied.one") : t("edit.applied", { count: line.places }),
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
    const count = line.changes.length;
    return lineElement("applied-line", "applied-line", [
      sparkle(),
      text(
        "min-w-0",
        count === 1 ? t("edit.review.one") : t("edit.review", { count }),
        "applied-count",
      ),
      dot(),
      button("undo-all", t("edit.review.undoAll")),
      dot(),
      button("done", t("edit.review.done")),
    ]);
  }
  return lineElement("applied-line", "applied-line", [
    sparkle(),
    text("min-w-0", t("edit.held"), "applied-count"),
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

/** An applied edit's changes shown again (Review changes): old words struck, new ones underlined, the doc left as it is. */
function changeMarks(state: EditorState, changes: readonly AppliedChange[]): Decoration[] {
  return changes.flatMap((change, index) => {
    const place = placeOfChange(state, change);
    if (!place) return [];
    const marks: Decoration[] = [];
    if (change.removed.size > 0) {
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
  });
}

function decorations(state: EditorState, line: AppliedLine, options: Options): DecorationSet {
  const { doc } = state;
  const marks: Decoration[] = line.kind === "review" ? changeMarks(state, line.changes) : [];
  const at = placeOf(doc, line.blockId);
  const shared = { side: -1, ignoreSelection: true, stopEvent: () => true };
  if (at !== null) {
    const count =
      line.kind === "applied" ? line.places : line.kind === "review" ? line.changes.length : 0;
    marks.push(
      Decoration.widget(at, () => mainLine(line, options), {
        ...shared,
        // Drawn again in the interface language once it changes.
        key: `applied:${line.kind}:${count}:${line.blockId}:${options.translate("edit.applied.undo")}`,
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
                  waitingOf(state.doc, shown.request) === 0
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
