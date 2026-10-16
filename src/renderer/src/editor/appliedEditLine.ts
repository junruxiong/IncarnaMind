/**
 * The lines over an edit the User asked for (#880; DESIGN.md, AI edits),
 * drawn in the text above the Block they are about. ProposalBars handles
 * their buttons, by `data-applied-action`. They go once the User edits.
 */
import { type Editor, Extension } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { type EditorState, Plugin, PluginKey } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";
import type { Language } from "../../../core/language";
import type { MessageKey, MessageParams } from "../../../shared/i18n";
import { SPARK_PATH } from "../components/iconPaths";
import { fromDocument, proposalsIn } from "./proposalReview";
import { blockAt } from "./writingPlace";

export type AppliedAction =
  | "undo"
  | "review"
  | "undo-all"
  | "done"
  | "apply"
  | "discard"
  | "remove-kept";

export interface KeptLine {
  blockId: string;
  count: number;
  /** Of more than one kind: named Blocks, not paragraphs. */
  mixed: boolean;
}

export type AppliedLine = (
  | {
      kind: "applied";
      /** Null at the Mind's end. */
      blockId: string | null;
      places: number;
      reason: string | null;
      reviewable: boolean;
      kept: KeptLine | null;
    }
  | { kind: "review"; blockId: string | null; request: string }
  | { kind: "held"; blockId: string | null; request: string }
) & {
  /** The language it was last drawn in, once that changed. */
  language?: Language;
};

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

function mainLine(doc: ProseMirrorNode, line: AppliedLine, options: Options): HTMLElement {
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
    if (line.reviewable) parts.push(dot(), button("review", t("edit.applied.review")));
    const element = lineElement("applied-line", "applied-line", parts);
    if (line.reason) element.title = t("proposal.why", { reason: line.reason });
    return element;
  }
  if (line.kind === "review") {
    const count = waitingOf(doc, line.request);
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

function decorations(doc: ProseMirrorNode, line: AppliedLine, options: Options): DecorationSet {
  const marks: Decoration[] = [];
  const at = placeOf(doc, line.blockId);
  const shared = { side: -1, ignoreSelection: true, stopEvent: () => true };
  if (at !== null) {
    const count = line.kind === "review" ? waitingOf(doc, line.request) : 0;
    marks.push(
      Decoration.widget(at, () => mainLine(doc, line, options), {
        ...shared,
        // Drawn again in the interface language once it changes.
        key: `applied:${line.kind}:${line.kind === "applied" ? line.places : count}:${line.blockId}:${options.translate("edit.applied.undo")}`,
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
            const meta = tr.getMeta(appliedLineKey) as AppliedLine | null | undefined;
            const shown =
              meta !== undefined ? meta : tr.docChanged && !fromDocument(tr) ? null : value.line;
            // Restored, applied or discarded one by one, a reviewed or held edit's changes are all gone.
            const gone =
              shown !== null &&
              shown.kind !== "applied" &&
              (shown !== value.line || tr.docChanged) &&
              waitingOf(state.doc, shown.request) === 0;
            const line = gone ? null : shown;
            if (!line) return { line: null, marks: DecorationSet.empty };
            if (line !== value.line || (tr.docChanged && fromDocument(tr))) {
              return { line, marks: decorations(state.doc, line, options) };
            }
            return tr.docChanged ? { line, marks: value.marks.map(tr.mapping, tr.doc) } : value;
          },
        },
        props: {
          decorations: (state) => appliedLineKey.getState(state)?.marks ?? null,
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
