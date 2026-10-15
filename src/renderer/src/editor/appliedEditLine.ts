/**
 * The lines over an edit the User asked for (#880; DESIGN.md, AI edits), in
 * the Mind's text, above the Block they are about:
 *
 * - over the first Block it changed, "Changed 3 places · Undo ⌘Z · Review
 *   changes", its reason on hover; while its changes are reviewed, "3 changes
 *   made by Edit with AI · Undo all · Done"; and for an edit waiting for the
 *   User, "This edit would remove most of what you selected, so it's waiting
 *   for you · Apply · Discard";
 * - over the first paragraph it left out and kept, "Kept 2 paragraphs the
 *   edit left out · Remove them".
 *
 * Their buttons are handled where the Mind is known (`AppliedEditActions`),
 * by `data-applied-action`. The lines go once the User edits the text.
 */
import { type Editor, Extension } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { type EditorState, Plugin, PluginKey, type Transaction } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";
import { ySyncPluginKey } from "@tiptap/y-tiptap";
import type { Language } from "../../../core/language";
import type { MessageKey, MessageParams } from "../../../shared/i18n";
import { SPARK_PATH } from "../components/iconPaths";
import { proposalsIn } from "./proposalReview";
import { blockAt } from "./writingPlace";

export type AppliedAction =
  | "undo"
  | "review"
  | "undo-all"
  | "done"
  | "apply"
  | "discard"
  | "remove-kept";

/** The paragraphs an edit left out and kept: the first one's Block, and how many. */
export interface KeptLine {
  blockId: string;
  count: number;
  /** Of more than one kind (paragraphs, headings, list items): named Blocks, not paragraphs. */
  mixed: boolean;
}

export type AppliedLine = (
  | {
      kind: "applied";
      /** The first Block it changed; null at the Mind's end. */
      blockId: string | null;
      places: number;
      reason: string | null;
      /** Whether its changes can be shown again as changes (Review changes). */
      reviewable: boolean;
      kept: KeptLine | null;
    }
  | { kind: "review"; blockId: string | null; request: string }
  | { kind: "held"; blockId: string | null; request: string }
) & {
  /** The interface language it was last drawn in, once that changed. */
  language?: Language;
};

interface LineState {
  line: AppliedLine | null;
  marks: DecorationSet;
}

type Translate = (key: MessageKey, params?: MessageParams) => string;

interface Options {
  translate: Translate;
  /** ⌘Z as this system writes it. */
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

function text(className: string, words: string, testId?: string): HTMLSpanElement {
  const span = document.createElement("span");
  span.className = className;
  span.textContent = words;
  if (testId) span.dataset.testid = testId;
  return span;
}

/** How many of the request's changes still wait in the text. */
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
      text("applied-keys", options.undoKeys()),
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

/** Where a line over this Block goes: just before it, or at the Mind's end. */
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

/** A change from the Mind's document (the core, Undo, an edit applying), not the User typing here. */
const fromDocument = (tr: Transaction) => !!tr.getMeta(ySyncPluginKey)?.isChangeOrigin;

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

/** The line over the edit the User asked for last, if it shows. */
export const appliedLineIn = (state: EditorState): AppliedLine | null =>
  appliedLineKey.getState(state)?.line ?? null;

/** Shows a line over an edit, or takes it away (null). */
export function setAppliedLine(editor: Editor, line: AppliedLine | null): void {
  if (editor.isDestroyed) return;
  editor.view.dispatch(editor.state.tr.setMeta(appliedLineKey, line));
}

/** Draws the lines again, in the interface language `language` now is. */
export function redrawAppliedLine(editor: Editor, language: Language): void {
  const line = appliedLineIn(editor.state);
  if (line && line.language !== language) setAppliedLine(editor, { ...line, language });
}
