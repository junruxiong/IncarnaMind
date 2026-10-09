import type { Editor } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import type { EditorState } from "@tiptap/pm/state";
import { TextSelection } from "@tiptap/pm/state";
import { type CSSProperties, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { ANSWER_BLOCK, QUESTION_BLOCK, type SearchScope } from "../../../core/api";
import { scopeAttributesOf } from "../../../shared/searchScope";

/** Where the hint shows: on an empty Mind's line, or on the empty line after an Answer. */
export type HintPlace = "empty" | "afterAnswer";

/** The hint's words, split around the part that starts a Question when clicked. */
export interface HintText {
  before: string;
  ask: string;
  after: string;
}

/** The empty line a hint shows on: its position, which hint, and whether the cursor is on it. */
interface HintLine {
  pos: number;
  place: HintPlace;
  /** Its button takes clicks only then: a first click on the line is for writing there. */
  armed: boolean;
}

const isEmptyLine = (node: ProseMirrorNode | null) =>
  node?.type.name === "paragraph" && node.content.size === 0;

/** A Mind with nothing in it: one empty line. (An empty Question is something.) */
export const isEmptyMind = (doc: ProseMirrorNode) =>
  doc.childCount === 1 && isEmptyLine(doc.firstChild);

/**
 * The line that shows a hint, if any: an empty Mind's line; the empty line
 * after an Answer at the end of the Mind; or one after an Answer elsewhere,
 * while it is being written in.
 */
export function hintLine(state: EditorState, focused: boolean): HintLine | null {
  const { doc, selection } = state;
  const isCurrent = (pos: number) => focused && selection.empty && selection.$from.pos === pos + 1;
  if (isEmptyMind(doc)) return { pos: 0, place: "empty", armed: isCurrent(0) };
  let found: HintLine | null = null;
  doc.forEach((node, offset, index) => {
    if (found || index === 0 || !isEmptyLine(node)) return;
    if (doc.child(index - 1).type.name !== ANSWER_BLOCK) return;
    const current = isCurrent(offset);
    if (index === doc.childCount - 1 || current) {
      found = { pos: offset, place: "afterAnswer", armed: current };
    }
  });
  return found;
}

/**
 * Starts a Question at the end of the Mind: on its empty last line, or on a
 * new line after its last Block, with a Search scope if given.
 */
export function startQuestionAtEnd(editor: Editor, scope: SearchScope | null = null): void {
  if (editor.isDestroyed) return;
  editor
    .chain()
    .focus()
    .command(({ tr, state }) => {
      const last = state.doc.lastChild;
      let end = state.doc.content.size;
      if (isEmptyLine(last) && last) {
        end -= last.nodeSize;
      } else {
        const paragraph = state.schema.nodes.paragraph?.create();
        if (!paragraph) return false;
        tr.insert(end, paragraph);
      }
      tr.setSelection(TextSelection.create(tr.doc, end + 1));
      return true;
    })
    .setQuestion()
    .updateAttributes(QUESTION_BLOCK, scope ? scopeAttributesOf(scope) : {})
    .scrollIntoView()
    .run();
}

/**
 * The hint on an empty line where the next Block is likely written: "Start
 * writing, or press ⌘J to ask a Question…", or after an Answer "Keep
 * writing, or press ⌘J to ask a follow-up". It is drawn over the line, not in
 * the editor's text (so typing there, IME composition too, is left alone),
 * and its "press ⌘J…" part is a button that turns the line into a Question.
 * The button takes clicks once the cursor is on the line: before, a click
 * anywhere on the line puts the cursor there, to write.
 */
export function EndHint({
  editor,
  text,
}: {
  editor: Editor;
  text: (place: HintPlace) => HintText;
}) {
  const layer = useRef<HTMLDivElement>(null);
  const [hint, setHint] = useState<(HintLine & { style: CSSProperties }) | null>(null);

  useEffect(() => {
    let frame = 0;
    // Measured in a frame, and drawn in that same frame, before it is painted:
    // the hint never shows over a line that has just moved or been written
    // on, e.g. under an Answer's error row as it appears.
    const show: typeof setHint = (next) => flushSync(() => setHint(next));
    const measure = () => {
      frame = 0;
      const origin = layer.current?.parentElement;
      if (!origin || editor.isDestroyed) return;
      const line = hintLine(editor.state, editor.isFocused);
      const dom = line ? editor.view.nodeDOM(line.pos) : null;
      if (!line || !(dom instanceof HTMLElement)) {
        show(null);
        return;
      }
      const box = origin.getBoundingClientRect();
      const rect = dom.getBoundingClientRect();
      const style = {
        top: Math.round(rect.top - box.top),
        left: Math.round(rect.left - box.left),
        width: Math.round(rect.width),
      };
      show((shown) =>
        shown &&
        shown.pos === line.pos &&
        shown.place === line.place &&
        shown.armed === line.armed &&
        shown.style.top === style.top &&
        shown.style.left === style.left &&
        shown.style.width === style.width
          ? shown
          : { ...line, style },
      );
    };
    const request = () => {
      if (!frame) frame = requestAnimationFrame(measure);
    };
    request();
    editor.on("transaction", request);
    editor.on("focus", request);
    editor.on("blur", request);
    const resizes = new ResizeObserver(request);
    resizes.observe(editor.view.dom);
    // Node views (Answers, KaTeX) draw after the transaction, moving the lines below them.
    const mutations = new MutationObserver(request);
    mutations.observe(editor.view.dom, { subtree: true, childList: true });
    return () => {
      cancelAnimationFrame(frame);
      editor.off("transaction", request);
      editor.off("focus", request);
      editor.off("blur", request);
      resizes.disconnect();
      mutations.disconnect();
    };
  }, [editor]);

  const words = hint ? text(hint.place) : null;
  return (
    <div ref={layer} className="contents">
      {hint && words && (
        <p
          data-testid="end-hint"
          data-place={hint.place}
          data-armed={hint.armed || undefined}
          className="end-hint"
          style={hint.style}
        >
          {words.before}
          <button
            type="button"
            data-testid="end-hint-ask"
            className={`end-hint-ask ${hint.armed ? "end-hint-ask--armed" : ""}`}
            // The editor keeps the focus, so the Question starts where its line is.
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => {
              editor
                .chain()
                .focus()
                .setTextSelection(hint.pos + 1)
                .setQuestion()
                .scrollIntoView()
                .run();
            }}
          >
            {words.ask}
          </button>
          {words.after}
        </p>
      )}
    </div>
  );
}
