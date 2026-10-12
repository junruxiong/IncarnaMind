import "katex/dist/katex.min.css";
import { EditorContent, ReactNodeViewRenderer, useEditor } from "@tiptap/react";
import { useEffect, useRef } from "react";
import type { NodeJSON } from "../../../../shared/blockJson";
import { lendAnswerEditor } from "../../answerEditors";
import { CitationView } from "../../editor/CitationView";
import { CitationNumbers } from "../../editor/citationNumbers";
import { CopyCitations } from "../../editor/copyCitations";
import { MarginChecks } from "../../editor/MarginChecks";
import { RegenerateAnswerContext } from "../../editor/mindContext";
import { noteExtensions } from "../../editor/noteSchema";
import { useTableFades } from "./useTableFades";

/** A Citation redraws when it changes, and when its number does (as in a Mind). */
const citationView = ReactNodeViewRenderer(CitationView, {
  as: "span",
  update: ({ oldNode, newNode, oldDecorations, newDecorations, updateProps }) => {
    if (newNode.type !== oldNode.type) return false;
    if (newNode !== oldNode || oldDecorations !== newDecorations) updateProps();
    return true;
  },
});

const docOf = (blocks: readonly NodeJSON[]) => ({ type: "doc", content: [...blocks] });

const editorPropsFor = (label: string) => ({
  attributes: {
    "data-testid": "assistant-answer-text",
    "aria-label": label,
    class: "mind-editor assistant-answer-text",
  },
});

/**
 * An Answer's text in the assistant (#201): its Blocks drawn by the note's own
 * schema and node views, read only, with its Citations numbered from 1 and
 * their checks in the margin beside it, as in a Mind. Text can be selected in
 * it, to copy or add just that part (`selectedNotes`). While the Answer is
 * written its Blocks are drawn again as they change, and only then.
 */
export function AnswerText({
  answerId,
  blocks,
  label,
  regenerate,
}: {
  answerId: string;
  blocks: readonly NodeJSON[];
  label: string;
  /** Writes the Answer again. */
  regenerate(): void;
}) {
  const shown = useRef(JSON.stringify(blocks));
  const editor = useEditor(
    {
      editable: false,
      shouldRerenderOnTransaction: false,
      content: docOf(blocks),
      editorProps: editorPropsFor(label),
      extensions: [...noteExtensions({ citationView }), CitationNumbers, CopyCitations],
    },
    [],
  );

  const frame = useRef<HTMLDivElement>(null);
  useTableFades(frame);

  useEffect(() => lendAnswerEditor(answerId, editor), [answerId, editor]);

  useEffect(() => {
    const next = JSON.stringify(blocks);
    if (next === shown.current || editor.isDestroyed) return;
    shown.current = next;
    editor.commands.setContent(docOf(blocks), { emitUpdate: false });
  }, [editor, blocks]);

  // When the language changes: relabelled.
  useEffect(() => {
    if (!editor.isDestroyed) editor.setOptions({ editorProps: editorPropsFor(label) });
  }, [editor, label]);

  return (
    // A quote not found offers to write the Answer again, from its Citation's card.
    <RegenerateAnswerContext.Provider value={regenerate}>
      <div ref={frame} className="assistant-answer-frame">
        <EditorContent editor={editor} />
        <MarginChecks editor={editor} />
      </div>
    </RegenerateAnswerContext.Provider>
  );
}
