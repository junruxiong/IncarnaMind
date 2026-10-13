import type { Editor } from "@tiptap/core";
import { useEditorState } from "@tiptap/react";
import { useEffect, useState } from "react";
import type { MindFocus } from "../../../core/api";
import { holdSelection } from "./heldSelection";
import { mindFocusOf, selectedText } from "./mindFocus";

/** Where the selection is, to tell a new one from the one the User left out. */
interface Selected {
  text: string;
  at: string;
}

const NOTHING: Selected = { text: "", at: "" };

/**
 * The selection in the Mind the assistant reads, as the composer asks about it
 * (#565): the quote its chip shows while there is one, kept drawn in the Mind
 * meanwhile, and what the Question is told of it. The User leaves it out with
 * the chip's ×; selecting something else brings it back.
 */
export function useMindSelection(
  mind: { editor: Editor; mindId: string } | null,
  /** The composer is saying what to change in the selection: it has its own quote. */
  editing: boolean,
) {
  const editor = mind?.editor ?? null;
  const selected =
    useEditorState({
      editor,
      selector: ({ editor: current }): Selected => {
        if (!current || current.isDestroyed) return NOTHING;
        const { from, to } = current.state.selection;
        return { text: selectedText(current.state), at: `${from}:${to}` };
      },
    }) ?? NOTHING;
  const [left, setLeft] = useState<{ mindId: string; at: string } | null>(null);
  const leftOut = left !== null && left.mindId === mind?.mindId && left.at === selected.at;
  const quote = editing || leftOut ? "" : selected.text;

  // Chosen again later, a passage the User left out is asked about again.
  useEffect(() => {
    if (selected.text === "") setLeft(null);
  }, [selected.text]);

  useEffect(() => {
    if (!editor || quote === "") return;
    holdSelection(editor, true);
    return () => holdSelection(editor, false);
  }, [editor, quote]);

  return {
    /** What the chip quotes; "" when there is no chip. */
    quote,
    leaveOut() {
      if (mind) setLeft({ mindId: mind.mindId, at: selected.at });
    },
    /** The Mind scrolls to the passage and selects it again. */
    show() {
      editor?.commands.focus(null, { scrollIntoView: true });
    },
    /** Where the User is in the Mind, with the selection only while it is asked about. */
    focus(): MindFocus | null {
      if (!editor || editor.isDestroyed) return null;
      const focus = mindFocusOf(editor.state);
      return quote === "" ? { ...focus, selection: "" } : focus;
    },
    /** A Question asked about the passage is the end of it: the selection goes, the cursor stays. */
    asked() {
      if (!editor || editor.isDestroyed || quote === "") return;
      if (selectedText(editor.state) !== quote) return;
      editor.commands.setTextSelection(editor.state.selection.to);
    },
  };
}
