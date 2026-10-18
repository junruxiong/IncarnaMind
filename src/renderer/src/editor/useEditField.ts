import type { VirtualElement } from "@floating-ui/dom";
import { type Editor, posToDOMRect } from "@tiptap/core";
import { useCallback, useEffect, useRef, useState } from "react";
import { useEdits } from "../editWithAi";
import { editTarget } from "./proposalReview";

/** What the Mind lends the selection menu for its Edit with AI… field. */
export interface EditWithAiField {
  mindId: string;
  /** Asks for the change as the assistant's composer does, applied in place. */
  send(editor: Editor, instruction: string): void;
  /** Moves the request and the selection to the composer; absent where there is no assistant. */
  handOff?(editor: Editor, instruction: string): void;
}

/** The field before a request, while the AI works on it, and after, when it is shown why it gave nothing. */
export type EditPhase = "idle" | "working" | "settled";

/**
 * The state of the selection menu's Edit with AI… field: what is typed, and
 * the request it sends. The cursor goes back to the text at once, so the User
 * can write on; the menu stays under the text as one line (`working`) until
 * the AI is done, then goes, or, if it gave nothing and the User hasn't moved,
 * comes back over the selection with the field and the reason.
 */
export function useEditField(editor: Editor, edit: EditWithAiField | undefined, hide: () => void) {
  const mindId = edit?.mindId;
  const [phase, setPhase] = useState<EditPhase>("idle");
  const [text, setText] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const working = useRef(false);
  const sent = useRef<{ from: number; to: number } | null>(null);
  const pending = useEdits((state) => (mindId ? state.pending[mindId] : undefined));

  const asked = () => {
    // Set before the request collapses the selection, which would otherwise close the menu.
    working.current = true;
    setPhase("working");
    editor.view.focus();
  };

  const send = () => {
    const instruction = text.trim();
    if (!edit || instruction === "" || working.current) return;
    if (useEdits.getState().pending[edit.mindId]) return;
    const { from, to } = editor.state.selection;
    asked();
    edit.send(editor, instruction);
    sent.current = { from, to };
  };

  useEffect(() => {
    if (phase !== "working" || pending) return;
    working.current = false;
    const kept = sent.current;
    const { selection } = editor.state;
    const problem = mindId ? useEdits.getState().problems[mindId] : undefined;
    // The request leaves the cursor at the end of the text; the User has moved it if it is elsewhere.
    if (problem && kept && !editor.isDestroyed && selection.empty && selection.from === kept.to) {
      const size = editor.state.doc.content.size;
      editor.commands.setTextSelection({
        from: Math.min(kept.from, size),
        to: Math.min(kept.to, size),
      });
      setPhase("settled");
      return;
    }
    hide();
  }, [phase, pending, editor, mindId, hide]);

  useEffect(() => {
    if (phase === "settled") input.current?.focus({ preventScroll: true });
  }, [phase]);

  const reset = useCallback(() => {
    working.current = false;
    sent.current = null;
    setPhase("idle");
    setText("");
  }, []);

  /** While the AI works the menu stays under the text it is working on, as it was. */
  const anchor = useCallback((): VirtualElement | null => {
    const target = working.current ? editTarget(editor.state) : null;
    if (!target) return null;
    const clamp = (pos: number) => Math.min(pos, editor.state.doc.content.size);
    return {
      getBoundingClientRect: () => posToDOMRect(editor.view, clamp(target.from), clamp(target.to)),
    };
  }, [editor]);

  return {
    phase,
    text,
    setText,
    input,
    working,
    send,
    asked,
    reset,
    anchor,
    focus: () => input.current?.focus({ preventScroll: true }),
  };
}
