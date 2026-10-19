import type { VirtualElement } from "@floating-ui/dom";
import { type Editor, posToDOMRect } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { noteBusy, setMenuShows, useEdits } from "../editWithAi";
import { holdSelection } from "./heldSelection";
import { editTarget } from "./proposalReview";

export interface EditWithAiField {
  mindId: string;
  send(editor: Editor, instruction: string): void;
  /** Absent where there is no assistant to hand the request to. */
  handOff?(editor: Editor, instruction: string): void;
}

export type EditPhase = "idle" | "working" | "settled";

// The working line goes after the Block's end, so it never lies over the text being changed.
function endOfBlock(doc: ProseMirrorNode, pos: number): number {
  const $pos = doc.resolve(pos);
  for (let depth = $pos.depth; depth > 0; depth--) {
    if ($pos.node(depth).isTextblock) return $pos.end(depth);
  }
  return pos;
}

/**
 * What the selection menu's Edit with AI… field is doing: what is typed, the
 * request it sends, and where that stands. While the AI works the menu is one
 * line under the text, with the keys: Esc stops, Enter does nothing. When it is
 * done the menu goes, or, if it gave nothing and the User hasn't moved, comes
 * back over the selection with the field and the reason.
 */
export function useEditField(editor: Editor, edit: EditWithAiField | undefined, hide: () => void) {
  const mindId = edit?.mindId;
  const [phase, setPhase] = useState<EditPhase>("idle");
  const [text, setText] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const pill = useRef<HTMLDivElement>(null);
  const working = useRef(false);
  const sent = useRef<{ from: number; to: number } | null>(null);
  const pending = useEdits((state) => (mindId ? state.pending[mindId] : undefined));

  const asked = () => {
    // Set before the request collapses the selection, which would otherwise close the menu.
    working.current = true;
    setPhase("working");
  };

  const send = () => {
    const instruction = text.trim();
    if (!edit || instruction === "" || working.current) return;
    if (useEdits.getState().pending[edit.mindId]) {
      noteBusy(edit.mindId, true);
      setPhase("settled");
      return;
    }
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

  useLayoutEffect(() => {
    if (phase === "working") pill.current?.focus({ preventScroll: true });
  }, [phase]);

  useEffect(() => {
    if (phase === "settled") input.current?.focus({ preventScroll: true });
  }, [phase]);

  useEffect(() => {
    if (!mindId || phase === "idle") return;
    setMenuShows(mindId, true);
    return () => setMenuShows(mindId, false);
  }, [mindId, phase]);

  const reset = useCallback(() => {
    working.current = false;
    sent.current = null;
    setPhase("idle");
    setText("");
    holdSelection(editor, false, "edit-field");
  }, [editor]);

  const anchor = useCallback((): VirtualElement | null => {
    const target = working.current ? editTarget(editor.state) : null;
    if (!target) return null;
    const { doc } = editor.state;
    const clamp = (pos: number) => Math.min(pos, doc.content.size);
    return {
      getBoundingClientRect: () =>
        posToDOMRect(editor.view, clamp(target.from), endOfBlock(doc, clamp(target.to))),
    };
  }, [editor]);

  return {
    phase,
    text,
    setText,
    input,
    pill,
    working,
    send,
    asked,
    reset,
    anchor,
    focus: () => input.current?.focus({ preventScroll: true }),
  };
}
