import type { Editor } from "@tiptap/core";
import { useEffect } from "react";
import type * as Y from "yjs";

/** Something a command acts on that lives in a component, there while the component is. */
export interface Handle<T> {
  current: T | null;
}

/** The window's one assistant (#253): what only it can do, such as open its History. */
export const assistantHandle: Handle<{ openHistory(): void }> = { current: null };

/** The Citation whose card is open: what "Open cited page" opens. */
export const citationHandle: Handle<{ open(): void }> = { current: null };

export const sidebarHandle: Handle<{ hidden: boolean; toggle(): void }> = { current: null };

/** The Mind the User last wrote in: its editor and its document, for the commands on its text. */
export const noteHandle: Handle<{ editor: Editor; mindId: string; doc: Y.Doc }> = {
  current: null,
};

/**
 * The Mind the assistant reads (#253), open in its editor: where Add to note
 * puts an Answer, at its cursor (`cursorInMind`) or at its end (#201), and
 * where a second Esc in the assistant goes back to (`returnToNote`).
 */
export const readMindHandle: Handle<{
  editor: Editor;
  mindId: string;
  doc: Y.Doc;
  /** Whether the Mind's cursor is where an Answer goes: else, at the Mind's end. */
  cursorInMind(): boolean;
  /** Puts the focus back in the note, where its cursor was, or at its end. */
  returnToNote(): void;
}> = { current: null };

/** Lends `value` to commands through `handle` while the calling component is mounted. */
export function useHandle<T>(handle: Handle<T>, value: T): void {
  useEffect(() => {
    handle.current = value;
    return () => {
      if (handle.current === value) handle.current = null;
    };
  }, [handle, value]);
}

/** The Transcript in view (#139): the line the User is on, which "Add transcript line to note" adds. */
export const transcriptHandle: Handle<{ hasLine(): boolean; addLine(): Promise<void> }> = {
  current: null,
};
