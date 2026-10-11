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

/** The open Mind's find bar: what "Find next" and "Find previous" do. */
export const findHandle: Handle<{ next(): void; previous(): void }> = { current: null };

export const sidebarHandle: Handle<{ hidden: boolean; toggle(): void }> = { current: null };

/** The Mind the User last wrote in: its editor and its document, for the commands on its text. */
export const noteHandle: Handle<{ editor: Editor; mindId: string; doc: Y.Doc }> = {
  current: null,
};

/** Lends `value` to commands through `handle` while the calling component is mounted. */
export function useHandle<T>(handle: Handle<T>, value: T): void {
  useEffect(() => {
    handle.current = value;
    return () => {
      if (handle.current === value) handle.current = null;
    };
  }, [handle, value]);
}
