import type { Editor } from "@tiptap/core";
import { useEffect } from "react";
import type * as Y from "yjs";

/** Something a command acts on that lives in a component, there while the component is. */
export interface Handle<T> {
  current: T | null;
}

export const composerHandle: Handle<{ focus(): void }> = { current: null };

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
