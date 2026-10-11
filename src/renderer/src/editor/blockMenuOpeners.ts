import type { Editor } from "@tiptap/react";

/** Opens a Block's menu by its ID, for the keyboard: each Mind editor's handle lends it while it is mounted. */
export const blockMenuOpeners = new WeakMap<Editor, (blockId: string) => void>();
