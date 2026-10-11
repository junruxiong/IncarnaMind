import type { Editor } from "@tiptap/react";

export const blockMenuOpeners = new WeakMap<Editor, (blockId: string) => void>();
