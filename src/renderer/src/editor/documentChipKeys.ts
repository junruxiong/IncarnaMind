import { Extension } from "@tiptap/core";
import { NodeSelection } from "@tiptap/pm/state";
import { DOCUMENT_CHIP_NODE } from "../../../core/api";
import { useAppStore } from "../store";

/** Opens the Document a chip names, if it is still there: whether it was. */
export function openChipDocument(documentId: string): boolean {
  const { documents, openDocument } = useAppStore.getState();
  if (!documents.some((document) => document.id === documentId)) return false;
  openDocument({ documentId });
  return true;
}

/**
 * A chip by keyboard: the arrows select it, as one thing, and Enter opens its
 * Document, rather than splitting the line where it is.
 */
export const DocumentChipKeys = Extension.create({
  name: "documentChipKeys",
  // Before Enter splits the Block.
  priority: 1000,

  addKeyboardShortcuts() {
    return {
      Enter: ({ editor }) => {
        const { selection } = editor.state;
        if (!(selection instanceof NodeSelection)) return false;
        if (selection.node.type.name !== DOCUMENT_CHIP_NODE) return false;
        openChipDocument(String(selection.node.attrs.documentId));
        return true;
      },
    };
  },
});
