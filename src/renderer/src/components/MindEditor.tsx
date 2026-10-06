import Collaboration from "@tiptap/extension-collaboration";
import { Placeholder } from "@tiptap/extensions";
import { EditorContent, useEditor } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import { useEffect, useRef, useState } from "react";
import * as Y from "yjs";
import { MIND_CONTENT_FIELD } from "../../../core/api";
import { core } from "../core";
import { useT } from "../i18n";
import { useAppStore } from "../store";

/** Marks changes that came from the core, so they aren't sent back to it. */
const FROM_CORE = Symbol("from the core");

/**
 * A local copy of the Mind's Yjs document, kept in step with the core's: it
 * starts from the core's full state, sends every local change to the core, and
 * applies the changes the core pushes. Null until the state has arrived.
 */
function useMindDocument(mindId: string): Y.Doc | null {
  const [doc, setDoc] = useState<Y.Doc | null>(null);
  const reportError = useAppStore((state) => state.reportError);

  useEffect(() => {
    const local = new Y.Doc();
    let active = true;
    // Listen before opening, so nothing pushed in between is missed. Yjs merges in any order.
    const stopListening = core.on("mind.update", ({ mindId: changed, update }) => {
      if (changed === mindId) Y.applyUpdate(local, update, FROM_CORE);
    });
    const send = (update: Uint8Array, origin: unknown) => {
      if (origin !== FROM_CORE) core.applyMindUpdate(mindId, update).catch(reportError);
    };
    local.on("update", send);
    core.openMind(mindId).then(
      ({ state }) => {
        if (!active) return;
        Y.applyUpdate(local, state, FROM_CORE);
        setDoc(local);
      },
      (error: unknown) => {
        if (active) reportError(error);
      },
    );

    return () => {
      active = false;
      stopListening();
      local.off("update", send);
      setDoc(null);
      core.closeMind(mindId).catch(() => undefined);
    };
  }, [mindId, reportError]);

  return doc;
}

/** The open Mind's Blocks, edited with Tiptap and saved through the core as they change. */
export function MindEditor({ mindId }: { mindId: string }) {
  const doc = useMindDocument(mindId);
  return doc ? <MindEditorView doc={doc} /> : null;
}

const editorPropsFor = (label: string) => ({
  attributes: { "data-testid": "mind-editor", "aria-label": label, class: "mind-editor" },
});

function MindEditorView({ doc }: { doc: Y.Doc }) {
  const t = useT();
  const placeholder = useRef("");
  placeholder.current = t("mind.editor.placeholder");
  const label = t("mind.editor.label");

  const editor = useEditor(
    {
      editorProps: editorPropsFor(label),
      shouldRerenderOnTransaction: false,
      extensions: [
        StarterKit.configure({
          // Collaboration brings its own undo, which only undoes this client's changes.
          undoRedo: false,
          // It would add a paragraph to Minds that end in another Block just by opening them.
          trailingNode: false,
        }),
        Collaboration.configure({ document: doc, field: MIND_CONTENT_FIELD }),
        Placeholder.configure({ placeholder: () => placeholder.current }),
      ],
    },
    [doc],
  );

  // When the language changes: relabel. Setting the props also redraws the placeholder.
  useEffect(() => {
    editor.setOptions({ editorProps: editorPropsFor(label) });
  }, [editor, label]);

  return <EditorContent editor={editor} />;
}
