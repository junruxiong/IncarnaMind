import "katex/dist/katex.min.css";
import Collaboration from "@tiptap/extension-collaboration";
import { Focus, Placeholder } from "@tiptap/extensions";
import { EditorContent, ReactNodeViewRenderer, useEditor } from "@tiptap/react";
import { useEffect, useRef, useState } from "react";
import * as Y from "yjs";
import { MIND_CONTENT_FIELD } from "../../../core/api";
import { core } from "../core";
import { BlockHandle } from "../editor/BlockHandle";
import { BlockCommands } from "../editor/blockCommands";
import { CodeBlockView } from "../editor/CodeBlockView";
import { FormatMenu } from "../editor/FormatMenu";
import { MathEditing, MathEditor } from "../editor/MathEditor";
import { noteExtensions } from "../editor/noteSchema";
import { SlashMenu } from "../editor/SlashMenu";
import { noteSlashItems } from "../editor/slashItems";
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

const editorPropsFor = (label: string, emptyFormula: string) => ({
  attributes: {
    "data-testid": "mind-editor",
    "aria-label": label,
    class: "mind-editor",
    // Shown in a formula that has no LaTeX yet (see styles.css).
    style: `--empty-formula: ${JSON.stringify(emptyFormula)}`,
  },
});

/**
 * Notes (see `noteExtensions`) in the Mind's Yjs document, with the slash menu,
 * the drag handle, the formatting menu and the LaTeX field.
 */
function MindEditorView({ doc }: { doc: Y.Doc }) {
  const t = useT();
  const placeholders = useRef({ empty: "", line: "" });
  placeholders.current = { empty: t("mind.editor.placeholder"), line: t("editor.placeholder") };
  const label = t("mind.editor.label");
  const emptyFormula = t("editor.math.empty");
  /** The position of the formula whose LaTeX is being edited. */
  const [editingMath, setEditingMath] = useState<number | null>(null);

  const editor = useEditor(
    {
      editorProps: editorPropsFor(label, emptyFormula),
      shouldRerenderOnTransaction: false,
      extensions: [
        ...noteExtensions({
          onEditMath: setEditingMath,
          codeBlockView: ReactNodeViewRenderer(CodeBlockView),
        }),
        Collaboration.configure({ document: doc, field: MIND_CONTENT_FIELD }),
        Placeholder.configure({
          placeholder: ({ editor: current }) =>
            current.isEmpty ? placeholders.current.empty : placeholders.current.line,
        }),
        Focus.configure({ className: "has-focus", mode: "shallowest" }),
        SlashMenu.configure({ items: () => noteSlashItems }),
        MathEditing.configure({ onEdit: setEditingMath }),
        BlockCommands,
      ],
    },
    [doc],
  );

  // When the language changes: relabel. Setting the props also redraws the placeholder.
  useEffect(() => {
    editor.setOptions({ editorProps: editorPropsFor(label, emptyFormula) });
  }, [editor, label, emptyFormula]);

  return (
    <>
      <EditorContent editor={editor} className="relative" />
      <BlockHandle editor={editor} />
      <FormatMenu editor={editor} />
      {editingMath !== null && (
        <MathEditor
          key={editingMath}
          editor={editor}
          pos={editingMath}
          onClose={() => setEditingMath(null)}
        />
      )}
    </>
  );
}
