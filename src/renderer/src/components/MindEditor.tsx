import "katex/dist/katex.min.css";
import Collaboration from "@tiptap/extension-collaboration";
import { Focus, Placeholder } from "@tiptap/extensions";
import { EditorContent, ReactNodeViewRenderer, useEditor, useEditorState } from "@tiptap/react";
import { useContext, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import * as Y from "yjs";
import { ANSWER_BLOCK, MIND_CONTENT_FIELD } from "../../../core/api";
import { translate } from "../../../shared/i18n";
import { useComposer } from "../composer";
import { core } from "../core";
import { AnswerView } from "../editor/AnswerView";
import { BlockHandle } from "../editor/BlockHandle";
import { BlockCommands } from "../editor/blockCommands";
import { CitationView } from "../editor/CitationView";
import { CodeBlockView } from "../editor/CodeBlockView";
import { Composer, type ComposerHandle } from "../editor/Composer";
import { CitationNumbers } from "../editor/citationNumbers";
import { EndHint, type HintPlace, type HintText, isEmptyMind } from "../editor/EndHint";
import { FormatMenu } from "../editor/FormatMenu";
import { MarginChecks } from "../editor/MarginChecks";
import { MathEditing, MathEditor } from "../editor/MathEditor";
import { ComposerDockContext, MindIdContext } from "../editor/mindContext";
import { noteExtensions } from "../editor/noteSchema";
import { QuestionView } from "../editor/QuestionView";
import { askInEditor, QUESTION_SHORTCUT_LABEL, QuestionCommands } from "../editor/questionCommands";
import { QuestionFold } from "../editor/questionFold";
import { ScopePicker } from "../editor/ScopePicker";
import { SlashMenu } from "../editor/SlashMenu";
import { noteSlashItems, skillSlashItems } from "../editor/slashItems";
import { SmartTypography } from "../editor/typography";
import { useT } from "../i18n";
import { skillDescription } from "../skills";
import { useAppStore } from "../store";
import { StartGuide, useGettingStarted, useIsExample } from "./GettingStarted";

/** Stands in for the hint's button in its translated words, which are split around it. */
const ASK = "\uE000";

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
  return doc ? <MindEditorView mindId={mindId} doc={doc} /> : null;
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
 * A Citation redraws when it changes, and when an edit renumbers it (its
 * number comes in a decoration, see `CitationNumbers`), not on every edit.
 */
const citationView = ReactNodeViewRenderer(CitationView, {
  as: "span",
  update: ({ oldNode, newNode, oldDecorations, newDecorations, updateProps }) => {
    if (newNode.type !== oldNode.type) return false;
    if (newNode !== oldNode || oldDecorations !== newDecorations) updateProps();
    return true;
  },
});

/**
 * Notes (see `noteExtensions`), Questions and Answers in the Mind's Yjs
 * document, with the slash menu, the "@" picker of a Question's Search scope,
 * the drag handle, the formatting menu, smart typography, the LaTeX field,
 * the margin column of Citation checks, and the composer at the column's
 * foot, where Questions are asked (drawn in the dock, `ComposerDockContext`).
 *
 * Every text starts at one edge: the column (styles.css, `.mind-column`) has
 * a left margin for the controls (the block handle, a Question's spark and
 * fold chevron) and a right margin for the checks.
 */
function MindEditorView({ mindId, doc }: { mindId: string; doc: Y.Doc }) {
  const t = useT();
  const isExample = useIsExample(mindId);
  const dock = useContext(ComposerDockContext);
  const composer = useRef<ComposerHandle>(null);
  /**
   * Whether the Mind's cursor is where the next Question goes: since the
   * editor last had the focus, only the composer (and its menus) has had it.
   * Otherwise a Question goes at the end of the Mind.
   */
  const cursorInMind = useRef(false);
  const placeholder = useRef("");
  placeholder.current = t("editor.placeholder");
  // The example Mind invites a Question of one's own at its end.
  const hintText = (place: HintPlace): HintText => {
    const key =
      place === "empty"
        ? "editor.hint.empty"
        : isExample
          ? "editor.hint.example"
          : "editor.hint.afterAnswer";
    const [before = "", after = ""] = t(key, { ask: ASK }).split(ASK);
    return { before, ask: t(`${key}.ask`, { shortcut: QUESTION_SHORTCUT_LABEL }), after };
  };
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
          questionView: ReactNodeViewRenderer(QuestionView),
          answerView: ReactNodeViewRenderer(AnswerView),
          citationView,
        }),
        CitationNumbers,
        Collaboration.configure({ document: doc, field: MIND_CONTENT_FIELD }),
        Placeholder.configure({
          // An empty Mind's line and the line after an Answer have their hint drawn over them (EndHint).
          placeholder: ({ editor: current, pos }) => {
            if (isEmptyMind(current.state.doc)) return "";
            const before = current.state.doc.resolve(pos).nodeBefore;
            return before?.type.name === ANSWER_BLOCK ? "" : placeholder.current;
          },
        }),
        Focus.configure({ className: "has-focus", mode: "shallowest" }),
        SmartTypography,
        SlashMenu.configure({
          items: (place) => {
            const { skills: all, settings } = useAppStore.getState();
            const language = settings?.language ?? "en";
            const skills = skillSlashItems(
              all,
              (skill) => skillDescription(skill, (key, params) => translate(language, key, params)),
              (skill) => {
                // For the Question asked next, from the composer, which goes on this line.
                useComposer.getState().update(mindId, { skill });
                useComposer.getState().requestFocus();
              },
            );
            return place === "question" ? skills : [...noteSlashItems, ...skills];
          },
        }),
        ScopePicker,
        MathEditing.configure({ onEdit: setEditingMath }),
        BlockCommands,
        QuestionCommands.configure({
          onAsk: (current, questionId) => void askInEditor(current, mindId, questionId),
        }),
        QuestionFold.configure({ mindId }),
      ],
    },
    [doc],
  );

  // When the language changes: relabel. Setting the props also redraws the placeholder.
  useEffect(() => {
    editor.setOptions({ editorProps: editorPropsFor(label, emptyFormula) });
  }, [editor, label, emptyFormula]);

  // The cursor is where the next Question goes while the focus is in the editor, or has gone
  // from it only to the composer; anywhere else, Questions go at the end of the Mind.
  useEffect(() => {
    const onFocusIn = (event: FocusEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) return;
      if (editor.view.dom.contains(target)) cursorInMind.current = true;
      else if (!dock?.contains(target)) cursorInMind.current = false;
    };
    document.addEventListener("focusin", onFocusIn);
    return () => document.removeEventListener("focusin", onFocusIn);
  }, [editor, dock]);

  // A Question asked for from outside the editor, e.g. "Ask your own Question" in Get
  // started, or "Ask about this Folder" in the Library, which gives its Search scope:
  // the composer takes the focus, the Question to go at the end of the Mind.
  const questionHere = useAppStore((state) => state.questionToStart?.mindId === mindId);
  const composerShown = dock !== null;
  useEffect(() => {
    if (!questionHere || !composerShown) return;
    composer.current?.focus(useAppStore.getState().questionToStart?.scope ?? null);
    useAppStore.getState().questionStarted();
  }, [questionHere, composerShown]);

  /** The hint's "press ⌘J…": the composer takes the focus, the Question to go on the hint's line. */
  const askOnLine = (pos: number) => {
    if (editor.isDestroyed) return;
    editor.commands.setTextSelection(pos + 1);
    cursorInMind.current = true;
    useComposer.getState().requestFocus();
  };

  // An empty Mind of the User's own shows the three steps while Get started is shown.
  const empty = useEditorState({
    editor,
    selector: ({ editor: current }) => isEmptyMind(current.state.doc),
  });
  const guide = useGettingStarted() !== null && empty && !isExample;

  return (
    <MindIdContext.Provider value={mindId}>
      <div className={`mind-editor-frame ${guide ? "mind-editor-frame--guide" : ""}`}>
        <EditorContent editor={editor} />
        <EndHint editor={editor} text={hintText} onAsk={askOnLine} />
        <MarginChecks editor={editor} />
      </div>
      {dock &&
        createPortal(
          <Composer
            ref={composer}
            editor={editor}
            doc={doc}
            mindId={mindId}
            cursorInMind={cursorInMind}
          />,
          dock,
        )}
      {guide && <StartGuide />}
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
    </MindIdContext.Provider>
  );
}
