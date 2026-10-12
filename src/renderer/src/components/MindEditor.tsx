import "katex/dist/katex.min.css";
import Collaboration from "@tiptap/extension-collaboration";
import { Focus, Placeholder } from "@tiptap/extensions";
import { Selection } from "@tiptap/pm/state";
import { EditorContent, ReactNodeViewRenderer, useEditor, useEditorState } from "@tiptap/react";
import { useContext, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import * as Y from "yjs";
import { ANSWER_BLOCK, MIND_CONTENT_FIELD, type MindAccess } from "../../../core/api";
import { citationStyle } from "../../../shared/citationStyles";
import { translate } from "../../../shared/i18n";
import { useCitationFormattingFor, useMindStyleChoice } from "../citationFormatting";
import { noteHandle, readMindHandle } from "../commands/handles";
import { shortcutLabel } from "../commands/keymap";
import { useComposer } from "../composer";
import { core } from "../core";
import { AnswerView } from "../editor/AnswerView";
import { BlockHandle } from "../editor/BlockHandle";
import { BlockCommands } from "../editor/blockCommands";
import { BlockOrigins } from "../editor/blockOrigins";
import { CitationStylePrompt } from "../editor/CitationStylePrompt";
import { CitationView } from "../editor/CitationView";
import { CodeBlockView } from "../editor/CodeBlockView";
import { Composer, type ComposerHandle } from "../editor/Composer";
import { ConversationLinks } from "../editor/ConversationLinks";
import { CitationClusters } from "../editor/citationClusters";
import { CitationNumbers, redrawFootnotes, setCitationForm } from "../editor/citationNumbers";
import { CopyCitations } from "../editor/copyCitations";
import { EndHint, type HintPlace, type HintText, isEmptyMind } from "../editor/EndHint";
import { selectionForEdit } from "../editor/editSelection";
import { FootnoteArea } from "../editor/FootnoteArea";
import { FootnotePopover } from "../editor/FootnotePopover";
import { type FootnoteLabel, footnoteView } from "../editor/FootnoteView";
import { FootnoteKeys } from "../editor/footnoteSchema";
import { MarginChecks } from "../editor/MarginChecks";
import { MathEditing, MathEditor } from "../editor/MathEditor";
import { ComposerDockContext, CountSlotContext, MindIdContext } from "../editor/mindContext";
import { NoteKeymap } from "../editor/noteKeymap";
import { noteExtensions } from "../editor/noteSchema";
import { ProposalBars } from "../editor/ProposalBars";
import { ProposalReview } from "../editor/proposalReview";
import { QuestionView } from "../editor/QuestionView";
import { askInEditor, QuestionCommands } from "../editor/questionCommands";
import { QuestionFold } from "../editor/questionFold";
import { ReferencesBlock } from "../editor/ReferencesBlock";
import { ScopePicker } from "../editor/ScopePicker";
import { type SelectionAction, SelectionMenu } from "../editor/SelectionMenu";
import { SlashMenu } from "../editor/SlashMenu";
import { noteSlashItems, skillSlashItems } from "../editor/slashItems";
import { Suggestions } from "../editor/suggestions";
import { TableMenu } from "../editor/TableMenu";
import { TableKeys } from "../editor/tableCommands";
import { SmartTypography } from "../editor/typography";
import { WordCount } from "../editor/WordCount";
import { requestEdit, startEditing, TRANSLATE_LANGUAGES } from "../editWithAi";
import { translateNow, useT } from "../i18n";
import { useMeetingOf } from "../meetings";
import { isWritingIn } from "../mindStatus";
import { skillDescription } from "../skills";
import { useAppStore } from "../store";
import { askForSuggestion, describeSuggestion } from "../suggestions";
import { StartGuide, useGettingStarted, useIsExample } from "./GettingStarted";
import { SparkLineIcon } from "./lineIcons";
import { MindNotice } from "./MindNotice";

/** Stands in for the hint's button in its translated words, which are split around it. */
const ASK = "\uE000";

/** Marks changes that came from the core, so they aren't sent back to it. */
const FROM_CORE = Symbol("from the core");

interface OpenedDocument {
  access: MindAccess;
  /** Null when the Mind isn't shown in the editor ("update-required"). */
  doc: Y.Doc | null;
}

/**
 * A local copy of the Mind's Yjs document, kept in step with the core's: it
 * starts from the core's full state, sends every local change to the core, and
 * applies the changes the core pushes. Null until the state has arrived.
 *
 * A Mind the core opens read-only (a newer version wrote it) is only a copy to
 * look at: the editor's binding deletes what its schema doesn't know from the
 * copy, and nothing is ever sent from it.
 */
function useMindDocument(mindId: string): OpenedDocument | null {
  const [opened, setOpened] = useState<OpenedDocument | null>(null);
  const reportError = useAppStore((state) => state.reportError);

  useEffect(() => {
    const local = new Y.Doc();
    let active = true;
    let editable = false;
    // Listen before opening, so nothing pushed in between is missed. Yjs merges in any order.
    const stopListening = core.on("mind.update", ({ mindId: changed, update }) => {
      if (changed === mindId) Y.applyUpdate(local, update, FROM_CORE);
    });
    const send = (update: Uint8Array, origin: unknown) => {
      if (origin !== FROM_CORE && editable) core.applyMindUpdate(mindId, update).catch(reportError);
    };
    local.on("update", send);
    core.openMind(mindId).then(
      ({ state, access }) => {
        if (!active) return;
        if (access === "update-required") {
          setOpened({ access, doc: null });
          return;
        }
        editable = access === "edit";
        Y.applyUpdate(local, state, FROM_CORE);
        setOpened({ access, doc: local });
      },
      (error: unknown) => {
        if (active) reportError(error);
      },
    );

    return () => {
      active = false;
      stopListening();
      local.off("update", send);
      setOpened(null);
      core.closeMind(mindId).catch(() => undefined);
    };
  }, [mindId, reportError]);

  return opened;
}

/** The open Mind's Blocks, edited with Tiptap and saved through the core as they change. */
export function MindEditor({ mindId }: { mindId: string }) {
  const opened = useMindDocument(mindId);
  if (!opened) return null;
  if (!opened.doc) return <MindNotice kind="update" />;
  return (
    <MindEditorView mindId={mindId} doc={opened.doc} readOnly={opened.access === "read-only"} />
  );
}

const editorPropsFor = (label: string, emptyFormula: string) => ({
  attributes: {
    "data-testid": "mind-editor",
    "aria-label": label,
    class: "mind-editor",
    // Shown in a formula that has no LaTeX yet (see styles.css).
    style: `--empty-formula: ${JSON.stringify(emptyFormula)}`,
  },
  // The cursor scrolled into view stays clear of the bar of changes at the Mind's top (ProposalBars).
  scrollMargin: { top: 48, right: 5, bottom: 5, left: 5 },
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
 * the drag handle, the formatting menu, the table menu, smart typography, the LaTeX field,
 * suggestions as the User writes (see `Suggestions`),
 * the margin column of Citation checks, and the composer, where Questions
 * are asked (drawn in the assistant, when it reads this Mind: `ComposerDockContext`).
 *
 * Every text starts at one edge: the column (styles.css, `.mind-column`) has
 * a left margin for the controls (the block handle, a Question's spark and
 * fold chevron) and a right margin for the checks.
 */
function MindEditorView({
  mindId,
  doc,
  readOnly,
}: {
  mindId: string;
  doc: Y.Doc;
  /** A newer version wrote the Mind: shown, never written to, and no composer. */
  readOnly: boolean;
}) {
  const t = useT();
  const isExample = useIsExample(mindId);
  const dock = useContext(ComposerDockContext);
  const countSlot = useContext(CountSlotContext);
  const composer = useRef<ComposerHandle>(null);
  /**
   * Whether the Mind's cursor is where Add to note puts an Answer (#201): since
   * the editor last had the focus, only the assistant (its composer, its
   * Answers, its menus) has had it. Otherwise an Answer goes at the end of the Mind.
   */
  const cursorInMind = useRef(false);
  const placeholder = useRef("");
  placeholder.current = t("editor.placeholder");
  // A Meeting's empty Mind waits for the agenda (as in Granola), with no Question to ask yet.
  const isMeeting = useMeetingOf(mindId) !== undefined;
  // The example Mind invites a Question of one's own at its end.
  const hintText = (place: HintPlace): HintText => {
    if (place === "empty" && isMeeting) {
      return { before: t("meetings.hint.agenda"), ask: "", after: "" };
    }
    const key =
      place === "empty"
        ? "editor.hint.empty"
        : isExample
          ? "editor.hint.example"
          : "editor.hint.afterAnswer";
    const [before = "", after = ""] = t(key, { ask: ASK }).split(ASK);
    return { before, ask: t(`${key}.ask`, { shortcut: shortcutLabel("composer.focus") }), after };
  };
  const label = t("mind.editor.label");
  const emptyFormula = t("editor.math.empty");
  /** How a footnote's marker is named, in the interface's language as it is now. */
  const footnoteLabel = useRef<FootnoteLabel>(() => "");
  footnoteLabel.current = (number, text) =>
    text
      ? t("editor.footnote.labelWithText", { number, text })
      : t("editor.footnote.label", { number });
  /** The position of the formula whose LaTeX is being edited. */
  const [editingMath, setEditingMath] = useState<number | null>(null);
  /** The Mind's citation style, followed as it changes (#289). */
  const styleChoice = useMindStyleChoice(doc);
  const { style } = styleChoice;
  const styleRef = useRef(style);
  styleRef.current = style;

  const editor = useEditor(
    {
      editable: !readOnly,
      editorProps: editorPropsFor(label, emptyFormula),
      shouldRerenderOnTransaction: false,
      extensions: [
        ...noteExtensions({
          onEditMath: setEditingMath,
          codeBlockView: ReactNodeViewRenderer(CodeBlockView),
          questionView: ReactNodeViewRenderer(QuestionView),
          answerView: ReactNodeViewRenderer(AnswerView),
          citationView,
          footnoteView: footnoteView(() => footnoteLabel.current),
        }),
        // Footnotes are numbered as the Mind's citation style counts its Citations (#289).
        CitationNumbers.configure({ form: () => citationStyle(styleRef.current).form }),
        CitationClusters,
        CopyCitations,
        FootnoteKeys,
        Collaboration.configure({ document: doc, field: MIND_CONTENT_FIELD }),
        // Notes added from an Answer, with their links back to it (#201).
        BlockOrigins.configure({ doc }),
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
        TableKeys,
        NoteKeymap,
        QuestionCommands.configure({
          onAsk: (current, questionId) => void askInEditor(current, mindId, questionId),
        }),
        QuestionFold.configure({ mindId }),
        ProposalReview.configure({ translate: translateNow }),
        Suggestions.configure({
          request: (request) => askForSuggestion(mindId, doc, request),
          cancel: (requestId) =>
            void core.cancelSuggestion(requestId).catch(useAppStore.getState().reportError),
          accepted: (suggestionId, characters) =>
            void core
              .acceptSuggestion({ suggestionId, characters })
              .catch(useAppStore.getState().reportError),
          busy: () => isWritingIn(mindId),
          describe: describeSuggestion,
          reportError: (error) => useAppStore.getState().reportError(error),
        }),
      ],
    },
    [doc, readOnly],
  );

  // The Citations and References in the Mind's citation style (#289), formatted off the
  // window's thread, and the footnotes numbered again when the style counts Citations otherwise.
  useCitationFormattingFor(editor, style, mindId);
  useEffect(() => {
    if (!editor.isDestroyed) setCitationForm(editor.view, citationStyle(style).form);
  }, [editor, style]);

  // The commands on the Mind's text (⌘K, ⌘↵ and ⌘⌫ on its changes) act on the Mind last written in.
  useEffect(() => {
    if (readOnly) return;
    const lend = () => {
      noteHandle.current = { editor, mindId, doc };
    };
    if (!noteHandle.current || noteHandle.current.editor.isDestroyed) lend();
    editor.on("focus", lend);
    return () => {
      editor.off("focus", lend);
      if (noteHandle.current?.editor === editor) noteHandle.current = null;
    };
  }, [editor, mindId, doc, readOnly]);

  // The Mind the assistant reads (it draws this Mind's composer): Add to note puts an Answer
  // here through this editor, and a second Esc in the assistant comes back here.
  const reads = dock !== null && !readOnly;
  useEffect(() => {
    if (!reads) return;
    const handle = {
      editor,
      mindId,
      doc,
      cursorInMind: () => cursorInMind.current,
      returnToNote() {
        if (editor.isDestroyed) return;
        // Where the cursor was, or at the end, at once: what is typed next goes there.
        // (Tiptap's `focus` waits a frame, and the first keys would go elsewhere.)
        if (!cursorInMind.current) {
          editor.commands.command(({ tr }) => {
            tr.setSelection(Selection.atEnd(tr.doc));
            return true;
          });
        }
        editor.view.focus();
        editor.view.dispatch(editor.state.tr.scrollIntoView());
      },
    };
    readMindHandle.current = handle;
    return () => {
      if (readMindHandle.current === handle) readMindHandle.current = null;
    };
  }, [reads, editor, mindId, doc]);

  // The selection menu's AI section: each comes back as changes to accept or reject.
  const aiActions = useMemo((): SelectionAction[] => {
    const available = (current: typeof editor) => selectionForEdit(current.state) !== null;
    const run = (action: "improve" | "shorten" | "explain") => (current: typeof editor) =>
      void requestEdit(current, mindId, doc, action);
    return [
      { id: "improve", label: "editor.ai.improve", icon: "✎", available, run: run("improve") },
      { id: "shorten", label: "editor.ai.shorten", icon: "≡", available, run: run("shorten") },
      {
        id: "translate",
        label: "editor.ai.translate",
        icon: "文",
        available,
        choices: TRANSLATE_LANGUAGES.map((language) => ({
          id: language.id,
          label: language.name,
          run: (current) => void requestEdit(current, mindId, doc, "translate", language.prompt),
        })),
        run: () => undefined,
      },
      { id: "explain", label: "editor.ai.explain", icon: "?", available, run: run("explain") },
      {
        id: "edit",
        label: "editor.ai.edit",
        icon: <SparkLineIcon className="size-3.5" />,
        keys: shortcutLabel("composer.focus"),
        available,
        run: (current) => {
          if (startEditing(current, mindId)) useComposer.getState().requestFocus();
        },
      },
    ];
  }, [mindId, doc]);

  // When the language changes: relabel. Setting the props also redraws the placeholder,
  // and the footnotes' markers are named anew.
  useEffect(() => {
    editor.setOptions({ editorProps: editorPropsFor(label, emptyFormula) });
    if (!editor.isDestroyed) redrawFootnotes(editor.view);
  }, [editor, label, emptyFormula]);

  // The cursor is where Add to note puts an Answer while the focus is in the editor, or has gone
  // from it only to the assistant the composer is in (its Answers, its menus, its header);
  // anywhere else, Answers go at the end of the Mind. A dialog that asks something on the way
  // (consent to send, say) gives the focus back: it doesn't count.
  useEffect(() => {
    const assistant = dock?.closest("[data-assistant]") ?? dock;
    const onFocusIn = (event: FocusEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) return;
      if (editor.view.dom.contains(target)) cursorInMind.current = true;
      else if (
        !assistant?.contains(target) &&
        !(target instanceof Element && target.closest("dialog"))
      ) {
        cursorInMind.current = false;
      }
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
  const guide = useGettingStarted() !== null && empty && !isExample && !readOnly;

  return (
    <MindIdContext.Provider value={mindId}>
      {countSlot && createPortal(<WordCount editor={editor} doc={doc} />, countSlot)}
      {readOnly && <MindNotice kind="read-only" />}
      <div className={`mind-editor-frame ${guide ? "mind-editor-frame--guide" : ""}`}>
        {!readOnly && <ProposalBars editor={editor} mindId={mindId} />}
        <EditorContent editor={editor} />
        {!readOnly && <EndHint editor={editor} text={hintText} onAsk={askOnLine} />}
        <MarginChecks editor={editor} />
        <ConversationLinks editor={editor} />
        {!readOnly && <TableMenu editor={editor} />}
        <FootnoteArea editor={editor} citationView={citationView} />
        {(style !== "numbered" || styleChoice.bibliography) && (
          <ReferencesBlock mindId={mindId} editor={editor} readOnly={readOnly} />
        )}
        {!readOnly && !styleChoice.chosen && !styleChoice.promptDismissed && (
          <CitationStylePrompt mindId={mindId} editor={editor} doc={doc} />
        )}
      </div>
      {dock &&
        !readOnly &&
        createPortal(<Composer ref={composer} editor={editor} doc={doc} mindId={mindId} />, dock)}
      {guide && <StartGuide />}
      {!readOnly && <BlockHandle editor={editor} />}
      {!readOnly && <SelectionMenu editor={editor} ai={aiActions} />}
      <FootnotePopover editor={editor} citationView={citationView} />
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
