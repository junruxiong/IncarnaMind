import "katex/dist/katex.min.css";
import Collaboration from "@tiptap/extension-collaboration";
import { Focus, Placeholder } from "@tiptap/extensions";
import { Selection } from "@tiptap/pm/state";
import { EditorContent, ReactNodeViewRenderer, useEditor, useEditorState } from "@tiptap/react";
import { useContext, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type * as Y from "yjs";
import { ANSWER_BLOCK, MIND_CONTENT_FIELD } from "../../../core/api";
import { writingTime } from "../../../shared/blockTimes";
import { citationStyle } from "../../../shared/citationStyles";
import { translate } from "../../../shared/i18n";
import { checkSentences } from "../checks";
import { useCitationFormattingFor, useMindStyleChoice } from "../citationFormatting";
import { noteHandle, readMindHandle } from "../commands/handles";
import { shortcutLabel } from "../commands/keymap";
import { startCompareDocuments, useComparisons } from "../comparisons";
import { chosenScope, useComposer } from "../composer";
import { core } from "../core";
import { AnswerView } from "../editor/AnswerView";
import { holdsAiText } from "../editor/aiText";
import { BlockHandle } from "../editor/BlockHandle";
import { BlockCommands } from "../editor/blockCommands";
import { BlockOrigins } from "../editor/blockOrigins";
import { ChartView } from "../editor/ChartView";
import { CitationStylePrompt } from "../editor/CitationStylePrompt";
import { CitationView } from "../editor/CitationView";
import { CodeBlockView } from "../editor/CodeBlockView";
import { ComparisonTableView } from "../editor/ComparisonTableView";
import { ContradictionLine } from "../editor/ContradictionLine";
import { CitationClusters } from "../editor/citationClusters";
import { CitationNumbers, redrawFootnotes, setCitationForm } from "../editor/citationNumbers";
import { ClaimChecks, setClaimContext } from "../editor/claimChecks";
import { useClickBelowNote } from "../editor/clickBelowNote";
import { comparisonCellView, comparisonRowView } from "../editor/comparisonViews";
import { CopyCitations } from "../editor/copyCitations";
import { DocumentChipView } from "../editor/DocumentChipView";
import { DocumentChipKeys } from "../editor/documentChipKeys";
import { EndHint, type HintPlace, type HintText, isEmptyMind } from "../editor/EndHint";
import { selectionForEdit } from "../editor/editSelection";
import { FindBar } from "../editor/FindBar";
import { FootnoteArea } from "../editor/FootnoteArea";
import { FootnotePopover } from "../editor/FootnotePopover";
import { type FootnoteLabel, footnoteView } from "../editor/FootnoteView";
import { FindInMind } from "../editor/findInMind";
import { FootnoteKeys } from "../editor/footnoteSchema";
import { HeldSelection } from "../editor/heldSelection";
import { LinkBubble } from "../editor/LinkBubble";
import { MarginChecks } from "../editor/MarginChecks";
import { MathEditing, MathEditor } from "../editor/MathEditor";
import { MindCheckAsk } from "../editor/MindCheckLine";
import { MindOutline } from "../editor/MindOutline";
import { MeetingNotes } from "../editor/meetingNotes";
import {
  CompareSheetSlotContext,
  ComposerDockContext,
  CountSlotContext,
  MindIdContext,
} from "../editor/mindContext";
import { NoteHealth } from "../editor/NoteHealth";
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
import { compareSlashItem, noteSlashItems, skillSlashItems } from "../editor/slashItems";
import { Suggestions } from "../editor/suggestions";
import { TableMenu } from "../editor/TableMenu";
import { TableKeys } from "../editor/tableCommands";
import { SmartTypography } from "../editor/typography";
import { useVerification, VerifyLine } from "../editor/VerifyLine";
import { wordAtomView, wordBlockView } from "../editor/WordAtomView";
import { WordCount } from "../editor/WordCount";
import { OriginalNotAdded, useWordFile, WordOrigin } from "../editor/WordOrigin";
import { useWordLookStyle } from "../editor/wordLookStyle";
import { setWritingPlace, WritingPlace } from "../editor/writingPlace";
import { requestEdit, startEditing, TRANSLATE_LANGUAGES } from "../editWithAi";
import { translateNow, useT } from "../i18n";
import { markSelectionAsMine } from "../markAsMine";
import { useMeetingOf, useMeetings } from "../meetings";
import { type OpenedCopy, openMindCopy } from "../mindCopies";
import { isWritingIn } from "../mindStatus";
import { useMindTools } from "../mindTools";
import { onMindWriting, writingPlaceOf } from "../mindWriting";
import { skillDescription } from "../skills";
import { useAppStore } from "../store";
import { askForSuggestion, describeSuggestion } from "../suggestions";
import { CompareSheet } from "./CompareSheet";
import { StartGuide, useGettingStarted, useIsExample } from "./GettingStarted";
import { CheckLineIcon, MineLineIcon, SparkLineIcon } from "./lineIcons";
import { MindNotice } from "./MindNotice";

/** Stands in for the hint's button in its translated words, which are split around it. */
const ASK = "\uE000";

/**
 * A local copy of the Mind's Yjs document, kept in step with the core's
 * (../mindCopies). Null until the state has arrived.
 *
 * A Mind the core opens read-only (a newer version wrote it) is only a copy to
 * look at: the editor's binding deletes what its schema doesn't know from the
 * copy, and nothing is ever sent from it.
 */
function useMindDocument(mindId: string): OpenedCopy | null {
  const [opened, setOpened] = useState<OpenedCopy | null>(null);
  const reportError = useAppStore((state) => state.reportError);

  useEffect(() => {
    const close = openMindCopy(mindId, setOpened, reportError);
    return () => {
      close();
      setOpened(null);
    };
  }, [mindId, reportError]);

  return opened;
}

const MIND_EXPLAINED_KEY = "incarnamind.mindExplained";

/** Whether the sentence saying what a Mind is has been shown and written past (#721). */
function mindExplained(): boolean {
  try {
    return localStorage.getItem(MIND_EXPLAINED_KEY) === "1";
  } catch {
    return true;
  }
}

function rememberMindExplained(): void {
  try {
    localStorage.setItem(MIND_EXPLAINED_KEY, "1");
  } catch {
    // Without storage the sentence shows again next time, which does no harm.
  }
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
 * A chart takes its own pointer and keys (its menus, its values' focus and
 * cards), so the editor doesn't select or type over it; a press on its edge
 * still selects it as a Block, and ⌘Z / ⌘⇧Z reach the editor's undo from it.
 */
const chartView = ReactNodeViewRenderer(ChartView, {
  stopEvent: ({ event }) => {
    const target = event.target;
    if (!(target instanceof Element) || !target.closest("[data-chart-ui]")) return false;
    if (event.type.startsWith("drag") || event.type === "drop") return false;
    if (event.type === "copy" || event.type === "cut" || event.type === "paste") return false;
    const undo =
      event instanceof KeyboardEvent &&
      (event.metaKey || event.ctrlKey) &&
      /^[zy]$/i.test(event.key) &&
      !(target instanceof HTMLInputElement);
    return !undo;
  },
});

/**
 * A comparison table (#98): its title line, heading row and "+ Question" take
 * their own pointer and keys, so the editor doesn't select or type over them;
 * its cells are content the editor draws, never typed in.
 */
const comparisonViews = {
  table: ReactNodeViewRenderer(ComparisonTableView, {
    stopEvent: ({ event }) => {
      const target = event.target;
      if (!(target instanceof Element) || !target.closest("[data-comparison-ui]")) return false;
      if (event.type.startsWith("drag") || event.type === "drop") return false;
      const undo =
        event instanceof KeyboardEvent &&
        (event.metaKey || event.ctrlKey) &&
        /^[zy]$/i.test(event.key) &&
        !(target instanceof HTMLInputElement);
      return !undo;
    },
  }),
  row: comparisonRowView,
  cell: comparisonCellView,
};

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
  const sheetSlot = useContext(CompareSheetSlotContext);
  const comparing = useComparisons((state) => state.sheets[mindId] !== undefined);
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
  const [explained, setExplained] = useState(mindExplained);
  const hintText = (place: HintPlace): HintText => {
    if (place === "empty" && isMeeting) {
      return { before: t("meetings.hint.agenda"), ask: "", after: "" };
    }
    const key =
      place === "empty"
        ? explained
          ? "editor.hint.empty"
          : "editor.hint.first"
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
  /** A Mind still filling in from a Word file can't be edited until it has (#104). */
  const importing = useWordFile(doc).progress !== null;
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
          documentChipView: ReactNodeViewRenderer(DocumentChipView, { as: "span" }),
          comparisonViews,
          chartView,
          wordAtomView,
          wordBlockView,
        }),
        // Footnotes are numbered as the Mind's citation style counts its Citations (#289).
        CitationNumbers.configure({ form: () => citationStyle(styleRef.current).form }),
        CitationClusters,
        ClaimChecks,
        CopyCitations,
        FootnoteKeys,
        DocumentChipKeys,
        MeetingNotes.configure({
          mindId,
          doc,
          time: () =>
            writingTime(
              useMeetings.getState().meetings.find((each) => each.mindId === mindId),
              Date.now(),
            ),
        }),
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
            const compare = compareSlashItem(
              (afterBlockId) => void startCompareDocuments(mindId, afterBlockId),
            );
            return place === "question" ? skills : [...noteSlashItems, compare, ...skills];
          },
        }),
        ScopePicker,
        MathEditing.configure({ onEdit: setEditingMath }),
        BlockCommands,
        FindInMind,
        HeldSelection,
        TableKeys,
        NoteKeymap,
        QuestionCommands.configure({
          onAsk: (current, questionId) => void askInEditor(current, mindId, questionId),
        }),
        QuestionFold.configure({ mindId }),
        ProposalReview.configure({ translate: translateNow }),
        WritingPlace.configure({ label: () => translateNow("mind.writing") }),
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

  useEffect(() => {
    if (!editor.isDestroyed) editor.setEditable(!readOnly && !importing);
  }, [editor, readOnly, importing]);

  // A sentence's check needs a re-check once a Document it read changes, or the Mind moves (#96).
  const documents = useAppStore((state) =>
    state.status.kind === "ready" ? state.documents : null,
  );
  const folderId = useAppStore(
    (state) => state.minds.find((each) => each.id === mindId)?.folderId ?? null,
  );
  const versions = useMemo(
    () => (documents ? new Map(documents.map((each) => [each.id, each.contentHash])) : null),
    [documents],
  );
  useEffect(() => {
    if (!editor.isDestroyed) setClaimContext(editor.view, { versions, folderId });
  }, [editor, versions, folderId]);

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

  // The selection menu's AI section: each comes back as changes to accept or reject; Mark as mine acts at once.
  const hasDocuments = useAppStore((state) => state.documents.length > 0);
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
        id: "check-against-documents",
        label: hasDocuments ? "editor.ai.checkAgainstDocuments" : "editor.ai.addDocumentsToCheck",
        icon: <CheckLineIcon className="size-3.5" />,
        available,
        run: (current) =>
          hasDocuments
            ? void checkSentences(current, mindId, doc, "check-against-documents")
            : void useAppStore.getState().pickDocuments(),
      },
      {
        id: "mark-as-mine",
        label: "editor.ai.markAsMine",
        icon: <MineLineIcon className="size-3.5" />,
        available: (current) => holdsAiText(current.state),
        run: (current) => void markSelectionAsMine(current),
      },
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
  }, [mindId, doc, hasDocuments]);

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

  // The assistant's one composer asks in this Mind while it reads it (`Assistant`).
  const asksHere = dock !== null && !readOnly;
  useEffect(() => {
    if (!asksHere) return;
    useComposer.setState({ seat: { editor, doc, mindId } });
    return () => {
      if (useComposer.getState().seat?.mindId === mindId) useComposer.setState({ seat: null });
    };
  }, [asksHere, editor, doc, mindId]);

  // The assistant writing changes into this Mind as they come (#202): "Writing…" where the next goes.
  useEffect(() => {
    setWritingPlace(editor, writingPlaceOf(mindId));
    return onMindWriting(({ mindId: changed, place }) => {
      if (changed === mindId) setWritingPlace(editor, place);
    });
  }, [editor, mindId]);

  // A Mind just made with its title given ("Create a new note called …", #202): its text has the cursor.
  const bodyHere = useAppStore((state) => state.bodyToFocus === mindId);
  useEffect(() => {
    if (!bodyHere || editor.isDestroyed) return;
    editor.commands.focus("start");
    useAppStore.getState().bodyFocused();
  }, [bodyHere, editor]);

  // A Question asked for from outside the editor, e.g. "Ask your own Question" in Get
  // started, or "Ask about this Folder" in the Library, which gives its Search scope:
  // the composer takes the focus, the Question to go at the end of the Mind.
  const questionHere = useAppStore((state) => state.questionToStart?.mindId === mindId);
  const composerShown = dock !== null;
  useEffect(() => {
    if (!questionHere || !composerShown) return;
    const scope = useAppStore.getState().questionToStart?.scope;
    if (scope) {
      const folderId =
        useAppStore.getState().minds.find((each) => each.id === mindId)?.folderId ?? null;
      useComposer.getState().update(mindId, { scope: chosenScope(scope, folderId) });
    }
    useComposer.getState().requestFocus();
    useAppStore.getState().questionStarted();
  }, [questionHere, composerShown, mindId]);

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
  // The first empty Mind says what a Mind is until something is written in it, and never again (#721).
  const wasEmpty = useRef(empty);
  useEffect(() => {
    if (wasEmpty.current && !empty && !explained) {
      rememberMindExplained();
      setExplained(true);
    }
    wasEmpty.current = empty;
  }, [empty, explained]);
  const findShown = useMindTools((state) => state.findMindId === mindId);
  const outlineShown = useMindTools((state) => state.outlineMindId === mindId);
  const guide = useGettingStarted() !== null && empty && !isExample && !readOnly;
  const verification = useVerification(doc);
  const frame = useRef<HTMLDivElement>(null);
  useClickBelowNote(editor, frame);
  const wordLook = useWordLookStyle(doc, mindId, editor);

  return (
    <MindIdContext.Provider value={mindId}>
      {countSlot &&
        createPortal(
          <>
            <WordOrigin doc={doc} editor={editor} />
            <OriginalNotAdded mindId={mindId} />
            {verification ? (
              <VerifyLine editor={editor} doc={doc} mindId={mindId} verification={verification} />
            ) : (
              <WordCount editor={editor} doc={doc}>
                <NoteHealth editor={editor} doc={doc} mindId={mindId} />
                <ContradictionLine doc={doc} mindId={mindId} />
                <MindCheckAsk mindId={mindId} />
              </WordCount>
            )}
          </>,
          countSlot,
        )}
      {readOnly && <MindNotice kind="read-only" />}
      <div
        ref={frame}
        className={`mind-editor-frame ${guide ? "mind-editor-frame--guide" : ""} ${wordLook?.className ?? ""}`}
      >
        {wordLook && <style>{wordLook.css}</style>}
        {!readOnly && <ProposalBars editor={editor} mindId={mindId} />}
        <EditorContent editor={editor} />
        {!readOnly && <EndHint editor={editor} text={hintText} onAsk={askOnLine} />}
        <MarginChecks editor={editor} mind={{ id: mindId, doc }} />
        {!readOnly && <TableMenu editor={editor} />}
        <FootnoteArea editor={editor} citationView={citationView} />
        {(style !== "numbered" || styleChoice.bibliography) && (
          <ReferencesBlock mindId={mindId} editor={editor} readOnly={readOnly} />
        )}
        {!readOnly && !styleChoice.chosen && !styleChoice.promptDismissed && (
          <CitationStylePrompt mindId={mindId} editor={editor} doc={doc} />
        )}
      </div>
      {sheetSlot &&
        comparing &&
        !readOnly &&
        createPortal(<CompareSheet mindId={mindId} doc={doc} editor={editor} />, sheetSlot)}
      {guide && <StartGuide />}
      {!readOnly && <BlockHandle editor={editor} />}
      {!readOnly && <SelectionMenu editor={editor} ai={aiActions} />}
      {!readOnly && <LinkBubble editor={editor} />}
      {findShown && <FindBar editor={editor} onClose={() => useMindTools.getState().hideFind()} />}
      {outlineShown && (
        <MindOutline editor={editor} onClose={() => useMindTools.getState().hideOutline()} />
      )}
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
