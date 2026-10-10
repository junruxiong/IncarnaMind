import "katex/dist/katex.min.css";
import { type Editor, getTextBetween, getTextSerializersFromSchema, isMacOS } from "@tiptap/core";
import Collaboration from "@tiptap/extension-collaboration";
import {
  EditorContent,
  NodeViewContent,
  NodeViewWrapper,
  type ReactNodeViewProps,
  ReactNodeViewRenderer,
  useEditor,
} from "@tiptap/react";
import {
  type KeyboardEvent,
  memo,
  type ReactNode,
  type RefObject,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type * as Y from "yjs";
import { BLOCK_ID_ATTRIBUTE, type BlockOrigin } from "../../../core/api";
import { blockReference } from "../../../core/references";
import { removeOrigins, setOrigins } from "../../../shared/blockOrigins";
import {
  type ChatSummary,
  chatBlocks,
  chatContaining,
  listChats,
  observeChats,
} from "../../../shared/mindChats";
import { withoutIds } from "../../../shared/noteBlocks";
import {
  hasSearchScope,
  SCOPE_ATTRIBUTES,
  type ScopeKind,
  scopeIds,
  searchScopeOf,
} from "../../../shared/searchScope";
import { useAnswers } from "../answers";
import { ResizeRod } from "../components/ResizeRod";
import { core } from "../core";
import { useLanguage, useT } from "../i18n";
import { useAppStore } from "../store";
import { type AddedToNote, type ThreadView, useThreads } from "../threads";
import { type AddTarget, AddToMindPicker } from "./AddToMindPicker";
import { AnswerView } from "./AnswerView";
import { addAtPlace, answerBlocks, checkedBlocks, placeInNote, removeFromNote } from "./addToNote";
import { findBlock } from "./blockCommands";
import { flashBlocks } from "./blockOrigins";
import { CitationView } from "./CitationView";
import { CodeBlockView } from "./CodeBlockView";
import { CitationNumbers } from "./citationNumbers";
import { MarginChecks } from "./MarginChecks";
import { MindIdContext } from "./mindContext";
import { noteExtensions } from "./noteSchema";
import { Popover } from "./Popover";
import { QuestionProblem, SkillChip } from "./QuestionView";
import { ScopeChips } from "./ScopeChips";
import { type ThreadActions, ThreadContext, useThread } from "./ThreadAnswerActions";

const CLOSED: ThreadView = { chatId: null, open: false, fresh: false, focusAnswer: null };

export function shownChatId(chats: readonly ChatSummary[], view: ThreadView): string | null {
  if (view.fresh) return null;
  if (view.chatId && chats.some((chat) => chat.id === view.chatId)) return view.chatId;
  return chats[0]?.id ?? null;
}

/** The Mind's chats, newest first, kept up to date (only when their list or titles change). */
export function useChats(doc: Y.Doc): ChatSummary[] {
  const [chats, setChats] = useState(() => listChats(doc));
  useEffect(() => {
    let last = JSON.stringify(listChats(doc));
    setChats(JSON.parse(last) as ChatSummary[]);
    return observeChats(doc, () => {
      const next = JSON.stringify(listChats(doc));
      if (next === last) return;
      last = next;
      setChats(JSON.parse(next) as ChatSummary[]);
    });
  }, [doc]);
  return chats;
}

const SparkIcon = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.75}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
    className="thread-spark"
  >
    <path d="M12 3.5c.55 4.6 3.4 7.45 8 8-4.6.55-7.45 3.4-8 8-.55-4.6-3.4-7.45-8-8 4.6-.55 7.45-3.4 8-8z" />
  </svg>
);

const HistoryIcon = () => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.75}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
    className="size-4"
  >
    <path d="M4 12a8 8 0 1 0 2.4-5.7" />
    <path d="M4 4.5v4h4" />
    <path d="M12 8v4.2l2.8 1.8" />
  </svg>
);

const FoldIcon = ({ open }: { open: boolean }) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.75}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
    className="size-4"
  >
    <path d={open ? "m6 9 6 6 6-6" : "m6 15 6-6 6 6"} />
  </svg>
);

function ThreadQuestionView({ node, updateAttributes }: ReactNodeViewProps) {
  const t = useT();
  const thread = useThread();
  const questionId =
    typeof node.attrs[BLOCK_ID_ATTRIBUTE] === "string" ? node.attrs[BLOCK_ID_ATTRIBUTE] : null;
  const forcedSkill = typeof node.attrs.forcedSkill === "string" ? node.attrs.forcedSkill : null;
  const skill = useAppStore((state) =>
    forcedSkill ? state.skills.find((each) => each.name === forcedSkill) : undefined,
  );
  const dropSkill = () => {
    updateAttributes({ forcedSkill: null });
    if (questionId) useAnswers.getState().dismiss(questionId);
  };
  const scope = searchScopeOf(node.attrs);
  const removeFromScope = (kind: ScopeKind, id: string) => {
    const left = scopeIds(scope, kind).filter((each) => each !== id);
    updateAttributes({ [SCOPE_ATTRIBUTES[kind]]: left.length > 0 ? left : null });
  };
  return (
    <NodeViewWrapper
      className="thread-question"
      data-testid="thread-question"
      data-question-id={questionId ?? undefined}
    >
      <div className="thread-question-bubble">
        <span contentEditable={false} className="sr-only">
          {t("thread.youAsked")}
        </span>
        {forcedSkill && (
          <SkillChip
            name={forcedSkill}
            state={!skill ? "removed" : skill.enabled ? "enabled" : "disabled"}
            onRemove={dropSkill}
          />
        )}
        {forcedSkill && " "}
        <NodeViewContent<"span"> as="span" className="thread-question-text" />
      </div>
      {hasSearchScope(scope) && (
        <div contentEditable={false} data-testid="question-meta" className="thread-question-meta">
          <ScopeChips
            scope={scope}
            onRemove={removeFromScope}
            shown={8}
            className="question-scope"
            testId="question-scope"
          />
        </div>
      )}
      <QuestionProblem
        questionId={questionId}
        onAskWithoutSkill={() => {
          dropSkill();
          if (thread && questionId) {
            void useAnswers.getState().ask(thread.mindId, questionId, false, thread.chatId);
          }
        }}
      />
    </NodeViewWrapper>
  );
}

/** A Citation redraws when it changes, and when an edit renumbers it (see MindEditor). */
const citationView = ReactNodeViewRenderer(CitationView, {
  as: "span",
  update: ({ oldNode, newNode, oldDecorations, newDecorations, updateProps }) => {
    if (newNode.type !== oldNode.type) return false;
    if (newNode !== oldNode || oldDecorations !== newDecorations) updateProps();
    return true;
  },
});

/**
 * The Questions and Answers of one chat (or, for a chat on its own, #203, of
 * the Mind itself), drawn with the editor's own node views from `fragment`,
 * never written to here: Answers come from the core. Their Citations are
 * numbered and checked in the right margin, as in a note.
 */
export const ThreadLog = memo(function ThreadLog({
  mindId,
  fragment,
  label,
  onEditor,
}: {
  mindId: string;
  fragment: Y.XmlFragment;
  label: string;
  onEditor(editor: Editor | null): void;
}) {
  const editor = useEditor(
    {
      editable: false,
      shouldRerenderOnTransaction: false,
      editorProps: {
        attributes: {
          "data-testid": "thread-editor",
          role: "log",
          "aria-label": label,
          class: "mind-editor thread-editor",
        },
      },
      extensions: [
        // The thread never writes: it reads Block ids, and gives none.
        ...noteExtensions({
          codeBlockView: ReactNodeViewRenderer(CodeBlockView),
          questionView: ReactNodeViewRenderer(ThreadQuestionView),
          answerView: ReactNodeViewRenderer(AnswerView),
          citationView,
          readOnly: true,
        }),
        CitationNumbers,
        Collaboration.configure({ fragment }),
      ],
    },
    [fragment],
  );

  useEffect(() => {
    onEditor(editor);
    return () => onEditor(null);
  }, [editor, onEditor]);

  return (
    <MindIdContext.Provider value={mindId}>
      <div className="thread-log-frame">
        <EditorContent editor={editor} />
        <MarginChecks editor={editor} />
      </div>
    </MindIdContext.Provider>
  );
});

function focusedAnswer(): string | null {
  const active = document.activeElement;
  const answer = active instanceof Element ? active.closest("[data-answer-id]") : null;
  return answer?.getAttribute("data-answer-id") ?? null;
}

function lastAnswer(editor: Editor | null): string | null {
  if (!editor || editor.isDestroyed) return null;
  let found: string | null = null;
  editor.state.doc.forEach((node) => {
    if (node.type.name === "answer" && typeof node.attrs[BLOCK_ID_ATTRIBUTE] === "string") {
      found = node.attrs[BLOCK_ID_ATTRIBUTE];
    }
  });
  return found;
}

const isCommand = (event: { metaKey: boolean; ctrlKey: boolean }) =>
  isMacOS() ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey;

interface Room {
  /** The Mind's height: by default the thread opens at no more than a third of it (D136). */
  mind: number;
  /** The tallest the thread can be: from the top of the Mind down to the composer. */
  most: number;
  /** The thread's header: the shortest it can be. */
  least: number;
  height: number;
}

const sameRoom = (a: Room | null, b: Room) =>
  a !== null &&
  a.mind === b.mind &&
  a.most === b.most &&
  a.least === b.least &&
  a.height === b.height;

/**
 * How tall the thread's log is: up to a third of the Mind (with its header)
 * as it fills, or the height the User dragged it to (`threadHeight`), within
 * the room there is.
 */
function logSize(room: Room, chosen: number | null): { height?: number; maxHeight: number } {
  const most = Math.max(room.least, room.most);
  if (chosen !== null) {
    const height = Math.min(Math.max(chosen, room.least), most) - room.least;
    return { height, maxHeight: height };
  }
  return {
    maxHeight: Math.min(Math.max(Math.round(room.mind / 3), room.least), most) - room.least,
  };
}

interface Picking {
  answerId: string;
  anchor: HTMLElement;
  /** What it adds, taken when the ▾ was pressed: the selection goes once the search takes the focus. */
  blocks: ReturnType<typeof answerBlocks>;
}

/**
 * The thread above the composer (#201, DESIGN.md D124; boards Chat-B1, B2,
 * C2): after a Question it rises from the composer over the lower part of the
 * Mind, joined to it, and the note stays as it is. Its header says what it is,
 * has the Mind's earlier chats and a button that folds it (Esc does too), to
 * its header; it never covers the composer. Under each Answer: Copy, Add to
 * note (here, at the cursor) with ▾ for another Mind, and Regenerate; once
 * added, "Added to note · Undo".
 */
export function ThreadPanel({
  mindId,
  doc,
  note,
  cursorInMind,
  focusComposer,
  dock = "bottom",
  headerActions,
  children,
}: {
  mindId: string;
  doc: Y.Doc;
  /** The Mind's note, where "Add to note" puts an Answer at the cursor. */
  note: Editor;
  /** Whether the note's cursor is where it goes; otherwise it goes at the end. */
  cursorInMind: RefObject<boolean>;
  focusComposer(): void;
  /**
   * Where the parent layout docks it: joined to the top of the composer at the
   * foot of the Mind ("bottom", the only one so far). Its placement is the
   * dock's CSS (`.thread-dock--bottom`), never this component's.
   */
  dock?: "bottom";
  /** More buttons for the header, before history and fold: e.g. a ⋯ menu. */
  headerActions?: ReactNode;
  children: ReactNode;
}) {
  const t = useT();
  const language = useLanguage();
  const chats = useChats(doc);
  const view = useThreads((state) => state.views[mindId]) ?? CLOSED;
  const added = useThreads((state) => state.added[mindId]) ?? null;
  const shownId = shownChatId(chats, view);
  const shown = chats.find((chat) => chat.id === shownId) ?? null;
  const fragment = useMemo(() => (shownId ? chatBlocks(doc, shownId) : null), [doc, shownId]);
  const [log, setLog] = useState<Editor | null>(null);
  const logRef = useRef<Editor | null>(null);
  logRef.current = log;
  const [picking, setPicking] = useState<Picking | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const historyButton = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLElement>(null);
  const head = useRef<HTMLElement>(null);
  const noteScroller = useRef<HTMLElement | null>(null);
  const [room, setRoom] = useState<Room | null>(null);
  const chosenHeight = useAppStore((state) => state.settings?.device.threadHeight ?? null);
  const bodyId = useId();
  const titleId = useId();
  const mindTitle = useAppStore(
    (state) => state.minds.find((mind) => mind.id === mindId)?.title ?? "",
  );

  const visible = shownId !== null || view.fresh;
  const open = view.open && visible;
  const update = (change: Partial<ThreadView>) => useThreads.getState().update(mindId, change);

  // As the thread's height changes: the card's one shadow, under it and the composer; room
  // under the note, so its end can scroll above the thread; and the note's cursor kept clear
  // of it, so typing near the foot of the note scrolls what is typed up above the thread.
  // biome-ignore lint/correctness/useExhaustiveDependencies: the panel comes and goes with `visible`.
  useLayoutEffect(() => {
    const element = panel.current;
    const dock = element?.closest<HTMLElement>(".composer-dock");
    const scroller = dock?.previousElementSibling;
    const composer = dock?.querySelector<HTMLElement>(".composer");
    const pane = dock?.parentElement;
    if (!element || !dock || !composer || !pane || !(scroller instanceof HTMLElement)) return;
    noteScroller.current = scroller;
    const place = () => {
      const height = Math.round(element.offsetHeight);
      // The thread's foot sits 8px into the dock, on the composer's top edge.
      const next: Room = {
        mind: pane.clientHeight,
        most: pane.clientHeight - dock.offsetHeight + 8,
        least: Math.round((head.current?.offsetHeight ?? 0) + element.clientTop),
        height,
      };
      setRoom((was) => (sameRoom(was, next) ? was : next));
      // As wide as the composer and over it, however the dock lays it out (a narrow Mind, a
      // scrollbar's gutter beside it).
      dock.style.setProperty("--thread-left", `${composer.offsetLeft}px`);
      dock.style.setProperty("--thread-width", `${composer.offsetWidth}px`);
      dock.style.setProperty("--thread-gutter", getComputedStyle(dock).paddingRight);
      dock.style.setProperty("--thread-height", `${height}px`);
      scroller.style.setProperty("--thread-room", `${height}px`);
      if (note.isDestroyed) return;
      note.view.setProps({
        scrollMargin: { top: 5, left: 5, right: 5, bottom: height + 16 },
        scrollThreshold: { top: 0, left: 0, right: 0, bottom: height + 16 },
      });
    };
    place();
    const observer = new ResizeObserver(place);
    for (const each of [element, composer, dock, pane]) observer.observe(each);
    return () => {
      observer.disconnect();
      for (const name of [
        "--thread-left",
        "--thread-width",
        "--thread-gutter",
        "--thread-height",
      ]) {
        dock.style.removeProperty(name);
      }
      scroller.style.removeProperty("--thread-room");
      // ProseMirror's own defaults.
      if (!note.isDestroyed) note.view.setProps({ scrollMargin: 5, scrollThreshold: 0 });
    };
  }, [visible, note]);

  const focusAnswer = view.focusAnswer;
  useEffect(() => {
    if (!focusAnswer) return;
    const update = (change: Partial<ThreadView>) => useThreads.getState().update(mindId, change);
    const chat = chatContaining(doc, focusAnswer);
    if (!chat) {
      update({ focusAnswer: null });
      return;
    }
    if (chat !== shownId || !view.open) {
      update({ chatId: chat, fresh: false, open: true });
      return;
    }
    if (!log || log.isDestroyed) return;
    // The Answer's view is drawn a moment after the editor has it: look for it each frame.
    let frame = 0;
    let tries = 0;
    const seek = () => {
      const found = log.isDestroyed ? null : findBlock(log.state.doc, focusAnswer);
      const element = found ? log.view.nodeDOM(found.pos) : null;
      const answer =
        element instanceof HTMLElement
          ? element.querySelector<HTMLElement>("[data-answer-id]")
          : null;
      if (!answer) {
        if (++tries < 60) frame = requestAnimationFrame(seek);
        return;
      }
      answer.tabIndex = -1;
      answer.scrollIntoView({ block: "nearest" });
      answer.focus({ preventScroll: true });
      answer.classList.add("thread-answer--sought");
      setTimeout(() => answer.classList.remove("thread-answer--sought"), 1600);
      update({ focusAnswer: null });
    };
    seek();
    return () => cancelAnimationFrame(frame);
  }, [focusAnswer, doc, shownId, view.open, log, mindId]);

  const chatTitle = shown?.title ?? "";

  const origin = (answerId: string): BlockOrigin => ({
    from: blockReference(mindId, answerId),
    label: chatTitle,
  });

  const setAdded = (next: AddedToNote | null) => useThreads.getState().setAdded(next, mindId);

  /** Into this Mind, at its cursor (or end), through its editor: ⌘Z in the note undoes it too. */
  const addHere = async (answerId: string, blocks: ReturnType<typeof answerBlocks>) => {
    if (!blocks || note.isDestroyed) return;
    const checked = await checkedBlocks(blocks, (input) => core.recheckCitation(input));
    if (note.isDestroyed) return;
    const added = addAtPlace(note, checked, placeInNote(note, cursorInMind.current === true));
    if (!added) return;
    const ids = added.blockIds;
    setOrigins(doc, ids, origin(answerId));
    setAdded({
      mindId,
      answerId,
      target: { mindId, title: mindTitle, here: true },
      blockIds: ids,
      emptyLine: added.emptyLine,
    });
    flashBlocks(note, ids);
    const first = ids[0] ? findBlock(note.state.doc, ids[0]) : null;
    const element = first ? note.view.nodeDOM(first.pos) : null;
    if (element instanceof HTMLElement) element.scrollIntoView({ block: "nearest" });
  };

  const addThere = async (
    answerId: string,
    blocks: NonNullable<ReturnType<typeof answerBlocks>>,
    target: Extract<AddTarget, { kind: "mind" | "new" }>,
  ) => {
    const mind =
      target.kind === "mind"
        ? target.mind
        : await core.createMind({ title: target.title.slice(0, 120) });
    const result = await core.addToNote({
      mindId: mind.id,
      blocks: withoutIds(blocks),
      origin: origin(answerId),
    });
    setAdded({
      mindId,
      answerId,
      target: { mindId: mind.id, title: mind.title, here: false, created: target.kind === "new" },
      blockIds: result.blockIds,
    });
  };

  const actions: ThreadActions = {
    mindId,
    chatId: shownId,
    added,
    async copy(answerId) {
      const editor = logRef.current;
      const found = editor && !editor.isDestroyed ? findBlock(editor.state.doc, answerId) : null;
      if (!editor || !found) return false;
      const range = { from: found.pos + 1, to: found.pos + found.node.nodeSize - 1 };
      // HTML that pastes back as Notes with Citations; text with each Citation's source in it.
      const { dom } = editor.view.serializeForClipboard(
        editor.state.doc.slice(range.from, range.to),
      );
      const text = getTextBetween(editor.state.doc, range, {
        blockSeparator: "\n\n",
        textSerializers: getTextSerializersFromSchema(editor.schema),
      });
      try {
        await navigator.clipboard.write([
          new ClipboardItem({
            "text/html": new Blob([dom.innerHTML], { type: "text/html" }),
            "text/plain": new Blob([text], { type: "text/plain" }),
          }),
        ]);
        return true;
      } catch (error) {
        useAppStore.getState().reportError(error);
        return false;
      }
    },
    add(answerId) {
      const editor = logRef.current;
      if (!editor) return;
      addHere(answerId, answerBlocks(editor.view, answerId)).catch((error: unknown) =>
        useAppStore.getState().reportError(error),
      );
    },
    pick(answerId, anchor) {
      const editor = logRef.current;
      if (!editor) return;
      setPicking({ answerId, anchor, blocks: answerBlocks(editor.view, answerId) });
    },
    undo() {
      const last = useThreads.getState().added[mindId];
      if (!last) return;
      setAdded(null);
      if (last.target.here) {
        if (note.isDestroyed) return;
        removeFromNote(note, { blockIds: last.blockIds, emptyLine: last.emptyLine ?? null });
        removeOrigins(doc, last.blockIds);
        return;
      }
      const { mindId: target, created } = last.target;
      core
        .undoAddToNote({ mindId: target, blockIds: last.blockIds })
        .then(() => (created ? core.deleteMind(target) : undefined))
        .catch((error: unknown) => useAppStore.getState().reportError(error));
    },
  };

  const choose = (target: AddTarget) => {
    const current = picking;
    setPicking(null);
    if (!current?.blocks) return;
    const run =
      target.kind === "here"
        ? addHere(current.answerId, current.blocks)
        : addThere(current.answerId, current.blocks, target);
    run
      .then(() => {
        // The answer's row now says it was added: its Undo takes the focus.
        requestAnimationFrame(() =>
          panel.current?.querySelector<HTMLElement>('[data-testid="thread-added-undo"]')?.focus(),
        );
      })
      .catch((error: unknown) => useAppStore.getState().reportError(error));
  };

  const fold = () => {
    useThreads.getState().fold(mindId);
    focusComposer();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (event.defaultPrevented || event.nativeEvent.isComposing) return;
    if (event.key === "Escape") {
      // A Document opened from a Citation here is the innermost thing: Esc closes it first.
      if (useAppStore.getState().viewerOpen) return;
      event.preventDefault();
      fold();
      return;
    }
    if (!isCommand(event) || event.altKey) return;
    const key = event.key.toLowerCase();
    if (key === "z" && !event.shiftKey && added) {
      event.preventDefault();
      actions.undo();
      return;
    }
    if (!event.shiftKey || (key !== "c" && key !== "a")) return;
    const answerId = focusedAnswer() ?? lastAnswer(logRef.current);
    if (!answerId) return;
    event.preventDefault();
    if (key === "c") void actions.copy(answerId);
    else actions.add(answerId);
  };

  const historyLabel = t("thread.history");
  const resize = (height: number | null) =>
    useAppStore.getState().previewLayout({ threadHeight: height });
  const keepHeight = (height: number | null) =>
    void useAppStore.getState().updateSettings({ device: { threadHeight: height } });
  return (
    <div data-dock={dock} className={`thread-dock thread-dock--${dock}`}>
      {visible && (
        <ThreadContext.Provider value={actions}>
          {/* The Mind's margins beside the thread are covered too: no bullet, handle or check of
              the Blocks under it shows past its edges. The wheel there still scrolls the Mind. */}
          <div
            aria-hidden="true"
            className="thread-backdrop"
            onWheel={(event) => noteScroller.current?.scrollBy({ top: event.deltaY })}
          />
          <div aria-hidden="true" className="thread-shadow" />
          <section
            ref={panel}
            aria-labelledby={titleId}
            data-testid="thread"
            data-open={open || undefined}
            onKeyDown={onKeyDown}
            className={open ? "thread-panel thread-panel--open" : "thread-panel"}
          >
            {open && room && (
              <ResizeRod
                axis="y"
                direction={-1}
                testId="thread-resize"
                label={t("thread.resize")}
                size={chosenHeight === null ? room.height : Math.min(chosenHeight, room.most)}
                min={room.least}
                max={room.most}
                onPreview={resize}
                onCommit={keepHeight}
                onReset={() => {
                  resize(null);
                  keepHeight(null);
                }}
                className="thread-resize"
              />
            )}
            <header ref={head} className="thread-head">
              <SparkIcon />
              <h2 id={titleId} className="thread-title">
                {t("thread.title")}
              </h2>
              <button
                type="button"
                data-testid="thread-subtitle"
                tabIndex={-1}
                onClick={() => (open ? undefined : update({ open: true }))}
                className="thread-subtitle"
              >
                {open || !shown ? t("thread.subtitle") : shown.title}
              </button>
              {headerActions}
              <button
                ref={historyButton}
                type="button"
                data-testid="thread-history"
                aria-label={historyLabel}
                title={historyLabel}
                aria-haspopup="menu"
                aria-expanded={historyOpen}
                onClick={() => setHistoryOpen((was) => !was)}
                className="thread-head-button"
              >
                <HistoryIcon />
              </button>
              <button
                type="button"
                data-testid="thread-fold"
                aria-expanded={open}
                aria-controls={bodyId}
                aria-label={open ? t("thread.fold") : t("thread.unfold")}
                title={open ? t("thread.foldTitle") : t("thread.unfold")}
                onClick={() => (open ? fold() : update({ open: true }))}
                className="thread-head-button"
              >
                <FoldIcon open={open} />
              </button>
            </header>
            <div id={bodyId} hidden={!open} className="thread-body">
              {open && fragment && (
                <div
                  className="thread-scroller"
                  data-testid="thread-scroller"
                  style={room ? logSize(room, chosenHeight) : undefined}
                >
                  <ThreadLog
                    key={shownId}
                    mindId={mindId}
                    fragment={fragment}
                    label={t("thread.log", { title: mindTitle || t("mind.untitled") })}
                    onEditor={setLog}
                  />
                </div>
              )}
              {open && !fragment && (
                <p data-testid="thread-empty" className="thread-empty">
                  {t("thread.empty")}
                </p>
              )}
            </div>
          </section>
          {historyOpen && historyButton.current && (
            <ChatHistory
              anchor={historyButton.current}
              chats={chats}
              shownId={shownId}
              language={language}
              onNew={() => {
                setHistoryOpen(false);
                useThreads.getState().startNew(mindId);
                focusComposer();
              }}
              onChoose={(chatId) => {
                setHistoryOpen(false);
                useThreads.getState().open(mindId, chatId);
              }}
              onClose={() => {
                setHistoryOpen(false);
                historyButton.current?.focus();
              }}
            />
          )}
          {picking && (
            <AddToMindPicker
              anchor={picking.anchor}
              mindId={mindId}
              newTitle={chatTitle}
              hereAt={cursorInMind.current ? "cursor" : "end"}
              onChoose={choose}
              onClose={() => {
                const anchor = picking.anchor;
                setPicking(null);
                anchor.focus();
              }}
            />
          )}
        </ThreadContext.Provider>
      )}
      {children}
    </div>
  );
}

function ChatHistory({
  anchor,
  chats,
  shownId,
  language,
  onNew,
  onChoose,
  onClose,
}: {
  anchor: HTMLElement;
  chats: readonly ChatSummary[];
  shownId: string | null;
  language: string;
  onNew(): void;
  onChoose(chatId: string): void;
  onClose(): void;
}) {
  const t = useT();
  const first = useRef<HTMLButtonElement>(null);
  const items = useRef<HTMLDivElement>(null);
  const moveFocus = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    event.preventDefault();
    const all = [...(items.current?.querySelectorAll<HTMLElement>('[role^="menuitem"]') ?? [])];
    const at = all.indexOf(document.activeElement as HTMLElement);
    const next = (at + (event.key === "ArrowDown" ? 1 : -1) + all.length) % all.length;
    all[next]?.focus();
  };
  return (
    <Popover
      anchor={anchor}
      placement="top-end"
      onDismiss={onClose}
      initialFocus={first}
      role="menu"
      aria-label={t("thread.history")}
      data-testid="thread-history-menu"
      className="editor-menu thread-history"
    >
      {/* biome-ignore lint/a11y/noStaticElementInteractions: ↑ and ↓ move between the menu's items. */}
      <div ref={items} onKeyDown={moveFocus} className="flex flex-col">
        <button
          ref={first}
          type="button"
          role="menuitem"
          data-testid="thread-new-chat"
          onClick={onNew}
          className="editor-menu-item"
        >
          {t("thread.newChat")}
        </button>
        {chats.length > 0 && (
          // biome-ignore lint/a11y/useSemanticElements: a rule between a menu's items, where an hr can't go.
          <div role="separator" className="add-to-separator" />
        )}
        {chats.map((chat) => (
          <button
            key={chat.id}
            type="button"
            role="menuitemradio"
            aria-checked={chat.id === shownId}
            data-testid="thread-history-chat"
            onClick={() => onChoose(chat.id)}
            className="editor-menu-item thread-history-chat"
          >
            <span className="thread-history-title" title={chat.title}>
              {chat.title || t("thread.untitledChat")}
            </span>
            <span className="thread-history-detail">
              {t(chat.questions === 1 ? "thread.historyOne" : "thread.historyDetail", {
                date: chat.createdAt
                  ? new Date(chat.createdAt).toLocaleDateString(language, {
                      day: "numeric",
                      month: "short",
                    })
                  : "",
                count: chat.questions,
              })}
            </span>
          </button>
        ))}
      </div>
    </Popover>
  );
}
