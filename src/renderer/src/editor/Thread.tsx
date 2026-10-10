import "katex/dist/katex.min.css";
import { type Editor, isMacOS } from "@tiptap/core";
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
import { type ThreadActions, ThreadContext } from "./ThreadAnswerActions";

const CLOSED: ThreadView = { chatId: null, open: false, fresh: false, focusAnswer: null };

/** The chat the thread shows: the one chosen if it is still there, else the newest; none for a new chat. */
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

/** A Question in the thread: the User's words in a bubble at the right (board Chat-B1). */
function ThreadQuestionView({ node }: ReactNodeViewProps) {
  const t = useT();
  const questionId =
    typeof node.attrs[BLOCK_ID_ATTRIBUTE] === "string" ? node.attrs[BLOCK_ID_ATTRIBUTE] : undefined;
  const skill = typeof node.attrs.forcedSkill === "string" ? node.attrs.forcedSkill : null;
  return (
    <NodeViewWrapper
      className="thread-question"
      data-testid="thread-question"
      data-question-id={questionId}
    >
      <div className="thread-question-bubble">
        <span contentEditable={false} className="sr-only">
          {t("thread.youAsked")}
        </span>
        {skill && (
          <span
            contentEditable={false}
            data-testid="question-skill"
            className="thread-question-skill"
          >
            {skill}
          </span>
        )}
        <NodeViewContent<"span"> as="span" className="thread-question-text" />
      </div>
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
          "aria-label": label,
          class: "mind-editor thread-editor",
        },
      },
      extensions: [
        // The thread never writes: no Block ids to give.
        ...noteExtensions({
          codeBlockView: ReactNodeViewRenderer(CodeBlockView),
          questionView: ReactNodeViewRenderer(ThreadQuestionView),
          answerView: ReactNodeViewRenderer(AnswerView),
          citationView,
        }).filter((extension) => extension.name !== "uniqueID"),
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

/** The Answer in the thread that holds the focus, if any. */
function focusedAnswer(): string | null {
  const active = document.activeElement;
  const answer = active instanceof Element ? active.closest("[data-answer-id]") : null;
  return answer?.getAttribute("data-answer-id") ?? null;
}

/** The newest Answer in the thread. */
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
}: {
  mindId: string;
  doc: Y.Doc;
  /** The Mind's note, where "Add to note" puts an Answer at the cursor. */
  note: Editor;
  /** Whether the note's cursor is where it goes; otherwise it goes at the end. */
  cursorInMind: RefObject<boolean>;
  focusComposer(): void;
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
  const bodyId = useId();
  const titleId = useId();
  const mindTitle = useAppStore(
    (state) => state.minds.find((mind) => mind.id === mindId)?.title ?? "",
  );

  const visible = shownId !== null || view.fresh;
  const open = view.open && visible;
  const update = (change: Partial<ThreadView>) => useThreads.getState().update(mindId, change);

  // Room under the note for the thread, so the note's end can scroll above it.
  // biome-ignore lint/correctness/useExhaustiveDependencies: the panel comes and goes with `visible`.
  useLayoutEffect(() => {
    const element = panel.current;
    const scroller = element?.closest(".composer-dock")?.previousElementSibling;
    if (!(scroller instanceof HTMLElement)) return;
    if (!element) {
      scroller.style.removeProperty("--thread-room");
      return;
    }
    const observer = new ResizeObserver(() => {
      scroller.style.setProperty("--thread-room", `${Math.round(element.offsetHeight)}px`);
    });
    observer.observe(element);
    return () => {
      observer.disconnect();
      scroller.style.removeProperty("--thread-room");
    };
  }, [visible]);

  // A link back from a Note: the chat with that Answer, up, the Answer in view and focused.
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
    const found = findBlock(log.state.doc, focusAnswer);
    if (!found) return;
    const element = log.view.nodeDOM(found.pos);
    const answer =
      element instanceof HTMLElement
        ? (element.querySelector<HTMLElement>("[data-answer-id]") ?? element)
        : null;
    if (!answer) return;
    answer.tabIndex = -1;
    answer.scrollIntoView({ block: "nearest" });
    answer.focus({ preventScroll: true });
    answer.classList.add("thread-answer--sought");
    setTimeout(() => answer.classList.remove("thread-answer--sought"), 1600);
    update({ focusAnswer: null });
  }, [focusAnswer, doc, shownId, view.open, log, mindId]);

  /** The chat's first Question, which names it in a Note's link back and a new Mind's title. */
  const chatTitle = shown?.title ?? "";

  const origin = (answerId: string): BlockOrigin => ({
    from: blockReference(mindId, answerId),
    label: chatTitle,
  });

  const setAdded = (next: AddedToNote | null) => useThreads.getState().setAdded(next, mindId);

  /** Into this Mind, at its cursor (or end), through its editor: ⌘Z in the note undoes it too. */
  const addHere = async (answerId: string, blocks: ReturnType<typeof answerBlocks>) => {
    if (!blocks || note.isDestroyed) return;
    const checked = await checkedBlocks(blocks);
    if (note.isDestroyed) return;
    const ids = addAtPlace(note, checked, placeInNote(note, cursorInMind.current === true));
    if (!ids) return;
    setOrigins(doc, ids, origin(answerId));
    setAdded({
      mindId,
      answerId,
      target: { mindId, title: mindTitle, here: true },
      blockIds: ids,
    });
    flashBlocks(note, ids);
    const first = ids[0] ? findBlock(note.state.doc, ids[0]) : null;
    const element = first ? note.view.nodeDOM(first.pos) : null;
    if (element instanceof HTMLElement) element.scrollIntoView({ block: "nearest" });
  };

  /** Into another Mind, at its end, through the core; a new one is made first. */
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
      const slice = editor.state.doc.slice(found.pos + 1, found.pos + found.node.nodeSize - 1);
      const { dom, text } = editor.view.serializeForClipboard(slice);
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
        removeFromNote(note, last.blockIds);
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

  if (!visible) return null;

  const historyLabel = t("thread.history");
  return (
    <ThreadContext.Provider value={actions}>
      <section
        ref={panel}
        aria-labelledby={titleId}
        data-testid="thread"
        data-open={open || undefined}
        onKeyDown={onKeyDown}
        className={open ? "thread-panel thread-panel--open" : "thread-panel"}
      >
        <header className="thread-head">
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
            <div className="thread-scroller" data-testid="thread-scroller">
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
  );
}

/** The thread's history (its header's clock): New chat, then the Mind's chats, newest first. */
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
