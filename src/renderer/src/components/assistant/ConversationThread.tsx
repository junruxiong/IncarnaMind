import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ConversationAnswer, ConversationQuestion } from "../../../../core/api";
import { mindName } from "../../../../shared/naming";
import { type AddTarget, addAnswer } from "../../addToNote";
import { answerBlocks } from "../../answerBlocks";
import { selectedNotes } from "../../answerEditors";
import { readsLine, stepsScope } from "../../assistantContext";
import { readMindHandle } from "../../commands/handles";
import { composerScope, NO_MIND, useComposer } from "../../composer";
import {
  conversationLabel,
  stopWriting,
  useAnswerOnItsWay,
  useConversations,
  useOpenConversation,
} from "../../conversations";
import { useT } from "../../i18n";
import { useModelSetup } from "../../modelSetup";
import { type ScopeChip, scopeChips } from "../../scope";
import { useAppStore } from "../../store";
import { interleave, type ThreadStart, threadTasks } from "../../taskGroups";
import { loadTasks, useTasks } from "../../tasks";
import { SearchIcon } from "../icons";
import { SparkLineIcon } from "../lineIcons";
import { AddToMindPicker } from "./AddToMindPicker";
import { WritingLine } from "./AnswerActions";
import { LiveLine } from "./AnswerSteps";
import { type AnswerKeeping, AssistantAnswer } from "./AssistantAnswer";
import { TaskProgress } from "./TaskProgress";

/** How long an Answer a link back led to stays flashed. */
const SOUGHT_MS = 1600;
/** How near the foot counts as being at it, so the thread keeps up with an Answer being written. */
const AT_FOOT_PX = 24;

/** What to add of an Answer: what is selected in it, or all of it as it ended. */
const notesOf = (answer: ConversationAnswer) =>
  selectedNotes(answer.id) ?? answerBlocks({ ...answer, status: "done" });

/**
 * The conversation the assistant shows (#201): the Questions as bubbles at the
 * right, as in Claude and ChatGPT, and the Answers with their steps,
 * Citations, checks and actions (`AssistantAnswer`); empty, what the next
 * Question reads and searches, and what asking here does. A Question shows
 * the moment it is sent, with its Answer's first step and Stop under it
 * before the core has checked the model, so asking never looks like nothing
 * happened. The Tasks of the Mind it reads show their progress where each
 * started, by time among the messages (D189). It keeps up with an Answer
 * being written while the User is at its foot, and brings an Answer a link
 * back led to into view, flashed.
 */
export function ConversationThread({ focusedMindId }: { focusedMindId: string | null }) {
  const t = useT();
  const messages = useConversations((state) => state.messages);
  const asking = useConversations((state) => state.asking);
  const waitingQuestion = useModelSetup((state) => state.waiting);
  const askingQuote = useConversations((state) => state.askingQuote);
  const onItsWay = useAnswerOnItsWay();
  const stopWanted = useConversations((state) => state.stopWanted);
  const sought = useConversations((state) => state.sought);
  const conversation = useOpenConversation();
  useEffect(loadTasks, []);
  const allTasks = useTasks((state) => state.tasks);
  const shownTask = useTasks((state) => state.shown);
  const thread = useThreadStart(conversation ?? null);
  const tasks = threadTasks(Object.values(allTasks), focusedMindId, thread, shownTask);
  const focusedMind = useAppStore((state) => state.minds.find((mind) => mind.id === focusedMindId));
  const focusedMindName = focusedMindId
    ? (focusedMind && mindName(focusedMind)) || t("mind.untitled")
    : null;
  const nextScope = useNextScope(focusedMindId);
  const reads = readsLine(t, focusedMind ? focusedMindName : null, nextScope);
  const [picking, setPicking] = useState<{
    answer: ConversationAnswer;
    anchor: HTMLElement;
    blocks: ReturnType<typeof notesOf>;
  } | null>(null);
  const log = useRef<HTMLDivElement>(null);
  const atFoot = useRef(true);

  // At the foot, it stays there as Answers grow; a scroll up by the User stops that.
  useEffect(() => {
    const scroller = log.current?.parentElement;
    if (!scroller) return;
    const onScroll = () => {
      atFoot.current =
        scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight <= AT_FOOT_PX;
    };
    scroller.addEventListener("scroll", onScroll);
    return () => scroller.removeEventListener("scroll", onScroll);
  }, []);
  // biome-ignore lint/correctness/useExhaustiveDependencies: each change to the messages may grow the thread.
  useLayoutEffect(() => {
    const scroller = log.current?.parentElement;
    if (scroller && (atFoot.current || asking) && !sought) {
      scroller.scrollTop = scroller.scrollHeight;
    }
  }, [messages, asking, sought]);

  // A link back led here: that Answer comes into view, flashed, once it shows (its conversation's
  // messages may come a moment after it is opened).
  const shownSought = useRef<string | null>(null);
  // biome-ignore lint/correctness/useExhaustiveDependencies: the Answer may show only once the messages come.
  useEffect(() => {
    if (!sought || shownSought.current === sought) return;
    const answer = log.current?.querySelector<HTMLElement>(`[data-answer-id="${sought}"]`);
    if (!answer) return;
    shownSought.current = sought;
    answer.scrollIntoView({ block: "nearest" });
    setTimeout(() => {
      shownSought.current = null;
      if (useConversations.getState().sought === sought) {
        useConversations.setState({ sought: null });
      }
    }, SOUGHT_MS);
  }, [sought, messages]);

  const add = (answer: ConversationAnswer, target: AddTarget, blocks = notesOf(answer)) =>
    void addAnswer({ answer, conversation, blocks }, target, focusedMindId === null);
  const keeping: AnswerKeeping = {
    focusedMindName,
    add(answer) {
      if (focusedMindId) add(answer, { kind: "mind", mindId: focusedMindId });
    },
    pick(answer, anchor) {
      setPicking({ answer, anchor, blocks: notesOf(answer) });
    },
  };

  const questions = new Map(
    messages.flatMap((message) =>
      message.kind === "question" ? [[message.id, message] as const] : [],
    ),
  );
  return (
    <div
      ref={log}
      role="log"
      aria-label={t("assistant.log", { title: conversationLabel(conversation) })}
      data-testid="assistant-log"
      className="assistant-log"
    >
      {messages.length === 0 && asking === null && !waitingQuestion && (
        <>
          <p data-testid="assistant-reads" title={reads} className="assistant-reads">
            <SearchIcon aria-hidden="true" className="assistant-reads-icon" />
            <span className="assistant-reads-text">{reads}</span>
          </p>
          <p data-testid="assistant-empty" className="assistant-empty">
            {t("assistant.empty")}
          </p>
        </>
      )}
      {interleave(messages, tasks).map((entry) => {
        if (entry.kind === "task") return <TaskProgress key={entry.task.id} task={entry.task} />;
        const message = entry.message;
        return message.kind === "question" ? (
          <QuestionBubble key={message.id} question={message} />
        ) : (
          <AssistantAnswer
            key={message.id}
            answer={message}
            question={questions.get(message.questionId)}
            sought={message.id === sought}
            keeping={keeping}
          />
        );
      })}
      {asking !== null && <QuestionBubble text={asking} quote={askingQuote} />}
      {waitingQuestion && (
        <>
          <QuestionBubble text={waitingQuestion.text} quote="" />
          <p data-testid="assistant-waiting-model" className="assistant-empty">
            {t("setup.waiting")}
          </p>
        </>
      )}
      {onItsWay && (
        // Its line shows at rest, as an Answer's being written does.
        <div
          data-testid="assistant-answer-on-its-way"
          data-status="streaming"
          className="assistant-answer"
        >
          <LiveLine
            step={
              focusedMind
                ? {
                    kind: "reading",
                    mind: { mindId: focusedMind.id, name: mindName(focusedMind), deleted: false },
                  }
                : { kind: "searching" }
            }
            scope={stepsScope(
              t,
              nextScope.filter((chip) => !chip.deleted),
            )}
          />
          {stopWanted ? (
            // Stopped before it started: it says so at once, and stops as it starts.
            <WritingLine
              status={
                <span data-testid="answer-stopped" className="assistant-answer-status">
                  {t("answer.status.stopped")}
                </span>
              }
            />
          ) : (
            <WritingLine onStop={stopWriting} />
          )}
        </div>
      )}
      {picking && (
        <AddToMindPicker
          anchor={picking.anchor}
          focusedMindId={focusedMindId}
          atCursor={readMindHandle.current?.cursorInMind() ?? false}
          newFolderId={conversation?.folderId ?? null}
          onChoose={(target) => {
            setPicking(null);
            add(picking.answer, target, picking.blocks);
          }}
          onClose={() => {
            const { anchor } = picking;
            setPicking(null);
            anchor.focus();
          }}
        />
      )}
    </div>
  );
}

/** A Question the User asked: their words in a bubble at the right, with the Skill forced on it. */
function QuestionBubble({
  question,
  text,
  quote,
}: {
  question?: ConversationQuestion;
  text?: string;
  quote?: string;
}) {
  const t = useT();
  const skill = question?.forcedSkill ?? null;
  const about = (question ? question.focus?.selection : quote) ?? "";
  return (
    <div
      data-testid="assistant-question"
      data-pending={question ? undefined : true}
      className="assistant-question"
    >
      <p className="assistant-question-bubble">
        <span className="sr-only">{t("assistant.youAsked")}</span>
        {about.trim() !== "" && (
          <span
            data-testid="assistant-question-quote"
            title={about}
            className="assistant-question-quote"
          >
            <span className="sr-only">{t("assistant.question.quote")}</span>“{about.trim()}”
          </span>
        )}
        {skill && (
          <span data-testid="assistant-question-skill" className="assistant-question-skill">
            <SparkLineIcon aria-hidden="true" className="size-3" />
            {skill}
          </span>
        )}
        {question?.text ?? text}
      </p>
    </div>
  );
}

/** What the next Question searches: the Search scope of the composer of the Mind in focus, or of the composer with none. */
function useNextScope(mindId: string | null): ScopeChip[] {
  const mind = useAppStore((state) => state.minds.find((each) => each.id === mindId));
  const draft = useComposer((state) => state.drafts[mindId ?? NO_MIND]);
  const folders = useAppStore((state) => state.folders);
  const groups = useAppStore((state) => state.library?.groups);
  const deletedGroups = useAppStore((state) => state.library?.deletedGroups);
  const tags = useAppStore((state) => state.tags);
  const documents = useAppStore((state) => state.documents);
  if (mindId !== null && !mind) return [];
  const scope = composerScope(
    draft ?? { text: "", scope: null, skill: null, pastes: [] },
    mind?.folderId ?? null,
  );
  return scopeChips({ folders, groups, deletedGroups, tags, documents }, scope);
}

/**
 * When the thread on screen began, and the Tasks it has shown running, for
 * the Tasks it shows: its conversation's start, or, for a new chat, when it
 * opened. A conversation that a new chat on screen became, with its first
 * Question, keeps the new chat's start and Tasks, so a Task started before
 * that Question stays where it was.
 */
function useThreadStart(conversation: { id: string; createdAt: string } | null): ThreadStart {
  const id = conversation?.id ?? null;
  const [thread, setThread] = useState(() => ({
    id,
    start: {
      since: conversation?.createdAt ?? new Date().toISOString(),
      seen: new Set<string>(),
    },
  }));
  if (thread.id === id) return thread.start;
  const born = conversation !== null && thread.id === null;
  const start: ThreadStart = born
    ? {
        since:
          thread.start.since < conversation.createdAt ? thread.start.since : conversation.createdAt,
        seen: thread.start.seen,
      }
    : { since: conversation?.createdAt ?? new Date().toISOString(), seen: new Set() };
  setThread({ id, start });
  return start;
}
