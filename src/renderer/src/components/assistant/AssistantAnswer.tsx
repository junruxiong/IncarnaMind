import { type ReactNode, useMemo, useState } from "react";
import type { ConversationAnswer, ConversationQuestion } from "../../../../core/api";
import { answerBlocks, hasText } from "../../answerBlocks";
import { AnswerWriting } from "../../answerPhase";
import { useAnswers } from "../../answers";
import { useApprovals, waitingFor } from "../../approvals";
import { scopeLine } from "../../assistantContext";
import { NO_SCOPE } from "../../composer";
import { regenerateAnswer } from "../../conversations";
import {
  AnswerError,
  ApprovalCard,
  CallRows,
  FiguresSummary,
  figuresOf,
  SEARCH,
  SearchesSummary,
  SearchList,
  UncalculatedLine,
} from "../../editor/AnswerView";
import { AskProblem } from "../../editor/AskProblem";
import { useT } from "../../i18n";
import { scopeChips } from "../../scope";
import { useAppStore } from "../../store";
import { AnswerActions } from "./AnswerActions";
import { AnswerText } from "./AnswerText";

/** What adding an Answer needs from the thread around it. */
export interface AnswerKeeping {
  /** The name of the Mind Add to note adds to; null with no Mind in focus (the ▾ opens). */
  focusedMindName: string | null;
  add(answer: ConversationAnswer): void;
  pick(answer: ConversationAnswer, anchor: HTMLElement): void;
}

/**
 * An Answer in the assistant (#201): no box, as an Answer in a Mind has none
 * (DESIGN.md, Answer), in the compact reading sizes. The Skills,
 * Connector calls and scripts it used above its text; its text with its
 * Citations and their checks in the margin; what went wrong, and the
 * approvals it waits for, under it; then its actions (`AnswerActions`), with
 * its model, its Question's Search scope and its searches.
 */
export function AssistantAnswer({
  answer,
  question,
  sought,
  keeping,
}: {
  answer: ConversationAnswer;
  question: ConversationQuestion | undefined;
  /** A link back from a Note led here: it is flashed. */
  sought: boolean;
  keeping: AnswerKeeping;
}) {
  const t = useT();
  const { text, citations, figures, status } = answer;
  const blocks = useMemo(
    () => answerBlocks({ text, citations, figures, status }),
    [text, citations, figures, status],
  );
  const writing = useAnswers((state) => state.writing.has(answer.id));
  const streaming = writing || status === "streaming";
  const problem = useAnswers((state) => state.blocked[answer.id]);
  const waiting = useApprovals((state) => state.waiting);
  const approvals = useMemo(() => waitingFor(waiting, answer.id), [waiting, answer.id]);
  const openSettings = useAppStore((state) => state.openSettings);
  const regenerate = () => void regenerateAnswer(answer.id);
  const scope = useScopeLine(question);
  const [searchesOpen, setSearchesOpen] = useState(false);
  const searches = answer.toolCalls.filter((call) => call.tool === SEARCH);
  // Written again, it starts with its searches closed.
  if (searchesOpen && searches.length === 0) setSearchesOpen(false);
  const written = hasText(blocks);
  const { calculated, uncalculated } = figuresOf(answer.toolCalls);

  const meta: ReactNode[] = [];
  if (status === "stopped") {
    meta.push(
      <span data-testid="answer-stopped" className="assistant-answer-status">
        {t("answer.status.stopped")}
      </span>,
    );
  }
  if (answer.model) {
    meta.push(
      <span data-testid="answer-model" className="answer-model">
        {answer.model.modelId}
      </span>,
    );
  }
  meta.push(<span data-testid="answer-scope">{scope}</span>);
  if (searches.length > 0) {
    meta.push(
      <SearchesSummary
        searches={searches}
        open={searchesOpen}
        onToggle={() => setSearchesOpen((open) => !open)}
      />,
    );
  }
  if (calculated > 0) meta.push(<FiguresSummary count={calculated} />);

  return (
    <article
      data-testid="assistant-answer"
      data-answer-id={answer.id}
      data-status={streaming ? "streaming" : status}
      data-sought={sought || undefined}
      aria-label={t("answer.label")}
      className="assistant-answer"
    >
      <CallRows calls={answer.toolCalls} approvals={approvals} />
      {answer.citationSupport === "none" && (
        <p data-testid="answer-no-citations" className="answer-info">
          {t("citation.unsupported")}
        </p>
      )}
      {(written || streaming) && (
        <AnswerText
          answerId={answer.id}
          blocks={blocks}
          label={t("answer.label")}
          regenerate={regenerate}
        />
      )}
      {!streaming && uncalculated > 0 && <UncalculatedLine count={uncalculated} />}
      {searchesOpen && (
        <div className="answer-searches-open">
          <SearchList searches={searches} />
        </div>
      )}
      {!streaming && status === "failed" && (
        <AnswerError
          kind={answer.errorKind ?? "unknown"}
          modelId={answer.model?.modelId ?? null}
          providerId={answer.model?.providerId ?? null}
          // A conversation doesn't keep when a plan's limit resets: the line says so without it.
          resetsAt={null}
          details={answer.errorMessage}
          onRetry={regenerate}
          onOpenSettings={openSettings}
        />
      )}
      {problem && (
        <AskProblem
          problem={problem}
          testIds={{
            notReady: "answer-not-ready",
            skillUnavailable: "answer-skill-unavailable",
            error: "answer-ask-error",
          }}
        />
      )}
      {approvals.length > 0 && (
        <div className="answer-approvals">
          {approvals.map((request) => (
            <ApprovalCard key={request.requestId} request={request} />
          ))}
        </div>
      )}
      <AnswerActions
        answer={answer}
        streaming={streaming}
        canKeep={written}
        writing={<AnswerWriting answerId={answer.id} waitingForApproval={approvals.length > 0} />}
        meta={meta}
        focusedMindName={keeping.focusedMindName}
        onAdd={() => keeping.add(answer)}
        onPick={(anchor) => keeping.pick(answer, anchor)}
      />
    </article>
  );
}

/** What an Answer's Question searched: "“Thesis”", or All Documents. */
function useScopeLine(question: ConversationQuestion | undefined): string {
  const t = useT();
  const folders = useAppStore((state) => state.folders);
  const groups = useAppStore((state) => state.library?.groups);
  const deletedGroups = useAppStore((state) => state.library?.deletedGroups);
  const tags = useAppStore((state) => state.tags);
  const documents = useAppStore((state) => state.documents);
  const chips = scopeChips(
    { folders, groups, deletedGroups, tags, documents },
    question?.scope ?? NO_SCOPE,
  );
  return scopeLine(t, chips);
}
