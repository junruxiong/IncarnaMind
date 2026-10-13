import { useMemo } from "react";
import type { ConversationAnswer, ConversationQuestion } from "../../../../core/api";
import { answerBlocks, hasText } from "../../answerBlocks";
import { useAnswers } from "../../answers";
import { useApprovals, waitingFor } from "../../approvals";
import { regenerateAnswer } from "../../conversations";
import { AnswerError, ApprovalCard, figuresOf, UncalculatedLine } from "../../editor/AnswerView";
import { AskProblem } from "../../editor/AskProblem";
import { useT } from "../../i18n";
import { useAppStore } from "../../store";
import { useUnreadFiles } from "../../unreadFiles";
import { UnreadFilesLine } from "../UnreadFilesLine";
import { AnswerActions } from "./AnswerActions";
import { AnswerSteps } from "./AnswerSteps";
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
 * (DESIGN.md, Answer), in the assistant's reading sizes. Its steps first
 * (#391), folded to one line; its text with its Citations and their checks
 * in the margin; what went wrong, and the approvals it waits for, under it;
 * then its actions (`AnswerActions`).
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
  const written = hasText(blocks);
  const { uncalculated } = figuresOf(answer.toolCalls);
  const unread = useUnreadFiles(question?.scope ?? null);

  return (
    <article
      data-testid="assistant-answer"
      data-answer-id={answer.id}
      data-status={streaming ? "streaming" : status}
      data-sought={sought || undefined}
      aria-label={t("answer.label")}
      className="assistant-answer"
    >
      <AnswerSteps
        answer={answer}
        question={question}
        streaming={streaming}
        waitingForApproval={approvals.length > 0}
        onRegenerate={regenerate}
      />
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
      {!streaming && status !== "failed" && <UnreadFilesLine files={unread} />}
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
        focusedMindName={keeping.focusedMindName}
        onAdd={() => keeping.add(answer)}
        onPick={(anchor) => keeping.pick(answer, anchor)}
      />
    </article>
  );
}
