import type { ConversationAnswer } from "../../../../core/api";
import { mindName } from "../../../../shared/naming";
import { notWrittenReason, waitingProblemOf } from "../../changeProblems";
import { useT } from "../../i18n";
import { showWaitingChanges, useWaitingIn } from "../../mindChanges";
import { useAppStore } from "../../store";
import { ProblemLine } from "../ProblemLine";

/**
 * Under an Answer that couldn't change a Mind because changes wait there:
 * how many, and Review, which shows the Mind at its first change. It goes
 * once none wait.
 */
export function WaitingChangesLine({ answer }: { answer: Pick<ConversationAnswer, "toolCalls"> }) {
  const t = useT();
  const problem = waitingProblemOf(answer);
  const mind = useAppStore((state) =>
    problem ? state.minds.find((each) => each.id === problem.mindId) : undefined,
  );
  const left = useWaitingIn(mind ? mind.id : null);
  if (!problem || !mind || left === 0) return null;
  const name = mindName(mind) || t("mind.untitled");
  const count = typeof left === "number" ? left : problem.count;
  const text =
    count === 0
      ? t("mindChanges.waiting.some", { mind: name })
      : count === 1
        ? t("mindChanges.waiting.one", { mind: name })
        : t("mindChanges.waiting", { mind: name, count });
  return (
    <ProblemLine
      role="status"
      testId="answer-waiting-changes"
      action={{
        label: t("mindChanges.review"),
        testId: "answer-waiting-review",
        onClick: () => showWaitingChanges(problem.mindId),
      }}
    >
      {text}
    </ProblemLine>
  );
}

/**
 * In place of a reply that claimed a change to a Mind none of its Tool calls
 * made, even once the model was asked again (#1559): "Nothing was written:
 * {reason} · Try again", the reason a write call gave for changing nothing,
 * else that none ran. An Answer that did change a Mind shows its card instead.
 */
export function NotWrittenLine({
  answer,
  onRetry,
}: {
  answer: Pick<ConversationAnswer, "steps" | "toolCalls">;
  onRetry(): void;
}) {
  const t = useT();
  const reason = notWrittenReason(t, answer);
  if (reason === null) return null;
  return (
    <ProblemLine
      role="status"
      testId="answer-not-written"
      action={{
        label: t("answer.notWritten.retry"),
        testId: "answer-not-written-retry",
        onClick: onRetry,
      }}
    >
      {t("answer.notWritten", { reason })}
    </ProblemLine>
  );
}
