import type { ConversationAnswer } from "../../../../core/api";
import { mindName } from "../../../../shared/naming";
import { unbackedClaimKey, waitingProblemOf } from "../../changeProblems";
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
 * Under a reply that claims a change to a Mind that none of its Tool calls
 * made, even once the model was asked again: it says so, with Regenerate.
 */
export function UnbackedClaimLine({
  answer,
  onRegenerate,
}: {
  answer: Pick<ConversationAnswer, "steps">;
  onRegenerate(): void;
}) {
  const t = useT();
  const key = unbackedClaimKey(answer.steps?.unbackedClaims);
  if (!key) return null;
  return (
    <ProblemLine
      role="status"
      testId="answer-unbacked-claim"
      action={{
        label: t("answer.regenerate"),
        testId: "answer-unbacked-regenerate",
        onClick: onRegenerate,
      }}
    >
      {t(key)}
    </ProblemLine>
  );
}
