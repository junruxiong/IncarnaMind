import type { ConversationAnswer } from "../../../../core/api";
import { mindName } from "../../../../shared/naming";
import { unbackedClaimKey, waitingProblemOf } from "../../changeProblems";
import { useT } from "../../i18n";
import { showWaitingChanges, useWaitingIn } from "../../mindChanges";
import { useAppStore } from "../../store";
import { useAnswerLine } from "../../undo";
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

/**
 * Under the Answer of a model that can't call Tools, to a request to write
 * into a Mind or change one: it says nothing was written, and Add to Mind puts
 * the Answer there, as the Answer's own Add to note does. It goes while the
 * Answer's added line shows.
 */
export function NoMindToolsLine({
  answerId,
  onAdd,
}: {
  answerId: string;
  onAdd(button: HTMLElement): void;
}) {
  const t = useT();
  const added = useAnswerLine(answerId);
  if (added) return null;
  return (
    <ProblemLine
      role="status"
      testId="answer-no-mind-tools"
      action={{
        label: t("answer.noMindTools.add"),
        testId: "answer-no-mind-tools-add",
        onClick: (event) => onAdd(event.currentTarget),
      }}
    >
      {t("answer.noMindTools")}
    </ProblemLine>
  );
}
