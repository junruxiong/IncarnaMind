import { useAnswers } from "./answers";
import { useT } from "./i18n";

/**
 * The meta line's word on an Answer being written: what it is doing, from the
 * core's "answer.phase" events ("Writing…" until the first one arrives), with
 * the pulsing accent dot; or, when it waits for the User (to approve a call,
 * or to allow the Question to go to the model's service), what it waits for,
 * with the attention dot.
 */
export function AnswerWriting({
  answerId,
  waitingForApproval,
}: {
  answerId: string | null;
  waitingForApproval: boolean;
}) {
  const t = useT();
  const phase = useAnswers((state) => (answerId ? state.phases[answerId] : undefined));
  const waiting = waitingForApproval || phase === "waiting-for-consent";
  return (
    <span
      data-testid="answer-writing"
      data-phase={phase ?? "writing"}
      className={`answer-writing ${waiting ? "answer-writing--waiting" : ""}`}
    >
      {waitingForApproval
        ? t("approvals.answer.waiting")
        : t(phase ? `answer.phase.${phase}` : "answer.status.streaming")}
    </span>
  );
}
