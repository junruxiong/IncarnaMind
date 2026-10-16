import type { ConversationAnswer } from "../../../../core/api";
import { waitText } from "../../answerWaitWords";
import { stopAnswer } from "../../conversations";
import { useT } from "../../i18n";
import { useModelNames } from "../../modelNames";

/**
 * Under the live line of an Answer with no words after 15 seconds: how long
 * its model usually takes, or that it is still working, with Stop (#899).
 */
export function AnswerWait({
  answer,
  usual,
}: {
  answer: ConversationAnswer;
  usual: number | null;
}) {
  const t = useT();
  const nameOf = useModelNames();
  const model = answer.model ? nameOf(answer.model.modelId, answer.model.providerId).name : null;
  return (
    <p data-testid="answer-wait" aria-live="polite" className="answer-steps-wait">
      {waitText(t, usual, model)}
      <span className="whitespace-nowrap">
        <span aria-hidden="true"> · </span>
        <button
          type="button"
          data-testid="answer-stop"
          onClick={() => stopAnswer(answer.id)}
          className="answer-steps-wait-stop"
        >
          {t("answer.stop")}
        </button>
      </span>
    </p>
  );
}
