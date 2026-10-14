import { useState } from "react";
import { firstReset, nextPlanModel } from "../../../shared/planLimits";
import { ProblemLine } from "../components/ProblemLine";
import { core } from "../core";
import { formatResetTime } from "../cost";
import { useLanguage, useT } from "../i18n";
import { useModelNames } from "../modelNames";
import { usePlanLimits } from "../planLimits";
import { useAppStore } from "../store";

/**
 * An Answer the ChatGPT plan's usage limit stopped (DESIGN.md, Problems): which
 * model, until when, and one action, "Answer with" the plan's next model with
 * limit left, which asks the same Question again in place. With none left,
 * when the first limit resets, and nothing to click.
 */
export function PlanLimitLine({
  modelId,
  providerId,
  details,
  onAnswerAgain,
}: {
  modelId: string;
  providerId: string | null;
  details: string | null;
  /** Asks the Question again; resolves once it is asked, or refused. */
  onAnswerAgain(): Promise<void>;
}) {
  const t = useT();
  const [asking, setAsking] = useState(false);
  const language = useLanguage();
  const nameOf = useModelNames();
  const plan = usePlanLimits();
  if (!plan) return null;
  const name = (id: string) => nameOf(id, providerId).name;
  const resetsAt = plan.limits.usedUp.find((limit) => limit.modelId === modelId)?.resetsAt;
  const next = nextPlanModel(plan.models, plan.limits, modelId);
  const first = firstReset(plan.limits);
  const line = next
    ? resetsAt
      ? t("answer.error.planModel", {
          model: name(modelId),
          time: formatResetTime(resetsAt, language),
        })
      : t("answer.error.planModel.forNow", { model: name(modelId) })
    : first
      ? t("answer.error.planModels", { time: formatResetTime(first, language) })
      : t("answer.error.planModels.forNow");
  const answerWith = next && {
    label: t("answer.error.answerWith", { model: name(next.id) }),
    testId: "answer-error-action",
    disabled: asking,
    onClick: () => {
      setAsking(true);
      core
        .answerWithPlanModel({ insteadOf: modelId, modelId: next.id })
        .then(onAnswerAgain)
        .catch(useAppStore.getState().reportError)
        .finally(() => setAsking(false));
    },
  };
  return (
    <div contentEditable={false} data-testid="answer-error" data-error-kind="plan-limit">
      <ProblemLine
        action={answerWith ?? undefined}
        title={details ?? undefined}
        className="mt-1 select-none"
      >
        {line}
      </ProblemLine>
    </div>
  );
}
