/**
 * What this session knows of the ChatGPT plan's usage limits, and the plan's
 * models in its order, for the Problem line under an Answer the limit stopped
 * and the model chip's "until" state. Asked for once something shows them.
 */
import { useEffect } from "react";
import { create } from "zustand";
import type { ChatGptPlanStatus, ProviderErrorKind } from "../../core/api";
import { useAnswers } from "./answers";
import { core } from "./core";
import { formatResetTime } from "./cost";
import { useLanguage } from "./i18n";

type PlanLimitsState = Pick<ChatGptPlanStatus, "models" | "limits">;

const usePlanStatus = create<{ plan: PlanLimitsState | null }>()(() => ({ plan: null }));

const keep = ({ models, limits }: ChatGptPlanStatus) =>
  usePlanStatus.setState({ plan: { models, limits } });

let asked = false;
const ask = () => {
  asked = true;
  core.getChatGptPlan().then(keep, (error: unknown) => {
    asked = false;
    console.error("The ChatGPT plan's limits couldn't be read.", error);
  });
};

core.on("chatGptPlan.changed", keep);

export function usePlanLimits(): PlanLimitsState | null {
  useEffect(() => {
    if (!asked) ask();
  }, []);
  return usePlanStatus((state) => state.plan);
}

/**
 * The plan's stand-in while it lasts, with when it ends as the User reads a
 * time (`untilTime`, null when the plan didn't say). At that time it ends,
 * and the chip shows the chosen model again.
 */
export function usePlanStandIn(): {
  insteadOf: string;
  modelId: string;
  untilTime: string | null;
} | null {
  const language = useLanguage();
  const standIn = usePlanLimits()?.limits.standIn ?? null;
  const until = standIn?.until ?? null;
  useEffect(() => {
    if (!until) return;
    const timer = setTimeout(ask, Math.max(0, Date.parse(until) - Date.now()));
    return () => clearTimeout(timer);
  }, [until]);
  if (!standIn || (until && Date.parse(until) <= Date.now())) return null;
  return {
    insteadOf: standIn.insteadOf,
    modelId: standIn.modelId,
    untilTime: until && formatResetTime(until, language),
  };
}

/**
 * Whether an Answer that failed with `kind` was stopped by the ChatGPT plan's
 * usage limit: an API key's rate limit, or Ollama Cloud's allowance, keeps its own Problem.
 */
export function useStoppedByPlanLimit(
  kind: ProviderErrorKind | null,
  providerId: string | null,
): boolean {
  const limit = kind === "plan-limit";
  const onPlan = useAnswers(
    (state) =>
      state.models?.find((group) => group.provider.id === providerId)?.provider.kind === "chatgpt",
  );
  useEffect(() => {
    if (limit) useAnswers.getState().loadModels();
  }, [limit]);
  return limit && onPlan;
}
