import type { ChatGptPlanLimits, ChatGptPlanModel } from "../core/api";

/**
 * The ChatGPT plan's first model, in the plan's own order, whose limit isn't
 * used up, passing over `failed`; null when none is left.
 */
export function nextPlanModel(
  models: readonly ChatGptPlanModel[],
  limits: ChatGptPlanLimits,
  failed: string,
): ChatGptPlanModel | null {
  const usedUp = new Set(limits.usedUp.map((limit) => limit.modelId));
  return models.find((model) => model.id !== failed && !usedUp.has(model.id)) ?? null;
}

/** When the plan's first used-up limit resets (ISO 8601); null when the plan said for none. */
export function firstReset(limits: ChatGptPlanLimits): string | null {
  const times = limits.usedUp.flatMap((limit) => (limit.resetsAt ? [limit.resetsAt] : []));
  return times.sort()[0] ?? null;
}
