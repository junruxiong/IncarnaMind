/**
 * The ChatGPT plan's usage limits as this session meets them: which models
 * the plan refused for their limit, until when, and the stand-in the User
 * chose with "Answer with …". Kept in memory only, so a new session starts
 * with none; the plan says again when a model is used up.
 */
import { type LanguageModelMiddleware, wrapLanguageModel } from "ai";
import type { ChatGptPlanLimits, ChatGptPlanModel } from "../../api";
import type { ChatLanguageModel } from "../models";
import { classifyProviderError } from "../providerErrors";

/** With no reset time from the plan, the chosen model is tried again this long after its last try. */
export const RETRY_WITHOUT_RESET_MS = 60 * 60_000;

type Wrapped = Parameters<NonNullable<LanguageModelMiddleware["wrapStream"]>>[0];
type StreamPart =
  Awaited<ReturnType<Wrapped["doStream"]>>["stream"] extends ReadableStream<infer Part>
    ? Part
    : never;

/**
 * The plan's first model, in the plan's own order, whose limit isn't used
 * up, passing over `failed`; null when none is left.
 */
export function nextPlanModel(
  models: readonly ChatGptPlanModel[],
  limits: ChatGptPlanLimits,
  failed: string,
): ChatGptPlanModel | null {
  const usedUp = new Set(limits.usedUp.map((limit) => limit.modelId));
  return models.find((model) => model.id !== failed && !usedUp.has(model.id)) ?? null;
}

/** When the first used-up limit resets (ISO 8601); null when the plan said for none. */
export function firstReset(limits: ChatGptPlanLimits): string | null {
  const times = limits.usedUp.flatMap((limit) => (limit.resetsAt ? [limit.resetsAt] : []));
  return times.sort()[0] ?? null;
}

export type PlanLimits = ReturnType<typeof createPlanLimits>;

export function createPlanLimits(options: {
  /** Milliseconds since the epoch. */
  now: () => number;
  onChange: () => void;
}) {
  const { now, onChange } = options;
  /** Each used-up model's reset time (epoch milliseconds), or null when the plan didn't say. */
  const usedUp = new Map<string, number | null>();
  let standIn: { insteadOf: string; modelId: string; triedAt: number } | null = null;

  /** Forgets limits whose reset time has passed, and a stand-in whose model can answer again. */
  const expire = () => {
    for (const [modelId, resetsAt] of usedUp) {
      if (resetsAt !== null && now() >= resetsAt) usedUp.delete(modelId);
    }
    if (standIn && !usedUp.has(standIn.insteadOf)) standIn = null;
  };

  const reached = (modelId: string, resetsAt: number | null) => {
    expire();
    usedUp.set(modelId, resetsAt ?? usedUp.get(modelId) ?? null);
    if (standIn?.insteadOf === modelId) standIn.triedAt = now();
    onChange();
  };

  const answered = (modelId: string) => {
    if (!usedUp.delete(modelId)) return;
    if (standIn?.insteadOf === modelId) standIn = null;
    onChange();
  };

  const heard = (modelId: string, error: unknown) => {
    const failure = classifyProviderError(error);
    if (failure.kind !== "plan-limit") return;
    reached(modelId, failure.resetsAt ? Date.parse(failure.resetsAt) : null);
  };

  const iso = (at: number | null) => (at === null ? null : new Date(at).toISOString());

  return {
    snapshot(): ChatGptPlanLimits {
      expire();
      return {
        usedUp: [...usedUp].map(([modelId, resetsAt]) => ({ modelId, resetsAt: iso(resetsAt) })),
        standIn: standIn && {
          insteadOf: standIn.insteadOf,
          modelId: standIn.modelId,
          until: iso(usedUp.get(standIn.insteadOf) ?? null),
        },
      };
    },

    /**
     * The model a Question for `modelId` goes to, and whether `modelId` is due
     * a try first (no reset time, and an hour since the last): the caller
     * tries it, and its answer or refusal settles the stand-in.
     */
    standInFor(modelId: string): { modelId: string; retry: boolean } | null {
      expire();
      if (standIn?.insteadOf !== modelId) return null;
      const retry =
        usedUp.get(modelId) === null && now() - standIn.triedAt >= RETRY_WITHOUT_RESET_MS;
      if (retry) standIn.triedAt = now();
      return { modelId: standIn.modelId, retry };
    },

    answerWith(failed: string, modelId: string): void {
      expire();
      const insteadOf = standIn?.modelId === failed ? standIn.insteadOf : failed;
      // An Answer stopped in an earlier session: its model counts as used up until it answers.
      if (!usedUp.has(failed)) usedUp.set(failed, null);
      standIn = modelId === insteadOf ? null : { insteadOf, modelId, triedAt: now() };
      onChange();
    },

    endStandIn(): void {
      if (!standIn) return;
      standIn = null;
      onChange();
    },

    /** Another account signed in, or none: what was known of the last one's limits doesn't hold. */
    clear(): void {
      if (usedUp.size === 0 && !standIn) return;
      usedUp.clear();
      standIn = null;
      onChange();
    },

    /** The plan's model, telling this its limit reached (and when it resets) and each answer it gives. */
    watch(model: ChatLanguageModel, modelId: string): ChatLanguageModel {
      return wrapLanguageModel({
        model,
        middleware: {
          specificationVersion: "v4",
          wrapGenerate: async ({ doGenerate }) => {
            try {
              const result = await doGenerate();
              answered(modelId);
              return result;
            } catch (error) {
              heard(modelId, error);
              throw error;
            }
          },
          wrapStream: async ({ doStream }) => {
            let result: Awaited<ReturnType<typeof doStream>>;
            try {
              result = await doStream();
            } catch (error) {
              heard(modelId, error);
              throw error;
            }
            const reader = result.stream.getReader();
            let failed = false;
            const stream = new ReadableStream<StreamPart>({
              async pull(controller) {
                try {
                  const { done, value } = await reader.read();
                  if (done) {
                    controller.close();
                    return;
                  }
                  if (value.type === "error") {
                    failed = true;
                    heard(modelId, value.error);
                  } else if (value.type === "finish" && !failed) {
                    answered(modelId);
                  }
                  controller.enqueue(value);
                } catch (error) {
                  heard(modelId, error);
                  controller.error(error);
                }
              },
              cancel: (reason) => reader.cancel(reason),
            });
            return { ...result, stream };
          },
        },
      });
    },
  };
}
