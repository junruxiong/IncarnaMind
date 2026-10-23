/**
 * How long each AI request and each local step took (#1624), for the speed baseline and the
 * budgets every AI flow is held to: per model request, the times to its first byte, first
 * reasoning, first text and first Tool call, and to its end, with its tokens and reasoning
 * setting; per local step (reranking, reading a scanned page, indexing a Document,
 * transcribing a stretch), its time. Each is tagged with its flow and step (./flows). Only
 * numbers and names are kept, never a request's or a reply's text; the records stay in this
 * computer's database, nothing here is sent anywhere, and those older than 30 days go.
 */
import { randomUUID } from "node:crypto";
import type { Database } from "../storage";
import { type FlowTag, flowNow } from "./flows";

const KEPT_DAYS = 30;

export interface ModelRequestTiming {
  dataFlow: string;
  providerKind: string;
  modelId: string;
  /** The reasoning setting it was asked with ("off", "summary", "answer", "careful"); null if none. */
  reasoning: string | null;
}

export type TimingOutcome = "finished" | "failed" | "stopped";

interface Usage {
  inputTokens?: number | undefined;
  cachedInputTokens?: number | undefined;
  outputTokens?: number | undefined;
}

export function createTimings(options: { db: Database; now: () => Date }) {
  const { db, now } = options;

  function insert(row: {
    kind: "model" | "local";
    tag: FlowTag | null;
    dataFlow: string | null;
    providerKind: string | null;
    modelId: string;
    reasoning: string | null;
    startedAt: string;
    firstByteMs: number | null;
    firstReasoningMs: number | null;
    firstTextMs: number | null;
    firstToolMs: number | null;
    endMs: number;
    usage: Usage | null;
    units: number | null;
    outcome: TimingOutcome;
  }): void {
    const at = now().toISOString();
    db.run(
      `INSERT INTO ai_timings (id, kind, flow, step, data_flow, provider_kind, model_id, reasoning,
         started_at, first_byte_ms, first_reasoning_ms, first_text_ms, first_tool_ms, end_ms,
         input_tokens, cached_input_tokens, output_tokens, units, outcome, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        randomUUID(),
        row.kind,
        row.tag?.flow ?? null,
        row.tag?.step ?? null,
        row.dataFlow,
        row.providerKind,
        row.modelId,
        row.reasoning,
        row.startedAt,
        row.firstByteMs,
        row.firstReasoningMs,
        row.firstTextMs,
        row.firstToolMs,
        row.endMs,
        row.usage?.inputTokens ?? null,
        row.usage?.cachedInputTokens ?? null,
        row.usage?.outputTokens ?? null,
        row.units,
        row.outcome,
        at,
        at,
      ],
    );
  }

  return {
    /**
     * A model request starting now, in the flow it runs in: `heard` each part of its reply as it
     * comes, then `end` once, with what the provider counted.
     */
    request(model: ModelRequestTiming) {
      const tag = flowNow();
      const startedAt = now().toISOString();
      const start = performance.now();
      const since = () => Math.round(performance.now() - start);
      let firstByteMs: number | null = null;
      let firstReasoningMs: number | null = null;
      let firstTextMs: number | null = null;
      let firstToolMs: number | null = null;
      let ended = false;
      return {
        heard(type: string): void {
          // The provider's stream opens with "stream-start" before a byte of the reply comes.
          if (type === "stream-start") return;
          firstByteMs ??= since();
          if (type === "reasoning-delta") firstReasoningMs ??= since();
          else if (type === "text-delta") firstTextMs ??= since();
          else if (type === "tool-input-start" || type === "tool-call") firstToolMs ??= since();
        },
        end(outcome: TimingOutcome, usage: Usage | null): void {
          if (ended) return;
          ended = true;
          insert({
            kind: "model",
            tag,
            dataFlow: model.dataFlow,
            providerKind: model.providerKind,
            modelId: model.modelId,
            reasoning: model.reasoning,
            startedAt,
            firstByteMs,
            firstReasoningMs,
            firstTextMs,
            firstToolMs,
            endMs: since(),
            usage,
            units: null,
            outcome,
          });
        },
      };
    },

    /** A step run elsewhere and timed there (local transcription's decoding), as `local` records one. */
    step(step: string, model: string, ms: number, units: number | null): void {
      insert({
        kind: "local",
        tag: flowNow() ?? { flow: step, step: null },
        dataFlow: null,
        providerKind: "local",
        modelId: model,
        reasoning: null,
        startedAt: new Date(now().getTime() - ms).toISOString(),
        firstByteMs: null,
        firstReasoningMs: null,
        firstTextMs: null,
        firstToolMs: null,
        endMs: Math.round(ms),
        usage: null,
        units,
        outcome: "finished",
      });
    },

    /** Times a step run on this computer (`step`, by `model`), with how much it worked through. */
    async local<T>(
      step: string,
      model: string,
      work: () => Promise<T>,
      unitsOf: (result: T) => number | null = () => null,
    ): Promise<T> {
      const tag = flowNow();
      const startedAt = now().toISOString();
      const start = performance.now();
      const record = (outcome: TimingOutcome, units: number | null) =>
        insert({
          kind: "local",
          tag: tag ?? { flow: step, step: null },
          dataFlow: null,
          providerKind: "local",
          modelId: model,
          reasoning: null,
          startedAt,
          firstByteMs: null,
          firstReasoningMs: null,
          firstTextMs: null,
          firstToolMs: null,
          endMs: Math.round(performance.now() - start),
          usage: null,
          units,
          outcome,
        });
      try {
        const result = await work();
        record("finished", unitsOf(result));
        return result;
      } catch (error) {
        record("failed", null);
        throw error;
      }
    },

    /** Removes the records older than 30 days, once a start. */
    prune(): void {
      const before = new Date(now().getTime() - KEPT_DAYS * 86_400_000).toISOString();
      db.run("DELETE FROM ai_timings WHERE started_at < ?", [before]);
    },
  };
}

export type Timings = ReturnType<typeof createTimings>;
