/**
 * The Run engine seam (docs/designs/agent-extensibility.md §4.5, ADR-0013):
 * the Tool-calling loop of one Run behind a small interface, so the library
 * that runs it can change without anything above it. AI SDK 7 implements it
 * today (see ./aiSdkEngine); the engine bake-off on `prototype/engine-bakeoff`
 * ran its contestants behind this same interface (ADR-0007), and every engine
 * must pass the contract suite (tests/core/runEngine.test.ts).
 *
 * Nothing here imports an agent library. The only library type that crosses
 * is the model layer's `ChatLanguageModel`. Above the engine stay the kinds
 * of Run (an Answer today), Tools and their providers, Effects and approvals
 * (decided in `gate`), the window's policy, storage and the UI.
 */
import type { ProviderError } from "../api";
import type { ChatLanguageModel } from "../providers/models";
import type { Tool } from "../tools";

/** Runs the Tool-calling loop for one Run. */
export interface RunEngine {
  /**
   * Streams one Run. Never throws: failures, and refusals before any output,
   * are events. When `signal` aborts, the stream ends at once, with neither
   * "finished" nor "failed".
   */
  run(request: RunRequest): AsyncIterable<RunEvent>;
}

/**
 * What an engine needs of a Tool (CONTEXT.md): its name, description and JSON
 * Schema, and its call. Its provider and Effects stay above the engine.
 *
 * A call's arguments are checked against `inputSchema` first, leniently,
 * taken as a small model meant them (a number or true/false sent as text, an
 * array or an object sent as JSON text, null for an optional one); the Tool
 * gets them as the model sent them. Arguments that don't match never reach
 * `gate` or the Tool: the call fails, and the model reads why.
 */
export type RunTool = Pick<Tool, "name" | "description" | "inputSchema" | "call">;

/** A call of a Tool, under the id the model gave it. */
export interface RunToolCall {
  id: string;
  tool: string;
  input: Record<string, unknown>;
}

/**
 * What `gate` decided. "run": the Tool is called. Otherwise `result` is what
 * the model reads instead, e.g. that the User denied the call: an ordinary
 * Tool result, not an error (its event and message have `ok: true`; whoever
 * denied it knows, and records that).
 */
export type GateDecision = { run: true } | { run: false; result: string };

/**
 * A model's context window (see ../answers/window): how a Run is kept within
 * it. Its policy is the caller's; the engine only applies it.
 */
export interface RunWindow {
  /**
   * Fits what the model will read of a call of `tool` (a Tool's result, or
   * what `gate` gave instead) to the room left, before it enters the history.
   */
  fitResult(text: string, tool: string): string;
  /** Asked before each model call: false offers it no Tools, so it must write. */
  canCallTools(): boolean;
  /** A model call ended, with what its provider counted; before its "step-finished" event. */
  stepFinished(usage: TokenUsage): void;
  /**
   * Before each model call after the first: the history to send instead of
   * `history` (e.g. old Tool results elided, or a summary). The Run's own
   * history is unchanged ("finished" has it whole), so each call is compacted
   * afresh from it. A window that compacts restarts its account of the room
   * left from what it returns. Left out, the history goes as it is (v1).
   */
  compact?(history: readonly RunMessage[]): readonly RunMessage[];
}

export interface RunRequest {
  /** From the model layer (`Core.prepareChatModel`): consent already handled. */
  model: ChatLanguageModel;
  instructions: string;
  /**
   * The history, oldest first, in IncarnaMind's own form. It may end in a
   * Tool result (a resumed Run): the engine continues from it.
   */
  messages: readonly RunMessage[];
  /** Offered by name, in this order; each name once. */
  tools: readonly RunTool[];
  /** Model calls at most. The last one is offered no Tools: it must write. */
  maxSteps: number;
  /**
   * Awaited before every Tool call, while the User may be asked: the Run's
   * approvals decide here, from the call's Effects (see ../approvals). Each
   * call is gated as it arrives, so the calls of one step can wait at once.
   * The engine stops waiting when `signal` aborts, and the Tool isn't called.
   * Rejecting fails the call as if the Tool had thrown: the model reads why.
   */
  gate(call: RunToolCall): Promise<GateDecision>;
  /**
   * Messages added while the Run goes on (e.g. the User's), asked for before
   * each model call after the first, so after the current Tool calls. Each is
   * sent once, where it was taken, and stays in the history after.
   */
  steering?(): readonly RunMessage[];
  /** The model's window, when it is known (a local model's, or a cloud model's own). */
  window?: RunWindow;
  /** Left out: the provider's default. A refusal comes back as a "refused" event. */
  temperature?: number;
  signal: AbortSignal;
}

/**
 * A message of a Run, in IncarnaMind's form, whichever engine ran it: what a
 * later Task's journal stores. A Tool result with `ok: false` was read by the
 * model as an error: the call failed.
 */
export type RunMessage =
  | { role: "user"; text: string }
  | { role: "assistant"; text: string; toolCalls: readonly RunToolCall[] }
  | { role: "tool"; id: string; tool: string; result: string; ok: boolean };

/**
 * What a Run streams. A model call's text, Tool calls and their results come
 * before its "step-finished", so whoever reads them knows what the call did
 * (an Answer takes text before a search as a preamble) when it ends.
 */
export type RunEvent =
  | { type: "text-delta"; text: string }
  /**
   * The model asked for a Tool. Its `gate`, then its call, run alongside the
   * stream, so they may begin before this event is read.
   */
  | ({ type: "tool-call" } & RunToolCall)
  /** What the model read of the call: `ok: false` if it failed (see `RunMessage`). */
  | { type: "tool-result"; id: string; ok: boolean }
  /** A model call ended; the text since the last one belongs to it. */
  | { type: "step-finished"; usage: TokenUsage }
  /** Done: the messages this Run added, oldest first, steering included. Nothing follows. */
  | { type: "finished"; messages: readonly RunMessage[] }
  /**
   * Before any output, the provider refused the Tools offered, the temperature
   * sent, or (with a `window`) a request too long for it, with the request's
   * size when the provider said. The caller decides whether to try again
   * otherwise. Nothing follows.
   */
  | { type: "refused"; what: "tools" | "temperature" | "too-long"; promptTokens?: number }
  /** The model or its provider failed. Nothing follows. */
  | { type: "failed"; error: ProviderError };

/** What a provider counted of a model call. */
export interface TokenUsage {
  inputTokens?: number;
  outputTokens?: number;
}
