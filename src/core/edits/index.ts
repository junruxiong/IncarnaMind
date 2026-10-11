/**
 * Edit with AI (#93): the User selects text and asks for a change (improve,
 * shorten, translate, explain, or in their own words), and the change comes
 * back as Proposals in the Mind, waiting for them to accept or reject it.
 *
 * 1. The selection is read from the Mind (../proposals/write), with its
 *    Citations as numbered markers, and checked against what the editor read.
 * 2. The chat model (the Mind's, else the default; thinking off, one pass)
 *    gets the passage, the paragraphs around it and the task (./prompt).
 * 3. Every Citation the reply keeps is checked again against its Document; one
 *    whose check comes out differently now comes back as a checked copy.
 * 4. The reply is compared with the passage word by word and written as
 *    Proposals, if the passage is still as it was. A reply that drops a
 *    Citation leaves it in a Proposal, for the User to see before accepting.
 *
 * Nothing leaves the computer before the User allows the "edits" flow to the
 * model's service, and in local mode only a model on this computer is used.
 * Each request can be stopped (`cancel`).
 */
import { randomUUID } from "node:crypto";
import { generateText } from "ai";
import { proposalRequestOf, recordProposalRequest } from "../../shared/proposals";
import { createWindowBudget } from "../answers/window";
import {
  type ChatModelChoice,
  type ChatReadiness,
  type EditAction,
  type EditWithAiResult,
  type ExternalService,
  editActions,
  type TextPlace,
} from "../api";
import type { Consent } from "../consent";
import { ChatNotReadyError, ConsentDeclinedError, InvalidInputError, isRecord } from "../errors";
import type { MindContent } from "../mindContent";
import { parseMindId } from "../minds";
import { proposeAfter, proposeRewrite, readSelection, type Selection } from "../proposals/write";
import type { PreparedChatModel } from "../providers/chat";
import { classifyProviderError } from "../providers/providerErrors";
import {
  EDIT_INSTRUCTIONS,
  EXPLAIN_INSTRUCTIONS,
  editPrompt,
  readEditReply,
  sameText,
} from "./prompt";

/** What the "edits" flow sends: the selected text, with the paragraphs it is in. */
export const EDITS_FLOW_SENDS = ["selection"] as const;

/** The most text one request edits, in characters: more is better done a part at a time. */
export const MAX_SELECTION_CHARACTERS = 12_000;
/** The most a request may say in the User's own words. */
const MAX_INSTRUCTION_CHARACTERS = 2_000;
/** A request that takes longer than this (a local model loading a long text) fails. */
const REQUEST_TIMEOUT_MS = 180_000;

type Attributes = Record<string, unknown>;

export interface EditsOptions {
  content: Pick<MindContent, "read" | "edit">;
  /** Throws unless the Mind exists and this version may write to it. */
  requireMind(mindId: string): void;
  /** The default chat model, used when the Mind has none of its own. */
  defaultModel(): ChatModelChoice | null;
  chat: {
    readiness(choice: ChatModelChoice | undefined, flow: "edits"): Promise<ChatReadiness>;
    prepareModel(choice: ChatModelChoice | undefined, flow: "edits"): Promise<PreparedChatModel>;
    /** The service the default chat model sends to; null with none, or one on this computer. */
    defaultService(): ExternalService | null;
  };
  consent: Consent;
  /** Local mode: only models on this computer. */
  localOnly(): boolean;
  /** A Citation's attributes checked again now, or null when its check comes out the same. */
  recheck(attributes: Attributes): Attributes | null;
  now(): Date;
}

interface ParsedInput {
  requestId: string;
  mindId: string;
  from: TextPlace;
  to: TextPlace;
  text: string;
  action: EditAction;
  instruction: string;
  model: ChatModelChoice | null;
}

function parsePlace(value: unknown, what: string): TextPlace {
  if (
    !isRecord(value) ||
    typeof value.blockId !== "string" ||
    value.blockId === "" ||
    typeof value.offset !== "number" ||
    !Number.isInteger(value.offset) ||
    value.offset < 0
  ) {
    throw new InvalidInputError(`${what} must be a Block's id and an offset into its text.`);
  }
  return { blockId: value.blockId, offset: value.offset };
}

function parseInput(input: unknown): ParsedInput {
  if (!isRecord(input)) throw new InvalidInputError("editWithAi expects an object.");
  const { requestId, text, action, instruction, model } = input;
  if (typeof requestId !== "string" || requestId === "") {
    throw new InvalidInputError("An Edit with AI request needs an id.");
  }
  if (typeof text !== "string") throw new InvalidInputError("text must be the selected text.");
  if (!editActions.some((each) => each === action)) {
    throw new InvalidInputError(`Unknown Edit with AI action "${String(action)}".`);
  }
  const words = typeof instruction === "string" ? instruction.trim() : "";
  if ((action === "translate" || action === "instruction") && words === "") {
    throw new InvalidInputError("Say what to change, or the language to translate into.");
  }
  if (words.length > MAX_INSTRUCTION_CHARACTERS) {
    throw new InvalidInputError("What to change is too long.");
  }
  const chosen =
    isRecord(model) &&
    typeof model.providerId === "string" &&
    model.providerId !== "" &&
    typeof model.modelId === "string" &&
    model.modelId !== ""
      ? { providerId: model.providerId, modelId: model.modelId }
      : null;
  return {
    requestId,
    mindId: parseMindId(input.mindId),
    from: parsePlace(input.from, "from"),
    to: parsePlace(input.to, "to"),
    text,
    action: action as EditAction,
    instruction: words,
    model: chosen,
  };
}

/** The Citations the reply keeps, by marker, that come back checked differently now. */
function freshCitations(
  selection: Selection,
  reply: string,
  recheck: EditsOptions["recheck"],
): Map<number, Attributes> {
  const fresh = new Map<number, Attributes>();
  const kept = new Set([...reply.matchAll(/\[(\d+)\]/g)].map((match) => Number(match[1])));
  selection.citations.forEach((attributes, index) => {
    if (!kept.has(index + 1)) return;
    const checked = recheck(attributes);
    if (checked) fresh.set(index + 1, checked);
  });
  return fresh;
}

export function createEdits(options: EditsOptions) {
  const { chat, content } = options;
  const running = new Map<string, AbortController>();

  options.consent.registry.register({
    id: "edits",
    sends: EDITS_FLOW_SENDS,
    async services() {
      const service = chat.defaultService();
      return service ? [service] : [];
    },
  });

  const readinessOf = (choice: ChatModelChoice | null) =>
    chat.readiness(choice ?? undefined, "edits");

  const notReady = (readiness: ChatReadiness): EditWithAiResult | null =>
    readiness.ready ? null : { kind: "not-ready", readiness };

  async function edit(input: ParsedInput, signal: AbortSignal): Promise<EditWithAiResult> {
    const read = content.read(input.mindId, (blocks) =>
      readSelection(blocks, input.from, input.to),
    );
    if (!read.ok) return { kind: read.reason === "has-proposals" ? "has-proposals" : "changed" };
    const { selection } = read;
    if (selection.plain !== input.text) return { kind: "changed" };
    if (selection.text === "") return { kind: "unchanged" };
    if (selection.characters > MAX_SELECTION_CHARACTERS) return { kind: "too-long" };

    const choice = input.model ?? options.defaultModel();
    const readiness = await readinessOf(choice);
    const blocked = notReady(readiness);
    if (blocked) return blocked;
    if (readiness.ready && readiness.provider.service && options.localOnly()) {
      return { kind: "local-mode" };
    }

    let prepared: PreparedChatModel;
    try {
      prepared = await chat.prepareModel(choice ?? undefined, "edits");
    } catch (error) {
      if (error instanceof ChatNotReadyError)
        return { kind: "not-ready", readiness: error.readiness };
      if (error instanceof ConsentDeclinedError) {
        return notReady(await readinessOf(choice)) ?? { kind: "cancelled" };
      }
      throw error;
    }
    if (signal.aborted) return { kind: "cancelled" };

    const explain = input.action === "explain";
    const instructions = explain ? EXPLAIN_INSTRUCTIONS : EDIT_INSTRUCTIONS;
    const prompt = editPrompt({
      action: input.action,
      instruction: input.instruction,
      passage: selection.text,
      context: selection.context,
    });
    // The reply is about as long as the passage; a translation into Chinese can count more.
    let maxOutputTokens = Math.min(16_000, Math.max(1_024, Math.ceil(selection.characters * 1.5)));
    if (prepared.window) {
      const budget = createWindowBudget(prepared.window);
      const expected = explain ? 512 : Math.ceil(budget.tokens(selection.text) * 1.5) + 64;
      if (expected > prepared.window.outputTokens) return { kind: "too-long" };
      if (budget.roomBeside([instructions, prompt], expected) < 0) return { kind: "too-long" };
      maxOutputTokens = prepared.window.outputTokens;
    }

    let replyText: string;
    try {
      const result = await generateText({
        model: prepared.model,
        instructions,
        prompt,
        maxOutputTokens,
        maxRetries: 1,
        abortSignal: AbortSignal.any([signal, AbortSignal.timeout(REQUEST_TIMEOUT_MS)]),
      });
      replyText = result.text;
    } catch (error) {
      if (signal.aborted) return { kind: "cancelled" };
      return { kind: "failed", error: classifyProviderError(error) };
    }
    if (signal.aborted) return { kind: "cancelled" };
    const reply = readEditReply(replyText, selection.text);
    if (reply === "") {
      return {
        kind: "failed",
        error: { kind: "provider", message: "The model gave nothing back." },
      };
    }
    if (!explain && sameText(reply, selection.text)) return { kind: "unchanged" };

    const fresh = explain ? new Map() : freshCitations(selection, reply, options.recheck);
    // The text may have been edited while the model worked: it is changed only as it was.
    const now = content.read(input.mindId, (blocks) => readSelection(blocks, input.from, input.to));
    if (!now.ok || now.selection.plain !== selection.plain) return { kind: "changed" };
    const request = randomUUID();
    const written = content.edit(input.mindId, (blocks) => {
      const again = readSelection(blocks, input.from, input.to);
      if (!again.ok) return null;
      const result = explain
        ? proposeAfter(blocks, again.selection, reply, request, randomUUID)
        : proposeRewrite(again.selection, reply, request, randomUUID, fresh);
      const doc = blocks.doc;
      if (result.proposals.length > 0 && doc && !proposalRequestOf(doc, request)) {
        recordProposalRequest(doc, request, {
          source: "edit-with-ai",
          providerId: prepared.provider.id,
          modelId: prepared.modelId,
          createdAt: options.now().toISOString(),
        });
      }
      return result;
    });
    if (!written) return { kind: "changed" };
    if (written.proposals.length === 0) return { kind: "unchanged" };
    return {
      kind: "proposed",
      request,
      proposals: written.proposals.length,
      droppedCitations: written.droppedCitations,
    };
  }

  return {
    async editWithAi(raw: unknown): Promise<EditWithAiResult> {
      const input = parseInput(raw);
      options.requireMind(input.mindId);
      running.get(input.requestId)?.abort();
      const controller = new AbortController();
      running.set(input.requestId, controller);
      try {
        return await edit(input, controller.signal);
      } finally {
        if (running.get(input.requestId) === controller) running.delete(input.requestId);
      }
    },

    cancel(requestId: unknown): void {
      if (typeof requestId !== "string") throw new InvalidInputError("requestId must be text.");
      running.get(requestId)?.abort();
    },

    /** The core is closing: requests still waiting stop. */
    stopAll(): void {
      for (const controller of running.values()) controller.abort();
      running.clear();
    },
  };
}

export type Edits = ReturnType<typeof createEdits>;
