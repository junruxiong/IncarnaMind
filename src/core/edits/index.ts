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
 *    Proposals. A reply that drops a Citation leaves it in a Proposal, for
 *    the User to see before accepting. If the User edited the passage while
 *    the model worked, the Proposals are written over the passage as the
 *    model read it and merged with the edits (../proposals/merge): those
 *    whose words the User changed arrive out of date (#94).
 *
 * A change out of date can be made again (`regenerateProposal`): the same
 * request, from the text it covers as it is now, the User's edits included.
 *
 * Nothing leaves the computer before the User allows the "edits" flow to the
 * model's service, and in local mode only a model on this computer is used.
 * Each request can be stopped (`cancel`).
 */
import { randomUUID } from "node:crypto";
import { generateText } from "ai";
import type * as Y from "yjs";
import {
  outOfDateProposals,
  pendingProposals,
  proposalRequestOf,
  recordProposalRequest,
} from "../../shared/proposals";
import { createWindowBudget } from "../answers/window";
import {
  type ChatModelChoice,
  type ChatReadiness,
  type EditAction,
  type EditWithAiResult,
  type ExternalService,
  editActions,
  MIND_CONTENT_FIELD,
  type TextPlace,
} from "../api";
import type { Consent } from "../consent";
import { ChatNotReadyError, ConsentDeclinedError, InvalidInputError, isRecord } from "../errors";
import type { MindContent } from "../mindContent";
import { parseMindId } from "../minds";
import { docOf, forkChanges } from "../proposals/merge";
import {
  proposeAfter,
  proposeRewrite,
  readSelection,
  type Selection,
  takeBackForRegenerate,
  type Written,
} from "../proposals/write";
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
  content: Pick<MindContent, "read" | "edit" | "state" | "apply">;
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

/** What a request asks the model to do, in which Mind. */
interface Ask {
  mindId: string;
  action: EditAction;
  instruction: string;
  model: ChatModelChoice | null;
}

interface ParsedInput extends Ask {
  requestId: string;
  from: TextPlace;
  to: TextPlace;
  text: string;
}

interface ParsedRegenerate {
  requestId: string;
  mindId: string;
  proposalId: string;
  model: ChatModelChoice | null;
}

/** Writes a request's Proposals for the selection read again where they go (`propose`). */
type Writer = (propose: (blocks: Y.XmlFragment, selection: Selection) => Written) => Written | null;

export function parsePlace(value: unknown, what: string): TextPlace {
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
  return {
    requestId,
    mindId: parseMindId(input.mindId),
    from: parsePlace(input.from, "from"),
    to: parsePlace(input.to, "to"),
    text,
    action: action as EditAction,
    instruction: words,
    model: parseModel(model),
  };
}

/** The model a request names: the Mind's own; null for the default chat model. */
export function parseModel(model: unknown): ChatModelChoice | null {
  return isRecord(model) &&
    typeof model.providerId === "string" &&
    model.providerId !== "" &&
    typeof model.modelId === "string" &&
    model.modelId !== ""
    ? { providerId: model.providerId, modelId: model.modelId }
    : null;
}

function parseRegenerate(input: unknown): ParsedRegenerate {
  if (!isRecord(input)) throw new InvalidInputError("regenerateProposal expects an object.");
  const { requestId, proposalId } = input;
  if (typeof requestId !== "string" || requestId === "") {
    throw new InvalidInputError("A Regenerate request needs an id.");
  }
  if (typeof proposalId !== "string" || proposalId === "") {
    throw new InvalidInputError("proposalId must be the change's id.");
  }
  return {
    requestId,
    mindId: parseMindId(input.mindId),
    proposalId,
    model: parseModel(input.model),
  };
}

/** Rejects a Proposal in `doc`, in a transaction of its own, and says where its text is now. */
function takeBack(doc: Y.Doc, blocks: Y.XmlFragment, proposalId: string) {
  const taken: { region: ReturnType<typeof takeBackForRegenerate> } = { region: null };
  doc.transact(() => {
    taken.region = takeBackForRegenerate(blocks, proposalId);
  });
  return taken.region;
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

  /**
   * Asks the model to change `selection` as `ask` says, and writes what comes
   * back as Proposals through `write`. `settle` runs when the model gives the
   * text back as it was.
   */
  async function run(
    ask: Ask,
    selection: Selection,
    write: Writer,
    signal: AbortSignal,
    settle?: () => void,
  ): Promise<EditWithAiResult> {
    if (selection.characters > MAX_SELECTION_CHARACTERS) return { kind: "too-long" };

    const choice = ask.model ?? options.defaultModel();
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

    const explain = ask.action === "explain";
    const instructions = explain ? EXPLAIN_INSTRUCTIONS : EDIT_INSTRUCTIONS;
    const prompt = editPrompt({
      action: ask.action,
      instruction: ask.instruction,
      passage: selection.text,
      context: selection.context,
      // Explain writes under the passage, not into it: the words around it don't matter.
      ...(!explain && { before: selection.before, after: selection.after }),
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
    if (!explain && sameText(reply, selection.text)) {
      settle?.();
      return { kind: "unchanged" };
    }

    const fresh = explain ? new Map() : freshCitations(selection, reply, options.recheck);
    const request = randomUUID();
    const written = write((blocks, again) => {
      const result = explain
        ? proposeAfter(blocks, again, reply, request, randomUUID)
        : proposeRewrite(again, reply, request, randomUUID, fresh);
      const doc = blocks.doc;
      if (result.proposals.length > 0 && doc && !proposalRequestOf(doc, request)) {
        recordProposalRequest(doc, request, {
          source: "edit-with-ai",
          providerId: prepared.provider.id,
          modelId: prepared.modelId,
          createdAt: options.now().toISOString(),
          action: ask.action,
          instruction: ask.instruction,
        });
      }
      return result;
    });
    if (!written) return { kind: "changed" };
    if (written.proposals.length === 0) return { kind: "unchanged" };
    // Merged with the User's edits, some may have gone with the text they were in, or be out of date.
    const arrived = content.read(ask.mindId, (blocks) => {
      const stale = outOfDateProposals(blocks);
      const mine = pendingProposals(blocks).filter((each) => each.request === request);
      return { count: mine.length, outOfDate: mine.filter((each) => stale.has(each.id)).length };
    });
    if (arrived.count === 0) return { kind: "changed" };
    return {
      kind: "proposed",
      request,
      proposals: arrived.count,
      droppedCitations: written.droppedCitations,
      outOfDate: arrived.outOfDate,
    };
  }

  /**
   * Writes Proposals into `fork`, a copy of the Mind made from `base`, and
   * merges what it changed into the Mind (see ../proposals/merge); null when
   * there was nothing to write, or the merge would change the User's text.
   */
  function mergeFrom<T>(
    mindId: string,
    base: Uint8Array,
    fork: Y.Doc,
    change: (blocks: Y.XmlFragment) => T | null,
  ): T | null {
    const changed: { result: T | null } = { result: null };
    fork.transact(() => {
      changed.result = change(fork.getXmlFragment(MIND_CONTENT_FIELD));
    });
    if (changed.result === null) return null;
    const update = forkChanges(base, fork, content.state(mindId));
    if (!update) return null;
    content.apply(mindId, update);
    return changed.result;
  }

  async function edit(input: ParsedInput, signal: AbortSignal): Promise<EditWithAiResult> {
    // The Mind as the model reads it: Proposals are written over it if the User edits the text meanwhile.
    const base = content.state(input.mindId);
    const read = content.read(input.mindId, (blocks) =>
      readSelection(blocks, input.from, input.to),
    );
    if (!read.ok) return { kind: read.reason === "has-proposals" ? "has-proposals" : "changed" };
    const { selection } = read;
    if (selection.plain !== input.text) return { kind: "changed" };
    if (selection.text === "") return { kind: "unchanged" };

    const reread = (blocks: Y.XmlFragment) => {
      const again = readSelection(blocks, input.from, input.to);
      return again.ok ? again.selection : null;
    };
    const write: Writer = (propose) => {
      const now = content.read(input.mindId, reread);
      if (now?.plain === selection.plain) {
        return content.edit(input.mindId, (blocks) => {
          const again = reread(blocks);
          return again ? propose(blocks, again) : null;
        });
      }
      // Edited while the model worked: written as the model read it, and merged with the edits.
      const fork = docOf(base);
      try {
        return mergeFrom(input.mindId, base, fork, (blocks) => {
          const then = reread(blocks);
          return then ? propose(blocks, then) : null;
        });
      } finally {
        fork.destroy();
      }
    };
    return run(input, selection, write, signal);
  }

  async function regenerate(
    input: ParsedRegenerate,
    signal: AbortSignal,
  ): Promise<EditWithAiResult> {
    const base = content.state(input.mindId);
    const fork = docOf(base);
    try {
      const blocks = fork.getXmlFragment(MIND_CONTENT_FIELD);
      const ref = pendingProposals(blocks).find((each) => each.id === input.proposalId);
      if (!ref) return { kind: "changed" };
      const made = proposalRequestOf(fork, ref.request);
      if (made?.source !== "edit-with-ai" || !made.action) {
        throw new InvalidInputError("Only a change from Edit with AI can be regenerated.");
      }
      // Taken back in the copy: in the Mind, only once the new changes come, or none are needed.
      const region = takeBack(fork, blocks, input.proposalId);
      if (!region) return { kind: "changed" };
      const { from, to } = region;
      const read = readSelection(blocks, from, to);
      if (!read.ok) return { kind: read.reason === "has-proposals" ? "has-proposals" : "changed" };
      /** The change goes: there is no change to make now. */
      const withdraw = () => {
        mergeFrom(input.mindId, base, fork, () => true);
      };
      if (read.selection.text === "") {
        withdraw();
        return { kind: "unchanged" };
      }
      const write: Writer = (propose) =>
        mergeFrom(input.mindId, base, fork, (forked) => {
          const again = readSelection(forked, from, to);
          return again.ok ? propose(forked, again.selection) : null;
        });
      return await run(
        {
          mindId: input.mindId,
          action: made.action,
          instruction: made.instruction ?? "",
          model: input.model,
        },
        read.selection,
        write,
        signal,
        withdraw,
      );
    } finally {
      fork.destroy();
    }
  }

  /** Runs a request that `cancel` can stop by `requestId`. */
  async function stoppable(
    requestId: string,
    work: (signal: AbortSignal) => Promise<EditWithAiResult>,
  ): Promise<EditWithAiResult> {
    running.get(requestId)?.abort();
    const controller = new AbortController();
    running.set(requestId, controller);
    try {
      return await work(controller.signal);
    } finally {
      if (running.get(requestId) === controller) running.delete(requestId);
    }
  }

  return {
    async editWithAi(raw: unknown): Promise<EditWithAiResult> {
      const input = parseInput(raw);
      options.requireMind(input.mindId);
      return stoppable(input.requestId, (signal) => edit(input, signal));
    },

    async regenerateProposal(raw: unknown): Promise<EditWithAiResult> {
      const input = parseRegenerate(raw);
      options.requireMind(input.mindId);
      return stoppable(input.requestId, (signal) => regenerate(input, signal));
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
