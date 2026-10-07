/**
 * Asking Questions (ADR-0007): the core builds the Question context from the
 * Mind's Yjs document, has the Answer engine stream an Answer, and writes it
 * into the document as it arrives, right after its Question, so every window
 * sees it grow: its text, its Citations (see ./citations), and the searches it
 * ran. It also pushes the Answer event stream for UI state and the evaluation.
 *
 * Everything here depends on the `AnswerEngine` port, not on the AI SDK.
 */
import { randomUUID } from "node:crypto";
import type * as Y from "yjs";
import {
  ANSWER_BLOCK,
  type AnswerAttributes,
  type AnswerToolCall,
  type AskResult,
  BLOCK_ID_ATTRIBUTE,
  type ChatModelChoice,
  type ChatReadiness,
  type CitationSupport,
  type ProviderError,
  QUESTION_BLOCK,
} from "../api";
import { ChatNotReadyError, InvalidInputError, isRecord, NotFoundError } from "../errors";
import type { createEventHub } from "../events";
import type { MindContent } from "../mindContent";
import type { PreparedChatModel } from "../providers/chat";
import { classifyProviderError } from "../providers/providerErrors";
import {
  contentHash,
  createElement,
  findBlock,
  plainText,
  syncContent,
  textAttribute,
  topLevelBlocks,
} from "./blocks";
import {
  type AnswerDocuments,
  createCitationSession,
  withCitations,
  withoutFootnoteDefinitions,
} from "./citations";
import { buildQuestionContext, type QuestionContext } from "./context";
import type { AnswerEngine } from "./engine";
import { markdownToBlocks } from "./markdown";
import { answerInstructions, documentInstructions } from "./prompt";

export type { AnswerDocuments } from "./citations";
export type {
  AnswerEngine,
  AnswerEngineEvent,
  AnswerMessage,
  AnswerRequest,
  AnswerTools,
  CitationRecordInput,
} from "./engine";
export { createAiSdkAnswerEngine } from "./engine";

/**
 * How often a streaming Answer is written into its Mind: often enough to look
 * live, seldom enough not to store one Yjs update per token.
 */
export const ANSWER_WRITE_INTERVAL_MS = 50;

type Outcome = { status: "done" | "stopped" } | { status: "failed"; error: ProviderError };

interface Generation {
  mindId: string;
  /** Writes what has arrived with the final status, now. Later calls do nothing. */
  finish(outcome: Outcome): void;
}

export interface AnswersOptions {
  content: MindContent;
  events: ReturnType<typeof createEventHub>;
  engine: AnswerEngine;
  /** Throws `NotFoundError` unless the Mind exists; returns its id. */
  requireMind(mindId: unknown): string;
  mindExists(mindId: string): boolean;
  providerExists(providerId: string): boolean;
  /** Whether Questions can be asked with this model (the default one when undefined). */
  readiness(choice: ChatModelChoice | undefined): Promise<ChatReadiness>;
  /** The model, once the User has accepted its data flow (see `Core.prepareChatModel`). */
  prepareModel(choice: ChatModelChoice): Promise<PreparedChatModel>;
  /** Searching the User's Documents, and what Citations point to. */
  documents: AnswerDocuments;
  reportError(error: unknown): void;
}

function parseId(value: unknown, what: string): string {
  if (typeof value !== "string" || value === "")
    throw new InvalidInputError(`${what} must be text.`);
  return value;
}

/** The Answer to a Question, wherever it is in the Mind. */
function answerFor(blocks: Y.XmlFragment, questionId: string): Y.XmlElement | null {
  return (
    topLevelBlocks(blocks).find(
      (block) => block.nodeName === ANSWER_BLOCK && block.getAttribute("questionId") === questionId,
    ) ?? null
  );
}

/** The model picked on a Question, if any. */
function pickedModel(question: Y.XmlElement): ChatModelChoice | null {
  const providerId = textAttribute(question, "providerId");
  const modelId = textAttribute(question, "modelId");
  return providerId && modelId ? { providerId, modelId } : null;
}

/** The User has changed the Answer since it was written. */
function isEdited(answer: Y.XmlElement): boolean {
  const hash = textAttribute(answer, "generatedHash");
  return hash !== null && hash !== contentHash(answer);
}

function setAttributes(element: Y.XmlElement, attributes: Partial<AnswerAttributes>): void {
  for (const [key, value] of Object.entries(attributes)) {
    if (value === null || value === undefined) element.removeAttribute(key);
    // biome-ignore lint/suspicious/noExplicitAny: Yjs types attribute values loosely.
    else element.setAttribute(key, value as any);
  }
}

/** Why a model couldn't be prepared, as the kind of error the Answer shows. */
function failureOf(error: unknown): ProviderError {
  if (error instanceof ChatNotReadyError) {
    const { reason } = error.readiness;
    const kind =
      reason === "consent-declined" ? reason : reason === "missing-api-key" ? "auth" : "model";
    return { kind, message: error.message };
  }
  return classifyProviderError(error);
}

export function createAnswers(options: AnswersOptions) {
  const { content, events, engine } = options;
  /** Answers being written, by id. */
  const active = new Map<string, Generation>();

  /** Marks Answers left "streaming" by an earlier run of the app as stopped: nothing writes them any more. */
  const settleOrphans = (mindId: string) => {
    const orphans = content.read(mindId, (blocks) =>
      topLevelBlocks(blocks).some(
        (block) =>
          block.nodeName === ANSWER_BLOCK &&
          block.getAttribute("status") === "streaming" &&
          !active.has(textAttribute(block, BLOCK_ID_ATTRIBUTE) ?? ""),
      ),
    );
    if (!orphans) return;
    content.edit(mindId, (blocks) => {
      for (const block of topLevelBlocks(blocks)) {
        const id = textAttribute(block, BLOCK_ID_ATTRIBUTE) ?? "";
        if (
          block.nodeName === ANSWER_BLOCK &&
          block.getAttribute("status") === "streaming" &&
          !active.has(id)
        ) {
          setAttributes(block, { status: "stopped", generatedHash: contentHash(block) });
        }
      }
    });
  };

  /** How each model gives Citations, learnt from its earlier Answers: by provider and model. */
  const supportByModel = new Map<string, CitationSupport>();
  const modelKey = (model: ChatModelChoice) => `${model.providerId}\n${model.modelId}`;

  function start(input: {
    mindId: string;
    answerId: string;
    questionId: string;
    model: ChatModelChoice;
    context: QuestionContext;
  }): void {
    const { mindId, answerId, model, context } = input;
    const controller = new AbortController();
    /** What the model wrote: Markdown, with Citation markers. */
    let markdown = "";
    let timer: ReturnType<typeof setTimeout> | null = null;
    let finished = false;
    let support: CitationSupport | null = null;
    const toolCalls: AnswerToolCall[] = [];
    const session = createCitationSession(options.documents, {
      onRecord: (marker, citation) =>
        events.emit("answer.citationAdded", { mindId, answerId, marker, citation }),
    });

    /** The Answer's Blocks: its Markdown, with each marker a Citation node. */
    const blocksOf = (final: boolean) =>
      withCitations(
        markdownToBlocks(withoutFootnoteDefinitions(markdown), { streaming: !final }),
        final ? (marker) => session.finalNode(marker) : (marker) => session.streamingNode(marker),
      );

    /** Writes the Answer as it stands; false if it is gone (deleted, or its Mind deleted). */
    const write = (outcome?: Outcome): boolean => {
      if (!options.mindExists(mindId)) return false;
      try {
        return content.edit(mindId, (blocks) => {
          const found = findBlock(blocks, ANSWER_BLOCK, answerId);
          if (!found) return false;
          const { element } = found;
          syncContent(element, blocksOf(outcome !== undefined));
          const calls = toolCalls.length > 0 ? JSON.stringify(toolCalls) : null;
          if (textAttribute(element, "toolCalls") !== calls)
            setAttributes(element, { toolCalls: calls });
          if (support && textAttribute(element, "citationSupport") !== support) {
            setAttributes(element, { citationSupport: support });
          }
          if (outcome) {
            setAttributes(element, {
              status: outcome.status,
              errorKind: outcome.status === "failed" ? outcome.error.kind : null,
              errorMessage: outcome.status === "failed" ? outcome.error.message : null,
              generatedHash: contentHash(element),
            });
          }
          return true;
        });
      } catch (error) {
        options.reportError(error);
        return false;
      }
    };

    const finish = (outcome: Outcome) => {
      if (finished) return;
      finished = true;
      if (timer) clearTimeout(timer);
      active.delete(answerId);
      controller.abort();
      // A search still running when the Answer stopped didn't finish.
      for (const call of toolCalls) {
        if (call.status === "running") call.status = "failed";
      }
      // The check runs once, now; its results are stored with each Citation.
      try {
        session.check();
      } catch (error) {
        options.reportError(error);
      }
      const written = write(outcome);
      if (outcome.status === "failed") {
        events.emit("answer.failed", { mindId, answerId, error: outcome.error });
      } else {
        events.emit("answer.finished", {
          mindId,
          answerId,
          status: written ? outcome.status : "stopped",
          ...session.summary(),
          citationSupport: support,
        });
      }
    };

    const writeSoon = () => {
      if (timer || finished) return;
      timer = setTimeout(() => {
        timer = null;
        if (!finished && !write()) finish({ status: "stopped" });
      }, ANSWER_WRITE_INTERVAL_MS);
    };

    active.set(answerId, { mindId, finish });
    events.emit("answer.started", { mindId, answerId, questionId: input.questionId, model });

    const base = answerInstructions(context.question);
    const run = async () => {
      let prepared: PreparedChatModel;
      try {
        prepared = await options.prepareModel(model);
      } catch (error) {
        finish({ status: "failed", error: failureOf(error) });
        return;
      }
      // Stopped while waiting, e.g. for consent: send nothing.
      if (finished) return;
      const { tools } = session;
      let outcome: Outcome = { status: "stopped" };
      for await (const event of engine.generate({
        instructions: (mode, passages) =>
          [base, documentInstructions(mode, tools.documentCount, passages)]
            .filter(Boolean)
            .join("\n\n"),
        messages: context.messages,
        question: context.question,
        model: prepared.model,
        tools,
        support: supportByModel.get(modelKey(model)),
        signal: controller.signal,
      })) {
        if (finished) break;
        switch (event.type) {
          case "support":
            support = event.support;
            supportByModel.set(modelKey(model), event.support);
            writeSoon();
            break;
          case "text-delta":
            markdown += event.text;
            events.emit("answer.delta", { mindId, answerId, text: event.text });
            writeSoon();
            break;
          case "text-retracted":
            markdown = markdown.slice(0, Math.max(0, markdown.length - event.length));
            writeSoon();
            break;
          case "tool-call-started": {
            const call: AnswerToolCall = {
              id: event.id,
              tool: event.tool,
              source: "documents",
              input: event.input,
              status: "running",
              resultCount: null,
            };
            toolCalls.push(call);
            events.emit("answer.toolCallStarted", { mindId, answerId, call: { ...call } });
            writeSoon();
            break;
          }
          case "tool-call-finished": {
            const call = toolCalls.find((each) => each.id === event.id);
            if (!call) break;
            call.status = event.ok ? "done" : "failed";
            call.resultCount = event.resultCount;
            events.emit("answer.toolCallFinished", { mindId, answerId, call: { ...call } });
            writeSoon();
            break;
          }
          case "finished":
            outcome = { status: "done" };
            break;
          case "failed":
            outcome = { status: "failed", error: event.error };
            break;
        }
      }
      finish(outcome);
    };
    run().catch((error: unknown) => {
      options.reportError(error);
      finish({ status: "failed", error: classifyProviderError(error) });
    });
  }

  async function ask(
    mindIdInput: unknown,
    questionIdInput: unknown,
    discardEdits: unknown,
  ): Promise<AskResult> {
    const mindId = options.requireMind(mindIdInput);
    const questionId = parseId(questionIdInput, "A Question id");
    if (discardEdits !== undefined && typeof discardEdits !== "boolean") {
      throw new InvalidInputError("discardEdits must be true or false.");
    }

    const picked = content.read(mindId, (blocks) => {
      const found = findBlock(blocks, QUESTION_BLOCK, questionId);
      if (!found) throw new NotFoundError("That Question isn't in this Mind.");
      if (plainText(found.element).trim() === "") {
        throw new InvalidInputError("Write the Question before asking it.");
      }
      return pickedModel(found.element);
    });
    // A model picked on a provider that has since been removed falls back to the default.
    const choice = picked && options.providerExists(picked.providerId) ? picked : undefined;
    const readiness = await options.readiness(choice);
    if (!readiness.ready) return { asked: false, reason: "not-ready", readiness };
    const model = { providerId: readiness.provider.id, modelId: readiness.modelId };

    // From here on nothing awaits, so the Mind can't change underneath.
    const current = content.read(mindId, (blocks) => answerFor(blocks, questionId));
    const currentId = current && textAttribute(current, BLOCK_ID_ATTRIBUTE);
    if (currentId) active.get(currentId)?.finish({ status: "stopped" });

    const written = content.edit(
      mindId,
      (blocks): { edited: string } | { answerId: string; context: QuestionContext } => {
        const question = findBlock(blocks, QUESTION_BLOCK, questionId);
        if (!question) throw new NotFoundError("That Question isn't in this Mind.");
        const context = buildQuestionContext(blocks, question.index);
        const attributes: Partial<AnswerAttributes> = {
          questionId,
          providerId: model.providerId,
          modelId: model.modelId,
          status: "streaming",
          errorKind: null,
          errorMessage: null,
          generatedHash: null,
          citationSupport: null,
          toolCalls: null,
        };

        const answer = answerFor(blocks, questionId);
        const answerId = answer && textAttribute(answer, BLOCK_ID_ATTRIBUTE);
        if (answer && answerId) {
          if (isEdited(answer) && discardEdits !== true) return { edited: answerId };
          // Regenerating: the same Block, where it is, with new content.
          setAttributes(answer, attributes);
          syncContent(answer, [{ type: "paragraph" }]);
          return { answerId, context };
        }

        const id = randomUUID();
        const element = createElement({
          type: ANSWER_BLOCK,
          attrs: { [BLOCK_ID_ATTRIBUTE]: id, ...attributes },
          content: [{ type: "paragraph" }],
        });
        const at = question.index + 1;
        blocks.insert(at, [element]);
        // Somewhere to go on writing, or to ask the next Question.
        if (at + 1 === blocks.length) blocks.insert(at + 1, [createElement({ type: "paragraph" })]);
        return { answerId: id, context };
      },
    );
    if ("edited" in written) return { asked: false, reason: "edited", answerId: written.edited };

    start({ mindId, answerId: written.answerId, questionId, model, context: written.context });
    return { asked: true, answerId: written.answerId };
  }

  return {
    ask(input: unknown): Promise<AskResult> {
      if (!isRecord(input)) throw new InvalidInputError("askQuestion expects an object.");
      return ask(input.mindId, input.questionId, input.discardEdits);
    },

    regenerate(input: unknown): Promise<AskResult> {
      if (!isRecord(input)) throw new InvalidInputError("regenerateAnswer expects an object.");
      const mindId = options.requireMind(input.mindId);
      const answerId = parseId(input.answerId, "An Answer id");
      const questionId = content.read(mindId, (blocks) => {
        const found = findBlock(blocks, ANSWER_BLOCK, answerId);
        if (!found) throw new NotFoundError("That Answer isn't in this Mind.");
        const id = textAttribute(found.element, "questionId");
        if (!id || !findBlock(blocks, QUESTION_BLOCK, id)) {
          throw new NotFoundError("The Question of this Answer has been deleted.");
        }
        return id;
      });
      return ask(mindId, questionId, input.discardEdits);
    },

    stop(input: unknown): void {
      if (!isRecord(input)) throw new InvalidInputError("stopAnswer expects an object.");
      const mindId = options.requireMind(input.mindId);
      const answerId = parseId(input.answerId, "An Answer id");
      const generation = active.get(answerId);
      if (generation?.mindId === mindId) generation.finish({ status: "stopped" });
      else settleOrphans(mindId);
    },

    /** Call before a Mind is opened: Answers a quit or crash left "streaming" become "stopped". */
    settleOrphans,

    /** Stops every Answer, keeping what was written, e.g. when the app quits. */
    stopAll(): void {
      for (const generation of [...active.values()]) generation.finish({ status: "stopped" });
    },
  };
}
