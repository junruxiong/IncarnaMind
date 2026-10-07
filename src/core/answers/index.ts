/**
 * Asking Questions (ADR-0007): the core builds the Question context from the
 * Mind's Yjs document, has the Answer engine stream an Answer, and writes it
 * into the document as it arrives, right after its Question, so every window
 * sees it grow: its text, its Citations (see ./citations), and the Tools it
 * called (searches, and the Skills it used). It also pushes the Answer event
 * stream for UI state and the evaluation.
 *
 * Skills: the instructions list the enabled Skills' names and descriptions,
 * and the model loads one with `use_skill` when it needs it. A Skill the
 * Question forces is loaded up front, and shown as a Tool call too.
 *
 * A Question with a Search scope searches only the Documents it covers,
 * resolved when the Question is asked (or its Answer regenerated). When the
 * scope covers no Document with Passages, the Answer says so, and nothing is
 * searched or sent.
 *
 * Everything here depends on the `AnswerEngine` port, not on the AI SDK.
 */
import { randomUUID } from "node:crypto";
import type * as Y from "yjs";
import { searchScopeOf } from "../../shared/searchScope";
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
  type SearchScope,
  type SkillAvailability,
} from "../api";
import type { WindowedPassage } from "../documents/search";
import { ChatNotReadyError, InvalidInputError, isRecord, NotFoundError } from "../errors";
import type { createEventHub } from "../events";
import type { MindContent } from "../mindContent";
import type { PreparedChatModel } from "../providers/chat";
import { classifyProviderError } from "../providers/providerErrors";
import type { SkillSession } from "../skills";
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
import type { AnswerEngine, AnswerSkillTools, AnswerTools, ExternalTool } from "./engine";
import { markdownToBlocks } from "./markdown";
import {
  answerInstructions,
  connectorInstructions,
  documentInstructions,
  loadedSkillText,
  signInNeededInstructions,
  skillInstructions,
} from "./prompt";

export type { AnswerDocuments } from "./citations";
export type {
  AnswerEngine,
  AnswerEngineEvent,
  AnswerMessage,
  AnswerRequest,
  AnswerSkillTools,
  AnswerTools,
  CitationRecordInput,
  ExternalTool,
  InstructionOptions,
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
  documents: DocumentsForAnswers;
  /** The ids of the live Documents a Search scope covers; null with no Search scope (every Document). */
  resolveScope(scope: SearchScope): string[] | null;
  /** What an Answer says when its Question's Search scope has no Documents to search. */
  emptyScopeAnswer(): string;
  /** The User's Skills. */
  skills: AnswerSkills;
  /** The Connector Tools an Answer may call: the read-only Tools of every Connector that is on and ready. */
  connectorTools(signal: AbortSignal): Promise<ExternalTool[]>;
  /** The names of Connectors that are on but wait for the User to sign in: their Tools are skipped. */
  connectorsNeedingSignIn?(): string[];
  reportError(error: unknown): void;
}

/** What Answers need of Skills (see ../skills). */
export interface AnswerSkills {
  /** Whether the Skill named `name` can be forced now. */
  availability(name: string): SkillAvailability;
  /** The Skills one Answer may use, with the forced one loaded. Throws if it can't be used. */
  openSession(forced: string | null): Promise<SkillSession>;
}

/** The Tool call that shows a forced Skill: loaded by the core, up front. */
const FORCED_SKILL_CALL = "forced-skill";

/**
 * What the core gives Answers from Documents: `AnswerDocuments`, counting and
 * searching only the Documents of a Search scope when given their ids (null:
 * every Document).
 */
export interface DocumentsForAnswers extends Pick<AnswerDocuments, "citationSource" | "pageTexts"> {
  searchableCount(documentIds: readonly string[] | null): number;
  search(
    query: string,
    documentIds: readonly string[] | null,
    signal?: AbortSignal,
  ): Promise<WindowedPassage[]>;
}

/** The Documents one Answer searches: those of its Question's Search scope, or all of them (null). */
function inScope(
  documents: DocumentsForAnswers,
  documentIds: readonly string[] | null,
): AnswerDocuments {
  return {
    searchableCount: () => documents.searchableCount(documentIds),
    search: (query, signal) => documents.search(query, documentIds, signal),
    citationSource: (passageId) => documents.citationSource(passageId),
    pageTexts: (documentId, from, to) => documents.pageTexts(documentId, from, to),
  };
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

/** The Skill forced on a Question, if any. */
const forcedSkillOf = (question: Y.XmlElement): string | null =>
  textAttribute(question, "forcedSkill") || null;

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
    /** The Documents of the Question's Search scope; null with no Search scope (every Document). */
    documentIds: string[] | null;
    /** The Search scope has no Documents to search: the Answer says so, and nothing is sent. */
    emptyScope: boolean;
    /** The Skill the Question forces, loaded up front; null if none. */
    forcedSkill: string | null;
  }): void {
    const { mindId, answerId, model, context, documentIds, forcedSkill } = input;
    const controller = new AbortController();
    /** What the model wrote: Markdown, with Citation markers. */
    let markdown = "";
    let timer: ReturnType<typeof setTimeout> | null = null;
    let finished = false;
    let support: CitationSupport | null = null;
    const toolCalls: AnswerToolCall[] = [];
    let skills: SkillSession | null = null;
    const session = createCitationSession(inScope(options.documents, documentIds), {
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
      skills?.release();
      // A Tool call still running when the Answer stopped didn't finish.
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
    const callStarted = (call: AnswerToolCall) => {
      toolCalls.push(call);
      events.emit("answer.toolCallStarted", { mindId, answerId, call: { ...call } });
      writeSoon();
    };
    const callFinished = (id: string, ok: boolean, resultCount: number | null) => {
      const call = toolCalls.find((each) => each.id === id);
      if (!call) return;
      call.status = ok ? "done" : "failed";
      call.resultCount = resultCount;
      events.emit("answer.toolCallFinished", { mindId, answerId, call: { ...call } });
      writeSoon();
    };

    /** Opens the Answer's Skills, loading the forced one, shown as its Tool call; false if it failed. */
    const openSkills = async (): Promise<boolean> => {
      if (forcedSkill) {
        callStarted({
          id: FORCED_SKILL_CALL,
          tool: "use_skill",
          source: "skill",
          input: { name: forcedSkill },
          status: "running",
          resultCount: null,
          forced: true,
        });
      }
      try {
        const session = await options.skills.openSession(forcedSkill);
        if (finished) {
          session.release();
          return false;
        }
        skills = session;
      } catch (error) {
        if (forcedSkill) callFinished(FORCED_SKILL_CALL, false, null);
        finish({
          status: "failed",
          error: {
            kind: "unknown",
            message: error instanceof Error ? error.message : String(error),
          },
        });
        return false;
      }
      if (forcedSkill) callFinished(FORCED_SKILL_CALL, true, null);
      return true;
    };

    const run = async () => {
      // Rather than search every Document, the Answer says the scope has none to search.
      if (input.emptyScope) {
        // On a later turn, as a model's Answer would arrive, so whoever asked hears every event.
        await new Promise((resolve) => setTimeout(resolve, 0));
        if (finished) return;
        const text = options.emptyScopeAnswer();
        markdown = text;
        events.emit("answer.delta", { mindId, answerId, text });
        finish({ status: "done" });
        return;
      }
      let prepared: PreparedChatModel;
      try {
        prepared = await options.prepareModel(model);
      } catch (error) {
        finish({ status: "failed", error: failureOf(error) });
        return;
      }
      // Stopped while waiting, e.g. for consent: send nothing.
      if (finished) return;
      if (!(await openSkills()) || !skills) return;
      const { listed, forced } = skills;
      const opened: SkillSession = skills;
      // The read-only Tools of the Connectors that are on, next to document search and the Skills.
      let external: ExternalTool[] = [];
      try {
        external = await options.connectorTools(controller.signal);
      } catch (error) {
        if (finished) return;
        options.reportError(error);
      }
      if (finished) return;
      // Connectors waiting for a sign-in offer nothing: the model is told, so the Answer can say why.
      const signInNeeded = options.connectorsNeedingSignIn?.() ?? [];
      const tools: AnswerTools = {
        get documentCount() {
          return session.tools.documentCount;
        },
        searchDocuments: (query, signal) => session.tools.searchDocuments(query, signal),
        cite: (records) => session.tools.cite(records),
        external,
      };
      const skillTools: AnswerSkillTools | null =
        listed.length > 0 || (forced && forced.files.length > 1)
          ? {
              loadable: listed.length > 0,
              useSkill: async (name) =>
                loadedSkillText(await opened.load(name), { withFiles: true }),
              readSkillFile: (skill, path) => opened.readFile(skill, path),
            }
          : null;
      let outcome: Outcome = { status: "stopped" };
      for await (const event of engine.generate({
        instructions: (
          mode,
          { passages, skillTools: withSkillTools = false, connectorTools = false } = {},
        ) =>
          [
            base,
            documentInstructions(mode, tools.documentCount, passages, documentIds !== null),
            skillInstructions(listed, forced, withSkillTools),
            connectorTools ? connectorInstructions(external, mode === "no-documents") : "",
            signInNeededInstructions(signInNeeded),
          ]
            .filter(Boolean)
            .join("\n\n"),
        messages: context.messages,
        question: context.question,
        model: prepared.model,
        tools,
        skills: skillTools,
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
          case "tool-call-started":
            callStarted(
              event.source
                ? {
                    id: event.id,
                    tool: event.tool,
                    source: "connector",
                    connector: { id: event.source.connectorId, name: event.source.connectorName },
                    input: event.input,
                    status: "running",
                    resultCount: null,
                  }
                : {
                    id: event.id,
                    tool: event.tool,
                    source: event.tool === "search_documents" ? "documents" : "skill",
                    input: event.input,
                    status: "running",
                    resultCount: null,
                  },
            );
            break;
          case "tool-call-finished":
            callFinished(event.id, event.ok, event.resultCount);
            break;
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

  /**
   * What an Answer to `question` is written from: its context, and its Search
   * scope as it is now, resolved to the Documents it covers now.
   */
  const askedWith = (context: QuestionContext, question: Y.XmlElement) => {
    const documentIds = options.resolveScope(searchScopeOf(question.getAttributes()));
    const emptyScope = documentIds !== null && options.documents.searchableCount(documentIds) === 0;
    return { context, documentIds, emptyScope };
  };

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

    const { picked, forcedSkill } = content.read(mindId, (blocks) => {
      const found = findBlock(blocks, QUESTION_BLOCK, questionId);
      if (!found) throw new NotFoundError("That Question isn't in this Mind.");
      if (plainText(found.element).trim() === "") {
        throw new InvalidInputError("Write the Question before asking it.");
      }
      return { picked: pickedModel(found.element), forcedSkill: forcedSkillOf(found.element) };
    });
    // A model picked on a provider that has since been removed falls back to the default.
    const choice = picked && options.providerExists(picked.providerId) ? picked : undefined;
    const readiness = await options.readiness(choice);
    if (!readiness.ready) return { asked: false, reason: "not-ready", readiness };
    // A forced Skill that is off or gone: say so rather than answer without it.
    const skillState = forcedSkill ? options.skills.availability(forcedSkill) : "enabled";
    if (forcedSkill && skillState !== "enabled") {
      return { asked: false, reason: "skill-unavailable", skill: forcedSkill, state: skillState };
    }
    const model = { providerId: readiness.provider.id, modelId: readiness.modelId };

    // From here on nothing awaits, so the Mind can't change underneath.
    const current = content.read(mindId, (blocks) => answerFor(blocks, questionId));
    const currentId = current && textAttribute(current, BLOCK_ID_ATTRIBUTE);
    if (currentId) active.get(currentId)?.finish({ status: "stopped" });

    /** What the Answer is written from: its Question context and Search scope, as they are now. */
    type Asked = { context: QuestionContext; documentIds: string[] | null; emptyScope: boolean };
    const written = content.edit(
      mindId,
      (blocks): { edited: string } | ({ answerId: string } & Asked) => {
        const question = findBlock(blocks, QUESTION_BLOCK, questionId);
        if (!question) throw new NotFoundError("That Question isn't in this Mind.");
        const context = buildQuestionContext(blocks, question.index);
        const asked = askedWith(context, question.element);
        const attributes: Partial<AnswerAttributes> = {
          questionId,
          // No model writes an Answer that says the Search scope is empty.
          providerId: asked.emptyScope ? null : model.providerId,
          modelId: asked.emptyScope ? null : model.modelId,
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
          return { answerId, ...asked };
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
        return { answerId: id, ...asked };
      },
    );
    if ("edited" in written) return { asked: false, reason: "edited", answerId: written.edited };

    start({ mindId, questionId, model, ...written, forcedSkill });
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
