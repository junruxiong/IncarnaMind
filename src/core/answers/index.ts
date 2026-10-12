/**
 * Asking Questions (ADR-0007): the core builds the Question context, has the
 * Answer engine stream an Answer, and writes it as it arrives, right after its
 * Question, so every window sees it grow: its text, its Citations (see
 * ./citations), and the Tools it called (searches, and the Skills it used). An
 * Answer has one of two homes: its Mind's Yjs document, or one of the
 * assistant's conversations (../conversations, ADR-0017), whose rows keep its
 * Markdown and Citations. It also pushes the Answer event stream for UI state
 * and the evaluation.
 *
 * Skills: the instructions list the enabled Skills' names and descriptions,
 * and the model loads one with `use_skill` when it needs it. A Skill the
 * Question forces is loaded up front, and shown as a Tool call too.
 *
 * Connector Tools and Skill scripts (`run_skill_script`, see
 * ../skills/scripts) go through one gate, which the Run engine awaits before
 * each call (see ../runs/engine): approvals decide from the call's Effects
 * whether it asks first (see ../approvals). One that asks pauses the Answer
 * there until the User decides; the call's card records the decision. A
 * denied call isn't made: the model is told so, and the Answer carries on. A
 * script's card also keeps how the run ended and what it wrote. Stopping the
 * Answer stops a running script.
 *
 * Third-party text (CONTEXT.md, #78): an Answer whose Question context holds
 * text someone else wrote (Blocks, or earlier Answers, with a third-party
 * origin), or whose Run reads a saved web page, asks before a call that writes
 * or sends data, even where the User chose "always"; its approval request
 * says why. The Answer keeps where that text came from (beside a Mind's
 * Blocks, `MIND_TEXT_ORIGINS_FIELD`, or on a conversation's Answer), so a
 * later Question with the Answer in its context counts as having read it too.
 *
 * A Question with a Search scope searches only the Documents it covers,
 * resolved when the Question is asked (or its Answer regenerated), and the
 * instructions name only those. When the scope covers no Document with
 * Passages, the Answer says so, and nothing is searched or sent.
 *
 * Everything here depends on the `AnswerEngine` port, not on the AI SDK.
 */
import { randomUUID } from "node:crypto";
import type * as Y from "yjs";
import { runsByDefault } from "../../shared/approvalRules";
import { withCitations } from "../../shared/blockJson";
import { recordProposalRequest, resolveProposals } from "../../shared/proposals";
import { searchScopeOf } from "../../shared/searchScope";
import { setTextOrigin } from "../../shared/textOrigins";
import {
  ANSWER_BLOCK,
  type AnswerAttributes,
  type AnswerPhase,
  type AnswerPlace,
  type AnswerSteps,
  type AnswerToolCall,
  type AnswerUsage,
  type AskResult,
  BLOCK_ID_ATTRIBUTE,
  type ChatModelChoice,
  type ChatReadiness,
  type CitationSupport,
  type ConversationAskResult,
  type ConversationQuestion,
  type CoreEventName,
  type CoreEvents,
  type Effect,
  type ExternalService,
  type MarkedCitation,
  MIND_CONTENT_FIELD,
  type ProviderError,
  QUESTION_BLOCK,
  type QuoteRetry,
  type SearchScope,
  type SkillAvailability,
  type SkillScriptRun,
  type ThirdPartyReason,
  type ThirdPartySource,
  type ToolCallApproval,
} from "../api";
import type { CallToDecide, ToolCallToApprove } from "../approvals";
import type { AnswerWrite, QuestionInput, Turn } from "../conversations";
import type { WindowedPassage } from "../documents/search";
import { ChatNotReadyError, InvalidInputError, isRecord, NotFoundError } from "../errors";
import type { createEventHub } from "../events";
import type { ExecAccess } from "../execution";
import type { MindContent } from "../mindContent";
import { plainText } from "../mindText";
import { docOf } from "../proposals/merge";
import { proposeBlocks } from "../proposals/write";
import { givesStructuredOutput } from "../providers/capabilities";
import type { PreparedChatModel } from "../providers/chat";
import { classifyProviderError } from "../providers/providerErrors";
import type { GateDecision } from "../runs/engine";
import type { SkillScript, SkillSession } from "../skills";
import {
  parseScriptArgs,
  type ScriptToRun,
  scriptNotRun,
  scriptResultText,
} from "../skills/scripts";
import { scriptCallInput, skillTools } from "../skills/tools";
import { asksForSources } from "../sources/intent";
import {
  SKILL_TOOLS,
  type Tool,
  type ToolCallContext,
  type ToolProvider,
  type ToolProviderKind,
} from "../tools";
import {
  contentHash,
  createElement,
  findBlock,
  type NodeJSON,
  syncContent,
  textAttribute,
  topLevelBlocks,
} from "./blocks";
import {
  type AnswerDocuments,
  createCitationSession,
  withoutFootnoteDefinitions,
} from "./citations";
import {
  conversationContext,
  conversationContextOf,
  type QuestionContext,
  questionContextOf,
} from "./context";
import type { AnswerEngine, AnswerTools, DocumentLanguage, GatedCall, ToolSet } from "./engine";
import { markdownToBlocks } from "./markdown";
import {
  answerInstructions,
  connectorInstructions,
  documentInstructions,
  LISTED_DOCUMENTS,
  type ListedDocuments,
  languageInstruction,
  loadedSkillText,
  signInNeededInstructions,
  skillInstructions,
  sourcesInstructions,
} from "./prompt";

export type { AnswerDocuments } from "./citations";
export { createAiSdkAnswerEngine } from "./engine";

/**
 * How often a streaming Answer is written into its Mind: often enough to look
 * live, seldom enough not to store one Yjs update per token.
 */
const ANSWER_WRITE_INTERVAL_MS = 50;

type Outcome = { status: "done" | "stopped" } | { status: "failed"; error: ProviderError };

interface Generation {
  home: AnswerHome;
  /** The model writing it. */
  model: ChatModelChoice;
  /** Writes what has arrived with the final status, now. Later calls do nothing. */
  finish(outcome: Outcome): void;
}

/**
 * Where an Answer is written as it streams: into its Mind's Yjs document, or into
 * a conversation's rows (ADR-0017). `place` is what its events and approval
 * requests say.
 */
interface AnswerHome {
  place: AnswerPlace;
  /** Writes the Answer as it stands; false when it is gone (or its Mind or conversation is). */
  write(answer: WrittenAnswer, outcome?: Outcome): boolean;
}

/** An Answer as it stands, for its home to write. */
interface WrittenAnswer {
  /** What the model wrote: Markdown, with Citation markers, its footnote definitions left out. */
  markdown: string;
  /** Its Blocks, each marker a Citation node: the final ones once it has ended. */
  blocks(): NodeJSON[];
  /** Its Citations, in the order of their markers in the text: the final ones once it has ended. */
  citations(): MarkedCitation[];
  toolCalls: AnswerToolCall[];
  support: CitationSupport | null;
  /** Where the third-party text its Run had read came from (#78); null: none. */
  textOrigin: ThirdPartySource | null;
  /** How it was reached: what an Answer in a conversation shows above its text (#391). */
  steps: AnswerSteps;
}

/** What Answers need of the assistant's conversations (see ../conversations). */
export interface AnswerConversations {
  /** Adds a Question, and its conversation with its first one, and its Answer, being written. */
  add(input: QuestionInput, answerModel: ChatModelChoice | null): Turn;
  /** The Question of an Answer, to write the Answer again. */
  questionOf(input: unknown): { answerId: string; question: ConversationQuestion };
  /** Empties an Answer, where it is, to write it again. */
  restartAnswer(answerId: string, answerModel: ChatModelChoice | null): Turn;
  /** Writes an Answer as it stands; false when it is gone: its conversation was deleted. */
  writeAnswer(answerId: string, write: AnswerWrite): boolean;
}

/** Why a Question wasn't asked, before anything was kept or sent. */
type NotAsked = Extract<AskResult, { reason: "not-ready" | "skill-unavailable" }>;

/**
 * An Answer written for an outside agent (see `answerOutside`): its Markdown
 * with Citation markers and its Citations, each checked; or why it wasn't
 * asked: the default chat model isn't ready, Local mode keeps the Question
 * from a model off this computer, or the User has yet to allow the model.
 */
export type OutsideAnswer =
  | { asked: false; reason: "not-ready" | "consent-needed" | "local-mode" }
  | {
      asked: true;
      status: "done" | "stopped" | "failed";
      markdown: string;
      citations: MarkedCitation[];
      error: ProviderError | null;
      /** Where the Question went: the chat model's service; null for a model on this computer. */
      service: ExternalService | null;
    };

/** A comparison table's cell as written (see `answerCell`). */
export interface CellAnswer {
  status: "done" | "stopped" | "failed";
  markdown: string;
  /** Its Blocks, each marker a Citation node. */
  blocks: NodeJSON[];
  citations: MarkedCitation[];
  error: ProviderError | null;
}

/** No Search scope: every Document. */
const NO_SCOPE: SearchScope = { folderIds: [], tagIds: [], documentIds: [] };

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
  /** The assistant's conversations, which Answers are written into too. */
  conversations: AnswerConversations;
  /**
   * The Tool providers the core registers beside the Documents and the
   * Skills: the User's Connectors, offering the Tools of every Connector that
   * is on and ready. Their Tools are offered to every Answer.
   */
  toolProviders: readonly ToolProvider[];
  /** The Connectors that are on but wait for the User to sign in: their Tools are skipped. */
  connectorsNeedingSignIn?(): { id: string; name: string }[];
  /** Asking the User before a Connector Tool or a Skill script runs. */
  approvals: AnswerApprovals;
  /** Running Skill scripts. */
  scripts: AnswerScripts;
  /**
   * Hears whether any Answer is being written, each time that changes: from
   * when the first one starts until none is left. Background model work gives
   * way to Answers (see ../backgroundQueue).
   */
  onWritingChange?(writing: boolean): void;
  reportError(error: unknown): void;
}

/** What Answers need of approvals (see ../approvals). */
export interface AnswerApprovals {
  /** Whether a call runs or asks the User first: from its Effects and the User's policy for it. */
  decide(call: CallToDecide): "run" | "ask";
  /** Asks and waits: true if allowed, false if denied. Rejects if `signal` aborts first, withdrawing the request. */
  request(call: ToolCallToApprove, signal: AbortSignal): Promise<boolean>;
}

/** What Answers need to run Skill scripts (see ../skills/scripts). */
export interface AnswerScripts {
  /** Whether Skill scripts may run: the User's switch in Settings. */
  enabled(): boolean;
  /** How long a script may run now, in seconds. */
  timeoutSeconds(): number;
  /** Throws, saying why, for a script that can't run here (its kind), before the User is asked. */
  check(script: string): void;
  /** What a run of a script of the Skill in `skillDir` can reach: from the Executor's sandbox level. */
  access(skillDir: string): ExecAccess;
  /** Runs it; rejects, saying why, when it can't start (e.g. its interpreter isn't installed). */
  run(request: ScriptToRun): Promise<SkillScriptRun>;
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

/** What the model is told when the User denies a call: a Tool result, so the Answer carries on. */
const deniedResult = (call: ToolCallToApprove) =>
  "script" in call
    ? `The User denied running ${call.script} of the Skill "${call.skill.name}", so it didn't run. Carry on without it, don't run it again, and say what wasn't done.`
    : `The User denied this call of ${call.tool} (from the Connector "${call.connector.name}"), so it wasn't made and nothing was sent. Carry on without it, don't call it again, and say what wasn't done.`;

/**
 * What the model reads for a call of a Tool whose side effect nothing can
 * approve yet: a later provider's (a Files Tool, say), before it can ask.
 */
const UNAPPROVED =
  "This call wasn't made: IncarnaMind can't ask the User about it yet, so nothing was changed or sent. Carry on without it and say what wasn't done.";

/** Why a call's Run counts as having read third-party text, for its approval request: absent when it doesn't. */
const readReasons = (
  thirdParty: readonly ThirdPartyReason[],
): { thirdParty?: ThirdPartyReason[] } =>
  thirdParty.length > 0 ? { thirdParty: [...thirdParty] } : {};

/** Where a Tool-call card says a call's Tool comes from: its provider's kind, as Minds store it. */
const CARD_SOURCES: Record<ToolProviderKind, AnswerToolCall["source"]> = {
  documents: "documents",
  skills: "skill",
  connector: "connector",
  sources: "sources",
};

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

/**
 * What the core gives Answers from Documents: `AnswerDocuments`, counting,
 * listing and searching only the Documents of a Search scope when given their
 * ids (null: every Document).
 */
export interface DocumentsForAnswers extends Pick<AnswerDocuments, "citationSource" | "pageTexts"> {
  searchableCount(documentIds: readonly string[] | null): number;
  /** How many Documents there are to search, and the names of the `limit` most recently added, newest first. */
  searchableNames(documentIds: readonly string[] | null, limit: number): ListedDocuments;
  /** The languages the Documents to search are in, the most common first, with how many are in each. */
  searchableLanguages(documentIds: readonly string[] | null): DocumentLanguage[];
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
    pageTexts: (documentId, contentHash, from, to) =>
      documents.pageTexts(documentId, contentHash, from, to),
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

/** How often, while a local model loads, Ollama is asked whether it is ready. */
const LOAD_POLL_MS = 1_000;

const CITING_ORDER: readonly CitationSupport[] = ["tools", "structured-output", "none"];

/**
 * How to start citing: what the model's capabilities say, unless an earlier
 * Answer found its provider refuses that (the engine stepped down), which wins.
 */
function startingSupport(
  known: CitationSupport | undefined,
  learnt: CitationSupport | undefined,
): CitationSupport | undefined {
  if (!known || !learnt) return learnt ?? known;
  return CITING_ORDER.indexOf(learnt) > CITING_ORDER.indexOf(known) ? learnt : known;
}

/** Why a model couldn't be prepared, as the kind of error the Answer shows. */
function failureOf(error: unknown): ProviderError {
  if (error instanceof ChatNotReadyError) {
    const { reason } = error.readiness;
    const kind =
      reason === "consent-declined" || reason === "local-mode"
        ? reason
        : reason === "missing-api-key"
          ? "auth"
          : "model";
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

  /**
   * How each model gives Citations, learnt from its earlier Answers: by
   * provider, model and, for a local model, its build (a model pulled again may do more).
   */
  const supportByModel = new Map<string, CitationSupport>();
  const modelKey = (model: ChatModelChoice, prepared: PreparedChatModel) =>
    `${model.providerId}\n${model.modelId}\n${prepared.revision ?? ""}`;

  function start(input: {
    home: AnswerHome;
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
    /** The User has yet to allow the chat flow to the model's service: they are asked first. */
    consentNeeded: boolean;
    /** Where the third-party text in its Question context came from (#78); empty: none. */
    contextOrigins: readonly ThirdPartySource[];
    /**
     * An Answer for an outside agent (#351), or a comparison table's cell
     * (#98): no window hears of it (it is in no Mind or conversation), and it
     * uses the Documents and Skills only: no Connector, and no Skill script,
     * so nothing it does waits for the User.
     */
    outside?: boolean;
    /** Which Tools its Run is offered (see `ToolSet`); absent: every Tool. */
    toolSet?: ToolSet;
  }): void {
    const { home, answerId, model, context, documentIds, forcedSkill } = input;
    const outside = input.outside === true;
    const { place } = home;
    const announce = <E extends CoreEventName>(event: E, payload: CoreEvents[E]) => {
      if (!outside) events.emit(event, payload);
    };
    const controller = new AbortController();
    /** What the model wrote: Markdown, with Citation markers. */
    let markdown = "";
    /**
     * Why its Run counts as having read third-party text, from its start: its
     * Question context holds some. The Run adds what its Tools read (see the
     * gate's `thirdParty`).
     */
    const contextReasons: ThirdPartyReason[] = input.contextOrigins.map((source) => ({
      via: "context",
      source,
    }));
    /**
     * Where the third-party text the Run had read came from: what its Question
     * context holds, else what it read first. Kept with the Answer, so a later
     * Question that has the Answer in its context counts as having read it too.
     * None for the Answer that says the Search scope is empty: no model wrote it.
     */
    let textOrigin: ThirdPartySource | null = input.emptyScope
      ? null
      : (input.contextOrigins[0] ?? null);
    let timer: ReturnType<typeof setTimeout> | null = null;
    let finished = false;
    let support: CitationSupport | null = null;
    /** Markers the engine put in for records the model gave without them. */
    let placedMarkers = 0;
    /** The one request for exact quotes, if the engine made it (see ./quoteRetry). */
    let quoteRetry: QuoteRetry | null = null;
    /** What the model calls counted, the cached input tokens too. */
    const usage: AnswerUsage = {
      inputTokens: 0,
      outputTokens: 0,
      cachedInputTokens: 0,
      cacheWriteTokens: 0,
    };
    const toolCalls: AnswerToolCall[] = [];
    /** When the Question went to the model, and when the Answer ended, in ms since the epoch. */
    let sentAt: number | null = null;
    let finishedAt: number | null = null;
    const phases: AnswerSteps["phases"] = [];
    /** The model's reasoning summary, kept only from a provider that summarizes it. */
    let reasoning = "";
    let reasoningStartedAt: number | null = null;
    let lastReasoningAt: number | null = null;
    /** When the text the Answer keeps began: a preamble taken back doesn't count. */
    let firstTextAt: number | null = null;
    const iso = (ms: number | null) => (ms === null ? null : new Date(ms).toISOString());
    const steps = (): AnswerSteps => {
      const summary = reasoning.trim();
      const thoughtUntil =
        firstTextAt !== null && reasoningStartedAt !== null && firstTextAt > reasoningStartedAt
          ? firstTextAt
          : (lastReasoningAt ?? Date.now());
      return {
        sentAt: iso(sentAt),
        finishedAt: iso(finishedAt),
        phases: [...phases],
        reasoning:
          summary && reasoningStartedAt !== null
            ? { summary, ms: Math.max(0, thoughtUntil - reasoningStartedAt) }
            : null,
      };
    };
    let skills: SkillSession | null = null;
    const session = createCitationSession(inScope(options.documents, documentIds), {
      onRecord: (marker, citation) =>
        announce("answer.citationAdded", { ...place, answerId, marker, citation }),
    });

    /** Writes the Answer as it stands into its home; false if it is gone. */
    const write = (outcome?: Outcome): boolean => {
      const final = outcome !== undefined;
      const text = withoutFootnoteDefinitions(markdown);
      // Final: a marker with no record is removed, and each is counted for the summary.
      const citationOf = (marker: number) =>
        final ? session.finalCitation(marker) : session.streamingCitation(marker);
      try {
        return home.write(
          {
            markdown: text,
            blocks: () =>
              withCitations(
                markdownToBlocks(text, { streaming: !final }),
                final
                  ? (marker) => session.finalNode(marker)
                  : (marker) => session.streamingNode(marker),
              ),
            citations: () => {
              const found: MarkedCitation[] = [];
              // The markers the Blocks would show, so none in code is taken for one.
              withCitations(markdownToBlocks(text, { streaming: !final }), (marker) => {
                const citation = citationOf(marker);
                if (citation && !found.some((each) => each.marker === marker)) {
                  found.push({ marker, ...citation });
                }
                return null;
              });
              return found;
            },
            toolCalls,
            support,
            textOrigin,
            steps: steps(),
          },
          outcome,
        );
      } catch (error) {
        options.reportError(error);
        return false;
      }
    };

    const finish = (outcome: Outcome) => {
      if (finished) return;
      finished = true;
      finishedAt = Date.now();
      if (timer) clearTimeout(timer);
      active.delete(answerId);
      if (active.size === 0) options.onWritingChange?.(false);
      controller.abort();
      skills?.release();
      // A Tool call still running when the Answer stopped didn't finish; one still waiting for
      // the User never ran (aborting above withdrew its request).
      for (const call of toolCalls) {
        if (call.status === "running") call.status = "failed";
        if (call.approval === "waiting") call.approval = "denied";
      }
      // The check runs once, now; its results are stored with each Citation.
      try {
        session.check();
      } catch (error) {
        options.reportError(error);
      }
      const written = write(outcome);
      if (outcome.status === "failed") {
        announce("answer.failed", { ...place, answerId, error: outcome.error });
      } else {
        announce("answer.finished", {
          ...place,
          answerId,
          status: written ? outcome.status : "stopped",
          ...session.summary(),
          placedMarkers,
          citationSupport: support,
          quoteRetry,
          usage,
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

    active.set(answerId, { home, model, finish });
    if (active.size === 1) options.onWritingChange?.(true);
    announce("answer.started", { ...place, answerId, questionId: input.questionId, model });

    const base = answerInstructions();
    const language = languageInstruction(context.question);
    const callStarted = (call: AnswerToolCall) => {
      // A call that asks the User first may already be there (see `ask`).
      if (toolCalls.some((each) => each.id === call.id)) return;
      toolCalls.push(call);
      announce("answer.toolCallStarted", { ...place, answerId, call: { ...call } });
      writeSoon();
    };
    const callFinished = (
      id: string,
      ok: boolean,
      resultCount: number | null,
      found: Pick<AnswerToolCall, "passages" | "error"> = {},
    ) => {
      const call = toolCalls.find((each) => each.id === id);
      if (!call) return;
      // A denied call returned the model a result, but never ran; a script that didn't exit with 0 failed.
      const ran = call.script === undefined || call.script.exitCode === 0;
      call.status = ok && call.approval !== "denied" && ran ? "done" : "failed";
      call.resultCount = resultCount;
      if (found.passages) call.passages = found.passages;
      if (found.error) call.error = found.error;
      announce("answer.toolCallFinished", { ...place, answerId, call: { ...call } });
      writeSoon();
    };
    const setApproval = (id: string, approval: ToolCallApproval) => {
      const call = toolCalls.find((each) => each.id === id);
      if (finished || !call) return;
      call.approval = approval;
      writeSoon();
    };
    /** Keeps how a script's run went on its card. */
    const setScriptRun = (id: string, run: SkillScriptRun) => {
      const call = toolCalls.find((each) => each.id === id);
      if (finished || !call) return;
      call.script = run;
      writeSoon();
    };

    /**
     * Asks the User about a call that may need it: approvals decide from the
     * call's Effects and the User's policy (see ../approvals). The wait races
     * the Answer's own signal: stopping the Answer ends it at once. The
     * model's call reaches here before the engine reports it, so its `card`
     * is recorded here, waiting, if it isn't already; the engine's report of
     * it then changes nothing. Resolves with null when the call may go ahead,
     * or with what the model reads when the User denied it.
     */
    const ask = async (call: ToolCallToApprove, card: AnswerToolCall): Promise<string | null> => {
      if (options.approvals.decide(call) === "run") return null;
      callStarted({ ...card, approval: "waiting" });
      // Already reported by the engine, after all: it waits too.
      setApproval(call.toolCallId, "waiting");
      const allowed = await options.approvals.request(call, controller.signal);
      setApproval(call.toolCallId, allowed ? "allowed" : "denied");
      return allowed ? null : deniedResult(call);
    };

    /** A Skill script that can't run: its card and the model say why. */
    const cantRun = (toolCallId: string, error: unknown): never => {
      const message = messageOf(error);
      setScriptRun(toolCallId, scriptNotRun(message));
      throw new Error(message);
    };

    /** Skill scripts the gate checked and let through, by call: their call runs what was checked. */
    const allowedScripts = new Map<string, { script: SkillScript; args: string[] }>();

    /**
     * `run_skill_script` at the gate: records its card, which shows the Skill,
     * the script and the arguments; checks the script (a script file of a
     * Skill this Answer may use, of a kind that can run here), so one that
     * can't run is refused without asking; then asks about the call's
     * `effects`, unless the Skill's scripts always run.
     */
    const scriptGate = async (
      session: SkillSession,
      { id: toolCallId, input, tainted, thirdParty }: GatedCall,
      effects: Effect[],
    ): Promise<string | null> => {
      const shown = scriptCallInput(input);
      const card: AnswerToolCall = {
        id: toolCallId,
        tool: SKILL_TOOLS.runScript,
        source: "skill",
        input: shown,
        status: "running",
        resultCount: null,
      };
      callStarted(card);
      let script: SkillScript;
      let args: string[];
      try {
        args = parseScriptArgs(input.args);
        script = await session.script(shown.skill, shown.script);
        options.scripts.check(script.path);
      } catch (error) {
        return cantRun(toolCallId, error);
      }
      const denied = await ask(
        {
          ...place,
          answerId,
          toolCallId,
          subject: { kind: "skill-script", skillId: script.skillId },
          skill: { id: script.skillId, name: script.skillName },
          tool: SKILL_TOOLS.runScript,
          script: script.path,
          args,
          effects,
          tainted,
          ...readReasons(thirdParty),
        },
        card,
      );
      if (denied === null) allowedScripts.set(toolCallId, { script, args });
      return denied;
    };

    /**
     * The gate the engine awaits before each Tool call (see `RunRequest.gate`):
     * a Connector's Tool asks first unless the User or its Effects say it
     * needn't; a Skill script is checked, then asks unless the Skill's scripts
     * always run (see `scriptGate`). Whether the Answer had read untrusted
     * content first (`tainted`), and third-party text (`thirdParty`), goes to
     * approvals with the call. Every other Tool runs if each of its Effects
     * runs without asking; a side effect of a provider whose calls nothing can
     * approve yet is refused (see `UNAPPROVED`).
     */
    const gate = async (tool: Tool, call: GatedCall): Promise<GateDecision> => {
      const { id, input, tainted, thirdParty } = call;
      let denied: string | null = null;
      if (tool.provider.kind === "connector") {
        const connector = { id: tool.provider.id, name: tool.provider.name };
        const name = tool.providerTool;
        const effects = tool.effects(input);
        denied = await ask(
          {
            ...place,
            answerId,
            toolCallId: id,
            subject: { kind: "tool", connectorId: connector.id, tool: name },
            connector,
            tool: name,
            title: tool.title ?? null,
            input,
            // Its Connector marks it read-only: it changes nothing.
            readOnly: effects.every((effect) => effect.action !== "write"),
            effects,
            tainted,
            ...readReasons(thirdParty),
          },
          {
            id,
            tool: name,
            source: "connector",
            connector,
            input,
            status: "running",
            resultCount: null,
          },
        );
      } else if (tool.name === SKILL_TOOLS.runScript && skills) {
        denied = await scriptGate(skills, call, tool.effects(input));
      } else if (!tool.effects(input).every(runsByDefault)) {
        // New side-effecting Tools ask unless a scope grant covers them (ADR-0013): until their
        // provider can ask, their side effects never run.
        options.reportError(
          new Error(`${tool.name} (${tool.provider.kind}) has a side effect nothing can approve.`),
        );
        denied = UNAPPROVED;
      }
      return denied === null ? { run: true } : { run: false, result: denied };
    };

    /**
     * `run_skill_script`'s call, once the gate let it through: runs the script
     * it checked, stopping it if the Answer stops. Its card keeps how the run
     * went, or why it couldn't run. A script the gate didn't let through never runs.
     */
    const runScript = async ({ toolCallId, signal }: ToolCallContext): Promise<string> => {
      const allowed = allowedScripts.get(toolCallId);
      allowedScripts.delete(toolCallId);
      if (!allowed) return cantRun(toolCallId, "The script wasn't allowed to run.");
      const { script, args } = allowed;
      // Turned off since the Answer started.
      if (!options.scripts.enabled()) {
        return cantRun(
          toolCallId,
          "Skill scripts are turned off in Settings, so the script didn't run.",
        );
      }
      const timeoutSeconds = options.scripts.timeoutSeconds();
      let run: SkillScriptRun;
      try {
        run = await options.scripts.run({
          skillDir: script.skillDir,
          script: script.path,
          args,
          timeoutMs: timeoutSeconds * 1000,
          signal: AbortSignal.any([signal, controller.signal]),
        });
      } catch (error) {
        return cantRun(toolCallId, error);
      }
      setScriptRun(toolCallId, run);
      return scriptResultText(script.path, run, timeoutSeconds);
    };

    /** Opens the Answer's Skills, loading the forced one, shown as its Tool call; false if it failed. */
    const openSkills = async (): Promise<boolean> => {
      if (forcedSkill) {
        callStarted({
          id: FORCED_SKILL_CALL,
          tool: SKILL_TOOLS.use,
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
        announce("answer.delta", { ...place, answerId, text });
        finish({ status: "done" });
        return;
      }
      // The phase the meta line shows: waiting for the User to allow the Question to go to the
      // model's service, searching, the model loading (a local model, until Ollama has it
      // loaded or it starts to answer), writing, or checking quotes (asking a local model once
      // more for the quotes the check didn't find).
      let consenting = input.consentNeeded;
      let activity: Extract<AnswerPhase, "searching" | "writing" | "checking-quotes"> = "writing";
      let loading = false;
      let responded = false;
      let shown: AnswerPhase | null = null;
      const showPhase = () => {
        const phase: AnswerPhase = consenting
          ? "waiting-for-consent"
          : activity !== "writing"
            ? activity
            : loading && !responded
              ? "loading"
              : "writing";
        if (finished || phase === shown) return;
        shown = phase;
        phases.push({ phase, at: new Date().toISOString() });
        announce("answer.phase", { ...place, answerId, phase });
      };
      // Nothing is sent before the User says: the Answer waits for them.
      if (consenting) showPhase();
      let prepared: PreparedChatModel;
      try {
        prepared = await options.prepareModel(model);
      } catch (error) {
        finish({ status: "failed", error: failureOf(error) });
        return;
      }
      // Stopped while waiting, e.g. for consent: send nothing.
      if (finished) return;
      if (consenting) {
        consenting = false;
        showPhase();
      }
      sentAt = Date.now();
      if (!(await openSkills()) || !skills) return;
      const { listed, forced } = skills;
      const opened: SkillSession = skills;
      // The Tools of the providers the core registered (the Connectors that are on), next to
      // the Documents' and the Skills'. A Connector's Tool asks the User first unless it needn't
      // (see `gate`).
      const provided: Tool[] = [];
      for (const provider of outside ? [] : options.toolProviders) {
        // Find sources' Tool only for a Question that asks for papers or other sources.
        if (provider.kind === "sources" && !asksForSources(context.question)) continue;
        try {
          provided.push(...(await provider.tools(controller.signal)));
        } catch (error) {
          if (finished) return;
          options.reportError(error);
        }
      }
      if (finished) return;
      const fromConnectors = provided.filter((tool) => tool.provider.kind === "connector");
      // Connectors waiting for a sign-in offer nothing. The Answer shows a card for each, which
      // didn't run, and the model is told, so the Answer can say why.
      const signInNeeded = outside ? [] : (options.connectorsNeedingSignIn?.() ?? []);
      for (const connector of signInNeeded) {
        const id = `sign-in:${connector.id}`;
        callStarted({
          id,
          tool: "sign_in",
          source: "connector",
          connector,
          input: {},
          status: "running",
          resultCount: null,
          signInRequired: true,
        });
        callFinished(id, false, null);
      }
      const documents: AnswerTools = {
        get documentCount() {
          return session.tools.documentCount;
        },
        // So the model can search again in the Documents' language (see the search Tool).
        documentLanguages: options.documents.searchableLanguages(documentIds),
        searchDocuments: (query, signal) => session.tools.searchDocuments(query, signal),
        cite: (records, citeOptions) => session.tools.cite(records, citeOptions),
        hasRecord: (marker) => session.tools.hasRecord?.(marker) ?? false,
        unfoundQuotes: () => session.tools.unfoundQuotes?.() ?? [],
        correctQuote: (marker, quote) => session.tools.correctQuote?.(marker, quote) ?? false,
      };
      // Scripts can run when a Skill has some, unless the User turned them off.
      const scripts = !outside && opened.hasScripts && options.scripts.enabled();
      const fromSkills: Tool[] =
        listed.length > 0 || (forced && forced.files.length > 1)
          ? skillTools({
              loadable: listed.length > 0,
              useSkill: async (name) =>
                loadedSkillText(await opened.load(name), { withFiles: true, scripts }),
              skill: (name) => opened.skill(name),
              readSkillFile: (skill, path) => opened.readFile(skill, path),
              ...(scripts && {
                scripts: {
                  access: (skillDir) => options.scripts.access(skillDir),
                  run: (_input, call) => runScript(call),
                },
              }),
            })
          : [];
      const watchLoading = async (loaded: () => Promise<boolean | null>) => {
        for (let first = true; !finished && !responded; first = false) {
          const ready = await loaded().catch(() => null);
          if (finished || responded) return;
          if (ready !== false) {
            loading = false;
            showPhase();
            return;
          }
          if (first) {
            loading = true;
            showPhase();
          }
          await new Promise((resolve) => setTimeout(resolve, LOAD_POLL_MS).unref?.());
        }
      };
      // Until a phase is known, the meta line says "Writing…".
      if (prepared.loaded) void watchLoading(prepared.loaded);

      const learntKey = modelKey(model, prepared);
      let outcome: Outcome = { status: "stopped" };
      for await (const event of engine.generate({
        instructions: (
          mode,
          {
            passages,
            skillTools: withSkillTools = false,
            connectorTools = false,
            sourceTools = false,
          } = {},
        ) =>
          [
            base,
            mode === "no-documents"
              ? ""
              : documentInstructions(
                  mode,
                  options.documents.searchableNames(documentIds, LISTED_DOCUMENTS),
                  passages,
                  documentIds !== null,
                ),
            skillInstructions(listed, forced, withSkillTools, scripts),
            connectorTools ? connectorInstructions(fromConnectors, mode === "no-documents") : "",
            sourcesInstructions(sourceTools),
            signInNeededInstructions(signInNeeded.map((connector) => connector.name)),
            language,
          ]
            .filter(Boolean)
            .join("\n\n"),
        messages: context.messages,
        question: context.question,
        model: prepared.model,
        documents,
        tools: [...provided, ...fromSkills],
        gate,
        toolSet: input.toolSet,
        thirdParty: contextReasons,
        support: startingSupport(prepared.support, supportByModel.get(learntKey)),
        structuredOutput: givesStructuredOutput(prepared.capabilities),
        takesTemperature: prepared.capabilities.facts.temperature,
        window: prepared.window,
        retryQuotes: prepared.retryQuotes,
        cacheKey: place.conversationId ?? place.mindId ?? undefined,
        signal: controller.signal,
      })) {
        if (finished) break;
        switch (event.type) {
          case "support":
            support = event.support;
            supportByModel.set(learntKey, event.support);
            // The model has begun to answer: it is loaded.
            responded = true;
            showPhase();
            writeSoon();
            break;
          case "phase":
            activity = event.phase;
            showPhase();
            break;
          case "markers-placed":
            placedMarkers += event.count;
            break;
          case "quotes-retried":
            quoteRetry = event.retry;
            break;
          case "usage":
            usage.inputTokens += event.usage.inputTokens ?? 0;
            usage.outputTokens += event.usage.outputTokens ?? 0;
            usage.cachedInputTokens += event.usage.cachedInputTokens ?? 0;
            usage.cacheWriteTokens += event.usage.cacheWriteTokens ?? 0;
            break;
          case "reasoning":
            // Raw chain of thought is dropped: only a provider's own summary is kept.
            if (!prepared.summarizesReasoning) break;
            reasoningStartedAt ??= Date.now();
            if (!event.text) break;
            lastReasoningAt = Date.now();
            reasoning += event.text;
            announce("answer.reasoning", { ...place, answerId, text: event.text });
            writeSoon();
            break;
          case "text-delta":
            responded = true;
            showPhase();
            firstTextAt ??= Date.now();
            markdown += event.text;
            announce("answer.delta", { ...place, answerId, text: event.text });
            writeSoon();
            break;
          case "text-retracted":
            markdown = markdown.slice(0, Math.max(0, markdown.length - event.length));
            if (markdown === "") firstTextAt = null;
            writeSoon();
            break;
          case "tool-call-started": {
            // The card says where the Tool comes from, from its provider, as Minds store it.
            const { provider } = event;
            callStarted({
              id: event.id,
              tool: event.tool,
              source: CARD_SOURCES[provider.kind],
              ...(provider.kind === "connector" && {
                connector: { id: provider.id, name: provider.name },
              }),
              input: event.input,
              status: "running",
              resultCount: null,
            });
            break;
          }
          case "tool-call-finished":
            callFinished(event.id, event.ok, event.resultCount, {
              ...(event.passages && { passages: [...event.passages] }),
              ...(event.error && { error: event.error }),
            });
            break;
          case "read-third-party":
            if (textOrigin === null) {
              textOrigin = event.source;
              writeSoon();
            }
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

  /** An Answer in its Mind: a Block right after its Question in the Mind's Yjs document. */
  const mindHome = (mindId: string, answerId: string): AnswerHome => ({
    place: { mindId },
    write(answer, outcome) {
      if (!options.mindExists(mindId)) return false;
      return content.edit(mindId, (blocks) => {
        const found = findBlock(blocks, ANSWER_BLOCK, answerId);
        if (!found) return false;
        const { element } = found;
        syncContent(element, answer.blocks());
        const calls = answer.toolCalls.length > 0 ? JSON.stringify(answer.toolCalls) : null;
        if (textAttribute(element, "toolCalls") !== calls)
          setAttributes(element, { toolCalls: calls });
        if (answer.support && textAttribute(element, "citationSupport") !== answer.support) {
          setAttributes(element, { citationSupport: answer.support });
        }
        // Beside the Blocks, by the Answer's id: a later Question above which it stands reads it.
        if (blocks.doc) setTextOrigin(blocks.doc, [answerId], answer.textOrigin);
        if (outcome) {
          setAttributes(element, {
            status: outcome.status,
            errorKind: outcome.status === "failed" ? outcome.error.kind : null,
            errorMessage: outcome.status === "failed" ? outcome.error.message : null,
            errorResetsAt: (outcome.status === "failed" && outcome.error.resetsAt) || null,
            generatedHash: contentHash(element),
          });
        }
        return true;
      });
    },
  });

  /**
   * An Answer the User edited, written again (#94): nothing is written while
   * it streams, and once it is finished it goes over the User's version as
   * Proposals of one request (../proposals/write, `proposeBlocks`), replacing
   * any still waiting in it. Its model, searches and how it cited are the new
   * Answer's; it counts as unedited again once every change is accepted.
   * Stopped, it writes nothing; failed, the Answer says why, as it is.
   */
  const changesHome = (mindId: string, answerId: string, model: ChatModelChoice): AnswerHome => ({
    place: { mindId },
    write(answer, outcome) {
      if (!options.mindExists(mindId)) return false;
      const there = content.read(mindId, (blocks) => !!findBlock(blocks, ANSWER_BLOCK, answerId));
      if (!there || !outcome || outcome.status === "stopped") return there;
      if (outcome.status === "failed") {
        return content.edit(mindId, (blocks) => {
          const found = findBlock(blocks, ANSWER_BLOCK, answerId);
          if (found) {
            setAttributes(found.element, {
              status: "failed",
              errorKind: outcome.error.kind,
              errorMessage: outcome.error.message,
            });
          }
          return !!found;
        });
      }
      const request = randomUUID();
      const wrote = content.edit(mindId, (blocks) => {
        const found = findBlock(blocks, ANSWER_BLOCK, answerId);
        if (!found) return false;
        const { element } = found;
        resolveProposals(element, () => true, "reject");
        const written = proposeBlocks(element, answer.blocks(), request, randomUUID);
        if (written.proposals.length > 0 && blocks.doc) {
          recordProposalRequest(blocks.doc, request, {
            source: "answer",
            providerId: model.providerId,
            modelId: model.modelId,
            createdAt: new Date().toISOString(),
            answerId,
          });
        }
        setAttributes(element, {
          status: "done",
          errorKind: null,
          errorMessage: null,
          providerId: model.providerId,
          modelId: model.modelId,
          toolCalls: answer.toolCalls.length > 0 ? JSON.stringify(answer.toolCalls) : null,
          citationSupport: answer.support,
        });
        return true;
      });
      if (!wrote) return false;
      // What it holds once every change is accepted is what the model wrote.
      const generated = docOf(content.state(mindId));
      try {
        const blocks = generated.getXmlFragment(MIND_CONTENT_FIELD);
        generated.transact(() => {
          resolveProposals(blocks, (ref) => ref.request === request, "accept");
        });
        const found = findBlock(blocks, ANSWER_BLOCK, answerId);
        const generatedHash = found ? contentHash(found.element) : null;
        content.edit(mindId, (live) => {
          const element = findBlock(live, ANSWER_BLOCK, answerId)?.element;
          if (element) setAttributes(element, { generatedHash });
        });
      } finally {
        generated.destroy();
      }
      return true;
    },
  });

  /** An Answer in one of the assistant's conversations: its row, with its Markdown and Citations. */
  const conversationHome = (conversationId: string, answerId: string): AnswerHome => ({
    place: { mindId: null, conversationId },
    write: (answer, outcome) =>
      options.conversations.writeAnswer(answerId, {
        text: answer.markdown,
        citations: answer.citations(),
        toolCalls: answer.toolCalls,
        support: answer.support,
        textOrigin: answer.textOrigin,
        steps: answer.steps,
        ...(outcome && {
          outcome: {
            status: outcome.status,
            error: outcome.status === "failed" ? outcome.error : null,
          },
        }),
      }),
  });

  /**
   * The model a Question is answered with, and whether the User has yet to allow its
   * data flow; or why it can't be asked now: Questions can't be asked yet, or the Skill
   * it forces is off or gone. A model picked on a provider since removed falls back to
   * the default.
   */
  async function readyModel(
    picked: ChatModelChoice | null,
    forcedSkill: string | null,
  ): Promise<{ model: ChatModelChoice; consentNeeded: boolean } | NotAsked> {
    const choice = picked && options.providerExists(picked.providerId) ? picked : undefined;
    const readiness = await options.readiness(choice);
    if (!readiness.ready) return { asked: false, reason: "not-ready", readiness };
    // A forced Skill that is off or gone: say so rather than answer without it.
    const skillState = forcedSkill ? options.skills.availability(forcedSkill) : "enabled";
    if (forcedSkill && skillState !== "enabled") {
      return { asked: false, reason: "skill-unavailable", skill: forcedSkill, state: skillState };
    }
    return {
      model: { providerId: readiness.provider.id, modelId: readiness.modelId },
      consentNeeded: readiness.consent === "needed",
    };
  }

  /**
   * Writes the Answer of a Question in a conversation, added or emptied in `turn`, from
   * the messages before it and its Search scope as it is now.
   */
  function startInConversation(
    turn: Turn,
    ready: { model: ChatModelChoice; consentNeeded: boolean },
    documentIds: string[] | null,
    emptyScope: boolean,
  ): ConversationAskResult {
    const { conversation, question, answer } = turn;
    const { context, origins } = conversationContextOf(turn.earlier, question.text);
    start({
      home: conversationHome(conversation.id, answer.id),
      answerId: answer.id,
      questionId: question.id,
      model: ready.model,
      context,
      contextOrigins: origins,
      documentIds,
      emptyScope,
      forcedSkill: question.forcedSkill,
      consentNeeded: ready.consentNeeded,
    });
    return { asked: true, conversation, questionId: question.id, answerId: answer.id };
  }

  /** The Documents of a Search scope as it is now, and whether it covers none to search. */
  const scopeNow = (scope: SearchScope | null) => {
    const documentIds = options.resolveScope(scope ?? NO_SCOPE);
    const emptyScope = documentIds !== null && options.documents.searchableCount(documentIds) === 0;
    return { documentIds, emptyScope };
  };

  /**
   * What an Answer to `question` is written from: its context, and its Search
   * scope as it is now, resolved to the Documents it covers now.
   */
  const askedWith = (context: QuestionContext, question: Y.XmlElement) => {
    const documentIds = options.resolveScope(searchScopeOf(question.getAttributes()));
    const emptyScope = documentIds !== null && options.documents.searchableCount(documentIds) === 0;
    return { context, documentIds, emptyScope };
  };

  async function ask(mindIdInput: unknown, questionIdInput: unknown): Promise<AskResult> {
    const mindId = options.requireMind(mindIdInput);
    const questionId = parseId(questionIdInput, "A Question id");

    const { picked, forcedSkill } = content.read(mindId, (blocks) => {
      const found = findBlock(blocks, QUESTION_BLOCK, questionId);
      if (!found) throw new NotFoundError("That Question isn't in this Mind.");
      if (plainText(found.element).trim() === "") {
        throw new InvalidInputError("Write the Question before asking it.");
      }
      return { picked: pickedModel(found.element), forcedSkill: forcedSkillOf(found.element) };
    });
    const ready = await readyModel(picked, forcedSkill);
    if ("asked" in ready) return ready;
    const { model } = ready;

    // From here on nothing awaits, so the Mind can't change underneath.
    const current = content.read(mindId, (blocks) => answerFor(blocks, questionId));
    const currentId = current && textAttribute(current, BLOCK_ID_ATTRIBUTE);
    if (currentId) active.get(currentId)?.finish({ status: "stopped" });

    /** What the Answer is written from: its Question context and Search scope, as they are now. */
    type Asked = {
      context: QuestionContext;
      contextOrigins: ThirdPartySource[];
      documentIds: string[] | null;
      emptyScope: boolean;
    };
    const written = content.edit(
      mindId,
      (blocks): { answerId: string; edited: boolean } & Asked => {
        const question = findBlock(blocks, QUESTION_BLOCK, questionId);
        if (!question) throw new NotFoundError("That Question isn't in this Mind.");
        const { context, origins } = questionContextOf(blocks, question.index);
        const asked = { ...askedWith(context, question.element), contextOrigins: origins };
        const attributes: Partial<AnswerAttributes> = {
          questionId,
          // No model writes an Answer that says the Search scope is empty.
          providerId: asked.emptyScope ? null : model.providerId,
          modelId: asked.emptyScope ? null : model.modelId,
          status: "streaming",
          errorKind: null,
          errorMessage: null,
          errorResetsAt: null,
          generatedHash: null,
          citationSupport: null,
          toolCalls: null,
        };

        const answer = answerFor(blocks, questionId);
        const answerId = answer && textAttribute(answer, BLOCK_ID_ATTRIBUTE);
        if (answer && answerId) {
          // The User's edits stay: the new Answer comes, once written, as changes over them.
          if (isEdited(answer)) return { answerId, edited: true, ...asked };
          // Regenerating: the same Block, where it is, with new content.
          setAttributes(answer, attributes);
          syncContent(answer, [{ type: "paragraph" }]);
          return { answerId, edited: false, ...asked };
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
        return { answerId: id, edited: false, ...asked };
      },
    );

    const { consentNeeded } = ready;
    start({
      home: written.edited
        ? changesHome(mindId, written.answerId, model)
        : mindHome(mindId, written.answerId),
      questionId,
      model,
      ...written,
      forcedSkill,
      consentNeeded,
    });
    return { asked: true, answerId: written.answerId };
  }

  return {
    ask(input: unknown): Promise<AskResult> {
      if (!isRecord(input)) throw new InvalidInputError("askQuestion expects an object.");
      return ask(input.mindId, input.questionId);
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
      return ask(mindId, questionId);
    },

    stop(input: unknown): void {
      if (!isRecord(input)) throw new InvalidInputError("stopAnswer expects an object.");
      const mindId = options.requireMind(input.mindId);
      const answerId = parseId(input.answerId, "An Answer id");
      const generation = active.get(answerId);
      if (generation?.home.place.mindId === mindId) generation.finish({ status: "stopped" });
      else settleOrphans(mindId);
    },

    /**
     * Asks a Question in a conversation (see `CoreApi.askInConversation`): nothing is
     * kept unless it can be asked.
     */
    async askInConversation(question: QuestionInput): Promise<ConversationAskResult> {
      const ready = await readyModel(question.model, question.forcedSkill);
      if ("asked" in ready) return ready;
      // From here on nothing awaits, so the conversation can't change underneath.
      const { documentIds, emptyScope } = scopeNow(question.scope);
      // No model writes an Answer that says the Search scope is empty.
      const turn = options.conversations.add(question, emptyScope ? null : ready.model);
      return startInConversation(turn, ready, documentIds, emptyScope);
    },

    /** Writes an Answer of a conversation again, in place (see `CoreApi.regenerateConversationAnswer`). */
    async regenerateInConversation(input: unknown): Promise<ConversationAskResult> {
      const { answerId, question } = options.conversations.questionOf(input);
      const ready = await readyModel(question.model, question.forcedSkill);
      if ("asked" in ready) return ready;
      active.get(answerId)?.finish({ status: "stopped" });
      const { documentIds, emptyScope } = scopeNow(question.scope);
      const turn = options.conversations.restartAnswer(answerId, emptyScope ? null : ready.model);
      return startInConversation(turn, ready, documentIds, emptyScope);
    },

    /** Stops an Answer of a conversation being written, keeping what it has. */
    stopInConversation(input: unknown): void {
      if (!isRecord(input))
        throw new InvalidInputError("stopConversationAnswer expects an object.");
      const conversationId = parseId(input.conversationId, "A conversation id");
      const generation = active.get(parseId(input.answerId, "An Answer id"));
      if (generation?.home.place.conversationId === conversationId) {
        generation.finish({ status: "stopped" });
      }
    },

    /**
     * Answers a Question for an outside agent (#351) from these Documents, on
     * the default chat model, as the assistant would but kept nowhere: the
     * Answer comes back once written, its Citations checked. Asks nothing of
     * the User: if the chat model isn't ready, or its data flow not allowed
     * yet, it says so instead. `signal` stops it, keeping what was written.
     */
    async answerOutside(
      question: string,
      documentIds: string[],
      signal: AbortSignal,
    ): Promise<OutsideAnswer> {
      const readiness = await options.readiness(undefined);
      if (!readiness.ready) {
        return {
          asked: false,
          reason: readiness.reason === "local-mode" ? "local-mode" : "not-ready",
        };
      }
      if (readiness.consent === "needed") return { asked: false, reason: "consent-needed" };
      const model = { providerId: readiness.provider.id, modelId: readiness.modelId };
      const { service } = readiness.provider;
      const answerId = randomUUID();
      return new Promise<OutsideAnswer>((resolve) => {
        start({
          home: {
            place: { mindId: null },
            write(answer, outcome) {
              if (outcome) {
                resolve({
                  asked: true,
                  status: outcome.status,
                  markdown: answer.markdown,
                  citations: answer.citations(),
                  error: outcome.status === "failed" ? outcome.error : null,
                  service,
                });
              }
              return true;
            },
          },
          answerId,
          questionId: answerId,
          model,
          context: conversationContext([], question),
          // The agent's Question is untrusted too, but nothing here can write or send (ADR-0013).
          contextOrigins: [],
          documentIds,
          emptyScope: options.documents.searchableCount(documentIds) === 0,
          forcedSkill: null,
          consentNeeded: false,
          outside: true,
        });
        const stop = () => active.get(answerId)?.finish({ status: "stopped" });
        if (signal.aborted) stop();
        else signal.addEventListener("abort", stop, { once: true });
      });
    },

    /**
     * Writes one cell of a comparison table (#98) from `documentIds` alone
     * (its row's Document: a one-Document Search scope), as a check Run,
     * offered the Documents' search and `cite` only, and kept nowhere: it
     * comes back once written, its Citations checked, as the Blocks a Mind
     * holds. The model is ready and allowed already. `signal` stops it.
     */
    answerCell(input: {
      context: QuestionContext;
      documentIds: string[];
      model: ChatModelChoice;
      signal: AbortSignal;
    }): Promise<CellAnswer> {
      const answerId = randomUUID();
      return new Promise<CellAnswer>((resolve) => {
        start({
          home: {
            place: { mindId: null },
            write(answer, outcome) {
              if (outcome) {
                resolve({
                  status: outcome.status,
                  markdown: answer.markdown,
                  blocks: answer.blocks(),
                  citations: answer.citations(),
                  error: outcome.status === "failed" ? outcome.error : null,
                });
              }
              return true;
            },
          },
          answerId,
          questionId: answerId,
          model: input.model,
          context: input.context,
          contextOrigins: [],
          documentIds: input.documentIds,
          emptyScope: options.documents.searchableCount(input.documentIds) === 0,
          forcedSkill: null,
          consentNeeded: false,
          outside: true,
          toolSet: "check",
        });
        const stop = () => active.get(answerId)?.finish({ status: "stopped" });
        if (input.signal.aborted) stop();
        else input.signal.addEventListener("abort", stop, { once: true });
      });
    },

    /** Stops every Answer being written in a conversation, e.g. before it is deleted. */
    stopConversation(conversationId: string): void {
      for (const generation of [...active.values()]) {
        if (generation.home.place.conversationId === conversationId) {
          generation.finish({ status: "stopped" });
        }
      }
    },

    /** Call before a Mind is opened: Answers a quit or crash left "streaming" become "stopped". */
    settleOrphans,

    /**
     * Local mode just turned on: stops every Answer being written by a model that isn't on
     * this computer, keeping what was written. It ends as failed with the "local-mode"
     * reason, so the Answer says why.
     */
    async stopCloud(): Promise<void> {
      for (const generation of [...active.values()]) {
        const readiness = await options.readiness(generation.model);
        if (readiness.ready || readiness.reason !== "local-mode") continue;
        generation.finish({
          status: "failed",
          error: { kind: "local-mode", message: "Stopped: local mode is on." },
        });
      }
    },

    /** Stops every Answer, keeping what was written, e.g. when the app quits. */
    stopAll(): void {
      for (const generation of [...active.values()]) generation.finish({ status: "stopped" });
    },
  };
}
