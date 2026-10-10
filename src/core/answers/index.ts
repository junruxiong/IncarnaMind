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
 * Connector Tools and Skill scripts (`run_skill_script`, see
 * ../skills/scripts) go through one gate, which the Run engine awaits before
 * each call (see ../runs/engine): approvals decide from the call's Effects
 * whether it asks first (see ../approvals). One that asks pauses the Answer
 * there until the User decides; the call's card records the decision. A
 * denied call isn't made: the model is told so, and the Answer carries on. A
 * script's card also keeps how the run ended and what it wrote. Stopping the
 * Answer stops a running script.
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
import { searchScopeOf } from "../../shared/searchScope";
import {
  ANSWER_BLOCK,
  type AnswerAttributes,
  type AnswerPhase,
  type AnswerToolCall,
  type AskResult,
  BLOCK_ID_ATTRIBUTE,
  type ChatModelChoice,
  type ChatReadiness,
  type CitationSupport,
  type Effect,
  type ProviderError,
  QUESTION_BLOCK,
  type QuoteRetry,
  type SearchScope,
  type SkillAvailability,
  type SkillScriptRun,
  type ToolCallApproval,
} from "../api";
import type { CallToDecide, ToolCallToApprove } from "../approvals";
import type { WindowedPassage } from "../documents/search";
import { ChatNotReadyError, InvalidInputError, isRecord, NotFoundError } from "../errors";
import type { createEventHub } from "../events";
import type { ExecAccess } from "../execution";
import type { MindContent } from "../mindContent";
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
import type { AnswerEngine, AnswerTools, DocumentLanguage, GatedCall } from "./engine";
import { markdownToBlocks } from "./markdown";
import {
  answerInstructions,
  connectorInstructions,
  documentInstructions,
  LISTED_DOCUMENTS,
  type ListedDocuments,
  loadedSkillText,
  signInNeededInstructions,
  skillInstructions,
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

/** Where a Tool-call card says a call's Tool comes from: its provider's kind, as Minds store it. */
const CARD_SOURCES: Record<ToolProviderKind, AnswerToolCall["source"]> = {
  documents: "documents",
  skills: "skill",
  connector: "connector",
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

  /**
   * How each model gives Citations, learnt from its earlier Answers: by
   * provider, model and, for a local model, its build (a model pulled again may do more).
   */
  const supportByModel = new Map<string, CitationSupport>();
  const modelKey = (model: ChatModelChoice, prepared: PreparedChatModel) =>
    `${model.providerId}\n${model.modelId}\n${prepared.revision ?? ""}`;

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
    /** The User has yet to allow the chat flow to the model's service: they are asked first. */
    consentNeeded: boolean;
  }): void {
    const { mindId, answerId, model, context, documentIds, forcedSkill } = input;
    const controller = new AbortController();
    /** What the model wrote: Markdown, with Citation markers. */
    let markdown = "";
    let timer: ReturnType<typeof setTimeout> | null = null;
    let finished = false;
    let support: CitationSupport | null = null;
    /** Markers the engine put in for records the model gave without them. */
    let placedMarkers = 0;
    /** The one request for exact quotes, if the engine made it (see ./quoteRetry). */
    let quoteRetry: QuoteRetry | null = null;
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
        events.emit("answer.failed", { mindId, answerId, error: outcome.error });
      } else {
        events.emit("answer.finished", {
          mindId,
          answerId,
          status: written ? outcome.status : "stopped",
          ...session.summary(),
          placedMarkers,
          citationSupport: support,
          quoteRetry,
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
    if (active.size === 1) options.onWritingChange?.(true);
    events.emit("answer.started", { mindId, answerId, questionId: input.questionId, model });

    const base = answerInstructions(context.question);
    const callStarted = (call: AnswerToolCall) => {
      // A call that asks the User first may already be there (see `ask`).
      if (toolCalls.some((each) => each.id === call.id)) return;
      toolCalls.push(call);
      events.emit("answer.toolCallStarted", { mindId, answerId, call: { ...call } });
      writeSoon();
    };
    const callFinished = (id: string, ok: boolean, resultCount: number | null) => {
      const call = toolCalls.find((each) => each.id === id);
      if (!call) return;
      // A denied call returned the model a result, but never ran; a script that didn't exit with 0 failed.
      const ran = call.script === undefined || call.script.exitCode === 0;
      call.status = ok && call.approval !== "denied" && ran ? "done" : "failed";
      call.resultCount = resultCount;
      events.emit("answer.toolCallFinished", { mindId, answerId, call: { ...call } });
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
      { id: toolCallId, input, tainted }: GatedCall,
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
          mindId,
          answerId,
          toolCallId,
          subject: { kind: "skill-script", skillId: script.skillId },
          skill: { id: script.skillId, name: script.skillName },
          tool: SKILL_TOOLS.runScript,
          script: script.path,
          args,
          effects,
          tainted,
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
     * always run (see `scriptGate`). Every other Tool runs. Whether the Answer
     * had read untrusted content first (`tainted`) goes to approvals with the call.
     */
    const gate = async (tool: Tool, call: GatedCall): Promise<GateDecision> => {
      const { id, input, tainted } = call;
      let denied: string | null = null;
      if (tool.provider.kind === "connector") {
        const connector = { id: tool.provider.id, name: tool.provider.name };
        const name = tool.providerTool;
        const effects = tool.effects(input);
        denied = await ask(
          {
            mindId,
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
        events.emit("answer.delta", { mindId, answerId, text });
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
        events.emit("answer.phase", { mindId, answerId, phase });
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
      if (!(await openSkills()) || !skills) return;
      const { listed, forced } = skills;
      const opened: SkillSession = skills;
      // The Tools of the providers the core registered (the Connectors that are on), next to
      // the Documents' and the Skills'. A Connector's Tool asks the User first unless it needn't
      // (see `gate`).
      const provided: Tool[] = [];
      for (const provider of options.toolProviders) {
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
      const signInNeeded = options.connectorsNeedingSignIn?.() ?? [];
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
      };
      // Scripts can run when a Skill has some, unless the User turned them off.
      const scripts = opened.hasScripts && options.scripts.enabled();
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
          { passages, skillTools: withSkillTools = false, connectorTools = false } = {},
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
            signInNeededInstructions(signInNeeded.map((connector) => connector.name)),
          ]
            .filter(Boolean)
            .join("\n\n"),
        messages: context.messages,
        question: context.question,
        model: prepared.model,
        documents,
        tools: [...provided, ...fromSkills],
        gate,
        support: startingSupport(prepared.support, supportByModel.get(learntKey)),
        window: prepared.window,
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
          case "text-delta":
            responded = true;
            showPhase();
            markdown += event.text;
            events.emit("answer.delta", { mindId, answerId, text: event.text });
            writeSoon();
            break;
          case "text-retracted":
            markdown = markdown.slice(0, Math.max(0, markdown.length - event.length));
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

    const consentNeeded = readiness.consent === "needed";
    start({ mindId, questionId, model, ...written, forcedSkill, consentNeeded });
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
