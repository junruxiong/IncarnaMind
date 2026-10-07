/**
 * The Answer engine: the one place that turns Question context into a
 * streamed Answer by calling a chat model. Everything else about Answers
 * (building the context, the Tools' work, Citations, writing the Answer into
 * the Mind, stop and regenerate, the event stream) depends only on this port,
 * so the agent layer behind it can change (the AI SDK today) without touching
 * the rest.
 *
 * How the model cites depends on what it can do (see `CitationSupport`):
 * - "tools": a Tool-calling loop (ADR-0007). The model searches the Documents
 *   with `search_documents` as often as it needs, and gives the records of its
 *   Citation markers with `cite`.
 * - "structured-output": for a model that can't call Tools, one search, its
 *   Passages in the instructions, and the Answer and its records returned as
 *   one JSON object. A follow-up Question is first rewritten by the model into
 *   a search query that stands on its own (see `searchQuery`).
 * - "none": for a model that can do neither, the same search, and a plain Answer.
 * The engine starts where it is told (or with Tools), and steps down when the
 * provider refuses Tools or structured output. A model may give records but
 * leave their markers out of its text: the engine then places them (see
 * ./markerPlacement).
 *
 * For a local model with a fixed context window, every request is kept
 * within it by IncarnaMind's own rules (see ./window), and one that can't fit
 * fails as "too-long"; nothing is left to the server to cut.
 *
 * Answers are written at a low temperature (see `ANSWER_TEMPERATURE`), except
 * with models that reject one; a provider that refuses it gets the request
 * again without, and that model gets none from then on.
 *
 * Skills add Tools to the loop: `use_skill` loads a Skill's instructions,
 * `read_skill_file` one of its files, and `run_skill_script` runs one of its
 * scripts (asking the User first, inside the call). The User's Connectors add
 * their Tools as external Tools (those that ask the User first wait inside their
 * `call`); their results aren't Passages, so they are never cited, and the
 * Answer shows each call as a Tool-call card. With
 * Skills or Connector Tools but no Documents, the loop runs with those alone;
 * a model that can't call Tools answers without.
 */
import {
  APICallError,
  generateText,
  jsonSchema,
  Output,
  parsePartialJson,
  RetryError,
  type StopCondition,
  stepCountIs,
  streamText,
  type ToolSet,
  tool,
  UnsupportedFunctionalityError,
} from "ai";
import type { AnswerPhase, CitationSupport, ProviderError } from "../api";
import type { ChatLanguageModel, ContextWindow } from "../providers/models";
import { classifyProviderError, contextOverflow } from "../providers/providerErrors";
import { earlierContext } from "./context";
import { missingMarkerEvents, textEdits } from "./markerPlacement";
import { SEARCH_QUERY_INSTRUCTIONS, searchQueryPrompt } from "./prompt";
import { createWindowBudget, SEARCH_RESERVE_TOKENS, type WindowBudget } from "./window";

/** One message of Question context. */
export interface AnswerMessage {
  role: "user" | "assistant";
  /** Markdown. */
  content: string;
}

/** A Citation record, as the model gives it for one marker. */
export interface CitationRecordInput {
  /** The marker's number: 1 for "[^1]". */
  marker: number;
  /** The Passage's id, as the search gave it (e.g. "P3"). */
  passage: string;
  /** The page or two consecutive pages the quote is on; left out for a Document without pages. */
  pageFrom?: number | null;
  pageTo?: number | null;
  /** A short quote, copied word for word from the Passage. */
  quote: string;
}

/** What a search gives the model. */
export interface SearchResultForModel {
  /** The Passages, formatted for the model, or a sentence saying there are none. */
  text: string;
  passageCount: number;
}

/**
 * A Tool from outside IncarnaMind, e.g. one of a Connector's: offered to the
 * model as it is, next to document search and the Skill Tools. Its result
 * isn't a Passage, so it can't be cited; the Answer shows the call as a
 * Tool-call card instead.
 */
export interface ExternalTool {
  /** The name the model calls it by: unique among the Answer's Tools, e.g. "github__search_issues". */
  name: string;
  description: string;
  /** A JSON Schema for its arguments (an object). */
  inputSchema: Record<string, unknown>;
  /** Where it comes from, for the Tool-call card: the Connector, and the Tool's own name there. */
  source: { connectorId: string; connectorName: string; tool: string };
  /** The Tool's display name, when its Connector gives one. */
  title?: string | null;
  /**
   * Its Connector says it only reads (a hint, not a fact): it runs without
   * asking the User unless they switched it to "ask". Otherwise it asks first.
   */
  readOnly: boolean;
  /**
   * Calls it; resolves with the text the model reads, rejects when it fails
   * (the model is told why). `call.toolCallId` is the id the model gave the
   * call, as in the "tool-call-started" event.
   */
  call(
    input: Record<string, unknown>,
    signal: AbortSignal,
    call?: { toolCallId: string },
  ): Promise<string>;
}

/** The Tools' work, done by the core: the engine only connects them to the model. */
export interface AnswerTools {
  /** Documents with Passages to search. With none, there is no document search. */
  readonly documentCount: number;
  /** The document-search Tool. */
  searchDocuments(query: string, signal?: AbortSignal): Promise<SearchResultForModel>;
  /** Takes Citation records; returns what to tell the model about them. */
  cite(records: readonly CitationRecordInput[]): string;
  /** Whether a valid record was taken for this marker: the engine places its marker if the model left it out. */
  hasRecord?(marker: number): boolean;
  /** Tools from the User's Connectors, offered to a model that can call Tools. */
  readonly external?: readonly ExternalTool[];
}

/** The Skill Tools' work, done by the core. */
export interface AnswerSkillTools {
  /** Offer `use_skill`: there are Skills the model may load, listed in the instructions. */
  readonly loadable: boolean;
  /** A Skill's full instructions and its list of files, for the model. Throws for an unknown Skill. */
  useSkill(name: string): Promise<string>;
  /** One of a Skill's files, as text. Throws for a path outside the Skill. */
  readSkillFile(skill: string, path: string): Promise<string>;
  /**
   * Runs one of a Skill's scripts (`run_skill_script`, with `{ skill, script,
   * args }`), asking the User first unless the Skill's scripts always run.
   * Resolves with what to tell the model (how it ended and what it wrote, or
   * that the User denied it); rejects with why it couldn't run. Absent when no
   * script can run: none of the Skills has one, or the User turned scripts off.
   */
  runScript?(
    input: Record<string, unknown>,
    signal: AbortSignal,
    call: { toolCallId: string },
  ): Promise<string>;
}

/** The Tool that runs a Skill's scripts. */
export const RUN_SKILL_SCRIPT_TOOL = "run_skill_script";

/**
 * A `run_skill_script` call's arguments as the Tool-call card shows them:
 * the Skill, the script and the arguments, as text.
 */
export function scriptCallInput(input: unknown): { skill: string; script: string; args: string[] } {
  const fields = isPlainObject(input) ? input : {};
  const text = (value: unknown) => (typeof value === "string" ? value : String(value ?? ""));
  return {
    skill: text(fields.skill),
    script: text(fields.script),
    args: Array.isArray(fields.args) ? fields.args.map(text) : [],
  };
}

export interface InstructionOptions {
  /** "structured-output" and "none": the Passages the search found (formatted, or a sentence saying there are none). */
  passages?: string;
  /** The model is offered the Skill Tools. */
  skillTools?: boolean;
  /** The model is offered Connector Tools (`AnswerTools.external`). */
  connectorTools?: boolean;
}

export interface AnswerRequest {
  /**
   * The instructions for a way of citing. "no-documents": the User has no
   * Documents to search.
   */
  instructions(mode: CitationSupport | "no-documents", options?: InstructionOptions): string;
  /** The Question context, oldest first; the last message is the User's and ends with the Question. */
  messages: AnswerMessage[];
  /**
   * The Question's own text: what a model that can't call Tools searches for,
   * once it has rewritten it into a query that stands on its own.
   */
  question: string;
  /** The model to answer with, from `Core.prepareChatModel`. */
  model: ChatLanguageModel;
  tools: AnswerTools;
  /** The Skill Tools, or null when there are no Skills to use. */
  skills: AnswerSkillTools | null;
  /** How this model is known to cite, from its capabilities or earlier Answers. Unknown: try Tools first. */
  support?: CitationSupport;
  /**
   * A local model's context window: every request is kept within it (see
   * ./window), and one that can't fit fails as "too-long". None for a cloud model.
   */
  window?: ContextWindow;
  /** Stops generating. The stream then ends, with neither "finished" nor "failed". */
  signal: AbortSignal;
}

export type AnswerEngineEvent =
  /** How the model gives Citations, once its provider has accepted the request. */
  | { type: "support"; support: CitationSupport }
  /** What the Answer is doing now: searching the Documents, or the model's turn. */
  | { type: "phase"; phase: Extract<AnswerPhase, "searching" | "writing"> }
  /** The engine put in `count` Citation markers the model left out of its text (see ./markerPlacement). */
  | { type: "markers-placed"; count: number }
  /** More of the Answer's text (Markdown, with Citation markers), in order. */
  | { type: "text-delta"; text: string }
  /**
   * The last `length` characters of text were only a preamble to a search
   * ("Let me look that up"), or are being replaced: remove them.
   */
  | { type: "text-retracted"; length: number }
  /**
   * A Tool call started: a search of the Documents (by the model, or by the
   * engine for a model without Tools, `{ query }`), a Skill Tool
   * (`use_skill` with `{ name }`, `read_skill_file` with `{ skill, path }`),
   * or an external Tool, with its `source` and the arguments the model sent.
   */
  | {
      type: "tool-call-started";
      id: string;
      tool: string;
      source?: ExternalTool["source"];
      input: Record<string, unknown>;
    }
  | { type: "tool-call-finished"; id: string; ok: boolean; resultCount: number | null }
  /** The Answer is complete. Nothing follows. */
  | { type: "finished" }
  /** The model or its provider failed. Nothing follows. */
  | { type: "failed"; error: ProviderError };

export interface AnswerEngine {
  /** Streams the Answer to the Question context as events. It never throws: failures are events. */
  generate(request: AnswerRequest): AsyncIterable<AnswerEngineEvent>;
}

export interface AiSdkAnswerEngineOptions {
  /** The most model calls one Answer makes in the Tool-calling loop. The last one must write. */
  maxSteps?: number;
}

/** Model calls per Answer: a few searches, the records, and the Answer, with room to spare. */
const MAX_STEPS = 10;

/**
 * The temperature Answers are written at. Providers default to about 1, which
 * samples freely: a model then tends to paraphrase the quotes of its
 * Citations, and a quote is checked word for word against the pages it cites,
 * so a paraphrase fails the check. A low temperature keeps the model to its
 * likeliest wording, which for a quote is the Passage's own, and keeps the
 * Answer close to the Passages. Not 0, greedy decoding: that makes some
 * models, small local ones especially, repeat themselves in long Answers. (The
 * old CLI used 0, the old backend 0.5.)
 */
export const ANSWER_TEMPERATURE = 0.2;

/** The part of a model id after a provider's prefix ("openai/o3" → "o3"), lowercased. */
const modelName = (modelId: string) => (modelId.split("/").at(-1) ?? "").toLowerCase();

/**
 * The temperature to send `model`, or undefined for a model that rejects one
 * or should run at its default:
 * - OpenAI's reasoning models (o1, o3, o4-mini…, and GPT-5 and later, except
 *   their "chat" models) reject a temperature. The AI SDK's OpenAI provider
 *   leaves it out for them, but an OpenAI-compatible server passes it on.
 * - Google wants Gemini 3 and later run at their default: lower makes them
 *   loop or reason worse.
 * Claude models that reject one (Opus 4.7 and later…) are left to the AI
 * SDK's Anthropic provider, which leaves it out for them. A provider that
 * refuses it for any other model gets the request again without (`generate`).
 */
export function answerTemperature(model: Pick<ChatLanguageModel, "modelId">): number | undefined {
  const name = modelName(model.modelId);
  if (/^o\d+(?:$|[-.])/.test(name)) return undefined;
  const gpt = /^gpt-(\d+)/.exec(name);
  if (gpt && Number(gpt[1]) >= 5 && !name.includes("chat")) return undefined;
  const gemini = /^gemini-(\d+)/.exec(name);
  if (gemini && Number(gemini[1]) >= 3) return undefined;
  return ANSWER_TEMPERATURE;
}

/** The provider refused this way of answering: try the next one. Internal to the engine. */
type Unsupported = { type: "unsupported" };
/** The provider refused the temperature: send the request again without one. Internal to the engine. */
type TemperatureRefused = { type: "temperature-refused" };
/**
 * A local model refused the first request as longer than its window: our
 * estimate was short. `promptTokens`: its real size, when Ollama said, where
 * `estimated` were expected. Internal to the engine.
 */
type Overflow = { type: "overflow"; promptTokens: number | null; estimated: number };
type Attempt = AsyncGenerator<AnswerEngineEvent | Unsupported | TemperatureRefused | Overflow>;
/** An attempt that has dealt with a refused temperature itself. */
type TemperedAttempt = AsyncGenerator<AnswerEngineEvent | Unsupported | Overflow>;
/** An attempt that has dealt with a refused temperature and a too-long first request itself. */
type FittedAttempt = AsyncGenerator<AnswerEngineEvent | Unsupported>;

const SEARCH_TOOL = "search_documents";
const CITE_TOOL = "cite";
const USE_SKILL_TOOL = "use_skill";
const READ_SKILL_FILE_TOOL = "read_skill_file";
/** Tool calls the Answer shows, and text before which was a preamble. `cite` isn't one: its records become Citations. */
const SHOWN_TOOLS: ReadonlySet<string> = new Set([
  SEARCH_TOOL,
  USE_SKILL_TOOL,
  READ_SKILL_FILE_TOOL,
  RUN_SKILL_SCRIPT_TOOL,
]);
const MARKER = /\[\^\d{1,4}\]/;

/**
 * Whether a provider refused a request because the model can't use `feature`:
 * e.g. Ollama's "model does not support tools", vLLM's "--enable-auto-tool-choice",
 * a server that rejects `response_format`, or OpenAI's "Unsupported parameter:
 * 'temperature'" for a reasoning model. Auth, rate limits and outages never count.
 */
function isUnsupportedFeature(
  error: unknown,
  feature: "tools" | "structured-output" | "temperature",
): boolean {
  const cause = RetryError.isInstance(error) ? error.lastError : error;
  const subject =
    feature === "tools"
      ? /tool|function/i
      : feature === "temperature"
        ? /temperature/i
        : /response_format|response format|json_schema|json schema|json mode|json_object|structured output|format/i;
  if (UnsupportedFunctionalityError.isInstance(cause)) return subject.test(cause.functionality);
  if (!APICallError.isInstance(cause)) return false;
  const status = cause.statusCode;
  if (status === undefined || [401, 403, 404, 408, 429].includes(status) || status >= 502) {
    return false;
  }
  const text = `${cause.message} ${cause.responseBody ?? ""}`;
  const refusal =
    /not support|unsupported|doesn't support|does not support|not enabled|not available|isn't available|requires --|requires the --|not allowed|is invalid|invalid value|unknown (?:field|parameter|argument)|unrecognized/i;
  // A temperature is also refused as deprecated, or as other than the default.
  const temperatureRefusal = /deprecated|only the default/i;
  return (
    subject.test(text) &&
    (refusal.test(text) || (feature === "temperature" && temperatureRefusal.test(text)))
  );
}

/**
 * What an attempt yields when its provider refused the request before anything
 * came back: the temperature, if one was sent, or else `feature`; or, for a
 * request sized to a window (`estimated` tokens), that it was too long. Null
 * for any other failure.
 */
function refusalOf(
  error: unknown,
  feature: "tools" | "structured-output" | null,
  temperature: number | undefined,
  estimated?: number,
): Unsupported | TemperatureRefused | Overflow | null {
  const overflow = estimated === undefined ? null : contextOverflow(error);
  if (overflow && estimated !== undefined) {
    return { type: "overflow", promptTokens: overflow.promptTokens, estimated };
  }
  if (temperature !== undefined && isUnsupportedFeature(error, "temperature")) {
    return { type: "temperature-refused" };
  }
  if (feature && isUnsupportedFeature(error, feature)) return { type: "unsupported" };
  return null;
}

const recordsSchema = {
  type: "array",
  items: {
    type: "object",
    properties: {
      marker: { type: "integer", description: "The marker's number: 1 for [^1]." },
      passage: { type: "string", description: "The Passage's id, e.g. P1." },
      pageFrom: {
        type: "integer",
        description: "The page the quote starts on. Leave out for a Passage without pages.",
      },
      pageTo: {
        type: "integer",
        description: "The page the quote ends on: the same page, or the next one.",
      },
      quote: {
        type: "string",
        description: "A short quote, copied word for word from the Passage.",
      },
    },
    required: ["marker", "passage", "quote"],
  },
} as const;

/** Records from the model's JSON, kept only where they have the right shape. */
function parseRecords(value: unknown): CitationRecordInput[] {
  // Some models send the array as a JSON string.
  if (typeof value === "string") {
    try {
      return parseRecords(JSON.parse(value));
    } catch {
      return [];
    }
  }
  if (!Array.isArray(value)) return [];
  return value.flatMap((item): CitationRecordInput[] => {
    if (typeof item !== "object" || item === null) return [];
    const { marker, passage, pageFrom, pageTo, quote } = item as Record<string, unknown>;
    const number = typeof marker === "string" ? Number.parseInt(marker, 10) : marker;
    if (typeof number !== "number" || typeof passage !== "string" || typeof quote !== "string") {
      return [];
    }
    const page = (value: unknown) =>
      typeof value === "number" ? value : typeof value === "string" ? Number(value) : null;
    return [{ marker: number, passage, pageFrom: page(pageFrom), pageTo: page(pageTo), quote }];
  });
}

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

/** The JSON a model is writing, parsed as far as it goes, even inside a Markdown code fence. */
const parseAnswerJson = (raw: string) =>
  parsePartialJson(raw.replace(/^\s*```(?:json)?[ \t]*\n?/i, "").replace(/\n?```\s*$/, ""));

/** The longest search query a rewrite may give, in characters: a query, not an Answer. */
const MAX_QUERY_LENGTH = 300;

/** The search query in a model's rewrite: its first line, without its thinking, a label or quotes. */
function queryIn(reply: string): string {
  const line =
    reply
      // An unclosed block was cut off mid-thought.
      .replace(/<think>[\s\S]*?(?:<\/think>|$)/gi, "")
      .split("\n")
      .map((each) => each.trim())
      .find(Boolean) ?? "";
  return line
    .replace(/^(?:search )?query:\s*/i, "")
    .replace(/^["'“”`]+|["'“”`]+$/g, "")
    .trim()
    .slice(0, MAX_QUERY_LENGTH);
}

/**
 * What a model that can't call Tools searches for: the Question, rewritten by
 * the model into one query that stands on its own, from the Question context
 * above it. So "What about its limitations?" searches for what "it" is, as
 * the old CLI's condense step did, but as one query: there is still one
 * search (ADR-0007). With nothing above the Question, the Question is the
 * query; and it is the query when the rewrite fails or comes back empty.
 */
async function searchQuery(
  request: AnswerRequest,
  temperature: number | undefined,
): Promise<string> {
  const { question, signal } = request;
  const earlier = earlierContext(request.messages, question);
  if (!earlier) return question;
  try {
    const { text } = await generateText({
      model: request.model,
      instructions: SEARCH_QUERY_INSTRUCTIONS,
      prompt: searchQueryPrompt(earlier, question),
      temperature,
      // A short query; a cut-off reply falls back to the Question.
      maxOutputTokens: 256,
      // A failure falls back to the Question at once; the Answer's own request still retries.
      maxRetries: 0,
      abortSignal: signal,
    });
    return queryIn(text) || question;
  } catch (error) {
    if (!signal.aborted) {
      console.error(`The Question couldn't be rewritten for search: ${messageOf(error)}`);
    }
    return question;
  }
}

/** The engine on the Vercel AI SDK. */
export function createAiSdkAnswerEngine(options: AiSdkAnswerEngineOptions = {}): AnswerEngine {
  const maxSteps = options.maxSteps ?? MAX_STEPS;
  /** Models whose provider refused a temperature, by provider and model: they get none. */
  const refusedTemperature = new Set<string>();
  /** What earlier Answers learnt of each local model's tokenizer (see ./window). */
  const tokenFactors = new Map<string, number>();

  return {
    async *generate(request) {
      const failed = (error: unknown): AnswerEngineEvent => ({
        type: "failed",
        error: classifyProviderError(error),
      });
      const { signal } = request;
      const modelKey = `${request.model.provider}\n${request.model.modelId}`;
      const temperature = () =>
        refusedTemperature.has(modelKey) ? undefined : answerTemperature(request.model);
      const budget = request.window
        ? createWindowBudget(request.window, tokenFactors.get(modelKey), (factor) =>
            tokenFactors.set(modelKey, factor),
          )
        : null;

      /**
       * Runs a way of answering at the Answer temperature, and if the provider
       * refuses it, again without one: the model gets none from then on.
       */
      async function* tempered(run: (temperature: number | undefined) => Attempt): TemperedAttempt {
        const sent = temperature();
        if (sent !== undefined) {
          let refused = false;
          for await (const event of run(sent)) {
            if (event.type === "temperature-refused") {
              refused = true;
              break;
            }
            yield event;
          }
          if (!refused || signal.aborted) return;
          refusedTemperature.add(modelKey);
        }
        for await (const event of run(undefined)) {
          if (event.type !== "temperature-refused") yield event;
        }
      }

      /**
       * Runs a way of answering sized to the window, and if the model refuses
       * its first request as too long after all, once more with the estimate
       * corrected by the model's own count; then the Answer fails as "too-long".
       */
      async function* fitted(make: () => TemperedAttempt): FittedAttempt {
        for (let tries = 0; ; tries++) {
          let overflow: Overflow | null = null;
          for await (const event of make()) {
            if (event.type === "overflow") {
              overflow = event;
              break;
            }
            yield event;
          }
          // Only a request sized to a window can overflow.
          if (!overflow || !budget || signal.aborted) return;
          if (tries === 0 && overflow.promptTokens) {
            budget.learn(overflow.promptTokens, overflow.estimated);
            continue;
          }
          yield { type: "failed", error: budget.tooLong(overflow.promptTokens) };
          return;
        }
      }

      /**
       * A way of answering with the Passages (if any) in its instructions:
       * structured output, or a plain Answer. Within a window, the Passages
       * that fit beside the Question, then the Question context that fits beside them.
       */
      const onePass = (mode: CitationSupport | "no-documents", passages?: string) =>
        fitted(async function* () {
          if (!budget) {
            const instructions = request.instructions(mode, { passages });
            yield* tempered((sent) =>
              mode === "structured-output"
                ? structured(request, instructions, sent)
                : plain(request, instructions, sent),
            );
            return;
          }
          const kept =
            passages === undefined
              ? undefined
              : budget.fitPassages(
                  passages,
                  request.instructions(mode, { passages: "" }),
                  request.question,
                );
          const instructions = request.instructions(mode, { passages: kept });
          const fit = budget.fit({
            instructions,
            messages: request.messages,
            question: request.question,
          });
          if (!fit.ok) {
            yield { type: "failed", error: fit.error };
            return;
          }
          const sized = { ...request, messages: fit.messages };
          const window: Sized = {
            estimated: fit.estimated,
            learn: (actual) => budget.learn(actual, fit.estimated),
          };
          yield* tempered((sent) =>
            mode === "structured-output"
              ? structured(sized, instructions, sent, window)
              : plain(sized, instructions, sent, window),
          );
        });

      // With no Documents there is nothing to search or cite: Skills and
      // Connector Tools alone, or a plain Answer.
      if (request.tools.documentCount === 0) {
        const external = request.tools.external?.length ?? 0;
        if ((request.skills || external > 0) && (request.support ?? "tools") === "tools") {
          let unsupported = false;
          for await (const event of fitted(() =>
            tempered((sent) => toolLoop(request, maxSteps, "no-documents", sent, budget)),
          )) {
            if (event.type === "unsupported") {
              unsupported = true;
              break;
            }
            yield event;
          }
          if (!unsupported || signal.aborted) return;
        }
        for await (const event of onePass("no-documents")) {
          if (event.type === "unsupported") return;
          yield event;
        }
        return;
      }

      // A model without Tools searches once, for the Question rewritten to stand on its own;
      // the result is kept for the next way of answering if the provider refuses this one.
      let searched: string | undefined;
      async function* searchOnce(): AsyncGenerator<AnswerEngineEvent, string> {
        if (searched !== undefined) return searched;
        const query = await searchQuery(request, temperature());
        signal.throwIfAborted();
        const id = "question-search";
        yield { type: "phase", phase: "searching" };
        yield { type: "tool-call-started", id, tool: SEARCH_TOOL, input: { query } };
        try {
          const result = await request.tools.searchDocuments(query, signal);
          yield { type: "tool-call-finished", id, ok: true, resultCount: result.passageCount };
          searched = result.text;
        } catch (error) {
          if (signal.aborted) throw error;
          console.error(error);
          yield { type: "tool-call-finished", id, ok: false, resultCount: null };
          searched = "The search of the User's Documents failed.";
        }
        yield { type: "phase", phase: "writing" };
        return searched;
      }

      const order: CitationSupport[] = ["tools", "structured-output", "none"];
      for (let mode = order.indexOf(request.support ?? "tools"); mode < order.length; mode++) {
        const support = order[mode] as CitationSupport;
        let attempt: FittedAttempt;
        try {
          if (support === "tools") {
            attempt = fitted(() =>
              tempered((sent) => toolLoop(request, maxSteps, "tools", sent, budget)),
            );
          } else {
            const passages: string = yield* searchOnce();
            if (signal.aborted) return;
            attempt = onePass(support, passages);
          }
        } catch (error) {
          if (signal.aborted) return;
          yield failed(error);
          return;
        }
        let announced = false;
        let unsupported = false;
        for await (const event of attempt) {
          if (event.type === "unsupported") {
            unsupported = true;
            break;
          }
          if (!announced && event.type !== "failed") {
            announced = true;
            yield { type: "support", support };
          }
          yield event;
        }
        if (!unsupported || signal.aborted) return;
      }
    },
  };
}

/** The Skill Tools, for the model. */
function skillTools(skills: AnswerSkillTools, signal: AbortSignal): ToolSet {
  const tools: ToolSet = {};
  if (skills.loadable) {
    tools[USE_SKILL_TOOL] = tool({
      description:
        "Load a Skill's full instructions by its name, before answering a Question the Skill is for. Returns the instructions and the Skill's files.",
      inputSchema: jsonSchema<{ name: string }>({
        type: "object",
        properties: { name: { type: "string", description: "The Skill's name, as listed." } },
        required: ["name"],
      }),
      execute: async ({ name }) => skills.useSkill(String(name ?? "")),
    });
  }
  tools[READ_SKILL_FILE_TOOL] = tool({
    description:
      "Read one of a Skill's files, such as a reference its instructions point to. Only files inside the Skill can be read.",
    inputSchema: jsonSchema<{ skill: string; path: string }>({
      type: "object",
      properties: {
        skill: { type: "string", description: "The Skill's name." },
        path: {
          type: "string",
          description: "The file's path inside the Skill, e.g. references/guide.md.",
        },
      },
      required: ["skill", "path"],
    }),
    execute: async ({ skill, path }) =>
      skills.readSkillFile(String(skill ?? ""), String(path ?? "")),
  });
  const { runScript } = skills;
  if (runScript) {
    tools[RUN_SKILL_SCRIPT_TOOL] = tool({
      description:
        "Run one of a Skill's scripts when its instructions say to, with arguments. It runs on the User's computer in a new, empty working folder; the Skill's own folder is in the SKILL_DIR environment variable. Python (.py), JavaScript (.js, .mjs) and shell (.sh) scripts can run. The User approves each run first and may deny it. Returns the exit code and what the script printed.",
      inputSchema: jsonSchema<{ skill: string; script: string; args?: string[] }>({
        type: "object",
        properties: {
          skill: { type: "string", description: "The Skill's name." },
          script: {
            type: "string",
            description: "The script's path inside the Skill, e.g. scripts/convert.py.",
          },
          args: {
            type: "array",
            items: { type: "string" },
            description: "The script's command-line arguments, in order.",
          },
        },
        required: ["skill", "script"],
      }),
      execute: (input, { abortSignal, toolCallId }) =>
        runScript(isPlainObject(input) ? input : {}, abortSignal ?? signal, { toolCallId }),
    });
  }
  return tools;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** The external Tools (a Connector's), for the model. Names IncarnaMind's own Tools use are skipped. */
function externalTools(external: ReadonlyMap<string, ExternalTool>, signal: AbortSignal): ToolSet {
  const tools: ToolSet = {};
  for (const each of external.values()) {
    tools[each.name] = tool({
      description: each.description,
      inputSchema: jsonSchema<Record<string, unknown>>(each.inputSchema),
      execute: (input, { abortSignal, toolCallId }) =>
        each.call(isPlainObject(input) ? input : {}, abortSignal ?? signal, { toolCallId }),
    });
  }
  return tools;
}

/** Within a window, the text a Tool returns is cut to the room left (search results are fitted by Passage). */
function withResultsFitted(tools: ToolSet, cut: (text: string) => string): ToolSet {
  const fitted: ToolSet = {};
  for (const [name, each] of Object.entries(tools)) {
    const run = each.execute;
    fitted[name] =
      !run || name === SEARCH_TOOL || name === CITE_TOOL
        ? each
        : {
            ...each,
            execute: async (input: unknown, options: Parameters<typeof run>[1]) => {
              const result: unknown = await run(input, options);
              return typeof result === "string" ? cut(result) : result;
            },
          };
  }
  return fitted;
}

/**
 * The Tool-calling loop: search, cite, answer; with Skills, load them as
 * needed; with Connectors, call their Tools where they help. "no-documents":
 * the Skill and Connector Tools alone, with nothing to cite. Within a
 * window, its requests are kept within it (see ./window).
 */
async function* toolLoop(
  request: AnswerRequest,
  maxSteps: number,
  mode: "tools" | "no-documents",
  temperature: number | undefined,
  budget: WindowBudget | null = null,
): Attempt {
  const { signal } = request;
  /** Passages each search gave, by Tool call. */
  const results = new Map<string, number>();
  /** The records the model gave, in order. */
  const cited: CitationRecordInput[] = [];
  /** The account of the window, once the first request is sized. */
  let loop: ReturnType<WindowBudget["loop"]> | null = null;
  const documentTools: ToolSet = {
    [SEARCH_TOOL]: tool({
      description:
        "Search the User's Documents. Returns the Passages that best match, each with an id, its Document and its pages. The search sees only the query, not the conversation: write it to stand on its own.",
      inputSchema: jsonSchema<{ query: string }>({
        type: "object",
        properties: {
          query: {
            type: "string",
            description:
              'What to look for: words likely to be in the Passages, in their language. Name what the Question refers to instead of pronouns or words that point back, such as "it", "they" or "that paper".',
          },
        },
        required: ["query"],
      }),
      execute: async ({ query }, { toolCallId, abortSignal }) => {
        const found = await request.tools.searchDocuments(String(query ?? ""), abortSignal);
        const result = loop ? loop.passages(found.text, found.passageCount) : found;
        results.set(toolCallId, result.passageCount);
        return result.text;
      },
    }),
    [CITE_TOOL]: tool({
      description:
        "Record the Citations of your Answer: for each marker such as [^1], the Passage, the page or two consecutive pages its quote is on, and a short quote copied word for word from the Passage.",
      inputSchema: jsonSchema<{ citations: unknown }>({
        type: "object",
        properties: { citations: recordsSchema },
        required: ["citations"],
      }),
      execute: async ({ citations }) => {
        const records = parseRecords(citations);
        cited.push(...records);
        return request.tools.cite(records);
      },
    }),
  };
  /** External Tools, by the name the model calls them; never one of IncarnaMind's own names. */
  const external = new Map(
    (request.tools.external ?? [])
      .filter((each) => !SHOWN_TOOLS.has(each.name) && each.name !== CITE_TOOL)
      .map((each) => [each.name, each]),
  );
  const offered: ToolSet = {
    ...externalTools(external, signal),
    ...(mode === "tools" ? documentTools : {}),
    ...(request.skills ? skillTools(request.skills, signal) : {}),
  };
  const instructions = request.instructions(mode, {
    skillTools: request.skills !== null,
    connectorTools: external.size > 0,
  });

  // Within a window: the Question context that fits beside the instructions, the Tools and a search.
  let messages = request.messages;
  let estimated: number | undefined;
  if (budget) {
    const fit = budget.fit({
      instructions,
      messages,
      question: request.question,
      tools: JSON.stringify(offered),
      reserve: SEARCH_RESERVE_TOKENS,
    });
    if (!fit.ok) {
      yield { type: "failed", error: fit.error };
      return;
    }
    ({ messages, estimated } = fit);
    loop = budget.loop(fit.estimated);
  }
  const tools = loop ? withResultsFitted(offered, (text) => loop?.result(text) ?? text) : offered;

  const result = streamText({
    model: request.model,
    instructions,
    messages,
    tools,
    temperature,
    stopWhen: stepCountIs(maxSteps) as StopCondition<ToolSet>,
    // The last step must write the Answer; so must a step with no room left for a Tool's result.
    prepareStep: ({ stepNumber }) =>
      stepNumber >= maxSteps - 1 || (loop !== null && !loop.canCallTools())
        ? { toolChoice: "none" }
        : {},
    abortSignal: signal,
    // Errors arrive as stream parts; don't also log them.
    onError: () => undefined,
  });

  /** Text of the current step, and of every step before it that was kept. */
  let stepText = "";
  let keptText = "";
  let stepTools: string[] = [];
  let produced = false;
  /** Searches running now: the Answer is "searching" while there are any. */
  let searching = 0;
  try {
    for await (const part of result.fullStream) {
      if (signal.aborted || part.type === "abort") return;
      switch (part.type) {
        case "start-step":
          stepText = "";
          stepTools = [];
          break;
        case "text-delta": {
          if (!part.text) break;
          produced = true;
          let text = part.text;
          // A new step's text goes on from the last kept text in a new paragraph.
          if (stepText === "" && keptText !== "" && !/\s$/.test(keptText)) text = `\n\n${text}`;
          stepText += text;
          yield { type: "text-delta", text };
          break;
        }
        case "tool-call": {
          produced = true;
          stepTools.push(part.toolName);
          const outside = external.get(part.toolName);
          if (part.toolName === SEARCH_TOOL && searching++ === 0) {
            yield { type: "phase", phase: "searching" };
          }
          if (SHOWN_TOOLS.has(part.toolName)) {
            yield {
              type: "tool-call-started",
              id: part.toolCallId,
              tool: part.toolName,
              input: shownInput(part.toolName, part.input),
            };
          } else if (outside) {
            yield {
              type: "tool-call-started",
              id: part.toolCallId,
              tool: outside.source.tool,
              source: outside.source,
              input: isPlainObject(part.input) ? part.input : {},
            };
          }
          break;
        }
        case "tool-result":
          if (SHOWN_TOOLS.has(part.toolName) || external.has(part.toolName)) {
            yield {
              type: "tool-call-finished",
              id: part.toolCallId,
              ok: true,
              resultCount: results.get(part.toolCallId) ?? null,
            };
          }
          if (part.toolName === SEARCH_TOOL && --searching === 0) {
            yield { type: "phase", phase: "writing" };
          }
          break;
        case "tool-error":
          if (SHOWN_TOOLS.has(part.toolName) || external.has(part.toolName)) {
            // A failed search is the app's problem; a Skill or file that isn't there, the
            // model's; a Connector's failure (or a declined consent) is told to the model.
            if (part.toolName === SEARCH_TOOL && !signal.aborted) console.error(part.error);
            yield { type: "tool-call-finished", id: part.toolCallId, ok: false, resultCount: null };
          }
          if (part.toolName === SEARCH_TOOL && --searching === 0) {
            yield { type: "phase", phase: "writing" };
          }
          break;
        case "finish-step": {
          loop?.stepFinished(part.usage);
          // Text before a search, a Skill or a Connector's Tool, or before records with no
          // marker in it, was a preamble.
          const preamble =
            stepTools.some((name) => SHOWN_TOOLS.has(name) || external.has(name)) ||
            (stepTools.includes(CITE_TOOL) && !MARKER.test(stepText));
          if (preamble && stepText) yield { type: "text-retracted", length: stepText.length };
          else keptText += stepText;
          stepText = "";
          break;
        }
        case "error": {
          const refused = produced ? null : refusalOf(part.error, "tools", temperature, estimated);
          yield refused ?? { type: "failed", error: classifyProviderError(part.error) };
          return;
        }
      }
    }
  } catch (error) {
    if (signal.aborted) return;
    const refused = produced ? null : refusalOf(error, "tools", temperature, estimated);
    yield refused ?? { type: "failed", error: classifyProviderError(error) };
    return;
  }
  if (signal.aborted) return;
  yield* missingMarkerEvents(keptText, cited, request.tools);
  yield { type: "finished" };
}

/** What a shown Tool call was asked, as text fields. */
function shownInput(toolName: string, input: unknown): Record<string, unknown> {
  const fields = (typeof input === "object" && input !== null ? input : {}) as Record<
    string,
    unknown
  >;
  const text = (value: unknown) => String(value ?? "");
  if (toolName === SEARCH_TOOL) return { query: text(fields.query) };
  if (toolName === USE_SKILL_TOOL) return { name: text(fields.name) };
  if (toolName === RUN_SKILL_SCRIPT_TOOL) return scriptCallInput(fields);
  return { skill: text(fields.skill), path: text(fields.path) };
}

/**
 * A request sized to a window: its estimated tokens, and what to learn from
 * the model's own count of it (see ./window).
 */
interface Sized {
  estimated: number;
  learn(actual: number): void;
}

/** The Answer and its records as one JSON object, streamed: its `answer` text as it grows. */
async function* structured(
  request: AnswerRequest,
  instructions: string,
  temperature: number | undefined,
  sized?: Sized,
): Attempt {
  const { signal } = request;
  const result = streamText({
    model: request.model,
    instructions,
    messages: request.messages,
    temperature,
    output: Output.object({
      schema: jsonSchema<{ answer: string; citations: unknown }>({
        type: "object",
        properties: {
          answer: {
            type: "string",
            description: "The Answer, in Markdown, with Citation markers.",
          },
          citations: recordsSchema,
        },
        required: ["answer", "citations"],
      }),
      name: "answer",
    }),
    abortSignal: signal,
    onError: () => undefined,
  });

  let raw = "";
  let emitted = "";
  const emit = function* (value: unknown): Generator<AnswerEngineEvent> {
    if (typeof value !== "string" || value === emitted) return;
    yield* textEdits(emitted, value);
    emitted = value;
  };
  const estimated = sized?.estimated;
  try {
    for await (const part of result.fullStream) {
      if (signal.aborted || part.type === "abort") return;
      if (part.type === "text-delta") {
        raw += part.text;
        const { value } = await parseAnswerJson(raw);
        yield* emit((value as { answer?: unknown } | undefined)?.answer);
      } else if (part.type === "finish-step" && part.usage.inputTokens) {
        sized?.learn(part.usage.inputTokens);
      } else if (part.type === "error") {
        const refused = emitted
          ? null
          : refusalOf(part.error, "structured-output", temperature, estimated);
        yield refused ?? { type: "failed", error: classifyProviderError(part.error) };
        return;
      }
    }
  } catch (error) {
    if (signal.aborted) return;
    const refused = emitted ? null : refusalOf(error, "structured-output", temperature, estimated);
    yield refused ?? { type: "failed", error: classifyProviderError(error) };
    return;
  }
  if (signal.aborted) return;
  const { value } = await parseAnswerJson(raw);
  const answer = (value as { answer?: unknown } | undefined)?.answer;
  // The model didn't answer in JSON after all: it can't do structured output.
  if (typeof answer !== "string" && !emitted) {
    yield { type: "unsupported" };
    return;
  }
  const records = parseRecords((value as { citations?: unknown } | undefined)?.citations);
  try {
    request.tools.cite(records);
  } catch (error) {
    console.error(`The Citations couldn't be recorded: ${messageOf(error)}`);
  }
  yield* emit(answer);
  // A small model often gives the records but leaves their markers out.
  if (typeof answer === "string") yield* missingMarkerEvents(emitted, records, request.tools);
  yield { type: "finished" };
}

/** A plain streamed Answer, with no Tools and no Citations. */
async function* plain(
  request: AnswerRequest,
  instructions: string,
  temperature: number | undefined,
  sized?: Sized,
): Attempt {
  const { signal } = request;
  const estimated = sized?.estimated;
  let produced = false;
  try {
    const result = streamText({
      model: request.model,
      instructions,
      messages: request.messages,
      temperature,
      abortSignal: signal,
      onError: () => undefined,
    });
    for await (const part of result.fullStream) {
      if (signal.aborted || part.type === "abort") return;
      if (part.type === "text-delta") {
        if (!part.text) continue;
        produced = true;
        yield { type: "text-delta", text: part.text };
      } else if (part.type === "finish-step" && part.usage.inputTokens) {
        sized?.learn(part.usage.inputTokens);
      } else if (part.type === "error") {
        const refused = produced ? null : refusalOf(part.error, null, temperature, estimated);
        yield refused ?? { type: "failed", error: classifyProviderError(part.error) };
        return;
      }
    }
  } catch (error) {
    if (signal.aborted) return;
    const refused = produced ? null : refusalOf(error, null, temperature, estimated);
    yield refused ?? { type: "failed", error: classifyProviderError(error) };
    return;
  }
  if (!signal.aborted) yield { type: "finished" };
}
