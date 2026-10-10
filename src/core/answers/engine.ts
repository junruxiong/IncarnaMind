/**
 * The Answer engine: the one place that turns Question context into a
 * streamed Answer by calling a chat model. Everything else about Answers
 * (building the context, the Tools' work, Citations, writing the Answer into
 * the Mind, stop and regenerate, the event stream) depends only on this port,
 * so the agent layer behind it can change without touching the rest.
 *
 * How the model cites depends on what it can do (see `CitationSupport`):
 * - "tools": a Tool-calling loop (ADR-0007), run by a Run engine (see
 *   ../runs/engine; AI SDK 7 by default). The model searches the Documents
 *   with `search_documents` as often as it needs, and gives the records of its
 *   Citation markers with `cite`. The Answer keeps what is its own on top of
 *   the engine's events: the search phase, preambles taken back, the records
 *   and the markers they place.
 * - "structured-output": for a model that can't call Tools, or a small local
 *   model, which doesn't cite in the loop (see `citingMode` in
 *   ../providers/ollamaModels): one search, its Passages in the instructions,
 *   and the Answer and its records returned as one JSON object. A follow-up
 *   Question is first rewritten by the model into a search query that stands
 *   on its own (see `searchQuery`).
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
 * again without, and that model gets none from then on. These retries, and
 * stepping down from Tools, are the Answer's: the Run engine only reports
 * what the provider refused.
 *
 * Every Tool the loop offers has one shape and comes from a Tool provider
 * (see ../tools): the Documents' two (made here, from `AnswerTools`), and
 * those the request brings: the Skills' (`use_skill`, `read_skill_file`,
 * `run_skill_script`, see ../skills/tools) and each Connector's. A call that
 * asks the User first waits in the request's `gate`, which the engine awaits
 * before each call. The gate is told whether the Run had read untrusted
 * content first (`GatedCall.tainted`): the loop records it, above the engine,
 * from the results of the Tools that declare them untrusted
 * (docs/designs/agent-extensibility.md §4.6). The loop names none of the
 * Tools but the Documents' own: every call shows as a Tool-call card, from
 * its provider, except `cite`, whose records become Citations; results that
 * aren't Passages are never cited. With Skills or Connector Tools but no
 * Documents, the loop runs with those alone; a model that can't call Tools
 * answers without.
 */
import { generateText, jsonSchema, Output, parsePartialJson, streamText } from "ai";
import type { AnswerPhase, CitationSupport, ProviderError } from "../api";
import type { ChatLanguageModel, ContextWindow } from "../providers/models";
import {
  classifyProviderError,
  contextOverflow,
  refusesFeature,
} from "../providers/providerErrors";
import { createAiSdkRunEngine } from "../runs/aiSdkEngine";
import type { GateDecision, RunEngine, RunMessage, RunToolCall, RunWindow } from "../runs/engine";
import { DOCUMENT_TOOLS, offeredTools, type Tool, type ToolProviderInfo } from "../tools";
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
  /**
   * Where the quote is, as the Passage's marks name it (ADR-0011): "p. 4",
   * "slide 4", "§ 2.1 Sensitivity", "Revenue, rows 12–14", "lines 120–134".
   */
  location?: string | null;
  /** For a PDF (or a deck), instead of `location`: the page or two consecutive pages the quote is on. */
  pageFrom?: number | null;
  pageTo?: number | null;
  /** A short quote, copied word for word from the Passage. */
  quote: string;
}

/** How records came, for `AnswerTools.cite`. */
export interface CiteOptions {
  /**
   * In structured output, which gets no word back to fix its records with
   * (see `structured`): a Passage named by its number alone ("1") is "P1",
   * and a quote that isn't on the pages its record names, but is word for
   * word on a page of the Passage it names, or on two consecutive ones, is
   * cited there, as `cite`'s feedback lets a model in the Tool loop do.
   */
  structured?: boolean;
}

/** What a search gives the model. */
export interface SearchResultForModel {
  /** The Passages, formatted for the model, or a sentence saying there are none. */
  text: string;
  passageCount: number;
  /**
   * Each Passage's relevance, in the order `text` has them (reading order):
   * 0 the most relevant. A window too small for all of them keeps the most
   * relevant (see ./window). Absent: their order is their relevance.
   */
  ranks?: number[];
}

/** A language the Documents to search are in, and how many of them are. */
export interface DocumentLanguage {
  /** In English, e.g. "Chinese". */
  language: string;
  documents: number;
}

/**
 * Searching and citing the Documents, done by the core: the engine makes the
 * Documents' Tools from it, and the ways of answering without Tools use it too.
 */
export interface AnswerTools {
  /** Documents with Passages to search. With none, there is no document search. */
  readonly documentCount: number;
  /**
   * The languages the Documents to search are in, the most common first: the
   * search Tool names them, so the model searches again in theirs when the
   * Question is in another language. Empty or absent when they can't be told.
   */
  readonly documentLanguages?: readonly DocumentLanguage[];
  /** The document-search Tool. */
  searchDocuments(query: string, signal?: AbortSignal): Promise<SearchResultForModel>;
  /** Takes Citation records; returns what to tell the model about them. */
  cite(records: readonly CitationRecordInput[], options?: CiteOptions): string;
  /** Whether a valid record was taken for this marker: the engine places its marker if the model left it out. */
  hasRecord?(marker: number): boolean;
}

export interface InstructionOptions {
  /** "structured-output" and "none": the Passages the search found (formatted, or a sentence saying there are none). */
  passages?: string;
  /** The model is offered the Skills' Tools. */
  skillTools?: boolean;
  /** The model is offered Connectors' Tools. */
  connectorTools?: boolean;
}

/** A Tool call at the Answer's gate. */
export interface GatedCall extends RunToolCall {
  /**
   * The Run had read untrusted content before this call: the result of a
   * Tool that declares `untrustedResult`, such as a search that found
   * Passages (docs/designs/agent-extensibility.md §4.6).
   */
  tainted: boolean;
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
  /** Searching and citing the Documents: the Documents' Tools, and the search of a model without Tools. */
  documents: AnswerTools;
  /**
   * The other Tools a model that can call Tools is offered, from their
   * providers: the Skills' and the Connectors' (see ../tools). Within a
   * window, what they return is cut to the room left.
   */
  tools: readonly Tool[];
  /**
   * Awaited before each call of an offered Tool (see `RunRequest.gate`): the
   * Answer's approvals decide there whether it asks the User first. Without
   * it, every call runs.
   */
  gate?(tool: Tool, call: GatedCall): Promise<GateDecision>;
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
   * A Tool call started, for its Tool-call card: any Tool but `cite`, or the
   * search the engine makes for a model without Tools. `tool` is its name at
   * its `provider` (a Connector's "create_issue"); `input`, what the card
   * shows of the arguments (e.g. a search's `{ query }`; a Connector's, as sent).
   */
  | {
      type: "tool-call-started";
      id: string;
      provider: ToolProviderInfo;
      tool: string;
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
  /** Runs the Tool-calling loop. Defaults to the one on AI SDK 7. */
  runEngine?: RunEngine;
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

/** Who provides document search and `cite`: IncarnaMind's own Documents. */
const DOCUMENTS_PROVIDER: ToolProviderInfo = {
  kind: "documents",
  id: "documents",
  name: "Documents",
};

/** "English", "English and Chinese", "English, Chinese and French". */
const listed = (items: readonly string[]) =>
  items.length < 2 ? (items[0] ?? "") : `${items.slice(0, -1).join(", ")} and ${items.at(-1)}`;

/**
 * The search Tool's description: what it returns, and the languages the
 * Documents are in, with how many are in each. Search finds a query best in
 * a Document's own language (ADR-0009), so a Question in another language is
 * searched in theirs too.
 */
export function searchToolDescription(languages: readonly DocumentLanguage[]): string {
  const base =
    "Search the User's Documents. Returns the Passages that best match, each with an id, its Document and where it is (pages, slides, sections, rows or lines). The search sees only the query, not the conversation: write it to stand on its own.";
  if (languages.length === 0) return base;
  const names = languages.map((each) =>
    languages.length > 1 ? `${each.language} (${each.documents})` : each.language,
  );
  return `${base} The Documents are in ${listed(names)}. Search finds Passages best in their own language: when the Question is in another language than the Documents it may be about, search with the query translated into their language too.`;
}
const MARKER = /\[\^\d{1,4}\]/;

/**
 * What a one-pass attempt yields when its provider refused the request before
 * anything came back: the temperature, if one was sent, or else `feature`;
 * or, for a request sized to a window (`estimated` tokens), that it was too
 * long. Null for any other failure. (The Tool loop's refusals come from its
 * Run engine, as "refused" events.)
 */
function refusalOf(
  error: unknown,
  feature: "structured-output" | null,
  temperature: number | undefined,
  estimated?: number,
): Unsupported | TemperatureRefused | Overflow | null {
  const overflow = estimated === undefined ? null : contextOverflow(error);
  if (overflow && estimated !== undefined) {
    return { type: "overflow", promptTokens: overflow.promptTokens, estimated };
  }
  if (temperature !== undefined && refusesFeature(error, "temperature")) {
    return { type: "temperature-refused" };
  }
  if (feature && refusesFeature(error, feature)) return { type: "unsupported" };
  return null;
}

const recordsSchema = {
  type: "array",
  items: {
    type: "object",
    properties: {
      marker: { type: "integer", description: "The marker's number: 1 for [^1]." },
      passage: { type: "string", description: "The Passage's id, e.g. P1." },
      location: {
        type: "string",
        description:
          'Where the quote is, as the Passage names it: e.g. "p. 4", "slide 4", "§ 2.1 Sensitivity", "Revenue, rows 12–14" or "lines 120–134". One place, or two in a row.',
      },
      pageFrom: {
        type: "integer",
        description:
          "For a PDF, instead of location: the page the quote starts on. Leave out for other Passages.",
      },
      pageTo: {
        type: "integer",
        description: "For a PDF: the page the quote ends on: the same page, or the next one.",
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
    const { marker, passage, location, pageFrom, pageTo, quote } = item as Record<string, unknown>;
    const number = typeof marker === "string" ? Number.parseInt(marker, 10) : marker;
    if (typeof number !== "number" || typeof passage !== "string" || typeof quote !== "string") {
      return [];
    }
    const page = (value: unknown) =>
      typeof value === "number" ? value : typeof value === "string" ? Number(value) : null;
    return [
      {
        marker: number,
        passage,
        ...(typeof location === "string" && location.trim() ? { location } : {}),
        pageFrom: page(pageFrom),
        pageTo: page(pageTo),
        quote,
      },
    ];
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

/**
 * The Answer engine: its Tool-calling loop on a Run engine, and the ways of
 * answering without Tools on the model layer (the Vercel AI SDK).
 */
export function createAiSdkAnswerEngine(options: AiSdkAnswerEngineOptions = {}): AnswerEngine {
  const maxSteps = options.maxSteps ?? MAX_STEPS;
  const engine = options.runEngine ?? createAiSdkRunEngine();
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
      const onePass = (mode: CitationSupport | "no-documents", found?: SearchResultForModel) =>
        fitted(async function* () {
          if (!budget) {
            const instructions = request.instructions(mode, { passages: found?.text });
            yield* tempered((sent) =>
              mode === "structured-output"
                ? structured(request, instructions, sent)
                : plain(request, instructions, sent),
            );
            return;
          }
          const kept =
            found === undefined
              ? undefined
              : budget.fitPassages(
                  found.text,
                  request.instructions(mode, { passages: "" }),
                  request.question,
                  found.ranks,
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
      if (request.documents.documentCount === 0) {
        if (request.tools.length > 0 && (request.support ?? "tools") === "tools") {
          let unsupported = false;
          for await (const event of fitted(() =>
            tempered((sent) => toolLoop(request, engine, maxSteps, "no-documents", sent, budget)),
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
      let searched: SearchResultForModel | undefined;
      async function* searchOnce(): AsyncGenerator<AnswerEngineEvent, SearchResultForModel> {
        if (searched !== undefined) return searched;
        const query = await searchQuery(request, temperature());
        signal.throwIfAborted();
        const id = "question-search";
        yield { type: "phase", phase: "searching" };
        yield {
          type: "tool-call-started",
          id,
          provider: DOCUMENTS_PROVIDER,
          tool: DOCUMENT_TOOLS.search,
          input: { query },
        };
        try {
          const result = await request.documents.searchDocuments(query, signal);
          yield { type: "tool-call-finished", id, ok: true, resultCount: result.passageCount };
          searched = result;
        } catch (error) {
          if (signal.aborted) throw error;
          console.error(error);
          yield { type: "tool-call-finished", id, ok: false, resultCount: null };
          searched = { text: "The search of the User's Documents failed.", passageCount: 0 };
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
              tempered((sent) => toolLoop(request, engine, maxSteps, "tools", sent, budget)),
            );
          } else {
            const found: SearchResultForModel = yield* searchOnce();
            if (signal.aborted) return;
            attempt = onePass(support, found);
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

/** What one go of the Tool-calling loop keeps of its calls of the Documents' Tools. */
interface DocumentToolsHooks {
  /** A search's result as the model gets it: within a window, the Passages that fit (see ./window). */
  fit(found: SearchResultForModel): SearchResultForModel;
  /** A search gave the model `count` Passages: its Tool-call card says so. */
  searched(toolCallId: string, count: number): void;
  /** The model gave these records: the engine places their markers if it leaves them out. */
  cited(records: readonly CitationRecordInput[]): void;
}

/**
 * The Documents as a Tool provider, for one go of the Tool-calling loop:
 * `search_documents`, which reads the Documents, and `cite`, whose records
 * become this Answer's Citations: nothing beyond it.
 */
export function documentTools(
  documents: AnswerTools,
  hooks: DocumentToolsHooks,
): { search: Tool; cite: Tool } {
  const own = (name: string) => ({ name, provider: DOCUMENTS_PROVIDER, providerTool: name });
  return {
    search: {
      ...own(DOCUMENT_TOOLS.search),
      description: searchToolDescription(documents.documentLanguages ?? []),
      inputSchema: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description:
              'What to look for: words likely to be in the Passages, in their language. Name what the Question refers to instead of pronouns or words that point back, such as "it", "they" or "that paper".',
          },
        },
        required: ["query"],
      },
      shownInput: ({ query }) => ({ query: String(query ?? "") }),
      effects: () => [{ action: "read", scope: { kind: "documents" } }],
      // The User didn't write their Documents.
      untrustedResult: true,
      async call({ query }, { toolCallId, signal }) {
        let found: SearchResultForModel;
        try {
          found = await documents.searchDocuments(String(query ?? ""), signal);
        } catch (error) {
          // A failed search is the app's problem, so it is logged; the model is told it failed.
          if (!signal.aborted) console.error(error);
          throw error;
        }
        const result = hooks.fit(found);
        hooks.searched(toolCallId, result.passageCount);
        return result.text;
      },
    },
    cite: {
      ...own(DOCUMENT_TOOLS.cite),
      description:
        "Record the Citations of your Answer: for each marker such as [^1], the Passage, where its quote is (a page, slide, section, or rows or lines, or two in a row), and a short quote copied word for word from the Passage.",
      inputSchema: {
        type: "object",
        properties: { citations: recordsSchema },
        required: ["citations"],
      },
      effects: () => [],
      untrustedResult: false,
      async call({ citations }) {
        const records = parseRecords(citations);
        hooks.cited(records);
        return documents.cite(records);
      },
    },
  };
}

/** What a call the gate lets through gets: it runs. */
const RUN: GateDecision = { run: true };

/** A message of Question context as a Run's. */
const runMessage = (message: AnswerMessage): RunMessage =>
  message.role === "user"
    ? { role: "user", text: message.content }
    : { role: "assistant", text: message.content, toolCalls: [] };

/**
 * The Tools' definitions as the window's estimate counts them: by name, each
 * with its description and JSON Schema (written out as the AI SDK's `ToolSet`
 * was before the Run engine, so requests are sized as they were).
 */
const definitions = (tools: readonly Tool[]) =>
  JSON.stringify(
    Object.fromEntries(
      tools.map((each) => [
        each.name,
        { description: each.description, inputSchema: { jsonSchema: each.inputSchema } },
      ]),
    ),
  );

/**
 * The Tool-calling loop, on a Run engine (see ../runs/engine): search, cite,
 * answer; with Skills, load them as needed; with Connectors, call their Tools
 * where they help. "no-documents": the Skill and Connector Tools alone, with
 * nothing to cite. Within a window, its requests are kept within it (see
 * ./window). A request the provider refused is the engine's "refused" event;
 * what to try next is `generate`'s.
 */
async function* toolLoop(
  request: AnswerRequest,
  engine: RunEngine,
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
  // The Documents' Tools fit what they return themselves: a search's Passages, and records take no room.
  const documents =
    mode === "tools"
      ? documentTools(request.documents, {
          fit: (found) =>
            loop ? loop.passages(found.text, found.passageCount, found.ranks) : found,
          searched: (toolCallId, count) => results.set(toolCallId, count),
          cited: (records) => cited.push(...records),
        })
      : null;
  const search = documents?.search ?? null;
  const cite = documents?.cite ?? null;
  /**
   * Whether this Run has read untrusted content: from the first call of a
   * Tool with `untrustedResult` that returned some, each call's gate is told.
   */
  let tainted = false;
  /** A Tool as the engine calls it: one with untrusted results taints the Run once it returns some. */
  const tracked = (tool: Tool): Tool =>
    tool.untrustedResult
      ? {
          ...tool,
          async call(input, context) {
            const result = await tool.call(input, context);
            // A search that found no Passages read nothing of the Documents.
            const read =
              tool === search ? (results.get(context.toolCallId) ?? 0) > 0 : result.trim() !== "";
            if (read) tainted = true;
            return result;
          },
        }
      : tool;
  /** The Tools offered, by the name the model calls them; never a Connector's with one of ours. */
  const offered = new Map(
    offeredTools([...(documents ? [documents.search, documents.cite] : []), ...request.tools]).map(
      (each) => [each.name, each],
    ),
  );
  const tools = [...offered.values()];
  const instructions = request.instructions(mode, {
    skillTools: request.tools.some((each) => each.provider.kind === "skills"),
    connectorTools: tools.some((each) => each.provider.kind === "connector"),
  });

  // Within a window: the Question context that fits beside the instructions, the Tools and a search.
  let messages = request.messages;
  let estimated: number | undefined;
  let window: RunWindow | undefined;
  if (budget) {
    const fit = budget.fit({
      instructions,
      messages,
      question: request.question,
      tools: definitions(tools),
      reserve: SEARCH_RESERVE_TOKENS,
    });
    if (!fit.ok) {
      yield { type: "failed", error: fit.error };
      return;
    }
    ({ messages, estimated } = fit);
    const account = budget.loop(fit.estimated);
    loop = account;
    window = {
      // What another provider's Tool returns (or the gate gives instead) is cut to the room left.
      fitResult: (text, tool) =>
        tool === search?.name || tool === cite?.name ? text : account.result(text),
      canCallTools: () => account.canCallTools(),
      stepFinished: ({ inputTokens, outputTokens }) =>
        account.stepFinished({ inputTokens, outputTokens }),
    };
  }

  /** Text of the current step, and of every step before it that was kept. */
  let stepText = "";
  let keptText = "";
  /** The Tools the current step called. */
  let stepTools: Tool[] = [];
  /** The Tool of each call, by its id. */
  const calls = new Map<string, Tool>();
  /** Searches running now: the Answer is "searching" while there are any. */
  let searching = 0;
  for await (const event of engine.run({
    model: request.model,
    instructions,
    messages: messages.map(runMessage),
    tools: tools.map(tracked),
    maxSteps,
    gate: async (call) => {
      const tool = offered.get(call.tool);
      return tool && request.gate ? request.gate(tool, { ...call, tainted }) : RUN;
    },
    window,
    temperature,
    signal,
  })) {
    if (signal.aborted) return;
    switch (event.type) {
      case "text-delta": {
        let { text } = event;
        // A new step's text goes on from the last kept text in a new paragraph.
        if (stepText === "" && keptText !== "" && !/\s$/.test(keptText)) text = `\n\n${text}`;
        stepText += text;
        yield { type: "text-delta", text };
        break;
      }
      case "tool-call": {
        const called = offered.get(event.tool);
        if (!called) break;
        calls.set(event.id, called);
        stepTools.push(called);
        if (called === search && searching++ === 0) {
          yield { type: "phase", phase: "searching" };
        }
        // Every call has a Tool-call card but `cite`'s: its records become Citations.
        if (called !== cite) {
          yield {
            type: "tool-call-started",
            id: event.id,
            provider: called.provider,
            tool: called.providerTool,
            input: called.shownInput ? called.shownInput(event.input) : event.input,
          };
        }
        break;
      }
      case "tool-result": {
        // A failed call (a Skill or file that isn't there, a Connector's failure or a declined
        // consent) is told to the model; its card says it failed.
        const called = calls.get(event.id);
        if (called && called !== cite) {
          yield {
            type: "tool-call-finished",
            id: event.id,
            ok: event.ok,
            resultCount: results.get(event.id) ?? null,
          };
        }
        if (called === search && --searching === 0) {
          yield { type: "phase", phase: "writing" };
        }
        break;
      }
      case "step-finished": {
        // Text before a call with a Tool-call card (a search, a Skill, a Connector's Tool), or
        // before records with no marker in it, was a preamble.
        const preamble =
          stepTools.some((each) => each !== cite) ||
          (cite !== null && stepTools.includes(cite) && !MARKER.test(stepText));
        if (preamble && stepText) yield { type: "text-retracted", length: stepText.length };
        else keptText += stepText;
        stepText = "";
        stepTools = [];
        break;
      }
      case "refused":
        // Too long only with a window, so sized to one.
        yield event.what === "tools"
          ? { type: "unsupported" }
          : event.what === "temperature"
            ? { type: "temperature-refused" }
            : {
                type: "overflow",
                promptTokens: event.promptTokens ?? null,
                estimated: estimated ?? 0,
              };
        return;
      case "failed":
        yield { type: "failed", error: event.error };
        return;
      case "finished":
        yield* missingMarkerEvents(keptText, cited, request.documents);
        yield { type: "finished" };
        return;
    }
  }
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
    // The answer is written: what `cite` would tell the model can't reach it. It places the records instead.
    request.documents.cite(records, { structured: true });
  } catch (error) {
    console.error(`The Citations couldn't be recorded: ${messageOf(error)}`);
  }
  yield* emit(answer);
  // A small model often gives the records but leaves their markers out.
  if (typeof answer === "string") yield* missingMarkerEvents(emitted, records, request.documents);
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
