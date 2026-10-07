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
 * - "structured-output": for a model that can't call Tools, one search with the
 *   Question's text, its Passages in the instructions, and the Answer and its
 *   records returned as one JSON object.
 * - "none": for a model that can do neither, the same search, and a plain Answer.
 * The engine starts where it is told (or with Tools), and steps down when the
 * provider refuses Tools or structured output.
 *
 * Skills add two Tools to the loop: `use_skill` loads a Skill's instructions
 * and `read_skill_file` one of its files. With Skills but no Documents, the
 * loop runs with those alone; a model that can't call Tools answers without.
 */
import {
  APICallError,
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
import type { CitationSupport, ProviderError } from "../api";
import type { ChatLanguageModel } from "../providers/models";
import { classifyProviderError } from "../providers/providerErrors";

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

/** The Tools' work, done by the core: the engine only connects them to the model. */
export interface AnswerTools {
  /** Documents with Passages to search. With none, the model gets no Tools. */
  readonly documentCount: number;
  /** The document-search Tool. */
  searchDocuments(query: string, signal?: AbortSignal): Promise<SearchResultForModel>;
  /** Takes Citation records; returns what to tell the model about them. */
  cite(records: readonly CitationRecordInput[]): string;
}

/** The Skill Tools' work, done by the core. */
export interface AnswerSkillTools {
  /** Offer `use_skill`: there are Skills the model may load, listed in the instructions. */
  readonly loadable: boolean;
  /** A Skill's full instructions and its list of files, for the model. Throws for an unknown Skill. */
  useSkill(name: string): Promise<string>;
  /** One of a Skill's files, as text. Throws for a path outside the Skill. */
  readSkillFile(skill: string, path: string): Promise<string>;
}

export interface InstructionOptions {
  /** "structured-output" and "none": the Passages the search found (formatted, or a sentence saying there are none). */
  passages?: string;
  /** The model is offered the Skill Tools. */
  skillTools?: boolean;
}

export interface AnswerRequest {
  /**
   * The instructions for a way of citing. "no-documents": the User has no
   * Documents to search.
   */
  instructions(mode: CitationSupport | "no-documents", options?: InstructionOptions): string;
  /** The Question context, oldest first; the last message is the User's and ends with the Question. */
  messages: AnswerMessage[];
  /** The Question's own text: what a model that can't call Tools searches for. */
  question: string;
  /** The model to answer with, from `Core.prepareChatModel`. */
  model: ChatLanguageModel;
  tools: AnswerTools;
  /** The Skill Tools, or null when there are no Skills to use. */
  skills: AnswerSkillTools | null;
  /** How this model is known to cite, from earlier Answers. Unknown: try Tools first. */
  support?: CitationSupport;
  /** Stops generating. The stream then ends, with neither "finished" nor "failed". */
  signal: AbortSignal;
}

export type AnswerEngineEvent =
  /** How the model gives Citations, once its provider has accepted the request. */
  | { type: "support"; support: CitationSupport }
  /** More of the Answer's text (Markdown, with Citation markers), in order. */
  | { type: "text-delta"; text: string }
  /**
   * The last `length` characters of text were only a preamble to a search
   * ("Let me look that up"), or are being replaced: remove them.
   */
  | { type: "text-retracted"; length: number }
  /**
   * A Tool call started: a search of the Documents (by the model, or by the
   * engine for a model without Tools, `{ query }`), or a Skill Tool
   * (`use_skill` with `{ name }`, `read_skill_file` with `{ skill, path }`).
   */
  | { type: "tool-call-started"; id: string; tool: string; input: Record<string, unknown> }
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

/** The provider refused this way of answering: try the next one. Internal to the engine. */
type Attempt = AsyncGenerator<AnswerEngineEvent | { type: "unsupported" }>;

const SEARCH_TOOL = "search_documents";
const CITE_TOOL = "cite";
const USE_SKILL_TOOL = "use_skill";
const READ_SKILL_FILE_TOOL = "read_skill_file";
/** Tool calls the Answer shows, and text before which was a preamble. `cite` isn't one: its records become Citations. */
const SHOWN_TOOLS: ReadonlySet<string> = new Set([
  SEARCH_TOOL,
  USE_SKILL_TOOL,
  READ_SKILL_FILE_TOOL,
]);
const MARKER = /\[\^\d{1,4}\]/;

/**
 * Whether a provider refused a request because the model can't use `feature`:
 * e.g. Ollama's "model does not support tools", vLLM's "--enable-auto-tool-choice",
 * or a server that rejects `response_format`. Auth, rate limits and outages never count.
 */
export function isUnsupportedFeature(
  error: unknown,
  feature: "tools" | "structured-output",
): boolean {
  const cause = RetryError.isInstance(error) ? error.lastError : error;
  const subject =
    feature === "tools"
      ? /tool|function/i
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
  return subject.test(text) && refusal.test(text);
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

/** The engine on the Vercel AI SDK. */
export function createAiSdkAnswerEngine(options: AiSdkAnswerEngineOptions = {}): AnswerEngine {
  const maxSteps = options.maxSteps ?? MAX_STEPS;

  return {
    async *generate(request) {
      const failed = (error: unknown): AnswerEngineEvent => ({
        type: "failed",
        error: classifyProviderError(error),
      });
      const { signal } = request;

      // With no Documents there is nothing to search or cite: Skills alone, or a plain Answer.
      if (request.tools.documentCount === 0) {
        if (request.skills && (request.support ?? "tools") === "tools") {
          let unsupported = false;
          for await (const event of toolLoop(request, maxSteps, "no-documents")) {
            if (event.type === "unsupported") {
              unsupported = true;
              break;
            }
            yield event;
          }
          if (!unsupported || signal.aborted) return;
        }
        for await (const event of plain(request, request.instructions("no-documents"))) {
          if (event.type === "unsupported") return;
          yield event;
        }
        return;
      }

      // A model without Tools searches once, with the Question's text; the result is kept
      // for the next way of answering if the provider refuses this one.
      let searched: string | undefined;
      async function* searchOnce(): AsyncGenerator<AnswerEngineEvent, string> {
        if (searched !== undefined) return searched;
        const id = "question-search";
        yield {
          type: "tool-call-started",
          id,
          tool: SEARCH_TOOL,
          input: { query: request.question },
        };
        try {
          const result = await request.tools.searchDocuments(request.question, signal);
          yield { type: "tool-call-finished", id, ok: true, resultCount: result.passageCount };
          searched = result.text;
        } catch (error) {
          if (signal.aborted) throw error;
          console.error(error);
          yield { type: "tool-call-finished", id, ok: false, resultCount: null };
          searched = "The search of the User's Documents failed.";
        }
        return searched;
      }

      const order: CitationSupport[] = ["tools", "structured-output", "none"];
      for (let mode = order.indexOf(request.support ?? "tools"); mode < order.length; mode++) {
        const support = order[mode] as CitationSupport;
        let attempt: Attempt;
        try {
          if (support === "tools") attempt = toolLoop(request, maxSteps, "tools");
          else {
            const passages: string = yield* searchOnce();
            if (signal.aborted) return;
            attempt =
              support === "structured-output"
                ? structured(request, request.instructions(support, { passages }))
                : plain(request, request.instructions(support, { passages }));
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
function skillTools(skills: AnswerSkillTools): ToolSet {
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
  return tools;
}

/**
 * The Tool-calling loop: search, cite, answer; with Skills, load them as
 * needed. "no-documents": the Skill Tools alone, with nothing to cite.
 */
async function* toolLoop(
  request: AnswerRequest,
  maxSteps: number,
  mode: "tools" | "no-documents",
): Attempt {
  const { signal } = request;
  /** Passages each search gave, by Tool call. */
  const results = new Map<string, number>();
  const documentTools: ToolSet = {
    [SEARCH_TOOL]: tool({
      description:
        "Search the User's Documents. Returns the Passages that best match, each with an id, its Document and its pages.",
      inputSchema: jsonSchema<{ query: string }>({
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "What to look for: words likely to be in the Passages, in their language.",
          },
        },
        required: ["query"],
      }),
      execute: async ({ query }, { toolCallId, abortSignal }) => {
        const result = await request.tools.searchDocuments(String(query ?? ""), abortSignal);
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
      execute: async ({ citations }) => request.tools.cite(parseRecords(citations)),
    }),
  };
  const tools: ToolSet = {
    ...(mode === "tools" ? documentTools : {}),
    ...(request.skills ? skillTools(request.skills) : {}),
  };

  const result = streamText({
    model: request.model,
    instructions: request.instructions(mode, { skillTools: request.skills !== null }),
    messages: request.messages,
    tools,
    stopWhen: stepCountIs(maxSteps) as StopCondition<ToolSet>,
    // The last step must write the Answer.
    prepareStep: ({ stepNumber }) => (stepNumber >= maxSteps - 1 ? { toolChoice: "none" } : {}),
    abortSignal: signal,
    // Errors arrive as stream parts; don't also log them.
    onError: () => undefined,
  });

  /** Text of the current step, and of every step before it that was kept. */
  let stepText = "";
  let keptText = "";
  let stepTools: string[] = [];
  let produced = false;
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
        case "tool-call":
          produced = true;
          stepTools.push(part.toolName);
          if (SHOWN_TOOLS.has(part.toolName)) {
            yield {
              type: "tool-call-started",
              id: part.toolCallId,
              tool: part.toolName,
              input: shownInput(part.toolName, part.input),
            };
          }
          break;
        case "tool-result":
          if (SHOWN_TOOLS.has(part.toolName)) {
            yield {
              type: "tool-call-finished",
              id: part.toolCallId,
              ok: true,
              resultCount: results.get(part.toolCallId) ?? null,
            };
          }
          break;
        case "tool-error":
          if (SHOWN_TOOLS.has(part.toolName)) {
            // A failed search is the app's problem; a Skill or file that isn't there, the model's.
            if (part.toolName === SEARCH_TOOL && !signal.aborted) console.error(part.error);
            yield { type: "tool-call-finished", id: part.toolCallId, ok: false, resultCount: null };
          }
          break;
        case "finish-step": {
          // Text before a search or a Skill, or before records with no marker in it, was a preamble.
          const preamble =
            stepTools.some((name) => SHOWN_TOOLS.has(name)) ||
            (stepTools.includes(CITE_TOOL) && !MARKER.test(stepText));
          if (preamble && stepText) yield { type: "text-retracted", length: stepText.length };
          else keptText += stepText;
          stepText = "";
          break;
        }
        case "error":
          if (!produced && isUnsupportedFeature(part.error, "tools")) {
            yield { type: "unsupported" };
            return;
          }
          yield { type: "failed", error: classifyProviderError(part.error) };
          return;
      }
    }
  } catch (error) {
    if (signal.aborted) return;
    if (!produced && isUnsupportedFeature(error, "tools")) {
      yield { type: "unsupported" };
      return;
    }
    yield { type: "failed", error: classifyProviderError(error) };
    return;
  }
  if (!signal.aborted) yield { type: "finished" };
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
  return { skill: text(fields.skill), path: text(fields.path) };
}

/** The Answer and its records as one JSON object, streamed: its `answer` text as it grows. */
async function* structured(request: AnswerRequest, instructions: string): Attempt {
  const { signal } = request;
  const result = streamText({
    model: request.model,
    instructions,
    messages: request.messages,
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
  const emit = async function* (value: unknown): AsyncGenerator<AnswerEngineEvent> {
    if (typeof value !== "string" || value === emitted) return;
    if (value.startsWith(emitted)) {
      yield { type: "text-delta", text: value.slice(emitted.length) };
    } else {
      if (emitted) yield { type: "text-retracted", length: emitted.length };
      yield { type: "text-delta", text: value };
    }
    emitted = value;
  };
  try {
    for await (const part of result.fullStream) {
      if (signal.aborted || part.type === "abort") return;
      if (part.type === "text-delta") {
        raw += part.text;
        const { value } = await parseAnswerJson(raw);
        yield* emit((value as { answer?: unknown } | undefined)?.answer);
      } else if (part.type === "error") {
        if (!emitted && isUnsupportedFeature(part.error, "structured-output")) {
          yield { type: "unsupported" };
          return;
        }
        yield { type: "failed", error: classifyProviderError(part.error) };
        return;
      }
    }
  } catch (error) {
    if (signal.aborted) return;
    if (!emitted && isUnsupportedFeature(error, "structured-output")) {
      yield { type: "unsupported" };
      return;
    }
    yield { type: "failed", error: classifyProviderError(error) };
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
  yield* emit(answer);
  try {
    request.tools.cite(parseRecords((value as { citations?: unknown } | undefined)?.citations));
  } catch (error) {
    console.error(`The Citations couldn't be recorded: ${messageOf(error)}`);
  }
  yield { type: "finished" };
}

/** A plain streamed Answer, with no Tools and no Citations. */
async function* plain(request: AnswerRequest, instructions: string): Attempt {
  const { signal } = request;
  try {
    const result = streamText({
      model: request.model,
      instructions,
      messages: request.messages,
      abortSignal: signal,
      onError: () => undefined,
    });
    for await (const part of result.fullStream) {
      if (signal.aborted || part.type === "abort") return;
      if (part.type === "text-delta") {
        if (part.text) yield { type: "text-delta", text: part.text };
      } else if (part.type === "error") {
        yield { type: "failed", error: classifyProviderError(part.error) };
        return;
      }
    }
  } catch (error) {
    if (signal.aborted) return;
    yield { type: "failed", error: classifyProviderError(error) };
    return;
  }
  if (!signal.aborted) yield { type: "finished" };
}
