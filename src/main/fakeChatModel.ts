/**
 * A scripted chat model for the smoke tests: no network and no keys. Only
 * test builds (`electron-vite build --mode test`) contain this file, and only
 * `INCARNAMIND_FAKE_CHAT=1` turns it on.
 *
 * With no Documents (no Tools offered), every Question gets the same short
 * Markdown Answer, streamed a few words at a time. With Documents, it does
 * what a Tool-calling model does: it searches for the Question, cites the line
 * of the first Passage that shares a word with the Question (on the page, or
 * at the Location, that line is on), then answers with a marker. A Question
 * with "misquote" in it gets a quote that isn't on the page; one with "each"
 * cites the first Passage of each Document found, with a marker for each; one
 * with "rows" quotes two lines, so a sheet's Citation covers a range of rows.
 *
 * With Skills: a Question that names a listed Skill gets it loaded with
 * use_skill first. An Answer that follows a Skill (forced, or loaded) starts
 * by saying which.
 *
 * With Connectors: a Question that names one of their Tools (e.g. "book_boat")
 * gets it called once first, its required arguments set to the Question's
 * last word; the Answer then starts with what the call gave back.
 *
 * With Skill scripts: a Question that names a listed Skill and one of its
 * scripts by path (e.g. "greeter scripts/hello.js") gets it run once first
 * with run_skill_script, with the Question's last word as its one argument;
 * the Answer then starts with what the script printed (or why it didn't run).
 *
 * For automatic tagging it tags a Document with every Tag whose name is a word in it.
 */
import { MockLanguageModelV4 } from "ai/test";
import type { ChatModelFactory } from "../core";
import { EXCERPT_END, EXCERPT_START } from "../core/tags/classify";

type StreamResult = Awaited<ReturnType<MockLanguageModelV4["doStream"]>>;
type StreamPart = StreamResult["stream"] extends ReadableStream<infer Part> ? Part : never;
type Prompt = Parameters<MockLanguageModelV4["doStream"]>[0]["prompt"];
type GenerateOptions = Parameters<MockLanguageModelV4["doGenerate"]>[0];

/** Time between two streamed words: slow enough for a test to watch the Answer grow. */
const WORD_DELAY_MS = 60;

/** The scripted Answer to a Question, when there are no Documents; it says which Skill it follows, if any. */
function fakeAnswer(question: string, skill: string | null = null): string {
  return [
    ...(skill ? [`Following the Skill ${skill}.`, ""] : []),
    `This is a **scripted** Answer to: ${question}`,
    "",
    "- It streams in a few words at a time",
    "- Math renders too: $E = mc^2$",
    "",
    "```js",
    'console.log("IncarnaMind");',
    "```",
    "",
    "That is all.",
  ].join("\n");
}

/** The scripted Answer when it cited something: one sentence with a marker. */
const CITED_ANSWER = "Your Documents answer this [^1].\n\nThat is all.";

/** The text of the last user message: the Question, at the end of its Question context. */
function lastQuestion(prompt: Prompt): string {
  const last = prompt.filter((message) => message.role === "user").at(-1);
  if (!last || typeof last.content === "string") return "";
  const text = last.content.map((part) => (part.type === "text" ? part.text : "")).join("");
  return text.split("\n\n").at(-1) ?? "";
}

/** The text results of the Tool calls so far, by Tool name, oldest first. */
function toolResults(prompt: Prompt, tool: string): string[] {
  const results: string[] = [];
  for (const message of prompt) {
    if (message.role !== "tool") continue;
    for (const part of message.content) {
      if (part.type === "tool-result" && part.toolName === tool && part.output.type === "text") {
        results.push(part.output.value);
      }
    }
  }
  return results;
}

/** The system prompt's text. */
function systemOf(prompt: Prompt): string {
  return prompt
    .filter((message) => message.role === "system")
    .map((message) => (typeof message.content === "string" ? message.content : ""))
    .join("\n");
}

/** The Skill the Answer follows: the one the Question forces, or one loaded with use_skill. */
function skillFollowed(prompt: Prompt): string | null {
  const forced = /the User chose the Skill "([^"]+)"/.exec(systemOf(prompt))?.[1];
  if (forced) return forced;
  const loaded = toolResults(prompt, "use_skill").at(-1);
  return (loaded && /<skill name="([^"]+)">/.exec(loaded)?.[1]) || null;
}

/** A listed Skill the Question names, to load before answering. */
function skillNamed(prompt: Prompt, question: string): string | null {
  const listed = [...systemOf(prompt).matchAll(/<skill name="([^"]+)">/g)].map((match) => match[1]);
  return listed.find((name) => name && question.toLowerCase().includes(name)) ?? null;
}

/** A Passage of a search result, as the model reads it. */
interface ShownPassage {
  id: string;
  document: string;
  /** A PDF's first page. */
  page: number | null;
  /** Any other kind's: where the Passage is, e.g. "slides 1–5" (ADR-0011). */
  location: string | null;
  text: string;
}

function passagesIn(result: string): ShownPassage[] {
  const pattern =
    /<passage id="([^"]+)" document="([^"]*)"(?: pages="(\d+)(?:-\d+)?")?(?: location="([^"]+)")?>\n([\s\S]*?)\n<\/passage>/g;
  return [...result.matchAll(pattern)].map((match) => ({
    id: match[1] as string,
    document: match[2] as string,
    page: match[3] ? Number(match[3]) : null,
    location: match[4] ?? null,
    text: match[5] ?? "",
  }));
}

/** A mark where a new Unit starts in a Passage: "[p. 4] ", "[slide 4] ", "[§ 2.1 Sensitivity] "… */
const UNIT_MARK = /^\[(p\. \d+|slides? [^\]]+|§[^\]]*|[^\]]*rows? [\d–-]+|lines? [\d–-]+)\] /;
/** Where a slide's speaker notes start in a Passage. */
const NOTES_MARK = /^\[speaker notes\] /;

/**
 * The record for a Passage: its line that shares the most words with the
 * Question, where that line is (the page, or the Unit a mark before it names,
 * or else the Passage's own location), and the line as the quote. A Question
 * that says "rows" quotes that line and the next, so it covers a range of rows.
 */
function recordFor(
  passage: ShownPassage,
  question: string,
  marker: number,
): { record: Record<string, unknown>; shared: number } | null {
  const words = question
    .toLowerCase()
    .split(/\W+/)
    .filter((word) => word.length >= 4);
  let page = passage.page;
  let location = passage.location;
  const lines: { line: string; page: number | null; location: string | null }[] = [];
  for (const raw of passage.text.split("\n")) {
    let rest = raw;
    const mark = UNIT_MARK.exec(rest);
    if (mark) {
      const label = mark[1] as string;
      if (label.startsWith("p. ")) page = Number(label.slice(3));
      else location = label;
      rest = rest.slice(mark[0].length);
    }
    rest = rest.replace(NOTES_MARK, "");
    lines.push({ line: rest.trim(), page, location });
  }
  let chosen = -1;
  let best = -1;
  lines.forEach(({ line }, index) => {
    if (!line) return;
    const shared = words.filter((word) => line.toLowerCase().includes(word)).length;
    if (shared > best) {
      best = shared;
      chosen = index;
    }
  });
  const picked = lines[chosen];
  if (!picked) return null;
  let quote = picked.line;
  const next = lines[chosen + 1];
  if (/\brows\b/i.test(question) && next?.line && next.location === picked.location) {
    quote = `${quote}\n${next.line}`;
  }
  if (/misquote/i.test(question)) quote = `${quote.replace(/[.!?]$/, "")} and tomorrow.`;
  const record = {
    marker,
    passage: passage.id,
    ...(picked.page !== null
      ? { pageFrom: picked.page, pageTo: picked.page }
      : picked.location
        ? { location: picked.location }
        : {}),
    quote,
  };
  return { record, shared: best };
}

/**
 * The records for a search result: for the first Passage, or with "each" in
 * the Question, for each Document found, its Passage with the line that
 * shares the most words with the Question, numbered in the order found.
 */
function recordsFor(result: string, question: string): Record<string, unknown>[] {
  const passages = passagesIn(result);
  if (!/\beach\b/i.test(question)) {
    const first = passages[0] && recordFor(passages[0], question, 1);
    return first ? [first.record] : [];
  }
  const documents = [...new Set(passages.map((passage) => passage.document))];
  return documents.flatMap((document, index) => {
    let best: { record: Record<string, unknown>; shared: number } | null = null;
    for (const passage of passages.filter((each) => each.document === document)) {
      const found = recordFor(passage, question, index + 1);
      if (found && (!best || found.shared > best.shared)) best = found;
    }
    return best ? [best.record] : [];
  });
}

const USAGE = {
  inputTokens: { total: 0, noCache: 0, cacheRead: undefined, cacheWrite: undefined },
  outputTokens: { total: 0, text: 0, reasoning: undefined },
};

const escapeRegExp = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * The scripted tagger: of the Tag names offered, those that appear as a word
 * (ignoring case) in the Document's excerpt, i.e. its name or text.
 */
function fakeTags(prompt: Prompt, offered: readonly string[]): string[] {
  const text = prompt
    .map((message) =>
      typeof message.content === "string"
        ? message.content
        : message.content.map((part) => (part.type === "text" ? part.text : "")).join(""),
    )
    .join("\n");
  // The instructions name the markers too: the excerpt is between the last ones.
  const excerpt = text.slice(text.lastIndexOf(EXCERPT_START), text.lastIndexOf(EXCERPT_END));
  return offered.filter((name) =>
    new RegExp(`(?<![\\p{L}\\p{N}])${escapeRegExp(name)}(?![\\p{L}\\p{N}])`, "iu").test(excerpt),
  );
}

/** The Tag names a tagging request's structured output may choose from. */
function offeredTags(options: GenerateOptions): string[] {
  const format = options.responseFormat;
  const schema = (format?.type === "json" ? format.schema : undefined) as
    | { properties?: { tags?: { items?: { enum?: unknown[] } } } }
    | undefined;
  const names: unknown[] = schema?.properties?.tags?.items?.enum ?? [];
  return names.filter((name): name is string => typeof name === "string");
}

type OfferedTool = NonNullable<Parameters<MockLanguageModelV4["doStream"]>[0]["tools"]>[number];
type Reply = { text: string } | { tool: string; input: unknown };

/** The last word of a Question, without punctuation: what a scripted call passes on. */
function lastWordOf(question: string): string {
  return (
    question
      .split(/\s+/)
      .filter(Boolean)
      .at(-1)
      ?.replace(/[^\p{L}\p{N}_-]/gu, "") ?? ""
  );
}

/**
 * A Connector Tool the Question names by its own name (after the Connector's
 * prefix, e.g. "book_boat" for "tides__book_boat"), with every required
 * argument set to the Question's last word.
 */
function connectorToolNamed(
  question: string,
  tools: readonly OfferedTool[],
): { tool: string; own: string; input: Record<string, string> } | null {
  const lastWord = lastWordOf(question);
  for (const each of tools) {
    if (each.type !== "function") continue;
    const own = each.name.split("__")[1];
    if (!own || !question.toLowerCase().includes(own.toLowerCase())) continue;
    const required = (each.inputSchema as { required?: unknown }).required;
    const names = Array.isArray(required)
      ? required.filter((name): name is string => typeof name === "string")
      : [];
    return {
      tool: each.name,
      own,
      input: Object.fromEntries(names.map((name) => [name, lastWord])),
    };
  }
  return null;
}

/** What a Tool call gave back, as text, whether it worked or not; undefined if it hasn't been called. */
function lastResult(prompt: Prompt, tool: string): string | undefined {
  let found: string | undefined;
  for (const message of prompt) {
    if (message.role !== "tool") continue;
    for (const part of message.content) {
      if (part.type !== "tool-result" || part.toolName !== tool) continue;
      const output = part.output;
      found =
        output.type === "text" || output.type === "error-text"
          ? output.value
          : JSON.stringify(output);
    }
  }
  return found;
}

/** A Skill script the Question names, with a listed Skill: to run with run_skill_script. */
function scriptNamed(
  prompt: Prompt,
  question: string,
  tools: readonly string[],
): { skill: string; script: string; args: string[] } | null {
  if (!tools.includes("run_skill_script")) return null;
  const script = /(\S+\.(?:js|mjs|cjs|py|sh))(?=\s|$)/.exec(question)?.[1];
  const skill = skillNamed(prompt, question);
  return script && skill ? { skill, script, args: [lastWordOf(question)] } : null;
}

/**
 * What the model does next. A Question that names a Skill script, or a
 * Connector Tool, gets it called once first, and the Answer starts with what
 * it said (e.g. that the User denied it).
 */
function nextReply(prompt: Prompt, tools: readonly OfferedTool[]): Reply {
  const names = tools.map((tool) => tool.name);
  const run = scriptNamed(prompt, lastQuestion(prompt), names);
  if (run) {
    const said = lastResult(prompt, "run_skill_script");
    if (said === undefined) return { tool: "run_skill_script", input: run };
    // What it printed, or the whole result when it didn't run.
    const printed = /<stdout>\n([\s\S]*?)\n?<\/stdout>/.exec(said)?.[1] ?? said;
    const reply = answerReply(prompt, names);
    return "text" in reply ? { text: `${run.script} said: ${printed}\n\n${reply.text}` } : reply;
  }
  const named = connectorToolNamed(lastQuestion(prompt), tools);
  if (!named) return answerReply(prompt, names);
  const said = lastResult(prompt, named.tool);
  if (said === undefined) return { tool: named.tool, input: named.input };
  const reply = answerReply(prompt, names);
  return "text" in reply ? { text: `${named.own} said: ${said}\n\n${reply.text}` } : reply;
}

/** What the model does next, apart from Connector Tools: call a Tool, or stream its Answer. */
function answerReply(prompt: Prompt, tools: readonly string[]): Reply {
  const question = lastQuestion(prompt);
  if (tools.includes("use_skill") && toolResults(prompt, "use_skill").length === 0) {
    const named = skillNamed(prompt, question);
    if (named) return { tool: "use_skill", input: { name: named } };
  }
  const skill = skillFollowed(prompt);
  if (!tools.includes("search_documents")) return { text: fakeAnswer(question, skill) };
  const searches = toolResults(prompt, "search_documents");
  if (searches.length === 0) return { tool: "search_documents", input: { query: question } };
  const cites = toolResults(prompt, "cite");
  if (cites.length === 0) {
    const records = recordsFor(searches.at(-1) ?? "", question);
    if (records.length > 0) return { tool: "cite", input: { citations: records } };
    return { text: fakeAnswer(question, skill) };
  }
  // The markers the last cite call recorded: "Recorded [^1], [^2]." and then any problems.
  const recordedLine = /^Recorded ([^.]*)\./.exec(cites.at(-1) ?? "")?.[1] ?? "";
  const recorded = [...new Set([...recordedLine.matchAll(/\[\^\d+\]/g)].map((match) => match[0]))];
  const answer =
    recorded.length > 1
      ? `Your Documents answer this ${recorded.join(" ")}.\n\nThat is all.`
      : CITED_ANSWER;
  return { text: skill ? `Following the Skill ${skill}.\n\n${answer}` : answer };
}

/** Test-only classifier: chooses a group whose name appears in the excerpt. */
function fakeStructuredOutput(options: GenerateOptions): string {
  const format = options.responseFormat;
  const schema = format?.type === "json" ? format.schema : undefined;
  if (schema?.properties?.groupId) {
    const message = options.prompt.find((part) => part.role === "user");
    const text =
      message && typeof message.content !== "string"
        ? message.content
            .filter((part) => part.type === "text")
            .map((part) => part.text)
            .join("")
        : "";
    const input = JSON.parse(text) as {
      groups: { id: string; name: string }[];
      tags?: { id: string; name: string }[];
      document: { name: string; text: string };
    };
    const document = `${input.document.name} ${input.document.text}`.toLowerCase();
    const selected = input.groups.find((group) => document.includes(group.name.toLowerCase()));
    return JSON.stringify({
      groupId: selected?.id ?? "__unsorted__",
      tags: (input.tags ?? [])
        .filter((tag) => document.includes(tag.name.toLowerCase()))
        .map((tag) => tag.id),
    });
  }
  return JSON.stringify({ tags: fakeTags(options.prompt, offeredTags(options)) });
}

export const createFakeChatModel: ChatModelFactory = (spec) =>
  new MockLanguageModelV4({
    provider: "incarnamind-fake",
    modelId: spec.modelId,
    // Structured output (automatic tagging) gets its Tags as JSON; anything else (a connection test) gets "OK".
    doGenerate: async (options) => ({
      content: [
        {
          type: "text",
          text: options.responseFormat?.type === "json" ? fakeStructuredOutput(options) : "OK",
        },
      ],
      finishReason: { unified: "stop", raw: undefined },
      usage: USAGE,
      warnings: [],
    }),
    doStream: async ({ prompt, tools, abortSignal }) => {
      const reply = nextReply(prompt, tools ?? []);
      const parts: StreamPart[] = [{ type: "stream-start", warnings: [] }];
      if ("tool" in reply) {
        parts.push(
          {
            type: "tool-call",
            toolCallId: `fake-${reply.tool}`,
            toolName: reply.tool,
            input: JSON.stringify(reply.input),
          },
          { type: "finish", finishReason: { unified: "tool-calls", raw: undefined }, usage: USAGE },
        );
      } else {
        const words = reply.text.split(/(?<=\s)/);
        parts.push(
          { type: "text-start", id: "answer" },
          ...words.map((word): StreamPart => ({ type: "text-delta", id: "answer", delta: word })),
          { type: "text-end", id: "answer" },
          { type: "finish", finishReason: { unified: "stop", raw: undefined }, usage: USAGE },
        );
      }
      const stream = new ReadableStream<StreamPart>({
        async pull(controller) {
          const part = parts.shift();
          if (!part) {
            controller.close();
            return;
          }
          if (part.type === "text-delta") {
            await new Promise((resolve) => setTimeout(resolve, WORD_DELAY_MS));
          }
          if (abortSignal?.aborted) {
            controller.error(new DOMException("The request was aborted.", "AbortError"));
            return;
          }
          controller.enqueue(part);
        },
      });
      return { stream };
    },
  });
