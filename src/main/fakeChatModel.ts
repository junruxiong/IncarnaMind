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
 * with "misquote" in it gets a quote that isn't on the page; one with "no page"
 * names no page, as a small model often doesn't; one with "each"
 * cites the first Passage of each Document found, with a marker for each; one
 * with "rows" quotes two lines, so a sheet's Citation covers a range of rows; one that
 * asks for it "as a table" gets a Markdown table with its markers in a cell; one that
 * asks "by quarter" (按季度) gets a table of quarterly figures, its marker in each cell
 * but the last, to chart (#436); "分季度" gets the same in Chinese, in 万元, and
 * "in millions" each figure as "$289.4 million".
 * One that asks how much something grew (增长) has the growth from the first
 * two numbers of its quote calculated, and writes it with its marker.
 * Before its search it reasons, a line that quotes the Question, as a model
 * that thinks does. A Question asked "slowly" waits a moment before each step
 * and writes its Answer word by word, so a test sees each step of the Answer.
 *
 * A Question with "blank reply" in it gets a reply with no text at all. A model whose id ends
 * "-not-found" fails as one its provider doesn't know.
 *
 * A Question about the User's notes or what they selected ("notes", "selected",
 * 笔记, 选中), asked beside a Mind, gets an Answer that starts by saying what it
 * read of the Mind: its name and first line, and the selected text.
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
 * A Question that names a script and a Connector Tool gets both, the script
 * first, and the Answer starts with what each said.
 *
 * With Find sources (a Question that names papers): it calls find_sources
 * once, for the words after "on" (and the year after "since"), then lists
 * the first three titles it found.
 *
 * With web search (#128): it searches the web for the Question, reads each
 * page found, one call each, then cites the first page it could read as it
 * cites a search's Passages. Asked to search with its provider's own tool,
 * it finds `FOUND_PAGES`, whatever the query.
 *
 * For automatic tagging it tags a Document with every Tag whose name is a word in it.
 * As the User writes, it suggests a Passage's line that shares words with their paragraph.
 *
 * Edit with AI (`fakeEdit`): Shorten drops filler words ("very", "really",
 * 非常, 真的…), Improve writing capitalises each sentence and writes "a lot
 * of" as "many", Translate… puts the text in capitals, Explain says what the
 * passage starts with, and an instruction to "replace X with Y" (把 X 换成 Y)
 * does so, or "drop the citations" (去掉引用) leaves their markers out.
 *
 * Make this a Skill (`fakeSkillDraft`): a draft whose "when to use it" and first
 * step quote the Mind's last Question, so a test sees that the Mind was read.
 */
import { APICallError } from "ai";
import { MockLanguageModelV4 } from "ai/test";
import type { ChatModelFactory } from "../core";
import { EXCERPT_END, EXCERPT_START } from "../core/tags/classify";

type StreamResult = Awaited<ReturnType<MockLanguageModelV4["doStream"]>>;
type StreamPart = StreamResult["stream"] extends ReadableStream<infer Part> ? Part : never;
type Prompt = Parameters<MockLanguageModelV4["doStream"]>[0]["prompt"];
type GenerateOptions = Parameters<MockLanguageModelV4["doGenerate"]>[0];

/** Time between two streamed words: slow enough for a test to watch the Answer grow. */
const WORD_DELAY_MS = 60;
/** For a Question asked "slowly": the wait before each step, and between two words of the Answer. */
const SLOW_STEP_MS = 1200;
const SLOW_WORD_MS = 250;
const SLOWLY = /\bslowly\b/i;

/** What the scripted model reasons before it searches: a line that quotes the Question. */
const reasoningFor = (question: string) => `Search the Documents for “${question}”.`;

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

/**
 * What the scripted provider's web search finds, whatever the query: pages the
 * smoke tests' stand-in web serves (INCARNAMIND_TEST_WEB), as any host does there.
 */
const FOUND_PAGES = [
  { url: "http://encyclopedia.test/tea", title: "Tea - The Open Encyclopedia" },
  { url: "http://news.test/members", title: "Tea's history, for members" },
];

/** The scripted Answer when it cited something: one sentence with a marker. */
const CITED_ANSWER = "Your Documents answer this [^1].\n\nThat is all.";

/** The scripted Answer, as a table, to a Question that asks for one: the markers in a cell. */
const tableAnswer = (markers: string) =>
  [
    "| Claim | Where |",
    "|:--|--:|",
    `| Your Documents answer this ${markers} | Documents |`,
    "| Nothing else <br> is claimed | — |",
    "",
    "That is all.",
  ].join("\n");

/** A Question that asks for its Answer as a table. */
const AS_A_TABLE = /\bas a table\b/i;

/** The scripted Answer to a Question that asks for figures by quarter: a table of them, each cited but the last. */
const figuresAnswer = (marker: string, written = (figure: string) => figure) =>
  [
    "| Quarter | 2024 | 2025 |",
    "|:--|--:|--:|",
    `| Q1 | ${written("289.4")} ${marker} | ${written("341.5")} ${marker} |`,
    `| Q2 | ${written("301.2")} ${marker} | ${written("362.0")} ${marker} |`,
    `| Q3 | ${written("309.8")} ${marker} | ${written("377.9")} ${marker} |`,
    `| Q4 | ${written("330.6")} ${marker} | ${written("401.2")} |`,
    "",
    "That is all.",
  ].join("\n");

/** The same, in Chinese, in 万元. */
const chineseFiguresAnswer = (marker: string) =>
  [
    "| 季度 | 2024 | 2025 |",
    "|:--|--:|--:|",
    `| 一季度 | 28,410.55 ${marker} | 35,120.08 ${marker} |`,
    `| 二季度 | 29,876.31 ${marker} | 37,005.62 ${marker} |`,
    `| 三季度 | 30,615.90 ${marker} | 38,770.43 ${marker} |`,
    `| 四季度 | 32,980.44 ${marker} | 41,444.43 |`,
    "",
    "以上。",
  ].join("\n");

/** A Question that asks for figures by quarter. */
const BY_QUARTER = /\bby quarter\b|按季度/i;
const BY_QUARTER_CHINESE = /分季度/;
const IN_MILLIONS = /\bin millions\b/i;

/** The text of the last user message: the Question, after what it is asked beside. */
function lastUserText(prompt: Prompt): string {
  const last = prompt.filter((message) => message.role === "user").at(-1);
  if (!last || typeof last.content === "string") return "";
  return last.content.map((part) => (part.type === "text" ? part.text : "")).join("");
}

/** The Question, at the end of its Question context. */
const lastQuestion = (prompt: Prompt): string => lastUserText(prompt).split("\n\n").at(-1) ?? "";

/** A Question about the User's notes or what they selected in them. */
const ABOUT_THE_MIND = /\bnotes?\b|\bselect(?:ed|ion)\b|笔记|选中/i;

/** What the Answer says it read of the Mind a Question about it was asked beside, if any. */
function mindRead(prompt: Prompt): string | null {
  if (!ABOUT_THE_MIND.test(lastQuestion(prompt))) return null;
  const text = lastUserText(prompt);
  const mind = /<mind name="([^"]*)">\n([\s\S]*?)<\/mind>/.exec(text);
  if (!mind) return null;
  const first = (mind[2] ?? "")
    .split("\n")
    .map((line) => line.replace(/^#+\s*/, "").trim())
    .find((line) => line !== "" && line !== "[…]")
    // A short Answer, however long the line.
    ?.slice(0, 80);
  const selection = /<selection>\n([\s\S]*?)\n<\/selection>/.exec(text)?.[1];
  return [
    first ? `Read “${mind[1]}”, which starts: ${first}` : `Read “${mind[1]}”, which is empty.`,
    ...(selection ? [`About what you selected: ${selection}`] : []),
  ].join("\n\n");
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
/** Where a slide's speaker notes, or a Word comment, start in a Passage. */
const NOTES_MARK = /^\[(?:speaker notes|comment(?: by [^\]]+)?)\] /;

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
    ...(picked.page !== null && !/\bno page\b/i.test(question)
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
type Reply = ({ text: string } | { tool: string; input: unknown }) & { reasoning?: string };

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

/** Every result of a Tool's calls so far, as text, whether they worked or not, oldest first. */
function allResults(prompt: Prompt, tool: string): string[] {
  const found: string[] = [];
  for (const message of prompt) {
    if (message.role !== "tool") continue;
    for (const part of message.content) {
      if (part.type !== "tool-result" || part.toolName !== tool) continue;
      const output = part.output;
      found.push(
        output.type === "text" || output.type === "error-text"
          ? output.value
          : JSON.stringify(output),
      );
    }
  }
  return found;
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

/**
 * What a Question asking for papers searches for: the words after "on", and
 * the year after "since" ("Find papers since 2021 on CRISPR off-target detection").
 */
function sourcesQuery(question: string): { query: string; from_year?: number } {
  const since = /\bsince (\d{4})\b/i.exec(question)?.[1];
  const topic = /\bon (.+?)[.?!]*$/i.exec(question)?.[1] ?? question;
  return { query: topic, ...(since && { from_year: Number(since) }) };
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
 * Connector Tool, or both, gets each called once first (the script first),
 * and the Answer starts with what each said (e.g. that the User denied it).
 */
function nextReply(prompt: Prompt, tools: readonly OfferedTool[]): Reply {
  const names = tools.map((tool) => tool.name);
  const question = lastQuestion(prompt);
  if (/blank reply/i.test(question)) return { text: "" };
  /** What the calls made first said, in order. */
  const said: string[] = [];
  const run = scriptNamed(prompt, question, names);
  if (run) {
    const result = lastResult(prompt, "run_skill_script");
    if (result === undefined) return { tool: "run_skill_script", input: run };
    // What it printed, or the whole result when it didn't run.
    const printed = /<stdout>\n([\s\S]*?)\n?<\/stdout>/.exec(result)?.[1] ?? result;
    said.push(`${run.script} said: ${printed}`);
  }
  const named = connectorToolNamed(question, tools);
  if (named) {
    const result = lastResult(prompt, named.tool);
    if (result === undefined) return { tool: named.tool, input: named.input };
    said.push(`${named.own} said: ${result}`);
  }
  // As a real model does: only a Question that asks for papers searches for them.
  if (names.includes("find_sources") && /\bpapers?\b/i.test(question)) {
    const found = lastResult(prompt, "find_sources");
    if (found === undefined) return { tool: "find_sources", input: sourcesQuery(question) };
    const titles = [...found.matchAll(/^\d+\. "([^"]+)"/gm)].slice(0, 3).map((match) => match[1]);
    return {
      text:
        titles.length > 0
          ? ["I found these papers:", "", ...titles.map((title) => `- ${title}`)].join("\n")
          : "I found no papers.",
    };
  }
  const mindLine = mindRead(prompt);
  if (mindLine) said.unshift(mindLine);
  if (names.includes("search_web")) {
    const searched = lastResult(prompt, "search_web");
    if (searched === undefined) return { tool: "search_web", input: { query: question } };
    const found = [...searched.matchAll(/^\d+\. .*?: (https?:\/\/\S+)/gm)].map((match) => match[1]);
    const reads = allResults(prompt, "read_web_page");
    const next = found[reads.length];
    if (next) return { tool: "read_web_page", input: { url: next, query: question } };
    const read = reads.find((each) => passagesIn(each).length > 0);
    if (!read) return { text: "I couldn't save any of the pages I found." };
    if (toolResults(prompt, "cite").length === 0) {
      return { tool: "cite", input: { citations: recordsFor(read, question) } };
    }
    return { text: CITED_ANSWER };
  }
  const reply = answerReply(prompt, names);
  return "text" in reply && said.length > 0 ? { text: [...said, reply.text].join("\n\n") } : reply;
}

/** The scripted provider's web search: it searches once and finds `FOUND_PAGES`, as Anthropic reports them. */
function searchParts(tool: string, query: string): StreamPart[] {
  return [
    { type: "stream-start", warnings: [] },
    {
      type: "tool-call",
      toolCallId: "fake-web-search",
      toolName: tool,
      input: JSON.stringify({ query }),
      providerExecuted: true,
    },
    {
      type: "tool-result",
      toolCallId: "fake-web-search",
      toolName: tool,
      result: FOUND_PAGES.map((page) => ({
        type: "web_search_result",
        url: page.url,
        title: page.title,
        pageAge: null,
        encryptedContent: "",
      })),
    },
    { type: "text-start", id: "answer" },
    { type: "text-delta", id: "answer", delta: "done" },
    { type: "text-end", id: "answer" },
    { type: "finish", finishReason: { unified: "stop", raw: undefined }, usage: USAGE },
  ];
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
  if (searches.length === 0) {
    return {
      tool: "search_documents",
      input: { query: question },
      reasoning: reasoningFor(question),
    };
  }
  const cites = toolResults(prompt, "cite");
  if (cites.length === 0) {
    const records = recordsFor(searches.at(-1) ?? "", question);
    if (records.length > 0) return { tool: "cite", input: { citations: records } };
    return { text: fakeAnswer(question, skill) };
  }
  if (tools.includes("calculate") && /\bgrow|增长/i.test(question)) {
    const calculated = toolResults(prompt, "calculate");
    if (calculated.length === 0) {
      const inputs = [1, 2].map((position) => ({
        name: `n${position}`,
        label: position === 1 ? "Later" : "Earlier",
        citation: 1,
        position,
      }));
      return {
        tool: "calculate",
        input: {
          figures: [{ marker: 1, label: "Growth", inputs, formula: "n1 / n2 - 1", as: "percent" }],
        },
      };
    }
    if (calculated.at(-1)?.startsWith("Recorded [=1]")) {
      return {
        text: /增长/.test(question) ? "增长了[=1][^1]。" : "It grew [=1] [^1].\n\nThat is all.",
      };
    }
  }
  // The markers the last cite call recorded: "Recorded [^1], [^2]." and then any problems.
  const recordedLine = /^Recorded ([^.]*)\./.exec(cites.at(-1) ?? "")?.[1] ?? "";
  const recorded = [...new Set([...recordedLine.matchAll(/\[\^\d+\]/g)].map((match) => match[0]))];
  const answer = BY_QUARTER_CHINESE.test(question)
    ? chineseFiguresAnswer(recorded[0] ?? "")
    : BY_QUARTER.test(question)
      ? figuresAnswer(
          recorded[0] ?? "",
          IN_MILLIONS.test(question) ? (figure) => `$${figure} million` : undefined,
        )
      : AS_A_TABLE.test(question)
        ? tableAnswer(recorded.join(" "))
        : recorded.length > 1
          ? `Your Documents answer this ${recorded.join(" ")}.\n\nThat is all.`
          : CITED_ANSWER;
  return { text: skill ? `Following the Skill ${skill}.\n\n${answer}` : answer };
}

/** A text's words of four letters or more, and its pairs of Chinese characters: what a line shares with it. */
function wordsAndPairs(text: string): string[] {
  const lower = text.toLowerCase();
  const words = lower.split(/[^\p{L}\p{N}]+/u).filter((word) => /^[a-z0-9]{4,}$/.test(word));
  const pairs: string[] = [];
  for (const run of lower.match(/\p{Script=Han}+/gu) ?? []) {
    for (let at = 0; at + 1 < run.length; at++) pairs.push(run.slice(at, at + 2));
  }
  return [...new Set([...words, ...pairs])];
}

/**
 * The scripted suggestion as the User writes (#282): of the Passages given,
 * the line that shares the most words with the paragraph and isn't in it
 * already, as the text and as its quote. A paragraph that says "misquote"
 * (or 错引) gets a quote that isn't in the Passage.
 */
function fakeSuggestion(options: GenerateOptions): string {
  const request = options.prompt
    .filter((message) => message.role === "user")
    .map((message) =>
      typeof message.content === "string"
        ? message.content
        : message.content.map((part) => (part.type === "text" ? part.text : "")).join(""),
    )
    .join("\n");
  const paragraph = /<paragraph>\n([\s\S]*?)\n<\/paragraph>/.exec(request)?.[1] ?? "";
  const wanted = wordsAndPairs(paragraph);
  let best: { passage: string; line: string; shared: number } | null = null;
  for (const passage of passagesIn(request)) {
    for (const raw of passage.text.split("\n")) {
      const line = raw.replace(UNIT_MARK, "").replace(NOTES_MARK, "").trim();
      if (!line || paragraph.includes(line)) continue;
      const shared = wanted.filter((word) => line.toLowerCase().includes(word)).length;
      if (shared > (best?.shared ?? 0)) best = { passage: passage.id, line, shared };
    }
  }
  if (!best) return JSON.stringify({ text: "", passage: "", quote: "" });
  // The line's first clauses that the paragraph has already said are left out: it goes on from there.
  const clauses = best.line.split(/(?<=[，,；;])/u);
  while (clauses.length > 1 && paragraph.includes((clauses[0] ?? "").replace(/[，,；;]$/u, ""))) {
    clauses.shift();
  }
  const text = clauses.join("").trim();
  const misquote = /misquote|错引/i.test(paragraph);
  return JSON.stringify({
    text,
    passage: best.passage,
    quote: misquote ? `${text.replace(/[.!?。！？]$/u, "")} and tomorrow.` : text,
  });
}

/** The scripted Make this a Skill: a draft in the form the instructions ask for (see ../core/skills/draft). */
function fakeSkillDraft(options: GenerateOptions): string | null {
  if (!/into a Skill/.test(systemOf(options.prompt))) return null;
  const request = options.prompt
    .filter((message) => message.role === "user")
    .map((message) =>
      typeof message.content === "string"
        ? message.content
        : message.content.map((part) => (part.type === "text" ? part.text : "")).join(""),
    )
    .join("\n");
  const questions = [...request.matchAll(/^\*\*Question:\*\* (.+)$/gm)].map((match) => match[1]);
  const last = questions.at(-1) ?? "the task";
  return [
    "NAME: scripted-review",
    `WHEN: Use when asked to do the same as "${last}".`,
    "STEPS:",
    `1. Do what the Question asks: ${last}`,
    "2. Read the Documents in scope.",
    "3. Cite each finding.",
  ].join("\n");
}

const FILLER = /(?:\b(?:very|really|truly|quite|actually|basically)\s+|非常|真的|其实|十分)/giu;

/** The scripted Edit with AI (see the top of this file): the passage's new text, or an explanation. */
function fakeEdit(options: GenerateOptions): string | null {
  const request = options.prompt
    .filter((message) => message.role === "user")
    .map((message) =>
      typeof message.content === "string"
        ? message.content
        : message.content.map((part) => (part.type === "text" ? part.text : "")).join(""),
    )
    .join("\n");
  const task = /<task action="([^"]+)">([\s\S]*?)<\/task>/.exec(request);
  if (!task) return null;
  const passage = /<passage>\n([\s\S]*?)\n<\/passage>/.exec(request)?.[1] ?? "";
  const asked = (task[2] ?? "").replace(/^Do what the User asks: /, "");
  switch (task[1]) {
    case "shorten":
      return passage.replace(FILLER, "");
    case "improve":
      return passage
        .replace(/\ba lot of\b/g, "many")
        .replace(
          /(^|[.!?]\s+)([a-z])/g,
          (_match, before: string, letter: string) => `${before}${letter.toUpperCase()}`,
        );
    case "translate":
      return passage.toUpperCase();
    case "explain": {
      const words = passage
        .replace(/\s*\[\d+\]/g, "")
        .split(/\s+/)
        .slice(0, 6)
        .join(" ");
      return `This passage says: ${words}…`;
    }
    default: {
      const replace =
        /replace\s+[“"]?(.+?)[”"]?\s+with\s+[“"]?(.+?)[”"]?\.?$/i.exec(asked) ??
        /把\s*[“"]?(.+?)[”"]?\s*换成\s*[“"]?(.+?)[”"]?。?$/u.exec(asked);
      if (replace?.[1] && replace[2] !== undefined)
        return passage.split(replace[1]).join(replace[2]);
      if (/drop the citations|去掉引用/i.test(asked)) return passage.replace(/\s*\[\d+\]/g, "");
      return `${passage} (edited)`;
    }
  }
}

/** Test-only classifier: chooses a group whose name appears in the excerpt. */
function fakeStructuredOutput(options: GenerateOptions): string {
  const format = options.responseFormat;
  const schema = format?.type === "json" ? format.schema : undefined;
  if (schema?.properties?.quote && schema.properties.passage) return fakeSuggestion(options);
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

/** The scripted reply to a request, as it streams: its reasoning, then a Tool call or its Answer. */
function replyParts(prompt: Prompt, tools: readonly OfferedTool[]): StreamPart[] {
  const reply = nextReply(prompt, tools);
  const parts: StreamPart[] = [{ type: "stream-start", warnings: [] }];
  if (reply.reasoning) {
    parts.push(
      { type: "reasoning-start", id: "thinking" },
      ...reply.reasoning
        .split(/(?<=\s)/)
        .map((word): StreamPart => ({ type: "reasoning-delta", id: "thinking", delta: word })),
      { type: "reasoning-end", id: "thinking" },
    );
  }
  if ("tool" in reply) {
    // Calls of one Tool after its first, such as a second page read, have ids of their own.
    const earlier = allResults(prompt, reply.tool).length;
    parts.push(
      {
        type: "tool-call",
        toolCallId: earlier === 0 ? `fake-${reply.tool}` : `fake-${reply.tool}-${earlier + 1}`,
        toolName: reply.tool,
        input: JSON.stringify(reply.input),
      },
      { type: "finish", finishReason: { unified: "tool-calls", raw: undefined }, usage: USAGE },
    );
    return parts;
  }
  const words = reply.text.split(/(?<=\s)/);
  parts.push(
    { type: "text-start", id: "answer" },
    ...words.map((word): StreamPart => ({ type: "text-delta", id: "answer", delta: word })),
    { type: "text-end", id: "answer" },
    { type: "finish", finishReason: { unified: "stop", raw: undefined }, usage: USAGE },
  );
  return parts;
}

export const createFakeChatModel: ChatModelFactory = (spec) =>
  new MockLanguageModelV4({
    provider: "incarnamind-fake",
    modelId: spec.modelId,
    // Structured output (automatic tagging) gets its Tags as JSON; Edit with AI its new text;
    // anything else (a connection test) gets "OK".
    doGenerate: async (options) => ({
      content: [
        {
          type: "text",
          text:
            options.responseFormat?.type === "json"
              ? fakeStructuredOutput(options)
              : (fakeSkillDraft(options) ?? fakeEdit(options) ?? "OK"),
        },
      ],
      finishReason: { unified: "stop", raw: undefined },
      usage: USAGE,
      warnings: [],
    }),
    doStream: async ({ prompt, tools, abortSignal }) => {
      // A model id ending "-not-found" is one its provider doesn't know, for a test of that Problem.
      if (spec.modelId.endsWith("-not-found")) {
        throw new APICallError({
          message: `The model ${spec.modelId} does not exist`,
          url: "https://fake.invalid/chat",
          requestBodyValues: {},
          statusCode: 404,
        });
      }
      const search = tools?.find((each) => each.type === "provider");
      const parts = search
        ? searchParts(search.name, lastQuestion(prompt))
        : replyParts(prompt, tools ?? []);
      const slow = !search && SLOWLY.test(lastQuestion(prompt));
      const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
      const stream = new ReadableStream<StreamPart>({
        async pull(controller) {
          const part = parts.shift();
          if (!part) {
            controller.close();
            return;
          }
          if (slow && part.type === "stream-start") await wait(SLOW_STEP_MS);
          if (part.type === "text-delta") await wait(slow ? SLOW_WORD_MS : WORD_DELAY_MS);
          if (slow && part.type === "reasoning-delta") await wait(WORD_DELAY_MS);
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
