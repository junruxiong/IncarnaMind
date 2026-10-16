/**
 * With no Documents (no Tools offered), every Question gets the same short
 * Markdown Answer, streamed a few words at a time. With Documents, it does
 * what a Tool-calling model does: it searches for the Question, cites the line
 * of the first Passage that shares a word with the Question (on the page, or
 * at the Location, that line is on), then answers with a marker. A Question
 * with "misquote" in it gets a quote that isn't on the page; one with "no page"
 * names no page, as a small model often doesn't; one with "each"
 * cites the first Passage of each Document found, with a marker for each; one
 * with "rows" quotes two lines, so a sheet's Citation covers a range of rows; one with
 * "quotation marks" (引号) quotes the clause after a line's colon or comma, and writes
 * it in quotation marks with the word before it, which the page doesn't have (#483); one that
 * asks for it "as a table" gets a Markdown table with its markers in a cell; one that
 * asks "by quarter" (按季度) gets a table of quarterly figures, its marker in each cell
 * but the last, to chart (#436); "分季度" gets the same in Chinese, in 万元, and
 * "in millions" each figure as "$289.4 million"; one that asks for "four columns" gets a
 * table of four, wider than the assistant (#612); one that asks for "two tables" (两个表) gets
 * two small ones; one that asks for "nine rows" (九行) gets
 * three peers' figures in nine rows, with the first column's labels; one that asks for "forty
 * citations" (四十条引用) gets forty sentences, each with a marker, and one that asks for "twelve in a
 * row" (连续十二条) gets one sentence with twelve markers side by side; one with "cited below" (or 引用在下) gets a
 * table whose one Citation stands in a paragraph under it, and one with "framed" (or 带说明)
 * gets a lead-in before its content and a closing note after it (#702).
 * One that asks how much something grew (增长) has the growth from the first
 * two numbers of its quote calculated, and writes it with its marker.
 * Before its search it reasons, a line that quotes the Question, as a model
 * that thinks does. A Question asked "slowly" waits a moment before each step
 * and writes its Answer word by word, so a test sees each step of the Answer.
 *
 * With Skills: a Question that names a listed Skill gets it loaded with
 * use_skill first. An Answer that follows a Skill (forced, or loaded) starts
 * by saying which.
 */

import { cellReply, isCell } from "./cells";
import { FORTY_CITATIONS, passagesIn, recordFor, recordsFor, TWELVE_IN_A_ROW } from "./passages";
import {
  lastQuestion,
  type Prompt,
  type Reply,
  type StreamPart,
  systemOf,
  toolResults,
  USAGE,
} from "./prompt";

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
export const CITED_ANSWER = "Your Documents answer this [^1].\n\nThat is all.";

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

/** The scripted Answer to a Question that asks for "four columns": a table wider than the assistant. */
const wideTableAnswer = (marker: string) =>
  [
    "| Quarter | Revenue 2024 | Revenue 2025 | Change |",
    "|:--|--:|--:|--:|",
    `| Q1 | 289.4 ${marker} | 341.5 | +18.0% |`,
    "| Q2 | 301.2 | 362.0 | +20.2% |",
    "| Q3 | 309.8 | 377.9 | +22.0% |",
    "",
    "That is all.",
  ].join("\n");

const TWO_TABLES = /\btwo tables\b|两个表/i;

function twoTablesAnswer(question: string): string {
  const chinese = /\p{Script=Han}/u.test(question);
  return (
    chinese
      ? [
          "| 季度 | 收入 |",
          "|:--|--:|",
          "| 一季度 | 289.4 |",
          "| 二季度 | 301.2 |",
          "",
          "| 地区 | 占比 |",
          "|:--|--:|",
          "| 北方 | 41% |",
          "| 南方 | 59% |",
          "",
          "以上。",
        ]
      : [
          "| Quarter | Revenue |",
          "|:--|--:|",
          "| Q1 | 289.4 |",
          "| Q2 | 301.2 |",
          "",
          "| Region | Share |",
          "|:--|--:|",
          "| North | 41% |",
          "| South | 59% |",
          "",
          "That is all.",
        ]
  ).join("\n");
}

const NINE_ROWS = /\bnine rows\b|九行/i;

function nineRowsAnswer(question: string, marker: string): string {
  const chinese = /\p{Script=Han}/u.test(question);
  const rows = chinese
    ? [
        ["营业收入（近十二个月）", "128,460 百万元", "94,120 百万元", "21,050 百万元"],
        ["上半年利润率", "40.3%", "39.5%", "37.9%"],
        ["第三季度利润率", "38.4%", "37.1%", "36.8%"],
        ["变化", "−1.9 个百分点", "−2.4 个百分点", "−1.1 个百分点"],
        ["息税折旧摊销前利润（近十二个月）", "10,280 百万元", "6,110 百万元", "1,490 百万元"],
        ["净债务", "24,670 百万元", "20,770 百万元", "3,880 百万元"],
        ["净杠杆率", "2.4×", "3.4×", "2.6×"],
        ["契约上限", "3.5×", "3.5×", "3.25×"],
        ["余量", "1.1×", "0.1×", "0.65×"],
      ]
    : [
        ["Revenue, LTM", "£12,846m", "£9,412m", "£2,105m"],
        ["Margin, H1", "40.3%", "39.5%", "37.9%"],
        ["Margin, Q3", "38.4%", "37.1%", "36.8%"],
        ["Change", "−1.9 pts", "−2.4 pts", "−1.1 pts"],
        ["EBITDA, LTM", "£1,028m", "£611m", "£149m"],
        ["Net debt", "£2,467m", "£2,077m", "£388m"],
        ["Net leverage", "2.4×", "3.4×", "2.6×"],
        ["Covenant limit", "3.5×", "3.5×", "3.25×"],
        ["Headroom", "1.1×", "0.1×", "0.65×"],
      ];
  const head = chinese
    ? ["指标", "哈罗食品", "明智超市", "考尔德杂货"]
    : ["Figure", "Harrow Foods", "Brightmart", "Calder Grocers"];
  return [
    `| ${head.join(" | ")} |`,
    "|:--|--:|--:|--:|",
    ...rows.map(([label, ...cells], index) => {
      const cited = cells.map((cell, column) =>
        index === 0 && column === 0 ? `${cell} ${marker}` : cell,
      );
      return `| ${label} | ${cited.join(" | ")} |`;
    }),
    "",
    chinese ? "变化和余量为计算所得。" : "Change and headroom are calculated.",
  ].join("\n");
}

function fortyCitationsAnswer(question: string): string {
  const chinese = /\p{Script=Han}/u.test(question);
  const sentences = Array.from({ length: 40 }, (_, index) =>
    chinese
      ? `第${index + 1}条说明潮水每天涨落两次，月亮的位置决定了涨落的时间[^${index + 1}]。`
      : `Statement ${index + 1} says the tide rises and falls twice a day, at hours that follow the moon [^${index + 1}].`,
  );
  const paragraphs: string[] = [];
  for (let at = 0; at < sentences.length; at += 8)
    paragraphs.push(sentences.slice(at, at + 8).join(chinese ? "" : " "));
  return paragraphs.join("\n\n");
}

const twelveInARowAnswer = (question: string) => {
  const markers = Array.from({ length: 12 }, (_, index) => `[^${index + 1}]`).join("");
  return /\p{Script=Han}/u.test(question)
    ? `潮水每天涨落两次${markers}。`
    : `The tide rises and falls twice a day ${markers}.`;
};

/** A Question whose table is cited under it, and one that is framed by a lead-in and a closing note (#702). */
const CITED_BELOW = /\bcited below\b|引用在下/i;
const FRAMED = /\bframed\b|带说明/i;

/** The table whose one Citation stands under it, in Chinese or English, with the framing if asked. */
function citedBelowAnswer(question: string, marker: string): string {
  const chinese = /\p{Script=Han}/u.test(question);
  const table = chinese
    ? ["| 季度 | 收入 |", "|:--|--:|", "| 一季度 | 289.4 |", "| 二季度 | 301.2 |"]
    : ["| Quarter | Revenue |", "|:--|--:|", "| Q1 | 289.4 |", "| Q2 | 301.2 |"];
  const framed = FRAMED.test(question);
  return [
    ...(framed ? [chinese ? "以下是草稿：" : "Here is a draft:", ""] : []),
    ...table,
    "",
    ...(CITED_BELOW.test(question) ? [marker, ""] : []),
    ...(framed ? [chinese ? "注：以上仅为草稿。" : "Note: this is only a draft."] : []),
  ].join("\n");
}

/** A Question that asks for four columns. */
const FOUR_COLUMNS = /\bfour columns\b/i;

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

/** The Skill the Answer follows: the one the Question forces, or one loaded with use_skill. */
function skillFollowed(prompt: Prompt): string | null {
  const forced = /the User chose the Skill "([^"]+)"/.exec(systemOf(prompt))?.[1];
  if (forced) return forced;
  const loaded = toolResults(prompt, "use_skill").at(-1);
  return (loaded && /<skill name="([^"]+)">/.exec(loaded)?.[1]) || null;
}

/** A listed Skill the Question names, to load before answering. */
export function skillNamed(prompt: Prompt, question: string): string | null {
  const listed = [...systemOf(prompt).matchAll(/<skill name="([^"]+)">/g)].map((match) => match[1]);
  return listed.find((name) => name && question.toLowerCase().includes(name)) ?? null;
}

/** The scripted provider's web search: it searches once and finds `FOUND_PAGES`, as Anthropic reports them. */
export function searchParts(tool: string, query: string): StreamPart[] {
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
export function answerReply(prompt: Prompt, tools: readonly string[]): Reply {
  const question = lastQuestion(prompt);
  if (tools.includes("search_documents") && isCell(prompt)) return cellReply(prompt, question);
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
  const firstPassage = passagesIn(searches.at(-1) ?? "")[0];
  const quoted = firstPassage && recordFor(firstPassage, question, 1)?.quoted;
  if (quoted) {
    return {
      text: /\p{Script=Han}/u.test(question)
        ? `文档中写道“${quoted}”[^1]。`
        : `Your Documents say “${quoted}” [^1].`,
    };
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
      : FOUR_COLUMNS.test(question)
        ? wideTableAnswer(recorded[0] ?? "")
        : TWO_TABLES.test(question)
          ? twoTablesAnswer(question)
          : NINE_ROWS.test(question)
            ? nineRowsAnswer(question, recorded[0] ?? "")
            : FORTY_CITATIONS.test(question)
              ? fortyCitationsAnswer(question)
              : TWELVE_IN_A_ROW.test(question)
                ? twelveInARowAnswer(question)
                : CITED_BELOW.test(question) || FRAMED.test(question)
                  ? citedBelowAnswer(question, recorded[0] ?? "")
                  : AS_A_TABLE.test(question)
                    ? tableAnswer(recorded.join(" "))
                    : recorded.length > 1
                      ? `Your Documents answer this ${recorded.join(" ")}.\n\nThat is all.`
                      : CITED_ANSWER;
  return { text: skill ? `Following the Skill ${skill}.\n\n${answer}` : answer };
}
