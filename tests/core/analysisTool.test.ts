/**
 * The analysis Tool (#447): an Answer's model writes Python over a sheet or
 * CSV it found and numbers quoted from Passages, IncarnaMind runs it in the
 * sandbox, and the figures it gives become the Answer's calculated figures.
 * Through the core's public interface, as the assistant uses it, with a
 * scripted chat model and the fake runtime.
 */
import { createHash } from "node:crypto";
import type { MockLanguageModelV4 } from "ai/test";
import { describe, expect, test } from "vitest";
import type { Core } from "../../src/core";
import { createFakeAnalysisRuntime, type FakeAnswer } from "../../src/core/analysis/fake";
import type { AnalysisJob } from "../../src/shared/analysis";
import { answerEnded, type ShownPassage, shownPassages } from "../helpers/citations";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";
import { scriptedModel, scriptedModels } from "../helpers/models";

const SALES_CSV = [
  "Region,Quarter,Revenue",
  "North,2026Q1,120",
  "North,2026Q2,150.5",
  "South,2026Q1,80",
  "South,2026Q2,n/a",
].join("\n");

/** A Tool call as a scripted model makes it. */
type Call = { tool: string; input: unknown };

/** A model that searches once, then makes each run's call from the Passages it found, then writes `answer`. */
function analysingModel(plan: {
  query: string;
  runs: ((passages: ShownPassage[]) => Call)[];
  answer: string;
}): MockLanguageModelV4 {
  return scriptedModel((call) => {
    const searches = call.results.filter((result) => result.tool === "search_documents");
    if (searches.length === 0) {
      return { calls: [{ tool: "search_documents", input: { query: plan.query } }] };
    }
    const ran = call.results.filter((result) => result.tool === "run_analysis").length;
    const next = plan.runs[ran];
    if (next) return { calls: [next(searches.flatMap((each) => shownPassages(each.text)))] };
    return { text: plan.answer };
  });
}

const passageOf = (passages: readonly ShownPassage[], name: string): string => {
  const found = passages.find((passage) => passage.document.startsWith(name));
  if (!found) throw new Error(`No Passage of ${name} was shown.`);
  return found.id;
};

/** A run over the whole sales sheet, with `code`. */
const salesRun =
  (code = 'result("total", inputs["sales"]["Revenue"].sum(), "Total revenue", "USD")') =>
  (passages: ShownPassage[]): Call => ({
    tool: "run_analysis",
    input: {
      title: "Revenue by region",
      inputs: [{ name: "sales", passage: passageOf(passages, "sales") }],
      code,
    },
  });

/** What each call of `tool` told the model, oldest first. */
function told(model: MockLanguageModelV4, tool: string): string[] {
  const last = model.doStreamCalls.at(-1);
  if (!last) throw new Error("The model wasn't asked anything.");
  const texts: string[] = [];
  for (const message of last.prompt) {
    if (message.role !== "tool") continue;
    for (const part of message.content) {
      if (part.type === "tool-result" && part.toolName === tool && part.output.type === "text") {
        texts.push(part.output.value);
      }
    }
  }
  return texts;
}

/** A core whose chat model is Claude Haiku 5.5, scripted, with the fake sandbox and these files as Documents. */
async function setUp(
  model: MockLanguageModelV4,
  answer: (job: AnalysisJob) => FakeAnswer,
  files: readonly { name: string; contents: string }[] = [
    { name: "sales.csv", contents: SALES_CSV },
  ],
) {
  const runtime = createFakeAnalysisRuntime({ answer });
  const core = startCore(await createTempDataFolder(), {
    createChatModel: scriptedModels(model).createChatModel,
    analysisRuntime: runtime,
  });
  const saved = await core.saveChatProvider({
    kind: "anthropic",
    apiKey: "sk-test",
    modelId: "claude-haiku-5-5",
  });
  if (saved.service) await core.allowDataFlow("chat", saved.service.id);
  const sources = await createTempDataFolder();
  await addAndProcess(
    core,
    await Promise.all(files.map((file) => writeSourceFile(sources, file.name, file.contents))),
  );
  return { core, runtime };
}

async function askAndRead(core: Core, text: string, analyse?: boolean) {
  const asked = await core.askInConversation({ text, ...(analyse && { analyse }) });
  if (!asked.asked) throw new Error(`The Question wasn't asked: ${JSON.stringify(asked)}`);
  const ended = await answerEnded(core, asked.answerId);
  if (ended.event !== "finished") throw new Error(`The Answer failed: ${JSON.stringify(ended)}`);
  const read = async () => {
    const content = await core.getConversation(asked.conversation.id);
    const answer = content.messages.find((message) => message.id === asked.answerId);
    if (answer?.kind !== "answer") throw new Error("No Answer.");
    return answer;
  };
  return { answer: await read(), read };
}

const totalOf = (): FakeAnswer => ({
  output: {
    results: [{ kind: "figure", name: "total", label: "Total revenue", unit: "USD", value: 350.5 }],
  },
});

describe("an Analysis an Answer runs", () => {
  test("reads the whole sheet into the sandbox, records each input's hash, and gives a figure the Answer shows, also when the conversation is reopened", async () => {
    const model = analysingModel({
      query: "revenue",
      runs: [salesRun()],
      answer: "Revenue came to [=1] in total.",
    });
    const { core, runtime } = await setUp(model, totalOf);

    const { answer, read } = await askAndRead(core, "What was the total revenue?");

    const [job] = runtime.jobs;
    const sales = job?.inputs[0];
    expect(sales).toMatchObject({
      kind: "table",
      name: "sales",
      columns: ["Region", "Quarter", "Revenue"],
      rows: [
        ["North", "2026Q1", 120],
        ["North", "2026Q2", 150.5],
        ["South", "2026Q1", 80],
        ["South", "2026Q2", "n/a"],
      ],
      rowNumbers: [2, 3, 4, 5],
    });
    const [analysis] = answer.analyses;
    expect(analysis).toMatchObject({ number: 1, title: "Revenue by region" });
    expect(analysis?.inputs[0]).toMatchObject({
      kind: "range",
      name: "sales",
      range: "A1:C5",
      rows: 4,
      columns: 3,
      cut: false,
      citation: {
        documentName: "sales",
        location: { kind: "rows", sheet: null, from: 1, to: 5 },
      },
    });
    const values = sales?.kind === "table" ? JSON.stringify([sales.columns, sales.rows]) : "";
    expect(analysis?.inputs[0]?.sha256).toBe(createHash("sha256").update(values).digest("hex"));
    expect(analysis?.codeSha256).toMatch(/^[0-9a-f]{64}$/);
    expect(answer.figures).toEqual([
      expect.objectContaining({
        marker: 1,
        value: 350.5,
        analysis: { analysis: 1, result: "total" },
      }),
    ]);
    expect(answer.toolCalls.find((call) => call.tool === "run_analysis")).toMatchObject({
      source: "analysis",
      status: "done",
      analysis: { number: 1, documents: ["sales"], end: "done" },
    });
    expect((await read()).analyses).toEqual(answer.analyses);
  });

  test("a result table written as [=table 1] on a line of its own is in the Answer as a table", async () => {
    const model = analysingModel({
      query: "revenue",
      runs: [salesRun('result("by_region", inputs["sales"].groupby("Region")["Revenue"].sum())')],
      answer: "By region:\n\n[=table 1]\n\nNorth sold most.",
    });
    const { core } = await setUp(model, () => ({
      output: {
        results: [
          {
            kind: "table",
            name: "by_region",
            label: "Revenue by region",
            unit: null,
            table: {
              columns: ["Region", "Revenue"],
              rows: [
                ["North", 270.5],
                ["South", 80],
              ],
              rowCount: 2,
              columnCount: 2,
            },
          },
        ],
      },
    }));

    const { answer } = await askAndRead(core, "Revenue by region?");

    expect(answer.text).toContain("| Region | Revenue |");
    expect(answer.text).toContain("| North | 270.5 |");
    expect(answer.text).not.toContain("[=table");
    expect(told(model, "run_analysis")[0]).toContain("[=table 1] Revenue by region");
  });

  test("the model is told the numbers written in the code, the cells that aren't numbers, and an error's own message", async () => {
    const traceback =
      "Traceback (most recent call last):\n  File \"<analysis>\", line 2, in <module>\nKeyError: 'Revnue'";
    const model = analysingModel({
      query: "revenue",
      runs: [salesRun("rate = 0.2\nprint(inputs['sales']['Revnue'])"), salesRun()],
      answer: "Revenue came to [=1].",
    });
    let runs = 0;
    const { core } = await setUp(model, () =>
      ++runs === 1
        ? {
            end: { kind: "error", message: traceback, line: 2 },
            output: {
              literals: [{ value: 0.2, line: 1, text: "0.2" }],
              notNumbers: [
                { input: "sales", column: "Revenue", count: 1, cells: [{ row: 5, value: "n/a" }] },
              ],
            },
          }
        : totalOf(),
    );

    await askAndRead(core, "What was the total revenue?");

    const [first] = told(model, "run_analysis");
    expect(first).toContain("Numbers written in the code: 0.2 (line 1)");
    expect(first).toContain("1 cell of sales's column Revenue isn't a number");
    expect(first).toContain("KeyError: 'Revnue'");
    expect(first).toContain("Fix the code and run it again");
  });

  test("a seventh run in one Answer is refused, in words the model acts on", async () => {
    const model = analysingModel({
      query: "revenue",
      runs: Array.from({ length: 7 }, () => salesRun()),
      answer: "Revenue came to [=1].",
    });
    const { core, runtime } = await setUp(model, totalOf);

    const { answer } = await askAndRead(core, "What was the total revenue?");

    expect(runtime.jobs).toHaveLength(6);
    expect(answer.analyses).toHaveLength(6);
    expect(told(model, "run_analysis").at(-1)).toContain(
      "You have run 6 analyses for this Question",
    );
  });

  test("a sheet longer than an Analysis reads is read only in part, and the model is told to say so", async () => {
    const rows = Array.from(
      { length: 20_050 },
      (_, index) => `R${index % 7},${index},${index * 2}`,
    );
    const model = analysingModel({
      query: "orders",
      runs: [
        (passages) => ({
          tool: "run_analysis",
          input: {
            title: "Orders",
            inputs: [{ name: "orders", passage: passageOf(passages, "orders") }],
            code: 'result("n", len(inputs["orders"]))',
          },
        }),
      ],
      answer: "There are [=1] orders.",
    });
    const { core } = await setUp(model, () => ({}), [
      { name: "orders.csv", contents: ["Region,Order,Value", ...rows].join("\n") },
    ]);

    const { answer } = await askAndRead(core, "How many orders?");

    expect(answer.analyses[0]?.inputs[0]).toMatchObject({ kind: "range", cut: true });
    expect(told(model, "run_analysis")[0]).toContain("cut: only its first");
  }, 60_000);

  test("it is offered only with a sheet or CSV to read, or when the User chose Analyse data", async () => {
    const offered: boolean[] = [];
    const model = scriptedModel((call) => {
      offered.push(call.tools.includes("run_analysis"));
      return { text: "No data." };
    });
    const { core } = await setUp(model, totalOf, [
      { name: "notes.md", contents: "# Notes\n\nThe board met in March." },
    ]);

    await askAndRead(core, "What did the board do?");
    await askAndRead(core, "Work it out", true);

    expect(offered.at(0)).toBe(false);
    expect(offered.at(-1)).toBe(true);
  });
});
