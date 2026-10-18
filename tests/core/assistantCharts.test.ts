import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { describe, expect, test } from "vitest";
import { type ChartAttributes, readChart } from "../../src/shared/charts";
import { pendingProposals } from "../../src/shared/proposals";
import { askAssistant, changesIn, lastResult } from "../helpers/assistant";
import { setUpWithDocuments, shownPassages } from "../helpers/citations";
import { note, readMind, resolveChanges, writeMind } from "../helpers/minds";
import { type ModelCall, type ScriptedReply, scriptedModel } from "../helpers/models";

/**
 * "Make a bar chart of revenue by quarter" (#1091), through the core's public
 * interface as the assistant asks: the model is told to write the cited table
 * and give chart, and the Mind gets the table and a chart drawn from it, each
 * value with its cell's Citation, as one change. A scatter chart of one
 * column of numbers keeps the table, and the model is told why in the chart's
 * own words. The model's choices are scripted; real models' are checked by hand.
 */

const QUARTERS = [
  ["Q1", "120"],
  ["Q2", "135"],
  ["Q3", "150"],
  ["Q4", "160"],
] as const;

const REVENUE = {
  name: "Annual report.md",
  contents: [
    "# Annual report",
    "",
    ...QUARTERS.map(([quarter, million]) => `Revenue in ${quarter} was €${million} million.`),
  ].join("\n\n"),
};

const quoteOf = (quarter: string, million: string) =>
  `Revenue in ${quarter} was €${million} million`;

const TABLE = [
  "| Quarter | Revenue |",
  "|---|---|",
  ...QUARTERS.map(
    ([quarter, million], index) => `| ${quarter} | €${million} million [^${index + 1}] |`,
  ),
].join("\n");

const called = (call: ModelCall, tool: string) => call.results.some((each) => each.tool === tool);

/** Searches, cites each quarter's revenue, then adds the table with `chart`. */
function chartingModel(chart: { kind?: string; title: string }, systems: string[] = []) {
  return scriptedModel((call): ScriptedReply => {
    if (call.system.includes("read_mind")) systems.push(call.system);
    const search = call.results.find((each) => each.tool === "search_documents");
    if (!search) return { calls: [{ tool: "search_documents", input: { query: "revenue" } }] };
    if (!called(call, "cite")) {
      const shown = shownPassages(search.text);
      const citations = QUARTERS.map(([quarter, million], index) => {
        const quote = quoteOf(quarter, million);
        return {
          marker: index + 1,
          passage: shown.find((each) => each.text.includes(quote))?.id,
          quote,
        };
      });
      return { calls: [{ tool: "cite", input: { citations } }] };
    }
    if (!called(call, "add_to_mind")) {
      return { calls: [{ tool: "add_to_mind", input: { chart, content: TABLE } }] };
    }
    return { text: "" };
  });
}

/** The Mind's top-level Blocks by type, and its charts as they read. */
function blocksOf(doc: ProseMirrorNode): {
  types: string[];
  charts: ChartAttributes[];
  ids: string[];
} {
  const types: string[] = [];
  const charts: ChartAttributes[] = [];
  const ids: string[] = [];
  doc.forEach((node) => {
    types.push(node.type.name);
    ids.push(String(node.attrs.id));
    if (node.type.name === "chart") charts.push(readChart(node.attrs));
  });
  return { types, charts, ids };
}

async function setUp(model: ReturnType<typeof scriptedModel>) {
  const set = await setUpWithDocuments(model, [REVENUE]);
  writeMind(set.client, [note("Notes on the year.")]);
  await set.client.settled();
  return set;
}

describe("A chart the User asks the assistant for (#1091)", { timeout: 60_000 }, () => {
  // protects=a request for a bar chart, in English or Chinese, tells the model to write the cited table and give chart, and the Mind gets the table, then a bar chart drawn from it, each value with its cell's checked Citation, as one change with the table; fails_when=the model isn't told, no chart comes, its values lose their Citations, or it isn't part of the table's change; no_claim=that a real model gives chart.
  test.each([
    ["make a bar chart of revenue by quarter", "Revenue by quarter"],
    ["画一个各季度收入的柱状图", "各季度收入"],
  ])("“%s” adds the cited table and a bar chart of it", async (request, title) => {
    const systems: string[] = [];
    const model = chartingModel({ kind: "bar", title }, systems);
    const { core, mind, client } = await setUp(model);

    const answer = await askAssistant(core, { text: request, mindId: mind.id });
    await client.settled();

    expect(systems.at(-1)).toContain("This request asks for a chart");
    expect(systems.at(-1)).toContain('kind "bar"');
    const { types, charts, ids } = blocksOf(readMind(client));
    expect(types).toEqual(["paragraph", "table", "chart"]);
    const [chart] = charts;
    expect(chart).toMatchObject({ kind: "bar", title, source: { kind: "table", blockId: ids[1] } });
    expect(chart?.data.rows.map((row) => [row.label, row.values[0]?.value])).toEqual(
      QUARTERS.map(([quarter, million]) => [quarter, Number(million) * 1e6]),
    );
    expect(chart?.data.rows.map((row) => row.values[0]?.citation?.check)).toEqual(
      QUARTERS.map(() => "found"),
    );
    const [change, ...more] = changesIn(answer);
    expect(more).toEqual([]);
    expect(change).toMatchObject({ tables: 1, charts: 1 });
    expect(new Set(pendingProposals(client.blocks).map((each) => each.request))).toEqual(
      new Set([change?.request]),
    );
  });

  // protects=the kind follows the request: "as a line chart" makes a line chart; fails_when=it is drawn as a bar chart; no_claim=none beyond protects.
  test("“as a line chart” makes a line chart", async () => {
    const systems: string[] = [];
    const model = chartingModel({ kind: "line", title: "Revenue by quarter" }, systems);
    const { core, mind, client } = await setUp(model);

    await askAssistant(core, {
      text: "Make a chart of revenue by quarter as a line chart",
      mindId: mind.id,
    });
    await client.settled();

    expect(systems.at(-1)).toContain('kind "line"');
    expect(blocksOf(readMind(client)).charts.map((chart) => chart.kind)).toEqual(["line"]);
  });

  // protects=taking the change back takes the table and its chart back together; fails_when=the chart, or the table, stays; no_claim=the window's own Undo step, which the e2e checks.
  test("taken back, the table and its chart go together", async () => {
    const model = chartingModel({ kind: "bar", title: "Revenue by quarter" });
    const { core, mind, client } = await setUp(model);
    await askAssistant(core, { text: "make a bar chart of revenue by quarter", mindId: mind.id });
    await client.settled();

    await resolveChanges(client, "reject");

    expect(blocksOf(readMind(client)).types).toEqual(["paragraph"]);
  });

  // protects=a scatter chart of a table with one column of numbers keeps the table, draws no chart, and the model is told why in the chart's own words; fails_when=a chart that can't draw is added, or the table is lost, or the model isn't told; no_claim=what a real model then says.
  test("a scatter chart of one column of numbers keeps the table and says it needs two", async () => {
    const model = chartingModel({ kind: "scatter", title: "Revenue by quarter" });
    const { core, mind, client } = await setUp(model);

    const answer = await askAssistant(core, {
      text: "make a scatter chart of revenue by quarter",
      mindId: mind.id,
    });
    await client.settled();

    expect(blocksOf(readMind(client)).types).toEqual(["paragraph", "table"]);
    expect(changesIn(answer)[0]).toMatchObject({ tables: 1 });
    expect(changesIn(answer)[0]?.charts).toBeUndefined();
    expect(lastResult(model)).toContain("A scatter chart needs two columns of numbers");
  });
});
