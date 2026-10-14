import { describe, expect, test } from "vitest";
import type { Reranker } from "../../src/core";
import { askAndFinish, setUpWithDocuments, shownPassages } from "../helpers/citations";
import { type ModelCall, scriptedModel } from "../helpers/models";

const TIDES = {
  name: "Tides.txt",
  contents: "Spring tides happen at new moon and at full moon.\n",
};

function countingReranker() {
  const reranked: string[] = [];
  const reranker: Reranker = async (query, candidates) => {
    reranked.push(query);
    return [...candidates];
  };
  return { reranked, reranker };
}

const searchingModel = () =>
  scriptedModel((call: ModelCall) => {
    if (call.tools.length === 0) {
      return { text: JSON.stringify({ answer: "Spring tides come at new moon.", citations: [] }) };
    }
    return call.results.length === 0
      ? { calls: [{ tool: "search_documents", input: { query: "spring tides" } }] }
      : { text: "Spring tides come at new moon." };
  });

const withoutTools = () =>
  scriptedModel((call: ModelCall) =>
    call.tools.length > 0
      ? { error: { status: 400, message: "this model does not support tools" } }
      : { text: JSON.stringify({ answer: "Spring tides come at new moon.", citations: [] }) },
  );

describe("Which searches rerank", () => {
  test("a model that calls Tools searches without the reranker", async () => {
    const { reranked, reranker } = countingReranker();
    const { core, client, mind } = await setUpWithDocuments(searchingModel(), [TIDES], {
      reranker,
    });

    const { finished } = await askAndFinish(core, client, mind.id, "When are spring tides?");

    expect(finished.citationSupport).toBe("tools");
    expect(reranked).toEqual([]);
  });

  test("an Answer from a model that searches once reranks", async () => {
    const { reranked, reranker } = countingReranker();
    const { core, client, mind } = await setUpWithDocuments(withoutTools(), [TIDES], {
      reranker,
    });

    const { finished } = await askAndFinish(core, client, mind.id, "When are spring tides?");

    expect(finished.citationSupport).toBe("structured-output");
    expect(reranked.length).toBeGreaterThan(0);
  });

  test("a check's search reranks", async () => {
    const { reranked, reranker } = countingReranker();
    const { core } = await setUpWithDocuments(searchingModel(), [TIDES], { reranker });

    const found = await core.searchEvidence("spring tides");

    expect(found.passages.length).toBeGreaterThan(0);
    expect(reranked).toEqual(["spring tides"]);
  });
});

describe("How many Passages a search hands the model", () => {
  /** Four long Documents that all talk about the lighthouse, so a search could hand over many Passages. */
  const LIGHTHOUSES = ["North", "South", "East", "West"].map((side) => ({
    name: `${side}.md`,
    contents: Array.from(
      { length: 24 },
      (_, section) =>
        `## Section ${section}\n\n${Array.from({ length: 14 }, (_, line) => `The ${side} lighthouse lamp ${section}.${line} burns all night over the coastal water.`).join(" ")}`,
    ).join("\n\n"),
  }));

  test("a model that calls Tools is handed 8, a model that searches once 12", async () => {
    let loop = 0;
    const calling = scriptedModel((call: ModelCall) => {
      if (call.tools.length === 0) {
        return { text: JSON.stringify({ answer: "A lamp.", citations: [] }) };
      }
      loop = Math.max(
        loop,
        ...call.results.map((result) => shownPassages(result.text ?? "").length),
      );
      return call.results.length === 0
        ? { calls: [{ tool: "search_documents", input: { query: "lighthouse lamp" } }] }
        : { text: "A lamp." };
    });
    const first = await setUpWithDocuments(calling, LIGHTHOUSES);
    await askAndFinish(first.core, first.client, first.mind.id, "Where is the lighthouse lamp?");

    let once = 0;
    const single = scriptedModel((call: ModelCall) => {
      if (call.tools.length > 0) {
        return { error: { status: 400, message: "this model does not support tools" } };
      }
      once = shownPassages(call.system).length;
      return { text: JSON.stringify({ answer: "A lamp.", citations: [] }) };
    });
    const second = await setUpWithDocuments(single, LIGHTHOUSES);
    await askAndFinish(second.core, second.client, second.mind.id, "lighthouse lamp");

    expect(loop).toBe(8);
    expect(once).toBe(12);
  });
});
