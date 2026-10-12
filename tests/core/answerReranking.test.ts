import { describe, expect, test } from "vitest";
import type { Reranker } from "../../src/core";
import { askAndFinish, setUpWithDocuments } from "../helpers/citations";
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
