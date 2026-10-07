import { describe, expect, test } from "vitest";
import type { Reranker } from "../../src/core";
import {
  centroidWindows,
  clusterByOverlap,
  documentWindows,
  SEARCH_TOOL_PARAMETERS,
  type SearchToolSources,
  searchDocumentsTool,
  type WindowedHit,
} from "../../src/core/documents/searchTool";
import {
  askAndFinish,
  citingModel,
  type ShownPassage,
  setUpWithDocuments,
} from "../helpers/citations";

/** A hit on the Passage at `position`, whose window range is the window of 3 ending at it. */
const hit = (position: number, score = 1): WindowedHit & { position: number } => ({
  position,
  windowFrom: Math.max(0, position - 2),
  windowTo: position,
  score,
});

/**
 * A long, repetitive logbook: every one of `count` long sections ends with
 * the same sentence about the lighthouse keeper, so each of its Passages is
 * a weaker match than one short note about the keeper.
 */
function logbook(count: number, book: number): string {
  return Array.from({ length: count }, (_, index) => {
    const day = index + 1;
    const filler = Array.from(
      { length: 18 },
      (_, line) => `Book ${book} day ${day} entry ${line} notes the ordinary coastal weather.`,
    ).join(" ");
    return `## Day ${day}\n\n${filler} The lighthouse keeper walked the shore.`;
  }).join("\n\n");
}

/** A Markdown Document of `count` long sections, the word "lighthouse" only in `marked`. */
function sections(count: number, marked: number, topic: string): string {
  return Array.from({ length: count }, (_, index) => {
    const number = index + 1;
    const special = number === marked ? ` The ${topic} lighthouse stands here.` : "";
    const filler = Array.from(
      { length: 18 },
      (_, line) => `Section ${number} sentence ${line} talks about ordinary coastal weather.`,
    ).join(" ");
    return `## Section ${number}\n\n${filler}${special}`;
  }).join("\n\n");
}

describe("The sliding-window clustering", () => {
  test("hits whose window ranges intersect, directly or through others, are one cluster, whatever their order", () => {
    const hits = [hit(20), hit(4), hit(11), hit(5), hit(13), hit(30)];

    const clusters = clusterByOverlap(hits).map((cluster) =>
      cluster.map((each) => (each as ReturnType<typeof hit>).position),
    );

    // 4 and 5 overlap; 11 and 13 overlap (windows 9–11 and 11–13); 20 and 30 are alone.
    expect(clusters).toEqual([[4, 5], [11, 13], [20], [30]]);
    expect(clusterByOverlap([...hits].reverse())).toEqual(clusterByOverlap(hits));
  });

  test("a cluster's centroid is the window that takes in most of it, so a hit comes with its neighbours", () => {
    // One hit at Passage 10 (windows 8–10): window 9, Passages 9 to 11, in the middle.
    expect(centroidWindows([hit(10)]).map((window) => window.window)).toEqual([9]);
    // Hits at 10 and 11 share windows 9 and 10; window 9 holds 9, 10 and 11.
    const pair = centroidWindows([hit(10), hit(11)]);
    expect(pair).toHaveLength(1);
    expect(pair[0]?.hits).toHaveLength(2);
    // A chain 10, 12, 14 can't fit one window of 3: the better match first, then 14 in the middle of its own.
    const chain = centroidWindows([hit(10, 3), hit(12, 1), hit(14, 1)]);
    expect(chain.map((window) => window.window)).toEqual([10, 13]);
    expect(chain.map((window) => window.hits.length)).toEqual([2, 1]);
  });

  test("a Document's windows come best first: by their best hit, then by the summed scores of the hits they take in", () => {
    const windows = documentWindows([hit(3, 0.2), hit(40, 0.5), hit(41, 0.4)]);

    expect(windows.map((window) => window.window)).toEqual([39, 2]);
    expect(windows[0]?.score).toBeCloseTo(0.9);
    expect(windows[0]?.best).toBeCloseTo(0.5);
    // Three weaker hits together don't outrank one better hit.
    const crowded = documentWindows([hit(3, 0.5), hit(40, 0.3), hit(41, 0.3), hit(42, 0.3)]);
    expect(crowded.map((window) => [window.window, window.hits.length])).toEqual([
      [2, 1],
      [40, 3],
    ]);
    // With the same best hit, the window that takes in more comes first.
    expect(documentWindows([hit(3, 0.5), hit(40, 0.5), hit(41, 0.3)])[0]?.window).toBe(39);
  });
});

describe("The document-search Tool", { timeout: 30_000 }, () => {
  test("returns the Passages around the best match, from the best Document first, in reading order", async () => {
    let shown: ShownPassage[] = [];
    const model = citingModel({
      query: "lighthouse",
      records: (passages) => {
        shown = passages;
        return [];
      },
      answer: "There is a lighthouse.",
    });
    const { core, client, mind } = await setUpWithDocuments(model, [
      { name: "Coast.md", contents: sections(24, 12, "red") },
      { name: "Weather.md", contents: sections(6, 0, "") },
    ]);
    const all = (await core.searchPassages("weather", { mode: "keyword", limit: 200 })).filter(
      (passage) => passage.documentName === "Coast",
    );
    // The model sees where each new section starts, e.g. "[§ Section 3]" (ADR-0011).
    const unmarked = (text: string) => text.replace(/\n\n\[(?:§[^\]\n]*|p\. \d+)\] /g, "\n\n");
    const positionOf = (text: string) =>
      all.find((passage) => passage.text === unmarked(text))?.position;

    await askAndFinish(core, client, mind.id, "Where is the lighthouse?");

    expect(shown.length).toBeLessThanOrEqual(SEARCH_TOOL_PARAMETERS.maxPassages);
    expect(shown[0]?.document).toBe("Coast");
    const fromCoast = shown.filter((passage) => passage.document === "Coast");
    // Coast's Passages come first, together, in reading order.
    expect(shown.slice(0, fromCoast.length)).toEqual(fromCoast);
    const positions = fromCoast.map((passage) => positionOf(passage.text) as number);
    expect(positions.every((position) => position !== undefined)).toBe(true);
    expect([...positions].sort((a, b) => a - b)).toEqual(positions);
    // The Passage with the match is there with its neighbours: a whole window of 3 around it.
    const match = fromCoast.findIndex((passage) => passage.text.includes("red lighthouse"));
    expect(match).toBeGreaterThanOrEqual(0);
    const at = positions[match] as number;
    const windowStarts = [at - 2, at - 1, at].filter((start) =>
      [start, start + 1, start + 2].every((position) => positions.includes(position)),
    );
    expect(windowStarts.length).toBeGreaterThan(0);
    // Every Passage has a short id to cite it by.
    expect(new Set(shown.map((passage) => passage.id)).size).toBe(shown.length);
    expect(shown.every((passage) => /^P\d+$/.test(passage.id))).toBe(true);
  });

  test("a short Document with the best match comes first, before long, repetitive ones whose weaker matches add up", async () => {
    let shown: ShownPassage[] = [];
    const model = citingModel({
      query: "lighthouse keeper",
      records: (passages) => {
        shown = passages;
        return [];
      },
      answer: "The keeper lit it.",
    });
    const { core, client, mind } = await setUpWithDocuments(model, [
      { name: "Log 1.md", contents: logbook(30, 1) },
      { name: "Log 2.md", contents: logbook(30, 2) },
      {
        name: "Keeper.md",
        contents: "The lighthouse keeper lit the lamp at dusk, and kept it burning all night.",
      },
    ]);
    // The note's Passage is the best match, by keyword and by vector search alike.
    for (const mode of ["keyword", "vector"] as const) {
      const [best] = await core.searchPassages("lighthouse keeper", { mode });
      expect(best?.documentName).toBe("Keeper");
    }

    await askAndFinish(core, client, mind.id, "Who lit the lighthouse lamp?");

    // Summed, the logbooks' windows of three weaker hits each came first, and
    // filled the 8 Passages: the note wasn't shown at all.
    expect(shown[0]?.document).toBe("Keeper");
    expect(shown.length).toBeLessThanOrEqual(SEARCH_TOOL_PARAMETERS.maxPassages);
    expect(new Set(shown.map((passage) => passage.document))).toEqual(
      new Set(["Keeper", "Log 1", "Log 2"]),
    );
  });

  test("a reranker plugged into the core reorders the hits before they are grouped", async () => {
    let shown: ShownPassage[] = [];
    const reranked: string[] = [];
    const reranker: Reranker = async (query, candidates) => {
      reranked.push(query);
      // A reranker that prefers the weather notes, whatever the search found.
      return candidates
        .map((candidate) => ({
          ...candidate,
          score: candidate.documentName === "Weather" ? 1 + candidate.score : candidate.score,
        }))
        .sort((a, b) => b.score - a.score);
    };
    const model = citingModel({
      query: "lighthouse",
      records: (passages) => {
        shown = passages;
        return [];
      },
      answer: "There is a lighthouse.",
    });
    const { core, client, mind } = await setUpWithDocuments(
      model,
      [
        { name: "Coast.md", contents: sections(24, 12, "red") },
        { name: "Weather.md", contents: sections(6, 0, "") },
      ],
      { reranker },
    );

    await askAndFinish(core, client, mind.id, "Where is the lighthouse?");

    expect(reranked).toEqual(["lighthouse"]);
    expect(shown[0]?.document).toBe("Weather");
  });

  test("with a reranker, hybrid search hands it the fused top 20, which are then all the hits", async () => {
    const asked: number[] = [];
    const reranked: number[] = [];
    const sources: SearchToolSources = {
      candidates: async (_query, limit) => {
        asked.push(limit);
        return Array.from({ length: limit }, (_, index) => ({
          seq: index,
          passageId: `p${index}`,
          documentId: `d${index}`,
          documentName: `Document ${index}`,
          documentKind: "markdown",
          contentHash: "hash",
          pageFrom: 1,
          pageTo: 1,
          position: 0,
          windowFrom: 0,
          windowTo: 0,
          text: `Passage ${index}`,
          score: 1 / (60 + index + 1),
        }));
      },
      window: () => [],
    };
    const reranker: Reranker = async (_query, candidates) => {
      reranked.push(candidates.length);
      return [...candidates].reverse();
    };

    await searchDocumentsTool(sources, "lighthouse");
    await searchDocumentsTool(sources, "lighthouse", { rerank: reranker });

    expect(SEARCH_TOOL_PARAMETERS.rerankCandidates).toBe(20);
    expect(asked).toEqual([SEARCH_TOOL_PARAMETERS.candidates, 20]);
    expect(reranked).toEqual([20]);
  });
});
