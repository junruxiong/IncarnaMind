import { describe, expect, test } from "vitest";
import type { Reranker } from "../../src/core";
import {
  centroidWindows,
  clusterByOverlap,
  documentWindows,
  SEARCH_TOOL_PARAMETERS,
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

  test("a Document's windows come best first, by the summed scores of the hits they take in", () => {
    const windows = documentWindows([hit(3, 0.2), hit(40, 0.5), hit(41, 0.4)]);

    expect(windows.map((window) => window.window)).toEqual([39, 2]);
    expect(windows[0]?.score).toBeCloseTo(0.9);
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
});
