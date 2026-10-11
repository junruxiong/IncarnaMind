/**
 * The dumps (eval/README.md, "Offline scoring"): what the writer keeps for a
 * query, and how the scorer orders and counts a dump's candidates without
 * running a model. The run needs the models; neither of these does.
 */
import { describe, expect, test } from "vitest";
import { buildDump } from "../../eval/lib/dumps";
import type { ExpectedPassage } from "../../eval/lib/evaluationSet";
import type { ScoringReranker } from "../../eval/lib/rerank";
import { GATING_MODE, runRetrieval } from "../../eval/lib/retrieval";
import {
  DUMP_FORMAT,
  type Dump,
  type DumpCandidate,
  type DumpQuery,
  type DumpQuestion,
  formatScore,
  isHit,
  isToolHit,
  main,
  type Order,
  parseDump,
  parseScores,
  type ScoreOptions,
  scoreDump,
} from "../../eval/score";
import { DATABASE_FILE } from "../../src/core";
import { openDatabase } from "../../src/core/storage";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, createSourceFolder, writeSourceFile } from "../helpers/documents";
import { buildPdf } from "../helpers/pdf";

const candidate = (index: number, over: Partial<DumpCandidate> = {}): DumpCandidate => ({
  passageId: `p${index}`,
  documentId: `d${index}`,
  position: 0,
  rank: index + 1,
  keywordScore: 10 - index,
  rerankScore: 0.5,
  hits: [false],
  rerankText: `text ${index}`,
  ...over,
});

const query = (candidates: DumpCandidate[], over: Partial<DumpQuery> = {}): DumpQuery => ({
  text: "q",
  translated: false,
  candidates,
  tool: { passages: [] },
  ...over,
});

const question = (id: string, queries: DumpQuery[], needed = 1, tags = {}): DumpQuestion => ({
  id,
  tags,
  needed,
  queries,
});

const dumpOf = (questions: DumpQuestion[]): Dump => ({
  format: DUMP_FORMAT,
  set: "test",
  createdAt: "2026-10-10T00:00:00.000Z",
  commit: "abc",
  embeddings: "off",
  keywordDepth: 150,
  rerankedDepth: 6,
  reranker: { id: "model", name: "Model" },
  groupBy: "kind",
  questions,
});

const options: ScoreOptions = { depth: 6, dropNeighbours: false };
const KEYWORD: Order = { kind: "keyword" };
const RERANK: Order = { kind: "rerank" };

/** Six candidates; the one at `hit` meets the hit rule, and the reranker scores it highest. */
const hitAt = (hit: number, extra: Partial<DumpQuery> = {}) =>
  query(
    Array.from({ length: 6 }, (_, index) =>
      candidate(index, {
        hits: [index === hit],
        rerankScore: index === hit ? 0.9 : 0.1 + index / 100,
      }),
    ),
    extra,
  );

describe("Scoring a dump", () => {
  test("keyword search's top k and the reranked top k differ where the reranker moves the hit", () => {
    const late = question("late", [hitAt(4)]);
    expect(isHit(late, KEYWORD, 5, options)).toBe(true);
    expect(isHit(late, KEYWORD, 3, options)).toBe(false);
    expect(isHit(late, RERANK, 1, options)).toBe(true);
    const missed = question("missed", [hitAt(5)]);
    expect(isHit(missed, KEYWORD, 8, options)).toBe(true);
    expect(isHit(missed, KEYWORD, 5, options)).toBe(false);
  });

  test("a rerank score ties keep keyword search's order", () => {
    const tied = question("tied", [
      query([
        candidate(0, { rerankScore: 0.4 }),
        candidate(1, { rerankScore: 0.4, hits: [true] }),
        candidate(2, { rerankScore: 0.4 }),
      ]),
    ]);
    expect(isHit(tied, RERANK, 1, options)).toBe(false);
    expect(isHit(tied, RERANK, 2, options)).toBe(true);
  });

  test("a Question needing two Passages is a hit when the top k of its queries, together, hold both", () => {
    const own = query([candidate(0, { hits: [true, false] }), candidate(1)]);
    const translated = query([candidate(0), candidate(1), candidate(2, { hits: [false, true] })], {
      translated: true,
    });
    const both = question("both", [own, translated], 2);
    expect(isHit(both, KEYWORD, 3, options)).toBe(true);
    expect(isHit(both, KEYWORD, 2, options)).toBe(false);
    expect(isHit(question("one", [own], 2), KEYWORD, 5, options)).toBe(false);
  });

  test("a blend of ranks sits between keyword search's order and the reranker's", () => {
    const late = question("late", [hitAt(4)]);
    const blend = (weight: number): Order => ({ kind: "blend", weight });
    expect(isHit(late, blend(1), 1, options)).toBe(true);
    expect(isHit(late, blend(0), 1, options)).toBe(false);
    expect(isHit(late, blend(0.5), 2, options)).toBe(true);
  });

  test("dropping a neighbour of a better-ranked Passage can lose the hit it held", () => {
    const next = question("next", [
      query([
        candidate(0, { documentId: "a", position: 5 }),
        candidate(1, { documentId: "a", position: 6, hits: [true], rerankScore: 0.9 }),
        candidate(2, { documentId: "b", position: 6 }),
      ]),
    ]);
    expect(isHit(next, RERANK, 1, options)).toBe(true);
    expect(isHit(next, RERANK, 3, { ...options, dropNeighbours: true })).toBe(false);
    const apart = question("apart", [
      query([
        candidate(0, { documentId: "a", position: 5 }),
        candidate(1, { documentId: "a", position: 7, hits: [true] }),
      ]),
    ]);
    expect(isHit(apart, KEYWORD, 2, { ...options, dropNeighbours: true })).toBe(true);
  });

  test("only the first `depth` candidates are read, and one without a score says so", () => {
    const deep = question("deep", [
      query([...hitAt(0).candidates, candidate(6, { rerankScore: null, hits: [true] })]),
    ]);
    expect(isHit(deep, RERANK, 10, options)).toBe(true);
    expect(() => isHit(deep, RERANK, 10, { ...options, depth: 7 })).toThrow(/no rerank score/);
  });

  test("another model's scores come from a file, aligned with the candidates", () => {
    const late = question("late", [hitAt(4)]);
    const scores = parseScores(
      JSON.stringify({
        model: "other",
        set: "test",
        scores: { "late/0": [0, 0, 0, 0, 5, 0] },
      }),
    );
    expect(isHit(late, { kind: "model", scores }, 1, options)).toBe(true);
    const reversed = parseScores(
      JSON.stringify({ model: "other", set: "test", scores: { "late/0": [6, 5, 4, 3, 2, 1] } }),
    );
    expect(isHit(late, { kind: "model", scores: reversed }, 1, options)).toBe(false);
    expect(() =>
      isHit(
        late,
        { kind: "model", scores: { ...scores, scores: { "late/0": [1, 2] } } },
        1,
        options,
      ),
    ).toThrow(/no score for candidate 3/);
    expect(() => parseScores("{}")).toThrow(/needs "model"/);
  });

  test("the search Tool's hit needs every Passage among those it handed the model, over all queries", () => {
    const handed = (...passages: boolean[][]): DumpQuery["tool"] => ({
      passages: passages.map((hits, index) => ({ passageId: `t${index}`, rank: index, hits })),
    });
    const tool = question("tool", [query([], { tool: handed([true, false], [false, false]) })], 2);
    expect(isToolHit(tool)).toBe(false);
    tool.queries.push(query([], { translated: true, tool: handed([false, true]) }));
    expect(isToolHit(tool)).toBe(true);
    expect(isToolHit(question("none", [query([])], 0))).toBe(false);
  });

  test("tallies overall and by tag, with the Tool's beside them, and prints them", () => {
    const dump = dumpOf([
      question(
        "a",
        [hitAt(0, { tool: { passages: [{ passageId: "p0", rank: 0, hits: [true] }] } })],
        1,
        {
          kind: "gating",
        },
      ),
      question("b", [hitAt(5)], 1, { kind: "gating" }),
      question("c", [hitAt(1)], 1, { kind: "paraphrase" }),
    ]);
    const score = scoreDump(dump, [KEYWORD, RERANK], [1, 5], options);
    expect(score.orders[0]?.byTop[0]?.all).toEqual({ hits: 1, total: 3 });
    expect(score.orders[1]?.byTop[0]?.all).toEqual({ hits: 3, total: 3 });
    expect(score.orders[1]?.byTop[0]?.groups).toEqual({
      gating: { hits: 2, total: 2 },
      paraphrase: { hits: 1, total: 1 },
    });
    expect(score.tool.all).toEqual({ hits: 1, total: 3 });
    const text = formatScore(score, dump);
    expect(text).toContain("keyword + rerank");
    expect(text).toContain("search Tool");
    expect(text).toContain("by kind");
  });

  test("rejects a file that isn't a dump of this format", () => {
    expect(() => parseDump("{}", "x.json")).toThrow(/x.json isn't a format 1 dump/);
  });

  test("the command prints the scores of the files it is given", () => {
    const printed: string[] = [];
    expect(() => main(["--top", "x"], (text) => printed.push(text))).toThrow(/takes numbers/);
    expect(() => main([], () => {})).toThrow(/Usage/);
    expect(() => main(["--nope"], () => {})).toThrow(/Unknown option/);
  });
});

describe("Writing a dump from a core", { timeout: 60_000 }, () => {
  test("keeps each query's candidates, texts, hit flags and the Tool's Passages, and scores as the run does", async () => {
    const dataDir = await createTempDataFolder();
    const sources = await createSourceFolder();
    const core = startCore(dataDir);
    const paths = await Promise.all([
      writeSourceFile(
        sources,
        "Harbour Report 2025.pdf",
        buildPdf([
          { lines: ["Harbour traffic", "The harbour handled 4,200 container ships in 2025."] },
          { lines: ["Dredging", "Dredging deepened the main channel to fourteen metres."] },
        ]),
      ),
      writeSourceFile(
        sources,
        "Ferry Review.pdf",
        buildPdf([
          { lines: ["Ferries", "The night ferry to the islands carried 90,000 passengers."] },
        ]),
      ),
    ]);
    const [harbour, ferry] = await addAndProcess(core, paths);
    const ids = new Map([
      ["harbour", harbour?.id as string],
      ["ferry", ferry?.id as string],
    ]);
    const channel: ExpectedPassage = {
      document: "harbour",
      pages: [2, 2],
      quote: "deepened the main channel to fourteen metres",
    };
    const passengers: ExpectedPassage = {
      document: "ferry",
      pages: [1, 1],
      quote: "carried 90,000 passengers",
    };
    // A reranker that prefers the last Passage keyword search found.
    const reversing: ScoringReranker = {
      mode: GATING_MODE,
      rerank: async (_query, passages) => [...passages].reverse(),
      rerankScored: async (_query, passages) =>
        [...passages].reverse().map((passage, index) => ({ ...passage, score: 1 / (index + 2) })),
      info: () => {
        throw new Error("not needed");
      },
      close: () => {},
    };
    const db = openDatabase(`${dataDir}/${DATABASE_FILE}`);
    try {
      const questions = [
        {
          id: "channel",
          tags: { kind: "gating" },
          expected: [channel],
          expectedDocumentIds: [ids.get("harbour") as string],
          queries: ["How deep was the main channel dredged?"],
        },
        {
          id: "ferry",
          tags: { kind: "cross-lingual" },
          expected: [passengers],
          expectedDocumentIds: [ids.get("ferry") as string],
          queries: ["夜间渡轮运送了多少乘客？", "How many passengers did the night ferry carry?"],
        },
      ];
      const dump = await buildDump(
        {
          set: "test",
          groupBy: "kind",
          commit: "abc",
          core,
          db,
          reranker: reversing,
          rerankerInfo: { id: "model", name: "Model" },
          log: () => {},
        },
        questions,
      );
      expect(parseDump(JSON.stringify(dump))).toEqual(dump);
      expect(dump).toMatchObject({ set: "test", embeddings: "off", keywordDepth: 150 });
      const [first, second] = dump.questions;
      expect(first?.needed).toBe(1);
      const own = first?.queries[0];
      expect(own?.candidates[0]).toMatchObject({
        rank: 1,
        hits: [true],
        rerankText: expect.stringContaining("Harbour Report 2025\n"),
      });
      expect(own?.candidates[0]?.rerankText).toContain("deepened the main channel");
      expect(own?.candidates[0]?.keywordScore).toBeGreaterThan(0);
      expect(own?.candidates.every((each) => each.rerankScore !== null)).toBe(true);
      expect(own?.tool.passages.some((each) => each.hits[0])).toBe(true);
      // Keyword search can't match the Chinese wording; the translation finds the Passage.
      expect(second?.queries.map((each) => each.translated)).toEqual([false, true]);
      expect(second?.queries[0]?.candidates).toEqual([]);
      expect(second?.queries[1]?.candidates[0]?.hits).toEqual([true]);

      const score = scoreDump(dump, [KEYWORD], [5], { depth: 60, dropNeighbours: false });
      const results = await runRetrieval(
        core,
        [
          {
            id: "channel",
            language: "en",
            crossLingual: false,
            question: "How deep was the main channel dredged?",
            expected: channel,
          },
        ],
        ids,
        ["keyword"],
      );
      expect(results[0]?.modes.keyword?.hit).toBe(true);
      expect(score.orders[0]?.byTop[0]?.groups.gating).toEqual({ hits: 1, total: 1 });
      expect(score.orders[0]?.byTop[0]?.groups["cross-lingual"]).toEqual({ hits: 1, total: 1 });
      expect(score.tool.all).toEqual({ hits: 2, total: 2 });
    } finally {
      db.close();
      core.close();
    }
  });
});
