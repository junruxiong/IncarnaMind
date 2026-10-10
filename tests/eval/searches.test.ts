/**
 * The other ways the evaluation finds what the reranker sees (eval/lib/searches.ts,
 * reported next to the gate, never gating): how each builds its candidates.
 * The evaluation itself needs the real models and isn't part of `npm test`.
 */
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import {
  bm25,
  type CorpusPassage,
  createCorpus,
  queryWeights,
  tokensOf,
} from "../../eval/lib/corpus";
import type { EvalQuestion } from "../../eval/lib/evaluationSet";
import {
  CACHE_FILE,
  type Generate,
  parseQueries,
  prepareQueries,
  queryPrompt,
} from "../../eval/lib/rewrites";
import {
  documentFirstCandidates,
  FEEDBACK,
  feedbackCandidates,
  feedbackTerms,
  fusedQueriesCandidates,
  fusePassages,
  smallToBigCandidates,
} from "../../eval/lib/searches";
import { buildSubChunkIndex, SUB_CHUNK_TOKENS, subChunksOf } from "../../eval/lib/subChunks";
import type { Core, PassageSearchResult } from "../../src/core";
import { openDatabase } from "../../src/core/storage";

let nextSeq = 1;
const passage = (
  documentId: string,
  text: string,
  overrides: Partial<CorpusPassage> = {},
): CorpusPassage => {
  const seq = nextSeq++;
  return {
    seq,
    passageId: `p${seq}`,
    documentId,
    documentName: documentId,
    pageFrom: 1,
    pageTo: 1,
    position: seq,
    text,
    ...overrides,
  };
};

/** A core whose keyword search returns `ranking` (cut to the limit asked), recording what it was asked. */
function fakeCore(ranking: (query: string) => readonly PassageSearchResult[]) {
  const asked: string[] = [];
  const core = {
    searchPassages: async (query: string, options: { mode: string; limit: number }) => {
      asked.push(`${query}: ${options.mode} ${options.limit}`);
      return ranking(query).slice(0, options.limit);
    },
  } as unknown as Core;
  return { core, asked };
}

const ids = (passages: readonly PassageSearchResult[]) => passages.map((each) => each.passageId);

describe("BM25 in memory", () => {
  test("ranks as FTS5's bm25() does, with a weight per query term", () => {
    const texts = [
      "Spring tides happen at full moon and at new moon.",
      "Neap tides happen at quarter moon.",
      "Tides tides tides: the sea rises and falls twice a day.",
      "Orders ship from the Leeds warehouse on Fridays.",
      "The warehouse in Leeds closes early on Fridays, and the loading bay is shut.",
    ];
    const db = openDatabase(":memory:");
    db.exec(
      "CREATE VIRTUAL TABLE t USING fts5 (text, content = '', tokenize = 'unicode61 remove_diacritics 2')",
    );
    texts.forEach((text, index) => {
      db.run("INSERT INTO t (rowid, text) VALUES (?, ?)", [
        BigInt(index + 1),
        tokensOf(text).join(" "),
      ]);
    });
    const index = bm25(texts.map((text, index) => ({ seq: index + 1, tokens: tokensOf(text) })));
    for (const query of ["spring tides moon", "Leeds warehouse", "tides fridays"]) {
      const fts = db
        .all<{ id: number; score: number }>(
          "SELECT rowid AS id, bm25(t) AS score FROM t WHERE t MATCH ? ORDER BY score",
          [[...queryWeights(query).keys()].map((term) => `"${term}"`).join(" OR ")],
        )
        .map((row) => ({ seq: row.id, score: -row.score }));
      const ours = index.search(queryWeights(query), 10);
      expect(ours.map((each) => each.seq)).toEqual(fts.map((each) => each.seq));
      ours.forEach((each, rank) => {
        expect(each.score).toBeCloseTo(fts[rank]?.score as number, 6);
      });
    }
    db.close();

    // A term's weight scales its part of the score.
    const weighted = index.search(
      new Map([
        ["tides", 1],
        ["warehouse", 3],
      ]),
      10,
    );
    expect(weighted[0]?.seq).toBe(4);
  });

  test("a Document's own index takes its term statistics from that Document alone", () => {
    nextSeq = 1;
    const corpus = createCorpus([
      passage("tides", "tides tides moon"),
      passage("tides", "tides warehouse"),
      passage("tides", "tides sun"),
      passage("orders", "warehouse orders"),
      passage("orders", "orders invoices"),
    ]);
    // Every Passage of the Document has "tides": it tells them apart by nothing there.
    expect(corpus.documentIndex("tides").idf("tides")).toBe(1e-6);
    expect(corpus.index.idf("tides")).toBeGreaterThan(0);
    expect(corpus.documentIndex("tides").size).toBe(3);
  });
});

describe("Fusing rankings", () => {
  const p = (id: string) => passage("d", id, { passageId: id });

  test("by reciprocal rank, each Passage once, ties in the order they first appear", () => {
    const fused = fusePassages(
      [
        [p("a"), p("b"), p("c")],
        [p("c"), p("d")],
      ],
      10,
    );
    // c is in both lists; b and d tie in second places, b's list first.
    expect(ids(fused)).toEqual(["c", "a", "b", "d"]);
    expect(ids(fusePassages([[p("a"), p("b")]], 1))).toEqual(["a"]);
  });

  test("a list's weight scales its share", () => {
    expect(
      ids(
        fusePassages(
          [
            [p("a"), p("b")],
            [p("x"), p("y")],
          ],
          4,
          [1, 0.5],
        ),
      ),
    ).toEqual(["a", "b", "x", "y"]);
  });
});

describe("Keyword + feedback terms", () => {
  test("adds weighted words of the top Passages, leaving out stopwords, numbers, one-letter words and the query's own", () => {
    nextSeq = 1;
    const top = [
      passage("tides", "Spring tides happen at full moon and new moon, in 2024 as in 1924."),
      passage("tides", "Spring tides: the moon and the sun pull together, so the tides are high."),
    ];
    const corpus = createCorpus([
      ...top,
      passage("tides", "Neap tides happen at the quarters."),
      ...["Leeds", "York", "Hull", "Bath", "Ely", "Wells"].map((town) =>
        passage("orders", `Orders ship from the ${town} warehouse on Fridays.`),
      ),
    ]);

    const terms = feedbackTerms(corpus, "When do spring tides happen?", top);

    expect(terms.length).toBeLessThanOrEqual(FEEDBACK.terms);
    // In both top Passages, three times in all: first.
    expect(terms[0]?.term).toBe("moon");
    expect(terms.map((each) => each.term)).toEqual(expect.arrayContaining(["full", "sun", "pull"]));
    for (const { term } of terms) {
      expect(["spring", "tides", "happen", "the", "and", "so", "2024", "a"]).not.toContain(term);
    }
    expect(terms.reduce((sum, each) => sum + each.weight, 0)).toBeCloseTo(1, 9);

    // A word every Passage of the library has tells nothing: it weighs next to nothing.
    const everywhere = createCorpus([
      ...top,
      passage("sky", "The moon rises."),
      passage("sky", "The moon sets."),
    ]);
    const weights = new Map(
      feedbackTerms(everywhere, "When do spring tides happen?", top).map((each) => [
        each.term,
        each.weight,
      ]),
    );
    expect(weights.get("moon") ?? 0).toBeLessThan(1e-3);
  });

  test("works on segmented Chinese: words of two characters or more", () => {
    nextSeq = 1;
    const top = [
      passage("zh", "大潮发生在满月和新月时。"),
      passage("zh", "满月时引力最强，潮差最大。"),
    ];
    const corpus = createCorpus([...top, passage("zh", "小潮出现在上弦月和下弦月。")]);

    const terms = feedbackTerms(corpus, "大潮什么时候出现？", top).map((each) => each.term);

    expect(terms).toContain("满月");
    expect(terms).not.toContain("大潮");
    for (const term of terms) expect([...term].length).toBeGreaterThanOrEqual(2);
  });

  test("searches again with them and fuses that with keyword search's own top 20", async () => {
    nextSeq = 1;
    const spring = passage("tides", "Spring tides happen at full moon and new moon.");
    const springAgain = passage("tides", "Spring tides are the largest, at full moon.");
    // Has none of the Question's words, only the feedback's.
    const bulge = passage("tides", "The moon's pull raises a bulge; at full moon it is largest.");
    const corpus = createCorpus([
      spring,
      springAgain,
      bulge,
      passage("orders", "Orders ship from the Leeds warehouse on Fridays."),
    ]);
    const { core, asked } = fakeCore(() => [spring, springAgain]);

    const found = await feedbackCandidates(core, corpus, "When do spring tides happen?");

    expect(asked).toEqual(["When do spring tides happen?: keyword 20"]);
    expect(ids(found.candidates).slice(0, 2)).toEqual([spring.passageId, springAgain.passageId]);
    expect(ids(found.candidates)).toContain(bulge.passageId);
    expect(found.queries).toContain("moon");
  });
});

describe("Keyword + rewrites and keyword + sub-questions", () => {
  test("search the Question and each of the model's queries, and fuse them", async () => {
    nextSeq = 1;
    const a = passage("d", "a");
    const b = passage("d", "b");
    const c = passage("d", "c");
    const { core, asked } = fakeCore((query) =>
      query === "Question?" ? [a, b] : query === "Rewrite one" ? [c, a] : [c],
    );

    const found = await fusedQueriesCandidates(core, "Question?", ["Rewrite one", "Rewrite two"]);

    expect(asked).toEqual([
      "Question?: keyword 20",
      "Rewrite one: keyword 20",
      "Rewrite two: keyword 20",
    ]);
    expect(ids(found.candidates)).toEqual([c.passageId, a.passageId, b.passageId]);
    expect(found.queries).toEqual(["Rewrite one", "Rewrite two"]);
  });

  test("a model's answer: one query a line, numbering and quotes off, the Question and repeats left out", () => {
    const answer =
      '1. "What is a few-shot setting?"\n2) few-shot: the term\n\n- What does "few-shot" mean?\n3. Few-shot setting definition\n4. Another';
    expect(parseQueries(answer, 'What does "few-shot" mean?', 2)).toEqual([
      "What is a few-shot setting?",
      "few-shot: the term",
    ]);
    expect(
      parseQueries("ESG 一词最早见于哪份报告？\n、", "ESG 这个说法最早出现在哪份报告里？", 2),
    ).toEqual(["ESG 一词最早见于哪份报告？"]);
    // The Question as it is, for a sub-question split: nothing else to search.
    expect(parseQueries("When are spring tides?", "When are spring tides?", 3)).toEqual([]);
  });

  test("the prompt names the Documents and their languages", () => {
    const prompt = queryPrompt("rewrites", "When are spring tides?", [
      { name: "Tides", language: "English" },
      { name: "潮汐", language: "Chinese" },
      { name: "Notes", language: null },
    ]);
    expect(prompt).toContain("- Tides (English)\n- 潮汐 (Chinese)\n- Notes\n");
    expect(prompt).toContain("Write 2 other ways to ask the question");
    expect(prompt.endsWith("Question: When are spring tides?")).toBe(true);
    expect(queryPrompt("sub-questions", "Q", [])).toContain("break it down into at most 3");
  });

  test("the model is asked once per Question and prompt; a later run reads its answers from the results folder", async () => {
    const resultsDir = await mkdtemp(join(tmpdir(), "query-rewrites-"));
    try {
      const prompts: string[] = [];
      const generate: Generate = async (prompt) => {
        prompts.push(prompt);
        return {
          text: "Spring tide dates\nWhen is the tidal range largest?",
          inputTokens: 80,
          outputTokens: 12,
        };
      };
      const questions = [
        { id: "en-01", question: "When are spring tides?" },
        { id: "en-02", question: "What are neap tides?" },
      ] as EvalQuestion[];
      const options = {
        search: "rewrites" as const,
        model: "anthropic/model",
        generate,
        questions,
        documents: [{ name: "Tides", language: "English" }],
        resultsDir,
        log: () => {},
      };

      const first = await prepareQueries(options);
      const second = await prepareQueries(options);

      expect(prompts).toHaveLength(2);
      expect(first.queries.get("en-01")).toEqual([
        "Spring tide dates",
        "When is the tidal range largest?",
      ]);
      expect(second.queries).toEqual(first.queries);
      expect(first.cost).toMatchObject({
        questions: 2,
        calls: 2,
        cached: 0,
        meanInputTokens: 80,
        meanOutputTokens: 12,
        meanQueries: 2,
      });
      expect(second.cost).toMatchObject({ calls: 0, cached: 2, meanInputTokens: 80 });
      // Another model, or another prompt, is asked afresh.
      await prepareQueries({ ...options, model: "openai/model" });
      await prepareQueries({ ...options, search: "sub-questions" });
      expect(prompts).toHaveLength(6);
      const cache = JSON.parse(await readFile(join(resultsDir, CACHE_FILE), "utf8"));
      expect(Object.keys(cache.entries)).toHaveLength(6);
    } finally {
      await rm(resultsDir, { recursive: true, force: true });
    }
  });
});

/** A line of about `tokens` approximate tokens (4 characters each). */
const line = (word: string, tokens: number) =>
  `${`${word} `.repeat(Math.ceil((tokens * 4) / (word.length + 1)))}`
    .slice(0, tokens * 4 - 1)
    .trim();

describe("Small-to-big", () => {
  test("cuts a Document into sub-chunks, each mapped to the Passages that hold it, or else those it overlaps", () => {
    const lines = ["one", "two", "three", "four", "five", "six"].map((word) => line(word, 60));
    const pages = [{ page: 1, text: lines.join("\n") }];
    const text = (from: number, to: number) => lines.slice(from - 1, to).join("\n");

    const held = subChunksOf(pages, [
      { seq: 1, text: text(1, 4) },
      { seq: 2, text: text(3, 6) },
    ]);
    // Two 60-token lines make a sub-chunk of at most 128 tokens.
    expect(held.chunks.map((chunk) => chunk.text)).toEqual([text(1, 2), text(3, 4), text(5, 6)]);
    expect(held.chunks.map((chunk) => chunk.passages)).toEqual([[1], [1, 2], [2]]);
    expect(held.unplaced).toBe(0);

    // Lines 3 and 4 are in no one Passage: the sub-chunk goes to both it overlaps.
    const straddling = subChunksOf(pages, [
      { seq: 1, text: text(1, 3) },
      { seq: 2, text: text(4, 6) },
    ]);
    expect(straddling.chunks.map((chunk) => chunk.passages)).toEqual([[1], [1, 2], [2]]);
    expect(SUB_CHUNK_TOKENS).toBe(128);
  });

  test("scores a Passage by its best sub-chunk; summing them favours Passages that match in several places", () => {
    nextSeq = 1;
    const focused = [line("filler", 60), "Spring tides come at full moon.", line("other", 60)].join(
      "\n",
    );
    const spread = [
      `Tides rise. ${line("tides", 20)}`,
      line("spare", 60),
      `Tides fall. ${line("tides", 20)}`,
      line("rest", 60),
    ].join("\n");
    const focusedPassage = passage("a", focused);
    const spreadPassage = passage("b", spread);
    const corpus = createCorpus([
      focusedPassage,
      spreadPassage,
      passage("c", "Orders ship from the Leeds warehouse on Fridays."),
    ]);
    const pages = new Map([
      ["a", [{ page: 1, text: focused }]],
      ["b", [{ page: 1, text: spread }]],
      ["c", [{ page: 1, text: "Orders ship from the Leeds warehouse on Fridays." }]],
    ]);
    const index = buildSubChunkIndex(corpus, (id) => pages.get(id) ?? []);
    try {
      expect(index.stats).toMatchObject({ maxTokens: 128, unplaced: 0 });
      expect(index.stats.subChunks).toBeGreaterThan(3);
      expect(index.stats.indexBytes).toBeGreaterThan(0);
      expect(index.stats.passageIndexBytes).toBeGreaterThan(0);

      const query = "spring tides full moon";
      expect(ids(smallToBigCandidates(index, query).candidates)).toEqual([
        focusedPassage.passageId,
        spreadPassage.passageId,
      ]);
      expect(ids(index.search("tides", 20, "sum"))[0]).toBe(spreadPassage.passageId);
    } finally {
      index.close();
    }
  });
});

describe("Document first", () => {
  test("searches inside the 3 Documents that hold most of keyword search's top 20, each with its own term statistics", async () => {
    nextSeq = 1;
    const a1 = passage("a", "tides tides tides tides moon");
    const a2 = passage("a", "tides warehouse");
    const a3 = passage("a", "tides sun");
    const b1 = passage("b", "warehouse orders");
    const c1 = passage("c", "tides invoices warehouse");
    const d1 = passage("d", "tides warehouse warehouse");
    const corpus = createCorpus([
      a1,
      a2,
      a3,
      b1,
      c1,
      d1,
      ...Array.from({ length: 6 }, (_, index) => passage("e", `unrelated ${index}`)),
    ]);
    // Keyword search's top: a's Passages twice, then b, c, and d last.
    const { core, asked } = fakeCore(() => [a1, b1, a3, c1, d1]);

    const found = await documentFirstCandidates(core, corpus, "tides warehouse");

    expect(asked).toEqual(["tides warehouse: keyword 20"]);
    expect(found.queries).toEqual(["a", "b", "c"]);
    // Inside a, every Passage has "tides": "warehouse" decides, so a2 comes first.
    expect(ids(found.candidates)[0]).toBe(a2.passageId);
    expect(ids(found.candidates)).not.toContain(d1.passageId);
    expect(ids(found.candidates)).toEqual(expect.arrayContaining([b1.passageId, c1.passageId]));
  });
});
