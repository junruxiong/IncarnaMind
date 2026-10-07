/**
 * How the grouping check (eval/grouping, #51) scores, chooses and reports,
 * on its real fixture set with generated vectors: one direction per subject,
 * and optionally a pull towards one "Chinese" direction, which splits the
 * English–Chinese pairs as a model favouring the language might. The check
 * itself needs the real model and isn't part of `npm test`; its decisions are.
 */
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, onTestFinished, test } from "vitest";
import {
  chooseVariant,
  evaluateVariant,
  fixturesReport,
  founderSample,
} from "../../eval/grouping/lib/check";
import {
  founderSheet,
  libraryFiles,
  SAMPLE_SIZE,
  shuffled,
  topicPreview,
} from "../../eval/grouping/lib/founder";
import { runIncremental } from "../../eval/grouping/lib/incremental";
import {
  type FounderReport,
  fixturesMarkdown,
  fixturesSummary,
  founderMarkdown,
  type RunInfo,
} from "../../eval/grouping/lib/report";
import {
  barFailures,
  type CasesScore,
  choose,
  scoreCases,
  scoreNameRows,
  summariseGrouping,
  topicsByKey,
} from "../../eval/grouping/lib/scoring";
import { loadGroupingSet } from "../../eval/grouping/lib/set";
import type { MeansTiming } from "../../eval/grouping/lib/timing";
import type { VariantId, VariantVectors } from "../../eval/grouping/lib/variants";
import { normalised, seededRandom } from "../../src/core/topics/grouping";

const root = fileURLToPath(new URL("../..", import.meta.url));
const set = loadGroupingSet(root);
const subjects = Object.keys(set.subjects);
const names = new Map(
  set.documents.map((document) => [
    document.key,
    document.path
      .split("/")
      .pop()
      ?.replace(/\.[^.]+$/, "") ?? "",
  ]),
);
const subjectOf = (key: string) => set.documents.find((document) => document.key === key)?.subject;

/**
 * A vector per Document: its subject's direction, seeded noise (none for the
 * `quiet` ones), and `zhPull` towards a shared "Chinese" direction.
 */
function vectorsFor({
  zhPull = 0,
  noise = 0.1,
  seed = 1,
  quiet = new Set<string>(),
} = {}): Map<string, Float32Array> {
  const dimensions = subjects.length + 2;
  const random = seededRandom(seed);
  return new Map(
    set.documents.map((document) => {
      const vector = new Float32Array(dimensions);
      const size = quiet.has(document.key) ? 0 : noise;
      for (let d = 0; d < dimensions; d++) vector[d] = (random() - 0.5) * 2 * size;
      const axis = subjects.indexOf(document.subject);
      vector[axis] = (vector[axis] as number) + 1;
      if (document.language === "zh")
        vector[dimensions - 1] = (vector[dimensions - 1] as number) + zhPull;
      return [document.key, normalised(vector) as Float32Array];
    }),
  );
}

const built = (id: VariantId, vectors: Map<string, Float32Array>): VariantVectors => ({
  id,
  vectors,
  seconds: 0.5,
  embeddings: id === "names-included" ? 0 : vectors.size,
});

const RUN: RunInfo = {
  startedAt: "2026-10-08T09:00:00.000Z",
  seconds: 120,
  commit: "abc1234",
  node: "v25.5.0",
  platform: "darwin arm64",
  cpu: "Apple M2 Max (12 cores)",
};

const MEANS: MeansTiming = {
  passages: 100_000,
  documents: 2_000,
  dimensions: 384,
  textLength: 1_500,
  databaseMb: 330,
  buildSeconds: 9,
  coldMs: [410, 380, 395],
  warmMs: [40, 38, 41],
  barMs: 300,
  passes: false,
};

describe("scoring", () => {
  test("a pair is a hit only in one Topic; a deck only beside a Document on its subject", () => {
    const topicOf = topicsByKey(
      [
        { members: ["attention-paper", "attention-zh", "attention-deck"] },
        { members: ["lm-gpt3", "tea-deck"] },
        { members: ["lm-zh", "tea-en"] },
      ],
      ["tea-zh", "esg-deck"],
    );
    const score = scoreCases(set, set.choosing, topicOf);
    const pair = (key: string) => score.pairs.cases.find((each) => each.documents[0] === key);
    expect(pair("attention-paper")).toMatchObject({ hit: true, why: null });
    expect(pair("lm-gpt3")?.hit).toBe(false);
    expect(pair("lm-gpt3")?.why).toMatch(/^apart: lm-gpt3 with tea; lm-zh with tea$/);
    expect(pair("tea-en")?.why).toBe("not grouped yet: tea-zh");
    expect(pair("esg-report")?.why).toMatch(/^no vector/);
    const deck = (key: string) => score.decks.cases.find((each) => each.documents[0] === key);
    expect(deck("attention-deck")?.hit).toBe(true);
    expect(deck("tea-deck")).toMatchObject({ hit: false, why: "with language-models" });
    expect(deck("esg-deck")).toMatchObject({ hit: false, why: "not grouped yet" });
    expect(score.hits).toBe(score.pairs.hits + score.decks.hits);
  });

  test("the shared-name row counts Documents on different subjects that share a Topic", () => {
    const topicOf = topicsByKey([
      { members: ["attention-zh", "gd-zh", "lm-zh", "lm-zh2"] },
      { members: ["sdg-zh", "esg-zh"] },
    ]);
    const [prefix] = scoreNameRows(set, names, topicOf);
    // attention/gd, attention/lm ×2, gd/lm ×2 in the first Topic; sdg/esg in the second. lm-zh/lm-zh2 share a subject.
    expect(prefix?.together).toBe(6);
    expect(prefix?.documents.length).toBe(17);
  });

  test("purity is the share of grouped Documents in a Topic whose main subject is theirs", () => {
    const topicOf = topicsByKey(
      [{ members: ["mars-en", "mars-zh", "photo-en"] }, { members: ["tea-en", "tea-zh"] }],
      ["quake-en"],
    );
    const summary = summariseGrouping(set, topicOf);
    expect(summary).toMatchObject({ topics: 2, grouped: 5, ungrouped: 1 });
    expect(summary.purity).toBeCloseTo(4 / 5, 10);
    expect(summary.contents).toEqual(["mars ×2, photosynthesis", "tea ×2"]);
  });

  test("the most hits wins; within one case, the lowest rank", () => {
    expect(
      choose(
        [
          { id: "a", hits: 8, total: 11, rank: 0 },
          { id: "b", hits: 10, total: 11, rank: 1 },
          { id: "c", hits: 9, total: 11, rank: 2 },
        ],
        "cost",
      ).id,
    ).toBe("b");
    const near = choose(
      [
        { id: "cheap", hits: 9, total: 11, rank: 0 },
        { id: "dear", hits: 10, total: 11, rank: 2 },
      ],
      "extra cost",
    );
    expect(near.id).toBe("cheap");
    expect(near.reason).toMatch(
      /within 1 case of the most hits \(9 against 10 of 11\), with the lowest extra cost/,
    );
    expect(
      choose(
        [
          { id: "first", hits: 5, total: 5, rank: 1 },
          { id: "second", hits: 5, total: 5, rank: 1 },
        ],
        "time",
      ).id,
    ).toBe("first");
  });

  test("the bars: 4 of 5 pairs and 3 of 5 decks or spreadsheets", () => {
    const score = (pairs: number, decks: number): CasesScore => ({
      pairs: { hits: pairs, total: 5, cases: [] },
      decks: { hits: decks, total: 5, cases: [] },
      hits: pairs + decks,
      total: 10,
    });
    expect(barFailures(score(4, 3))).toEqual([]);
    expect(barFailures(score(3, 3))).toEqual([
      "Held-out pairs: 3 of 5 land in one Topic; the bar is 4 of 5.",
    ]);
    expect(barFailures(score(5, 2))).toEqual([
      "Held-out decks and spreadsheets: 2 of 5 land with a Document on their subject; the bar is 3 of 5.",
    ]);
  });
});

describe("the variants", () => {
  test("Documents clearly on their subjects pass every case, with at least as many clusters as subjects", () => {
    const result = evaluateVariant(
      set,
      built("names-included", vectorsFor()),
      names,
      subjects.length,
    );
    expect(result.k).toBe(subjects.length);
    expect(result.heldOut.hits).toBe(result.heldOut.total);
    expect(result.choosing.hits).toBe(result.choosing.total);
    expect(result.summary.purity).toBe(1);
  });

  test("a pull towards the language splits the pairs, and the decks with an English Document stay", () => {
    const result = evaluateVariant(
      set,
      built("names-included", vectorsFor({ zhPull: 2 })),
      names,
      subjects.length,
    );
    expect(result.heldOut.pairs.hits).toBeLessThan(result.heldOut.pairs.total);
    expect(barFailures(result.heldOut).length).toBeGreaterThan(0);
  });

  test("the cheapest variant wins a near-tie on the choosing set", () => {
    const clean = vectorsFor();
    const variants = (["names-included", "names-removed", "name-free"] as const).map((id) =>
      evaluateVariant(
        set,
        built(id, id === "names-included" ? vectorsFor({ zhPull: 0.15, seed: 2 }) : clean),
        names,
        subjects.length,
      ),
    );
    const chosen = chooseVariant(variants);
    const best = Math.max(...variants.map((variant) => variant.choosing.hits));
    const winner = variants.find((variant) => variant.id === chosen.id);
    expect(winner?.choosing.hits).toBeGreaterThanOrEqual(best - 1);
    // Among those within a case, the order of cost: names included, names removed, name-free.
    const order: VariantId[] = ["names-included", "names-removed", "name-free"];
    const near = variants.filter((variant) => variant.choosing.hits >= best - 1).map((v) => v.id);
    expect(chosen.id).toBe(order.find((id) => near.includes(id)));
  });
});

describe("the incremental cases", () => {
  test("late arrivals are placed by the threshold: with their subject, or out when their subject is new", () => {
    // On their subject's direction exactly: above any Topic member's 10th percentile.
    const result = runIncremental(set, vectorsFor({ quiet: new Set(set.lateArrivals) }));
    expect(result.initial.documents).toBe(set.documents.length - set.lateArrivals.length);
    for (const arrival of result.lateArrivals) {
      const expected = subjectOf(arrival.key) === "pharma" ? "stays out" : "joins its subject";
      expect(arrival, arrival.key).toMatchObject({ expected, correct: true });
    }
  });

  test("the User's rename, moves and new Topic survive the Regroup", () => {
    const result = runIncremental(set, vectorsFor());
    expect(result.corrections[0]).toMatch(/^Renamed topic-\d+, the Topic holding tea-en\.$/);
    // No pair was split, so the last-resort move makes the move.
    expect(
      result.corrections.some((line) =>
        line.startsWith("Moved pharma-code from Not grouped yet into topic-"),
      ),
    ).toBe(true);
    expect(result.corrections).toContain(
      'Made "My data tables" (user-topic-1) by moving quake-sheet and inflation-sheet into it.',
    );
    expect(result.checks.length).toBeGreaterThanOrEqual(5);
    expect(result.checks.every((check) => check.kept)).toBe(true);
    expect(result.regroup.reclustered).toBe(true);
  });

  test("split pairs are put together by moving the Chinese Document, and the moves are kept", () => {
    const result = runIncremental(set, vectorsFor({ zhPull: 2 }));
    const moves = result.corrections.filter((line) => line.startsWith("Moved"));
    expect(moves).toHaveLength(set.corrections.moves);
    for (const move of moves) expect(move).toMatch(/^Moved \S+-zh\d? from /);
    expect(result.checks.every((check) => check.kept)).toBe(true);
  });
});

describe("the report", () => {
  const variantsFor = (zhPull: number) =>
    (["names-included", "names-removed", "name-free"] as const).map((id) =>
      evaluateVariant(set, built(id, vectorsFor({ zhPull })), names, subjects.length),
    );

  test("passes, recommends building, and names O5's fallback when the means miss 300 ms", () => {
    const variants = variantsFor(0);
    const report = fixturesReport({
      set,
      run: RUN,
      names,
      passages: new Map(),
      processing: { passages: 1500, seconds: 60 },
      variants,
      incremental: runIncremental(set, vectorsFor()),
      classifiers: { skipped: "no classifier was given." },
      timing: {
        kMeans: { documents: 5000, dimensions: 384, k: 40, seconds: 3.2, runs: [] },
        means: MEANS,
      },
    });
    expect(report.result).toBe("pass");
    expect(report.chosen.id).toBe("names-included");
    expect(report.bars.pairs).toEqual({ hits: 5, total: 5, needed: 4 });
    expect(report.recommendation[0]).toMatch(/^Build the grouping as designed/);
    expect(report.recommendation.join("\n")).toMatch(/Take O5's named fallback/);
    const markdown = fixturesMarkdown(report);
    for (const heading of [
      "# Grouping check",
      "**Result: pass**",
      "## The three Document vectors",
      "### Documents whose names share a part (R3)",
      "## Incremental cases, with the chosen variant",
      "## The classifier fallback (R0)",
      "Skipped: no classifier was given.",
      "## Timing",
      "## The founder's 30-Document sample",
      "INCARNAMIND_EVAL_GROUPING_FOLDER",
      "## The fixture set",
    ]) {
      expect(markdown).toContain(heading);
    }
    expect(fixturesSummary(report, join(root, "eval/results/grouping-x"), root)).toContain(
      "Result: pass",
    );
  });

  test("fails, with the bars it missed, when the pairs split", () => {
    const report = fixturesReport({
      set,
      run: RUN,
      names,
      passages: new Map(),
      processing: { passages: 1500, seconds: 60 },
      variants: variantsFor(2),
      incremental: runIncremental(set, vectorsFor({ zhPull: 2 })),
      classifiers: { skipped: "no classifier was given." },
      timing: { skipped: "INCARNAMIND_EVAL_GROUPING_TIMING=0" },
    });
    expect(report.result).toBe("fail");
    expect(report.failures[0]).toMatch(
      /^Held-out pairs: \d of 5 land in one Topic; the bar is 4 of 5\.$/,
    );
    expect(report.recommendation[0]).toMatch(/^The held-out bars aren't met/);
    expect(fixturesMarkdown(report)).toContain("**Result: fail**");
  });
});

describe("the founder's sample", () => {
  test("30 random Documents, each with its Topic in each grouping, blind to the variant", () => {
    const variants: VariantVectors[] = [
      built("names-included", vectorsFor()),
      built("names-removed", vectorsFor({ seed: 3 })),
      built("name-free", vectorsFor({ zhPull: 1 })),
    ];
    const sample = founderSample(variants, names, 51);
    expect(sample.sample).toBe(SAMPLE_SIZE);
    expect(sample.rows).toHaveLength(SAMPLE_SIZE * 3);
    expect(Object.keys(sample.letters)).toEqual(["A", "B", "C"]);
    expect(new Set(Object.values(sample.letters).map((letter) => letter.id)).size).toBe(3);
    for (const row of sample.rows) {
      expect(row.topic).toMatch(
        /^(With \d+ other Documents?: |Not grouped yet|A Topic of its own)/,
      );
      expect(row.topic).not.toContain(`${row.document};`);
    }
    // The same seed draws the same sample.
    expect(founderSample(variants, names, 51).rows).toEqual(sample.rows);
    const report: FounderReport = {
      kind: "founder",
      run: RUN,
      folder: "/library",
      files: 47,
      ready: 47,
      notReady: 0,
      passages: 900,
      processingSeconds: 40,
      letters: sample.letters,
      sample: sample.sample,
      sheet: "founder-sample.csv",
    };
    expect(founderMarkdown(report)).toContain("founder-sample.csv");
  });

  test("the Topic preview shows the Documents nearest its centre, leaving out the one judged", () => {
    const at = (x: number, y: number) => normalised([x, y]) as Float32Array;
    const centroid = at(1, 0);
    const members = [
      { key: "judged", name: "Judged", vector: at(1, 0) },
      ...Array.from({ length: 10 }, (_, index) => ({
        key: `m${index}`,
        name: `Member ${index}`,
        vector: at(1, index / 10),
      })),
    ];
    expect(topicPreview(members, centroid, "judged")).toBe(
      "With 10 other Documents: Member 0; Member 1; Member 2; Member 3; Member 4; Member 5; Member 6; Member 7; and 2 more",
    );
    expect(topicPreview(members.slice(0, 1), centroid, "judged")).toBe("A Topic of its own");
  });

  test("the sheet is UTF-8 with a byte-order mark, one row per Document and grouping", () => {
    const sheet = founderSheet([
      {
        document: "维基百科-地震",
        file: "zh/维基百科-地震.pdf",
        grouping: "A",
        topic: 'With 1 other Document: "Quakes", 2023',
      },
    ]);
    expect(sheet.startsWith("﻿document,file,grouping,its Topic,right Topic? (y/n),note\r\n")).toBe(
      true,
    );
    expect(sheet).toContain(
      '维基百科-地震,zh/维基百科-地震.pdf,A,"With 1 other Document: ""Quakes"", 2023",,\r\n',
    );
  });

  test("a seeded shuffle is a repeatable permutation", () => {
    const items = Array.from({ length: 20 }, (_, index) => index);
    const once = shuffled(items, 7);
    expect(shuffled(items, 7)).toEqual(once);
    expect([...once].sort((a, b) => a - b)).toEqual(items);
    expect(once).not.toEqual(items);
  });

  test("lists the files IncarnaMind indexes, at any depth, leaving out hidden ones; a limit draws at random", async () => {
    const folder = await mkdtemp(join(tmpdir(), "incarnamind-founder-"));
    onTestFinished(() => rm(folder, { recursive: true, force: true }));
    await mkdir(join(folder, "papers", "2024"), { recursive: true });
    await mkdir(join(folder, ".git"));
    for (const name of [
      "notes.md",
      "papers/a.pdf",
      "papers/2024/b.docx",
      "papers/2024/table.xlsx",
      "photo.jpg",
      ".hidden.md",
      ".git/config.txt",
    ]) {
      await writeFile(join(folder, name), "x");
    }
    expect(libraryFiles(folder, null, 1).map((file) => file.key)).toEqual([
      "notes.md",
      "papers/2024/b.docx",
      "papers/2024/table.xlsx",
      "papers/a.pdf",
    ]);
    const limited = libraryFiles(folder, 2, 1);
    expect(limited).toHaveLength(2);
    expect(libraryFiles(folder, 2, 1)).toEqual(limited);
    expect(limited[0]?.path).toBe(join(folder, limited[0]?.key as string));
  });
});
