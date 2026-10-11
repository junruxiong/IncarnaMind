/**
 * `npm run eval:score -- <dump.json>...` (eval/README.md, "Offline scoring"): scores
 * other result orders on the dumps `npm run eval:dump` writes, without running
 * a model: keyword search's top k, the built-in reranking model's top k, a
 * blend of the two ranks, another model's scores from a file, and what the
 * search Tool handed the model. It is one file that Node runs as it is
 * (`node eval/score.ts`), and the dump format is defined here: the writer
 * (./lib/dumps) and the tests import it.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

export const DUMP_FORMAT = 1;

/** One of keyword search's best Passages for a query, as a reranker reads it. */
export interface DumpCandidate {
  passageId: string;
  documentId: string;
  /** The Passage's position in its Document, from 0: neighbours are one apart and overlap. */
  position: number;
  /** Keyword search's rank, from 1. */
  rank: number;
  /** BM25 (higher is better); null when the query has no keyword match. */
  keywordScore: number | null;
  /** The built-in reranking model's score, 0 to 1; null past `rerankedDepth`. */
  rerankScore: number | null;
  /** For each Passage the Question needs, whether this one meets the hit rule for it. */
  hits: boolean[];
  /** The Document's name, then the text, normalised: what a reranker reads. */
  rerankText: string;
}

/** What the search Tool handed the model for a query: its Passages, in reading order. */
export interface DumpTool {
  passages: { passageId: string; rank: number; hits: boolean[] }[];
}

export interface DumpQuery {
  text: string;
  /** The Question's hand-written translation, searched as a second query. */
  translated: boolean;
  candidates: DumpCandidate[];
  tool: DumpTool;
}

export interface DumpQuestion {
  id: string;
  /** How reports group it, e.g. { kind: "gating", language: "en" }. */
  tags: Record<string, string>;
  /** How many Passages the answer needs: the length of every `hits`. */
  needed: number;
  queries: DumpQuery[];
}

export interface Dump {
  format: typeof DUMP_FORMAT;
  /** "gating", "formats" or "hard". */
  set: string;
  createdAt: string;
  commit: string;
  /** Always "off": the dumps are keyword search and the search Tool as it runs without embeddings. */
  embeddings: "off";
  /** How many keyword candidates each query holds. */
  keywordDepth: number;
  /** How many of them have a `rerankScore`: what the search Tool hands its reranker. */
  rerankedDepth: number;
  reranker: { id: string; name: string };
  /** The tag the report splits by first. */
  groupBy: string;
  questions: DumpQuestion[];
}

/** Another model's scores for a dump, one number per candidate, higher is better. */
export interface ScoresFile {
  model: string;
  /** The dump's `set`. */
  set: string;
  /** Keyed "<Question id>/<index of the query in the Question>", in the dump's candidate order. */
  scores: Record<string, number[]>;
}

function parseJson<Parsed>(text: string, source: string): Parsed {
  try {
    return JSON.parse(text) as Parsed;
  } catch (error) {
    throw new Error(`${source} isn't JSON: ${error instanceof Error ? error.message : error}`);
  }
}

export function parseDump(text: string, source = "dump"): Dump {
  const dump = parseJson<Dump>(text, source);
  if (dump.format !== DUMP_FORMAT || !Array.isArray(dump.questions)) {
    throw new Error(`${source} isn't a format ${DUMP_FORMAT} dump.`);
  }
  return dump;
}

export function parseScores(text: string, source = "scores file"): ScoresFile {
  const file = parseJson<ScoresFile>(text, source);
  if (typeof file.model !== "string" || typeof file.set !== "string" || !file.scores) {
    throw new Error(`${source} needs "model", "set" and "scores" (eval/README.md).`);
  }
  return file;
}

/** The ways to order a query's candidates. */
export type Order =
  | { kind: "keyword" }
  | { kind: "rerank" }
  /** The built-in model's. A blend of ranks: `weight` for the reranker, the rest for keyword search. */
  | { kind: "blend"; weight: number }
  /** Another model's scores, from a file. */
  | { kind: "model"; scores: ScoresFile }
  | { kind: "model-blend"; scores: ScoresFile; weight: number };

export interface ScoreOptions {
  /** How many of keyword search's best a reranker reads. */
  depth: number;
  /** Drop a candidate next to a better-ranked one of the same Document before reranking. */
  dropNeighbours: boolean;
}

/** Reciprocal rank fusion's k, as hybrid search uses. */
const BLEND_K = 60;

export function orderName(order: Order, options: ScoreOptions): string {
  const base = (() => {
    switch (order.kind) {
      case "keyword":
        return "keyword";
      case "rerank":
        return "keyword + rerank";
      case "blend":
        return `blend ${order.weight}`;
      case "model":
        return `keyword + ${order.scores.model}`;
      case "model-blend":
        return `blend ${order.weight} of ${order.scores.model}`;
    }
  })();
  return options.dropNeighbours ? `${base}, no neighbours` : base;
}

const ranksOf = (scores: readonly number[]) => {
  const byScore = scores
    .map((score, index) => ({ score, index }))
    .sort((a, b) => b.score - a.score || a.index - b.index);
  const ranks: number[] = [];
  byScore.forEach(({ index }, position) => {
    ranks[index] = position + 1;
  });
  return ranks;
};

/** The pool a reranker reads: keyword search's first `depth`, without neighbours of better ones if asked. */
function pool(query: DumpQuery, options: ScoreOptions): number[] {
  const indices: number[] = [];
  const kept: DumpCandidate[] = [];
  for (const [index, candidate] of query.candidates.slice(0, options.depth).entries()) {
    if (
      options.dropNeighbours &&
      kept.some(
        (other) =>
          other.documentId === candidate.documentId &&
          Math.abs(other.position - candidate.position) === 1,
      )
    ) {
      continue;
    }
    kept.push(candidate);
    indices.push(index);
  }
  return indices;
}

/** Another model's scores for a query, or the built-in model's, for the candidates in `indices`. */
function scoresFor(
  order: Order,
  question: DumpQuestion,
  queryIndex: number,
  indices: readonly number[],
): number[] {
  const query = question.queries[queryIndex] as DumpQuery;
  if (order.kind === "model" || order.kind === "model-blend") {
    const given = order.scores.scores[`${question.id}/${queryIndex}`];
    if (!given)
      throw new Error(`${order.scores.model}: no scores for ${question.id}/${queryIndex}.`);
    return indices.map((index) => {
      const score = given[index];
      if (score === undefined || !Number.isFinite(score)) {
        throw new Error(
          `${order.scores.model}: no score for candidate ${index + 1} of ${question.id}/${queryIndex}.`,
        );
      }
      return score;
    });
  }
  return indices.map((index) => {
    const score = query.candidates[index]?.rerankScore;
    if (score === null || score === undefined) {
      throw new Error(
        `${question.id}/${queryIndex}: candidate ${index + 1} has no rerank score; use --depth up to the dump's rerankedDepth, or give --scores for more.`,
      );
    }
    return score;
  });
}

/** The candidates' indices in the query, best first. */
export function orderQuery(
  order: Order,
  question: DumpQuestion,
  queryIndex: number,
  options: ScoreOptions,
): number[] {
  const query = question.queries[queryIndex] as DumpQuery;
  const indices = pool(query, options);
  if (order.kind === "keyword") return indices;
  const scores = scoresFor(order, question, queryIndex, indices);
  if (order.kind === "rerank" || order.kind === "model") {
    return indices
      .map((index, at) => ({ index, score: scores[at] as number }))
      .sort((a, b) => b.score - a.score || a.index - b.index)
      .map(({ index }) => index);
  }
  const reranked = ranksOf(scores);
  return indices
    .map((index, at) => ({
      index,
      fused:
        order.weight / (BLEND_K + (reranked[at] as number)) +
        (1 - order.weight) / (BLEND_K + at + 1),
    }))
    .sort((a, b) => b.fused - a.fused || a.index - b.index)
    .map(({ index }) => index);
}

/** Every Passage the answer needs is in the top k of some query's order. */
export function isHit(
  question: DumpQuestion,
  order: Order,
  topK: number,
  options: ScoreOptions,
): boolean {
  const found = new Array<boolean>(question.needed).fill(false);
  question.queries.forEach((query, queryIndex) => {
    for (const index of orderQuery(order, question, queryIndex, options).slice(0, topK)) {
      (query.candidates[index] as DumpCandidate).hits.forEach((hit, needed) => {
        if (hit) found[needed] = true;
      });
    }
  });
  return question.needed > 0 && found.every(Boolean);
}

/** Every Passage the answer needs is among those the search Tool handed the model, over the Question's queries. */
export function isToolHit(question: DumpQuestion): boolean {
  const found = new Array<boolean>(question.needed).fill(false);
  for (const query of question.queries) {
    for (const passage of query.tool.passages) {
      passage.hits.forEach((hit, needed) => {
        if (hit) found[needed] = true;
      });
    }
  }
  return question.needed > 0 && found.every(Boolean);
}

export interface Tally {
  hits: number;
  total: number;
}

const tallyOf = (questions: readonly DumpQuestion[], hit: (question: DumpQuestion) => boolean) => ({
  hits: questions.filter(hit).length,
  total: questions.length,
});

export interface Scored {
  name: string;
  /** Hits by top k, over all Questions of the dump, and by the tag's values. */
  byTop: { topK: number; all: Tally; groups: Record<string, Tally> }[];
}

export interface DumpScore {
  set: string;
  groupBy: string;
  orders: Scored[];
  tool: { all: Tally; groups: Record<string, Tally> };
}

function groupsOf(
  questions: readonly DumpQuestion[],
  tag: string,
  hit: (question: DumpQuestion) => boolean,
): Record<string, Tally> {
  const groups = new Map<string, DumpQuestion[]>();
  for (const question of questions) {
    const value = question.tags[tag] ?? "(none)";
    groups.set(value, [...(groups.get(value) ?? []), question]);
  }
  return Object.fromEntries([...groups].map(([value, group]) => [value, tallyOf(group, hit)]));
}

export function scoreDump(
  dump: Dump,
  orders: readonly Order[],
  topKs: readonly number[],
  options: ScoreOptions,
  groupBy = dump.groupBy,
): DumpScore {
  return {
    set: dump.set,
    groupBy,
    orders: orders.map((order) => ({
      name: orderName(order, options),
      byTop: topKs.map((topK) => {
        const hit = (question: DumpQuestion) => isHit(question, order, topK, options);
        return {
          topK,
          all: tallyOf(dump.questions, hit),
          groups: groupsOf(dump.questions, groupBy, hit),
        };
      }),
    })),
    tool: {
      all: tallyOf(dump.questions, isToolHit),
      groups: groupsOf(dump.questions, groupBy, isToolHit),
    },
  };
}

const cell = ({ hits, total }: Tally) => `${hits}/${total}`;

/** The scores as text: Questions hit by order and top k, then by the tag's values at top 5 (or the first top k). */
export function formatScore(score: DumpScore, dump: Dump): string {
  const topKs = score.orders[0]?.byTop.map((entry) => entry.topK) ?? [];
  const width = Math.max(...score.orders.map((order) => order.name.length), 20);
  const lines = [
    `${score.set}: ${dump.questions.length} Questions, ${dump.keywordDepth} keyword candidates each, reranked by ${dump.reranker.name} (${dump.rerankedDepth} deep), embeddings off`,
    `${"".padEnd(width)}  ${topKs.map((topK) => `top ${topK}`.padStart(8)).join("")}`,
    ...score.orders.map(
      (order) =>
        `${order.name.padEnd(width)}  ${order.byTop.map((entry) => cell(entry.all).padStart(8)).join("")}`,
    ),
    `${"search Tool (Passages handed to the model)".padEnd(width)}  ${cell(score.tool.all).padStart(8)}`,
  ];
  const values = Object.keys(score.orders[0]?.byTop[0]?.groups ?? {});
  if (values.length > 1) {
    const top = topKs.includes(5) ? 5 : (topKs[0] ?? 5);
    lines.push("", `by ${score.groupBy}, top ${top} (and the search Tool):`);
    const valueWidth = Math.max(...values.map((value) => value.length));
    const names = [...score.orders.map((order) => order.name), "search Tool"];
    const column = Math.max(...names.map((name) => name.length)) + 2;
    lines.push(`${"".padEnd(valueWidth)}  ${names.map((name) => name.padStart(column)).join("")}`);
    for (const value of values) {
      const cells = score.orders.map((order) => {
        const entry = order.byTop.find((each) => each.topK === top) ?? order.byTop[0];
        return cell(entry?.groups[value] ?? { hits: 0, total: 0 });
      });
      cells.push(cell(score.tool.groups[value] ?? { hits: 0, total: 0 }));
      lines.push(
        `${value.padEnd(valueWidth)}  ${cells.map((each) => each.padStart(column)).join("")}`,
      );
    }
  }
  return lines.join("\n");
}

const USAGE = `Usage: node eval/score.ts <dump.json>... [options]

  --top 5,8,10        the top k to count (default 5,8,10)
  --depth 60          how many of keyword search's best a reranker reads (default: the dump's)
  --blend 0.5,0.7     also blend keyword rank and reranker rank, with this weight for the reranker
  --no-neighbours     drop a candidate next to a better-ranked one of its Document before reranking
  --scores file.json  another model's scores for a dump (repeat for each dump; see eval/README.md)
  --by tag            split by this tag instead of the dump's own
  --json              print the scores as JSON`;

function numbers(value: string | undefined, name: string): number[] {
  const parsed = (value ?? "").split(",").map((each) => Number(each.trim()));
  if (parsed.length === 0 || parsed.some((each) => !Number.isFinite(each))) {
    throw new Error(`${name} takes numbers separated by commas.\n${USAGE}`);
  }
  return parsed;
}

export function main(argv: readonly string[], print: (text: string) => void = console.log): void {
  const dumps: string[] = [];
  const scoreFiles: string[] = [];
  let topKs = [5, 8, 10];
  let depth: number | null = null;
  let blends: number[] = [];
  let dropNeighbours = false;
  let by: string | undefined;
  let json = false;
  for (let at = 0; at < argv.length; at++) {
    const arg = argv[at] as string;
    const next = () => argv[++at];
    if (arg === "--top") topKs = numbers(next(), arg);
    else if (arg === "--depth") depth = numbers(next(), arg)[0] ?? null;
    else if (arg === "--blend") blends = numbers(next(), arg);
    else if (arg === "--no-neighbours") dropNeighbours = true;
    else if (arg === "--scores") scoreFiles.push(next() ?? "");
    else if (arg === "--by") by = next();
    else if (arg === "--json") json = true;
    else if (arg.startsWith("--")) throw new Error(`Unknown option ${arg}.\n${USAGE}`);
    else dumps.push(arg);
  }
  if (dumps.length === 0) throw new Error(USAGE);
  const models = scoreFiles.map((file) => parseScores(readFileSync(file, "utf8"), file));
  const results = dumps.map((file) => {
    const dump = parseDump(readFileSync(file, "utf8"), file);
    const options: ScoreOptions = { depth: depth ?? dump.rerankedDepth, dropNeighbours };
    const orders: Order[] = [
      { kind: "keyword" },
      { kind: "rerank" },
      ...blends.map((weight): Order => ({ kind: "blend", weight })),
    ];
    for (const scores of models.filter((each) => each.set === dump.set)) {
      orders.push({ kind: "model", scores });
      for (const weight of blends) orders.push({ kind: "model-blend", scores, weight });
    }
    return { dump, score: scoreDump(dump, orders, topKs, options, by) };
  });
  for (const file of models) {
    if (!results.some(({ dump }) => dump.set === file.set)) {
      throw new Error(
        `${file.model}: its scores are for the "${file.set}" set, not among the dumps.`,
      );
    }
  }
  if (json)
    print(
      JSON.stringify(
        results.map(({ score }) => score),
        null,
        2,
      ),
    );
  else print(results.map(({ dump, score }) => formatScore(score, dump)).join("\n\n"));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    main(process.argv.slice(2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  }
}
