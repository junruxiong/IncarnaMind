/**
 * The hard tier's Questions (eval/hard/questions.json): each tagged with one
 * difficulty, a domain and a language, with the Passages its answer needs.
 * Several come from public benchmarks (CUAD, Qasper, FinanceBench), their
 * evidence mapped to the pages the app stores; the rest were written for the
 * library. Read and checked here; checked against the stored text by
 * `textProblems`, at the start of each run.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { quoteInUnits } from "../../../src/shared/locations";
import { findQuote } from "../../../src/shared/quoteMatch";
import { sameRun, type UnitText } from "../../../src/shared/units";
import { DOMAINS, type Domain, type HardLanguage, LANGUAGES, type Manifest } from "./manifest";

/**
 * - easy: the answer is in the Documents' own words;
 * - paraphrase: asked in words unlike the passage's;
 * - table-number: a number read from a table;
 * - multi-page: needs passages on two or more pages of one Document;
 * - cross-document: needs passages from two or more Documents;
 * - unanswerable: the library doesn't say: an Answer should cite nothing;
 * - cross-lingual: asked in the other language from its Document's;
 * - near-duplicate: two versions of a Document differ, and the right one must be found.
 */
export const DIFFICULTIES = [
  "easy",
  "paraphrase",
  "table-number",
  "multi-page",
  "cross-document",
  "unanswerable",
  "cross-lingual",
  "near-duplicate",
] as const;
export type Difficulty = (typeof DIFFICULTIES)[number];

export const BENCHMARKS = ["CUAD", "Qasper", "FinanceBench"] as const;

export interface HardExpected {
  /** A key of the manifest's Documents. */
  document: string;
  /** The first and last Unit, from 1: one Unit, or two consecutive ones, as a Citation names them. */
  pages: [number, number];
  /** Text the Passage must contain, matched as the Citation check matches quotes. */
  quote: string;
}

export interface HardQuestion {
  id: string;
  language: HardLanguage;
  domain: Domain;
  difficulty: Difficulty;
  question: string;
  /** Every Passage the answer needs; none for an unanswerable Question. */
  expected: HardExpected[];
  /** Cross-lingual Questions only: the Question translated by hand into its Documents' language. */
  translatedQuery?: string;
  /** A short answer, for people reading the report. */
  answer?: string;
  /** Where the Question comes from, when a benchmark: its name and its own id. */
  benchmark?: { name: (typeof BENCHMARKS)[number]; id: string };
}

export interface HardQuestionSet {
  source: string;
  questions: HardQuestion[];
}

/** The Questions, relative to the repository root. */
export const QUESTIONS = "eval/hard/questions.json";

const isPages = (pages: unknown): pages is [number, number] =>
  Array.isArray(pages) &&
  pages.length === 2 &&
  pages.every((page) => Number.isInteger(page) && page >= 1) &&
  (pages[0] as number) <= (pages[1] as number) &&
  (pages[1] as number) - (pages[0] as number) <= 1;

const overlap = (a: [number, number], b: [number, number]) => a[0] <= b[1] && b[0] <= a[1];

/** Every problem with a Question, against the manifest; empty when it is sound. */
export function questionProblems(question: HardQuestion, manifest: Manifest): string[] {
  const problems: string[] = [];
  const id = question.id || "(no id)";
  const problem = (message: string) => problems.push(`${id}: ${message}`);
  const documents = new Map(manifest.documents.map((document) => [document.key, document]));
  if (!question.id || !/^[a-z0-9-]+$/.test(question.id)) problem("the id must be lowercase kebab.");
  if (!LANGUAGES.includes(question.language)) problem("unknown language.");
  if (!DOMAINS.includes(question.domain)) problem(`unknown domain "${question.domain}".`);
  if (!DIFFICULTIES.includes(question.difficulty)) {
    problem(`unknown difficulty "${question.difficulty}".`);
  }
  if (typeof question.question !== "string" || !question.question.trim()) problem("no question.");
  if (!Array.isArray(question.expected)) {
    problem("expected must be a list.");
    return problems;
  }
  for (const expected of question.expected) {
    if (!documents.has(expected.document)) problem(`unknown Document "${expected.document}".`);
    if (!isPages(expected.pages)) {
      problem("expected pages must be [first, last], one Unit or two consecutive ones.");
    }
    if (typeof expected.quote !== "string" || !expected.quote.trim()) problem("no quote.");
  }
  const keys = [...new Set(question.expected.map((expected) => expected.document))];
  const languages = keys.map((key) => documents.get(key)?.language);
  const domains = keys.map((key) => documents.get(key)?.domain);
  if (keys.length > 0 && !domains.includes(question.domain)) {
    problem("the domain isn't any expected Document's.");
  }
  if (question.translatedQuery !== undefined && question.difficulty !== "cross-lingual") {
    problem("only a cross-lingual Question has a translatedQuery.");
  }
  if (question.benchmark && !BENCHMARKS.includes(question.benchmark.name)) {
    problem("unknown benchmark.");
  }

  switch (question.difficulty) {
    case "unanswerable":
      if (question.expected.length > 0) problem("an unanswerable Question expects no Passage.");
      break;
    case "multi-page": {
      if (question.expected.length < 2 || keys.length !== 1) {
        problem("a multi-page Question expects two or more Passages of one Document.");
      }
      const ranges = question.expected.map((expected) => expected.pages);
      if (
        ranges.some((a, i) =>
          ranges.some((b, j) => i < j && isPages(a) && isPages(b) && overlap(a, b)),
        )
      ) {
        problem("a multi-page Question's Passages must be on different pages.");
      }
      break;
    }
    case "cross-document":
      if (keys.length < 2) problem("a cross-document Question expects two or more Documents.");
      break;
    case "cross-lingual":
      if (question.expected.length === 0) problem("a cross-lingual Question expects a Passage.");
      if (languages.some((language) => language === question.language)) {
        problem("a cross-lingual Question is asked in the other language from its Documents'.");
      }
      if (typeof question.translatedQuery !== "string" || !question.translatedQuery.trim()) {
        problem("a cross-lingual Question needs a translatedQuery.");
      }
      // With a translation in the Question's own language at hand, the answer needs no other language.
      for (const key of keys) {
        const translated = manifest.documents.some(
          (document) =>
            document.language === question.language &&
            (document.translationOf === key || documents.get(key)?.translationOf === document.key),
        );
        if (translated) problem(`${key} has a translation in the Question's language.`);
      }
      break;
    case "near-duplicate": {
      if (question.expected.length === 0) problem("a near-duplicate Question expects a Passage.");
      for (const key of keys) {
        const group = documents.get(key)?.group;
        const versions = manifest.documents.filter(
          (document) =>
            group !== undefined &&
            document.group === group &&
            document.language === question.language,
        );
        if (versions.length < 2) {
          problem(`${key} has no other version in its language for a near-duplicate Question.`);
        }
      }
      break;
    }
    default:
      if (question.expected.length === 0) problem("expects no Passage.");
  }
  if (
    question.difficulty !== "cross-lingual" &&
    languages.some((language) => language !== undefined && language !== question.language)
  ) {
    problem("asked in another language from its Documents': tag it cross-lingual.");
  }
  return problems;
}

/** Reads the Questions and fails, saying why, if any isn't sound. */
export function loadHardQuestions(
  root: string,
  manifest: Manifest,
  source = QUESTIONS,
): HardQuestionSet {
  const raw = JSON.parse(readFileSync(join(root, source), "utf8")) as { questions?: unknown };
  if (!Array.isArray(raw.questions)) throw new Error(`${source}: no questions.`);
  const questions = raw.questions as HardQuestion[];
  const problems = questions.flatMap((question) => questionProblems(question, manifest));
  const ids = questions.map((question) => question.id);
  const twice = ids.filter((id, index) => ids.indexOf(id) !== index);
  if (twice.length > 0) problems.push(`ids used twice: ${[...new Set(twice)].join(", ")}`);
  if (problems.length > 0) throw new Error(`${source}:\n- ${problems.join("\n- ")}`);
  return { source, questions };
}

/** What the run reads back for a Document: its stored Units and Passages. */
export interface StoredText {
  units: readonly UnitText[];
  passages: readonly { pageFrom: number | null; pageTo: number | null; text: string }[];
}

/**
 * Why a Question can't be scored on the text the app stored, if it can't:
 * each quote must be on its expected Units, on no other Unit of its Document,
 * and inside a Passage that covers them; a near-duplicate's quote must be in
 * no other version of its Document. Empty when it can be scored.
 */
export function textProblems(
  question: HardQuestion,
  stored: (key: string) => StoredText | undefined,
  manifest: Pick<Manifest, "documents">,
): string[] {
  const problems: string[] = [];
  for (const expected of question.expected) {
    const text = stored(expected.document);
    if (!text) {
      problems.push(`${expected.document} isn't in the library.`);
      continue;
    }
    const [first, last] = expected.pages;
    const inRange = text.units.filter(
      (unit) => unit.page !== null && unit.page >= first && unit.page <= last,
    );
    if (inRange.length !== last - first + 1) {
      problems.push(
        `${expected.document} has no Unit ${first === last ? first : `${first}–${last}`}.`,
      );
      continue;
    }
    if (inRange.length === 2 && !sameRun(inRange[0] as UnitText, inRange[1] as UnitText)) {
      problems.push(`${expected.document}: Units ${first}–${last} can't be cited together.`);
    }
    if (!quoteInUnits(inRange, expected.quote)) {
      problems.push(
        `${expected.document}: the quote isn't on Unit ${first}${last > first ? `–${last}` : ""}.`,
      );
      continue;
    }
    const elsewhere = text.units.filter(
      (unit) =>
        (unit.page === null || unit.page < first || unit.page > last) &&
        quoteInUnits([unit], expected.quote),
    );
    if (elsewhere.length > 0) {
      problems.push(
        `${expected.document}: the quote is also on Unit ${elsewhere.map((unit) => unit.page).join(", ")}.`,
      );
    }
    const covered = text.passages.some(
      (passage) =>
        passage.pageFrom !== null &&
        passage.pageTo !== null &&
        passage.pageFrom <= first &&
        passage.pageTo >= last &&
        findQuote(passage.text, expected.quote) !== null,
    );
    if (!covered)
      problems.push(`${expected.document}: no Passage that covers the quote's Units holds it.`);
    if (question.difficulty === "near-duplicate") {
      const document = manifest.documents.find((each) => each.key === expected.document);
      const versions = manifest.documents.filter(
        (each) => document?.group && each.group === document.group && each.key !== document.key,
      );
      for (const version of versions) {
        const other = stored(version.key);
        if (other?.units.some((unit) => quoteInUnits([unit], expected.quote))) {
          problems.push(`${expected.document}: the quote is in ${version.key} too.`);
        }
      }
    }
  }
  return problems;
}
