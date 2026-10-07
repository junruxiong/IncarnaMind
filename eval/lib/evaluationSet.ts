/**
 * The evaluation set (eval/retrieval/questions.json): the Documents to add,
 * and the Questions with the Passage each one should find (ADR-0009).
 */
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

export type EvalLanguage = "en" | "zh";

export interface ExpectedPassage {
  /** A key of `EvaluationSet.documents`. */
  document: string;
  /** The first and last page the Passage must cover, from 1 (pdf.js's page index, not the printed label). */
  pages: [number, number];
  /** Text the Passage must contain, after both are normalised. */
  quote: string;
}

export interface EvalQuestion {
  id: string;
  /** The language the Question is asked in. */
  language: EvalLanguage;
  /** Asked in one language about a Document in the other: reported apart, never gating. */
  crossLingual: boolean;
  question: string;
  expected: ExpectedPassage;
}

export interface EvalDocument {
  key: string;
  /** Absolute. */
  path: string;
}

export interface EvaluationSet {
  /** The file it was read from, relative to the repository root. */
  source: string;
  hitRule: string;
  documents: EvalDocument[];
  questions: EvalQuestion[];
}

const EVALUATION_SET = "eval/retrieval/questions.json";

const LANGUAGES: readonly EvalLanguage[] = ["en", "zh"];

interface RawSet {
  hitRule?: unknown;
  documents?: unknown;
  questions?: unknown;
}

function fail(message: string): never {
  throw new Error(`${EVALUATION_SET}: ${message}`);
}

function readQuestion(raw: unknown, documents: ReadonlySet<string>): EvalQuestion {
  const question = raw as Partial<EvalQuestion> & { crossLingual?: unknown };
  const { id, language, expected } = question;
  if (typeof id !== "string" || !id) fail("every question needs an id.");
  if (!LANGUAGES.includes(language as EvalLanguage)) fail(`${id}: unknown language.`);
  if (typeof question.question !== "string" || !question.question.trim()) {
    fail(`${id}: no question text.`);
  }
  if (!expected || !documents.has(expected.document)) fail(`${id}: unknown expected document.`);
  const [from, to] = Array.isArray(expected.pages) ? expected.pages : [];
  if (!Number.isInteger(from) || !Number.isInteger(to) || (from as number) > (to as number)) {
    fail(`${id}: expected.pages must be [first, last].`);
  }
  if (typeof expected.quote !== "string" || !expected.quote.trim()) fail(`${id}: no quote.`);
  return {
    id,
    language: language as EvalLanguage,
    crossLingual: question.crossLingual === true,
    question: question.question,
    expected: {
      document: expected.document,
      pages: [from, to] as [number, number],
      quote: expected.quote,
    },
  };
}

export function loadEvaluationSet(root: string): EvaluationSet {
  const raw = JSON.parse(readFileSync(join(root, EVALUATION_SET), "utf8")) as RawSet;
  if (typeof raw.documents !== "object" || raw.documents === null) fail("no documents.");
  const documents = Object.entries(raw.documents as Record<string, unknown>).map(([key, path]) => {
    if (typeof path !== "string") fail(`${key}: the path must be text.`);
    const absolute = join(root, path);
    if (!existsSync(absolute)) fail(`${key}: ${path} doesn't exist.`);
    return { key, path: absolute };
  });
  if (!Array.isArray(raw.questions)) fail("no questions.");
  const keys = new Set(documents.map((document) => document.key));
  const questions = raw.questions.map((question) => readQuestion(question, keys));
  const ids = new Set(questions.map((question) => question.id));
  if (ids.size !== questions.length) fail("question ids must be unique.");
  return {
    source: EVALUATION_SET,
    hitRule: typeof raw.hitRule === "string" ? raw.hitRule : "",
    documents,
    questions,
  };
}
