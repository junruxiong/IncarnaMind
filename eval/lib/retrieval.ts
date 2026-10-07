/**
 * Retrieval: each Question searched through the core's `searchPassages`, in
 * each search mode, scored with ADR-0009's hit rule. A Question is a hit when
 * one of the top 5 Passages (a) belongs to the expected Document, (b) covers
 * the expected pages, and (c) contains the expected quote, matched as the
 * Citation check matches quotes (both normalised by the shared normaliser).
 */
import type { Core, PassageSearchResult, SearchMode } from "../../src/core";
import { findQuote } from "../../src/shared/quoteMatch";
import type { EvalLanguage, EvalQuestion, ExpectedPassage } from "./evaluationSet";

export const TOP_K = 5;

/** Ranks are looked for this deep, to show near misses; only the top 5 count as hits. */
export const RANK_DEPTH = 20;

export const SEARCH_MODES: readonly SearchMode[] = ["hybrid", "keyword", "vector"];

/** The gating mode: what the search Tool runs. */
export const GATING_MODE: SearchMode = "hybrid";

/** The v1 design's bar: 16 of 20 overall, and 8 of 10 in each language. */
const RETRIEVAL_TARGET = { share: 0.8 } as const;

/** One retrieved Passage, against the expected one. */
export interface RetrievedPassage {
  documentName: string;
  pageFrom: number | null;
  pageTo: number | null;
  rightDocument: boolean;
  coversPages: boolean;
  hasQuote: boolean;
}

export interface ModeResult {
  /** A Passage in the top 5 is a hit. */
  hit: boolean;
  /** The rank of the first Passage that is a hit, from 1, within the top 20; null if none is. */
  rank: number | null;
  /** The top 5. */
  top: RetrievedPassage[];
}

export interface QuestionResult {
  id: string;
  language: EvalLanguage;
  crossLingual: boolean;
  question: string;
  modes: Partial<Record<SearchMode, ModeResult>>;
}

export interface Tally {
  hits: number;
  total: number;
}

/** A mode's hits: per language and overall over the gating Questions, and over the cross-lingual ones. */
export interface ModeSummary {
  en: Tally;
  zh: Tally;
  core: Tally;
  crossLingual: Tally;
}

export interface RetrievalRun {
  /** "built-in" or the cloud model, e.g. "openai/text-embedding-3-small". */
  embedding: string;
  /** Only the built-in model gates. */
  gating: boolean;
  passageCount: number;
  processingSeconds: number;
  questions: QuestionResult[];
  summary: Partial<Record<SearchMode, ModeSummary>>;
}

export function checkPassage(
  passage: PassageSearchResult,
  expected: ExpectedPassage,
  expectedDocumentId: string,
): RetrievedPassage {
  const [first, last] = expected.pages;
  return {
    documentName: passage.documentName,
    pageFrom: passage.pageFrom,
    pageTo: passage.pageTo,
    rightDocument: passage.documentId === expectedDocumentId,
    coversPages:
      passage.pageFrom !== null &&
      passage.pageTo !== null &&
      passage.pageFrom <= first &&
      passage.pageTo >= last,
    hasQuote: findQuote(passage.text, expected.quote) !== null,
  };
}

export const isHit = (passage: RetrievedPassage) =>
  passage.rightDocument && passage.coversPages && passage.hasQuote;

/** Searches each Question in each mode and scores the top 5 (and finds the first hit in the top 20). */
export async function runRetrieval(
  core: Core,
  questions: readonly EvalQuestion[],
  documentIds: ReadonlyMap<string, string>,
  modes: readonly SearchMode[] = SEARCH_MODES,
): Promise<QuestionResult[]> {
  const results: QuestionResult[] = [];
  for (const question of questions) {
    const expectedId = documentIds.get(question.expected.document);
    if (!expectedId) throw new Error(`${question.id}: its Document wasn't added.`);
    const result: QuestionResult = {
      id: question.id,
      language: question.language,
      crossLingual: question.crossLingual,
      question: question.question,
      modes: {},
    };
    for (const mode of modes) {
      // A deeper search returns the same top 5: each list's top 50 is fused before the limit.
      const found = await core.searchPassages(question.question, { mode, limit: RANK_DEPTH });
      const checked = found.map((passage) => checkPassage(passage, question.expected, expectedId));
      const index = checked.findIndex(isHit);
      result.modes[mode] = {
        hit: index >= 0 && index < TOP_K,
        rank: index >= 0 ? index + 1 : null,
        top: checked.slice(0, TOP_K),
      };
    }
    results.push(result);
  }
  return results;
}

const tally = (results: readonly QuestionResult[], mode: SearchMode): Tally => ({
  hits: results.filter((result) => result.modes[mode]?.hit).length,
  total: results.length,
});

export function summarise(
  results: readonly QuestionResult[],
  modes: readonly SearchMode[] = SEARCH_MODES,
): Partial<Record<SearchMode, ModeSummary>> {
  const core = results.filter((result) => !result.crossLingual);
  const summary: Partial<Record<SearchMode, ModeSummary>> = {};
  for (const mode of modes) {
    summary[mode] = {
      en: tally(
        core.filter((result) => result.language === "en"),
        mode,
      ),
      zh: tally(
        core.filter((result) => result.language === "zh"),
        mode,
      ),
      core: tally(core, mode),
      crossLingual: tally(
        results.filter((result) => result.crossLingual),
        mode,
      ),
    };
  }
  return summary;
}

const meets = ({ hits, total }: Tally) =>
  total > 0 && hits >= Math.ceil(total * RETRIEVAL_TARGET.share);

/** Why the gating mode misses the bar; empty when it passes. */
export function retrievalFailures(summary: ModeSummary | undefined): string[] {
  if (!summary) return [`No ${GATING_MODE} results.`];
  const failures: string[] = [];
  const need = ({ total }: Tally) => Math.ceil(total * RETRIEVAL_TARGET.share);
  for (const [label, count] of [
    ["overall", summary.core],
    ["English", summary.en],
    ["Chinese", summary.zh],
  ] as const) {
    if (!meets(count)) {
      failures.push(
        `Retrieval (${GATING_MODE}, built-in model), ${label}: ${count.hits} of ${count.total}, needs ${need(count)}.`,
      );
    }
  }
  return failures;
}
