/**
 * Whether an Answer's searches handed the model the Passages its Question
 * needs, beside whether the Answer cites them (#488): a model that was never
 * handed a Passage can't cite it, and one that was may still not.
 */
import type { PassageSearchResult } from "../../src/core";
import type { CitationRecord } from "./citations";
import type { EvalQuestion, ExpectedPassage } from "./evaluationSet";
import { checkPassage, isHit } from "./retrieval";

/** Of the Passages a Question needs: how many there are, were handed to the model, and are cited. */
export interface ExpectedCounts {
  needed: number;
  handed: number;
  cited: number;
}

/** What a library knows of the evaluation set's Documents. */
export interface DocumentLookup {
  /** The Document's id and name for a key of the set. */
  (key: string): { id: string; name: string } | undefined;
}

/** Every Passage the Question needs: none for an unanswerable one. */
export function neededPassages(question: EvalQuestion): readonly ExpectedPassage[] {
  if (question.expectedAll) return question.expectedAll;
  return question.expected.document ? [question.expected] : [];
}

const overlaps = (a: readonly [number, number], b: readonly [number, number]) =>
  a[0] <= b[1] && b[0] <= a[1];

/** Null for a Question that needs no Passage. */
export function expectedCounts(
  question: EvalQuestion,
  lookup: DocumentLookup,
  handed: readonly PassageSearchResult[],
  citations: readonly CitationRecord[],
): ExpectedCounts | null {
  const needed = neededPassages(question);
  if (needed.length === 0) return null;
  let handedCount = 0;
  let citedCount = 0;
  for (const expected of needed) {
    const document = lookup(expected.document);
    if (!document) continue;
    if (handed.some((passage) => isHit(checkPassage(passage, expected, document.id)))) {
      handedCount++;
    }
    const cited = citations.some(
      (citation) =>
        citation.outcome === "found" &&
        citation.documentName === document.name &&
        citation.pageFrom !== null &&
        citation.pageTo !== null &&
        overlaps([citation.pageFrom, citation.pageTo], expected.pages),
    );
    if (cited) citedCount++;
  }
  return { needed: needed.length, handed: handedCount, cited: citedCount };
}

/** Answers whose searches handed the model every Passage the Question needs, and whose Citations cover them all. */
export interface HandedSummary {
  /** Answers to Questions that need a Passage. */
  answers: number;
  handed: number;
  cited: number;
}

export function summariseHanded(
  answers: readonly { expected?: ExpectedCounts | null }[],
): HandedSummary | null {
  const counts = answers.flatMap((answer) => (answer.expected ? [answer.expected] : []));
  if (counts.length === 0) return null;
  return {
    answers: counts.length,
    handed: counts.filter((each) => each.handed === each.needed).length,
    cited: counts.filter((each) => each.cited === each.needed).length,
  };
}
