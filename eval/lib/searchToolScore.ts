/**
 * The search Tool's score (the gate, ./retrieval): for each Question, whether
 * the Passages the Tool hands the model hold the expected one, as a one-search
 * Answer is handed them (keyword search's top 20 reranked, 12 Passages), with
 * embeddings off. It runs the Tool's own code over the run's data folder, as
 * the dumps do (./dumps), so both give the same numbers.
 */
import type { Core } from "../../src/core";
import type { Database } from "../../src/core/storage";
import { DUMP_KEYWORD_DEPTH, keywordCandidates, toolPassages } from "./dumps";
import type { EvalQuestion } from "./evaluationSet";
import type { ScoringReranker } from "./rerank";
import { KEYWORD_RERANK_DEPTH, type QuestionResult, scoreHanded, TOOL_MODE } from "./retrieval";

/** Adds the Tool's score to `results`: each Question searched, and its translated query too. */
export async function runSearchTool(
  { core, db, reranker }: { core: Core; db: Database; reranker: ScoringReranker },
  questions: readonly EvalQuestion[],
  documentIds: ReadonlyMap<string, string>,
  results: readonly QuestionResult[],
): Promise<void> {
  for (const question of questions) {
    const result = results.find((each) => each.id === question.id);
    const expectedId = documentIds.get(question.expected.document);
    if (!result || !expectedId)
      throw new Error(`${question.id}: nothing to add the search Tool to.`);
    const handed = async (query: string) => {
      const { passages } = await keywordCandidates({ core, db }, query, DUMP_KEYWORD_DEPTH);
      const reranked = await reranker.rerankScored(query, passages.slice(0, KEYWORD_RERANK_DEPTH));
      const scores = new Map(reranked.map((passage) => [passage.passageId, passage.score]));
      return scoreHanded(
        await toolPassages(db, query, passages, scores),
        question.expected,
        expectedId,
      );
    };
    result.modes[TOOL_MODE] = await handed(question.question);
    if (question.translatedQuery && result.translated) {
      result.translated[TOOL_MODE] = await handed(question.translatedQuery);
    }
  }
}
