/**
 * Citation quality on the hard tier, with a chat model: each Question asked
 * once, in a Mind of its own, through the gating set's Citation part
 * (eval/lib/citations), and its Answers' Citations scored per difficulty.
 * Besides the gating set's figures: how many Answers cite a Document the
 * Question needs (for a near-duplicate one, the right version, and how many
 * cite another version), and, for the unanswerable Questions, how many are
 * answered without a Citation, as they should be.
 */

import {
  type AnswerRecord,
  type CitationRun,
  type GroupSummary,
  runCitations,
  summariseGroup,
} from "../../lib/citations";
import type { ChatSettings, EvalConfig } from "../../lib/config";
import type { EvalQuestion } from "../../lib/evaluationSet";
import type { Library } from "../../lib/library";
import type { Log } from "../../lib/log";
import { type Manifest, siblingsOf } from "./manifest";
import { DIFFICULTIES, type Difficulty, type HardQuestion } from "./questions";

export interface DifficultyCitations extends GroupSummary {
  /** Answers with a "found" Citation of a Document the Question needs. */
  citingExpected: number;
  /** Near-duplicate Questions: Answers citing another version of the Document. */
  citingOtherVersion: number;
  /** Unanswerable Questions: Answers that finished with no Citation. */
  withoutCitation: number;
}

export interface HardCitations {
  run: CitationRun;
  byDifficulty: Partial<Record<Difficulty, DifficultyCitations>>;
}

/** As the gating set's Citation part takes Questions: it reads only the id, language and wording. */
export const asEvalQuestion = (question: HardQuestion): EvalQuestion => ({
  id: question.id,
  language: question.language,
  crossLingual: question.difficulty === "cross-lingual",
  question: question.question,
  expected: question.expected[0] ?? { document: "", pages: [1, 1], quote: "" },
});

/** The figures for a group of Answers to the hard tier's Questions. */
export function summariseDifficulty(
  answers: readonly AnswerRecord[],
  questions: ReadonlyMap<string, HardQuestion>,
  documentName: (key: string) => string | undefined,
  manifest: Pick<Manifest, "documents">,
): DifficultyCitations {
  const cites = (answer: AnswerRecord, keys: readonly string[]) => {
    const names = new Set(keys.map(documentName).filter((name) => name !== undefined));
    return answer.citations.some(
      (citation) => citation.outcome === "found" && names.has(citation.documentName),
    );
  };
  let citingExpected = 0;
  let citingOtherVersion = 0;
  let withoutCitation = 0;
  for (const answer of answers) {
    const question = questions.get(answer.questionId);
    if (!question) continue;
    const keys = [...new Set(question.expected.map((expected) => expected.document))];
    if (cites(answer, keys)) citingExpected++;
    if (question.difficulty === "near-duplicate") {
      const others = keys.flatMap((key) => siblingsOf(manifest, key).map((each) => each.key));
      if (
        cites(
          answer,
          others.filter((key) => !keys.includes(key)),
        )
      )
        citingOtherVersion++;
    }
    if (answer.status === "done" && answer.citations.length === 0) withoutCitation++;
  }
  return { ...summariseGroup(answers), citingExpected, citingOtherVersion, withoutCitation };
}

/** Asks every Question once and scores the Answers per difficulty. Never gating. */
export async function runHardCitations(
  library: Library,
  questions: readonly HardQuestion[],
  chat: ChatSettings,
  config: Pick<EvalConfig, "answerTimeoutMs">,
  manifest: Pick<Manifest, "documents">,
  log: Log,
): Promise<HardCitations> {
  const run = await runCitations(
    library,
    questions.map(asEvalQuestion),
    chat,
    { ...config, minCitations: 1, maxRounds: 1 },
    log,
  );
  const byId = new Map(questions.map((question) => [question.id, question]));
  const documentName = (key: string) => library.documents.get(key)?.name;
  const byDifficulty: HardCitations["byDifficulty"] = {};
  for (const difficulty of DIFFICULTIES) {
    const answers = run.answers.filter(
      (answer) => byId.get(answer.questionId)?.difficulty === difficulty,
    );
    if (answers.length > 0) {
      byDifficulty[difficulty] = summariseDifficulty(answers, byId, documentName, manifest);
    }
  }
  // The hard tier never gates: the gating set's targets don't apply.
  return { run: { ...run, gating: false, failures: [] }, byDifficulty };
}
