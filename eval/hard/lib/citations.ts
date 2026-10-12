/**
 * Citation quality on the hard tier, with a chat model: each Question asked
 * once, in a Mind of its own, through the gating set's Citation part
 * (eval/lib/citations), and its Answers' Citations scored per difficulty.
 * Besides the gating set's figures: how many Answers cite a Document the
 * Question needs (for a near-duplicate one, the right version, and how many
 * cite another version), and, for the unanswerable Questions, how many are
 * answered without a Citation, and how many say the Documents don't answer
 * them (./declines), with or without Citations. For the false-premise and
 * disagreeing Questions, how many Answers follow the Answer rules of #548
 * (./answerRules): the premise flagged first, the newer Document leading.
 */

import {
  type AnswerRecord,
  type CitationRun,
  type GroupSummary,
  runCitations,
  summariseGroup,
} from "../../lib/citations";
import type { ChatSettings, EvalConfig } from "../../lib/config";
import { declines } from "../../lib/declines";
import type { EvalLanguage, EvalQuestion } from "../../lib/evaluationSet";
import type { Library } from "../../lib/library";
import type { Log } from "../../lib/log";
import { flagsPremise, newerLeads, opening } from "./answerRules";
import { type Manifest, siblingsOf } from "./manifest";
import { DIFFICULTIES, type Difficulty, type HardQuestion } from "./questions";

export interface DifficultyCitations extends GroupSummary {
  /** Answers with a "found" Citation of a Document the Question needs. */
  citingExpected: number;
  /** Near-duplicate Questions: Answers citing another version of the Document. */
  citingOtherVersion: number;
  /** Unanswerable Questions: Answers that finished with no Citation. */
  withoutCitation: number;
  /** Unanswerable Questions: Answers the phrase rule found saying the Documents don't answer it. */
  declines: number;
  /** The same per language, of the Answers to the unanswerable Questions. */
  declinesByLanguage: Record<EvalLanguage, { answers: number; declines: number }>;
  /** False-premise Questions: Answers whose opening says the Documents don't support the premise. */
  premiseFlagged: number;
  /** Disagreeing Questions: Answers giving what the newer Document says before what the older one says. */
  newerLeads: number;
  /** Disagreeing Questions: Answers that give what the older Document says too. */
  namesOlder: number;
}

/** What the Answer rules (./answerRules) found in one Answer to a false-premise or disagreeing Question. */
export interface AnswerRuleResult {
  questionId: string;
  difficulty: "false-premise" | "disagreeing";
  /**
   * False premise: flagged in the Answer's opening. Disagreeing: what the
   * newer Document says comes first, and the Answer cites that Document.
   */
  followed: boolean;
  /** Disagreeing: the Answer gives what the older Document says too. */
  namesOlder: boolean;
  /** The Answer's first paragraph, for a reviewer. */
  opening: string;
}

export interface HardCitations {
  run: CitationRun;
  byDifficulty: Partial<Record<Difficulty, DifficultyCitations>>;
  /** Each Answer to a false-premise or disagreeing Question, as the Answer rules read it. */
  rules: AnswerRuleResult[];
}

/**
 * What the Answer rules find in an Answer, for a false-premise or disagreeing
 * Question; null for any other. `documentName`: a Document's name, by its key.
 */
export function answerRuleResult(
  answer: Pick<AnswerRecord, "questionId" | "text" | "citations">,
  question: HardQuestion,
  documentName: (key: string) => string | undefined,
): AnswerRuleResult | null {
  const base = { questionId: answer.questionId, opening: opening(answer.text) };
  if (question.difficulty === "false-premise") {
    return {
      ...base,
      difficulty: "false-premise",
      followed: flagsPremise(answer.text),
      namesOlder: false,
    };
  }
  if (question.difficulty === "disagreeing" && question.newer && question.older) {
    const { leads, namesOlder } = newerLeads(answer.text, question.newer, question.older);
    const newer = documentName(question.newer.document);
    const citesNewer = answer.citations.some((citation) => citation.documentName === newer);
    return { ...base, difficulty: "disagreeing", followed: leads && citesNewer, namesOlder };
  }
  return null;
}

/** As the gating set's Citation part takes Questions: it reads only the id, language and wording. */
export const asEvalQuestion = (question: HardQuestion): EvalQuestion => ({
  id: question.id,
  language: question.language,
  crossLingual: question.difficulty === "cross-lingual",
  question: question.question,
  expected: question.expected[0] ?? { document: "", pages: [1, 1], quote: "" },
  expectedAll: question.expected,
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
  let premiseFlagged = 0;
  let leading = 0;
  let namesOlder = 0;
  const declinesByLanguage: DifficultyCitations["declinesByLanguage"] = {
    en: { answers: 0, declines: 0 },
    zh: { answers: 0, declines: 0 },
  };
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
    if (question.difficulty === "unanswerable") {
      declinesByLanguage[answer.language].answers++;
      if (answer.declines) declinesByLanguage[answer.language].declines++;
    }
    const rule = answerRuleResult(answer, question, documentName);
    if (rule?.difficulty === "false-premise" && rule.followed) premiseFlagged++;
    if (rule?.difficulty === "disagreeing") {
      if (rule.followed) leading++;
      if (rule.namesOlder) namesOlder++;
    }
  }
  return {
    ...summariseGroup(answers),
    citingExpected,
    citingOtherVersion,
    withoutCitation,
    declines: declinesByLanguage.en.declines + declinesByLanguage.zh.declines,
    declinesByLanguage,
    premiseFlagged,
    newerLeads: leading,
    namesOlder,
  };
}

/** Asks every Question once and scores the Answers per difficulty. Never gating. */
export async function runHardCitations(
  library: Library,
  questions: readonly HardQuestion[],
  chat: ChatSettings,
  config: Pick<EvalConfig, "answerTimeoutMs"> &
    Partial<Pick<EvalConfig, "rerankDepth" | "maxPassages">>,
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
  run.answers = run.answers.map((answer) =>
    byId.get(answer.questionId)?.difficulty === "unanswerable"
      ? { ...answer, declines: declines(answer.text) }
      : answer,
  );
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
  const rules = run.answers.flatMap((answer) => {
    const question = byId.get(answer.questionId);
    const rule = question ? answerRuleResult(answer, question, documentName) : null;
    return rule ? [rule] : [];
  });
  // The hard tier never gates: the gating set's targets don't apply.
  return { run: { ...run, gating: false, failures: [] }, byDifficulty, rules };
}
