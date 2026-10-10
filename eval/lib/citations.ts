/**
 * Citation quality: each Question asked through the core with a real chat
 * model, in a Mind of its own, and each Answer's Citations read back from the
 * Mind, as the editor shows them.
 *
 * For every Citation the core's check result is kept ("found" or why not),
 * and each "not found" quote is looked for again under a looser
 * normalisation, to tell the check's misses (the quote is on the cited
 * pages) from wrong pages and from quotes that aren't in the Document.
 */
import { randomUUID } from "node:crypto";
import { getSchema } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { yXmlFragmentToProseMirrorRootNode } from "@tiptap/y-tiptap";
import * as Y from "yjs";
import {
  ANSWER_BLOCK,
  type AnswerFinished,
  BLOCK_ID_ATTRIBUTE,
  CITATION_NODE,
  type CitationAttributes,
  type CitationCheck,
  type CitationCheckReason,
  type CitationSupport,
  type Core,
  createOllamaModels,
  MIND_CONTENT_FIELD,
  type OllamaModels,
  type ProviderError,
  QUESTION_BLOCK,
} from "../../src/core";
import { outputTokensFor } from "../../src/core/providers/ollamaModels";
import { noteExtensions } from "../../src/renderer/src/editor/noteSchema";
import { normaliseText } from "../../src/shared/text";
import type { ChatSettings, EvalConfig } from "./config";
import type { EvalLanguage, EvalQuestion } from "./evaluationSet";
import type { Library } from "./library";
import type { Log } from "./log";

/** The v1 design's targets, per language, over at least `EvalConfig.minCitations` Citations. */
export const CITATION_TARGETS = {
  /** At least this share of Citations show "Quote found". */
  found: 0.9,
  /** At most this share of Citations show "not found" for a quote that is on the cited pages. */
  falseNotFound: 0.05,
  /** At least this share of "found" quotes support their sentence, as a reviewer judges (the sheet). */
  supports: 0.8,
} as const;

/**
 * What became of a Citation:
 * - "found": the check found the quote on the cited pages;
 * - "false-not-found": "not found", yet the quote is on the cited pages under the looser normalisation;
 * - "wrong-page": "not found", and the quote is on other pages of the Document;
 * - "not-in-document": "not found", and the quote isn't in the Document at all (e.g. paraphrased);
 * - "page-range": "not found" because the cited pages break the page-range rule;
 * - "cant-check": the cited pages have no text.
 */
export type CitationOutcome =
  | "found"
  | "false-not-found"
  | "wrong-page"
  | "not-in-document"
  | "page-range"
  | "cant-check";

export const CITATION_OUTCOMES: readonly CitationOutcome[] = [
  "found",
  "false-not-found",
  "wrong-page",
  "not-in-document",
  "page-range",
  "cant-check",
];

export interface CitationRecord {
  /** The sentence of the Answer the Citation is anchored in, without its Citations. */
  sentence: string;
  quote: string;
  documentName: string;
  pageFrom: number | null;
  pageTo: number | null;
  check: CitationCheck;
  checkReason: CitationCheckReason | null;
  outcome: CitationOutcome;
}

export interface AnswerRecord {
  questionId: string;
  language: EvalLanguage;
  crossLingual: boolean;
  round: number;
  question: string;
  status: "done" | "stopped" | "failed" | "timed-out";
  error: ProviderError | null;
  citationSupport: CitationSupport | null;
  /** What the model searched for. */
  searches: string[];
  droppedMarkers: number;
  droppedRecords: number;
  /** Each sentence of the Answer (headings and code left out), and whether a Citation is anchored in it. */
  sentences: { text: string; cited: boolean }[];
  citations: CitationRecord[];
  seconds: number;
}

/** "en" and "zh" are the gating Questions in each language; "crossLingual" the rest. */
export type CitationGroup = "en" | "zh" | "crossLingual";

export interface GroupSummary {
  answers: number;
  /** Answers that failed or timed out. */
  failedAnswers: number;
  /** Answers with at least one Citation. */
  citedAnswers: number;
  /** Share of Answers with at least one Citation; null without Answers. */
  citedAnswerShare: number | null;
  /** The median time an Answer took, in seconds; null without Answers. */
  medianSeconds: number | null;
  citations: number;
  outcomes: Record<CitationOutcome, number>;
  /** Share of Citations showing "Quote found"; null without Citations. */
  foundShare: number | null;
  /** Share of Citations that are "false-not-found"; null without Citations. */
  falseNotFoundShare: number | null;
  sentences: number;
  citedSentences: number;
  /** Share of sentences with a Citation; null without sentences. */
  coverage: number | null;
  droppedMarkers: number;
  droppedRecords: number;
  citationSupport: Partial<Record<CitationSupport | "unknown", number>>;
}

export interface CitationRun {
  /** e.g. "anthropic/claude-sonnet-4-6". */
  model: string;
  /** Where Questions went, e.g. "Anthropic"; null for a model on this computer. */
  service: string | null;
  /**
   * A cloud model gates; a model on this computer (Ollama) is only reported,
   * and so is a run that asked only some Questions (`subset`).
   */
  gating: boolean;
  /**
   * The ids of the Questions asked, when the run asked only some
   * (INCARNAMIND_EVAL_QUESTIONS): a short check, never gating. Null: all.
   */
  subset: string[] | null;
  /** What the run set instead of the app's choice ("ollama" only), e.g. "num_ctx 8192"; empty for none. */
  overrides: string[];
  minCitations: number;
  rounds: number;
  answers: AnswerRecord[];
  summary: Record<CitationGroup, GroupSummary>;
  /** Why the gating targets are missed; empty when met (or when not gating). */
  failures: string[];
}

/** The schema of the renderer's editor, to read Answers the way it shows them. */
const mindSchema = getSchema(noteExtensions());

/** Stands for a Citation in an Answer's text while it is split into sentences (a private-use character). */
const MARK = "\uE000";
const HAS_WORD = /[\p{L}\p{N}]/u;
const LEADING_MARKS = new RegExp(`^[\\s${MARK}]*`, "u");

const count = (text: string, character: string) => text.split(character).length - 1;

/**
 * Splits one text block of an Answer into sentences, each with the Citations
 * anchored in it (`text` holds a MARK for each, in order). A Citation placed
 * just after a sentence's full stop counts for that sentence.
 */
export function sentencesOf<T>(
  text: string,
  citations: readonly T[],
  language: string,
): { text: string; citations: T[] }[] {
  const sentences: { text: string; citations: T[] }[] = [];
  let next = 0;
  let carried: T[] = [];
  for (const { segment } of new Intl.Segmenter(language, { granularity: "sentence" }).segment(
    text,
  )) {
    const marks = citations.slice(next, next + count(segment, MARK));
    next += marks.length;
    const words = segment.replaceAll(MARK, "").replace(/\s+/g, " ").trim();
    const previous = sentences.at(-1);
    if (!HAS_WORD.test(words)) {
      if (previous) previous.citations.push(...marks);
      else carried.push(...marks);
      continue;
    }
    const leading = count(segment.match(LEADING_MARKS)?.[0] ?? "", MARK);
    if (previous && leading > 0) previous.citations.push(...marks.splice(0, leading));
    sentences.push({ text: words, citations: [...carried, ...marks] });
    carried = [];
  }
  return sentences;
}

/** The Answer's sentences, with the Citations anchored in each. Headings and code are left out. */
function answerSentences(
  answer: ProseMirrorNode,
  language: string,
): { text: string; citations: CitationAttributes[] }[] {
  const sentences: { text: string; citations: CitationAttributes[] }[] = [];
  answer.descendants((node) => {
    if (!node.isTextblock) return true;
    if (node.type.name === "heading" || node.type.name === "codeBlock") return false;
    let text = "";
    const citations: CitationAttributes[] = [];
    node.forEach((child) => {
      if (child.isText) text += child.text ?? "";
      else if (child.type.name === CITATION_NODE) {
        text += MARK;
        citations.push(child.attrs as CitationAttributes);
      } else text += " ";
    });
    sentences.push(...sentencesOf(text, citations, language));
    return false;
  });
  return sentences;
}

/**
 * The looser normalisation: the shared one, then accents dropped, lower case,
 * and only letters and digits kept, so punctuation, spacing, hyphens and
 * quote marks can't stop a match.
 */
function looseText(text: string): string {
  return normaliseText(text)
    .normalize("NFKD")
    .replace(/\p{M}/gu, "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]/gu, "");
}

type PageText = { page: number | null; text: string };

/** What became of a Citation (see `CitationOutcome`), given its Document's stored pages. */
export function outcomeOf(
  citation: Pick<CitationAttributes, "check" | "checkReason" | "quote" | "pageFrom" | "pageTo">,
  pages: readonly PageText[],
): CitationOutcome {
  if (citation.check === "found") return "found";
  if (citation.check !== "not-found") return "cant-check";
  if (citation.checkReason === "pages-outside-passage" || citation.checkReason === "too-many-pages")
    return "page-range";
  const quote = looseText(citation.quote ?? "");
  if (!quote) return "not-in-document";
  const { pageFrom, pageTo } = citation;
  const cited = pages.filter(
    ({ page }) =>
      pageFrom === null || pageTo === null || (page !== null && page >= pageFrom && page <= pageTo),
  );
  const onPages = (some: readonly PageText[]) =>
    looseText(some.map((page) => page.text).join("\n")).includes(quote);
  if (onPages(cited)) return "false-not-found";
  return onPages(pages) ? "wrong-page" : "not-in-document";
}

/**
 * "ollama" only: the app's lookup of models in Ollama, with the window and
 * the citing mode the run asks for (`ChatSettings.numCtx` and `citing`)
 * instead of the app's choice; the output cap follows the window, as in the
 * app. Null when the run asks for neither.
 */
export function evalOllamaModels(
  chat: ChatSettings | null,
  models: OllamaModels = createOllamaModels(),
): OllamaModels | null {
  if (!chat || (chat.numCtx === null && chat.citing === null)) return null;
  const { numCtx, citing } = chat;
  return {
    async describe(baseUrl, model) {
      const profile = await models.describe(baseUrl, model);
      if (!profile) return null;
      return {
        ...profile,
        ...(citing !== null && { support: citing }),
        ...(numCtx !== null && {
          settings: {
            ...profile.settings,
            numCtx,
            outputTokens: outputTokensFor(numCtx, profile.settings.think),
          },
        }),
      };
    },
    loaded: (baseUrl, model, numCtx) => models.loaded(baseUrl, model, numCtx),
  };
}

/** Writes a Question into the Mind, as the editor would store it, through the public interface. */
async function writeQuestion(core: Core, mindId: string, id: string, text: string): Promise<void> {
  const doc = new Y.Doc();
  Y.applyUpdate(doc, (await core.openMind(mindId)).state);
  const before = Y.encodeStateVector(doc);
  const question = new Y.XmlElement(QUESTION_BLOCK);
  question.setAttribute(BLOCK_ID_ATTRIBUTE, id);
  question.insert(0, [new Y.XmlText(text)]);
  doc.getXmlFragment(MIND_CONTENT_FIELD).push([question]);
  await core.applyMindUpdate(mindId, Y.encodeStateAsUpdate(doc, before));
  doc.destroy();
}

/** The Answer as the editor reads it from the Mind. */
async function readAnswer(
  core: Core,
  mindId: string,
  answerId: string,
): Promise<ProseMirrorNode | null> {
  const doc = new Y.Doc();
  Y.applyUpdate(doc, (await core.openMind(mindId)).state);
  const root = yXmlFragmentToProseMirrorRootNode(
    doc.getXmlFragment(MIND_CONTENT_FIELD),
    mindSchema,
  );
  doc.destroy();
  let found: ProseMirrorNode | null = null;
  root.forEach((block) => {
    if (block.type.name === ANSWER_BLOCK && block.attrs[BLOCK_ID_ATTRIBUTE] === answerId) {
      found = block;
    }
  });
  return found;
}

type Ending = { finished: AnswerFinished } | { failed: ProviderError };

/** Asks one Question in a new Mind and reads its Answer back. */
async function askOne(
  library: Library,
  question: EvalQuestion,
  round: number,
  timeoutMs: number,
): Promise<AnswerRecord> {
  const { core } = library;
  const started = Date.now();
  const mind = await core.createMind({ title: `Evaluation ${question.id}, round ${round}` });
  const questionId = randomUUID();
  await writeQuestion(core, mind.id, questionId, question.question);

  const searches: string[] = [];
  const ended = new Promise<Ending>((resolve) => {
    const stops = [
      core.on("answer.finished", (event) => {
        if (event.mindId !== mind.id) return;
        for (const stop of stops) stop();
        resolve({ finished: event });
      }),
      core.on("answer.failed", (event) => {
        if (event.mindId !== mind.id) return;
        for (const stop of stops) stop();
        resolve({ failed: event.error });
      }),
      core.on("answer.toolCallStarted", (event) => {
        const query = event.call.input.query;
        if (event.mindId === mind.id && typeof query === "string") searches.push(query);
      }),
    ];
  });
  const asked = await core.askQuestion({ mindId: mind.id, questionId });
  if (!asked.asked) {
    throw new Error(`${question.id} couldn't be asked: ${JSON.stringify(asked)}`);
  }
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    void core.stopAnswer({ mindId: mind.id, answerId: asked.answerId });
  }, timeoutMs);
  const ending = await ended;
  clearTimeout(timer);

  const answer = await readAnswer(core, mind.id, asked.answerId);
  await core.closeMind(mind.id);
  const sentences = answer ? answerSentences(answer, question.language) : [];
  const citations = sentences.flatMap((sentence) =>
    sentence.citations.map((citation): CitationRecord => {
      const pages = citation.documentId ? library.pageTexts(citation.documentId) : [];
      const checked = {
        quote: citation.quote ?? "",
        pageFrom: citation.pageFrom ?? null,
        pageTo: citation.pageTo ?? null,
        check: citation.check,
        checkReason: citation.checkReason ?? null,
      };
      return {
        sentence: sentence.text,
        documentName: citation.documentName ?? "",
        ...checked,
        outcome: outcomeOf(checked, pages),
      };
    }),
  );
  const finished = "finished" in ending ? ending.finished : null;
  return {
    questionId: question.id,
    language: question.language,
    crossLingual: question.crossLingual,
    round,
    question: question.question,
    status: timedOut ? "timed-out" : (finished?.status ?? "failed"),
    error: "failed" in ending ? ending.failed : null,
    citationSupport: finished?.citationSupport ?? null,
    searches,
    droppedMarkers: finished?.droppedMarkers ?? 0,
    droppedRecords: finished?.droppedRecords ?? 0,
    sentences: sentences.map((sentence) => ({
      text: sentence.text,
      cited: sentence.citations.length > 0,
    })),
    citations,
    seconds: (Date.now() - started) / 1000,
  };
}

const groupOf = (answer: Pick<AnswerRecord, "crossLingual" | "language">): CitationGroup =>
  answer.crossLingual ? "crossLingual" : answer.language;

const share = (part: number, whole: number) => (whole > 0 ? part / whole : null);

function median(values: readonly number[]): number | null {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length === 0) return null;
  return sorted.length % 2 === 1
    ? (sorted[middle] as number)
    : ((sorted[middle - 1] as number) + (sorted[middle] as number)) / 2;
}

export function summariseGroup(answers: readonly AnswerRecord[]): GroupSummary {
  const citations = answers.flatMap((answer) => answer.citations);
  const outcomes = Object.fromEntries(CITATION_OUTCOMES.map((outcome) => [outcome, 0])) as Record<
    CitationOutcome,
    number
  >;
  for (const citation of citations) outcomes[citation.outcome]++;
  const sentences = answers.flatMap((answer) => answer.sentences);
  const cited = sentences.filter((sentence) => sentence.cited).length;
  const citedAnswers = answers.filter((answer) => answer.citations.length > 0).length;
  const citationSupport: GroupSummary["citationSupport"] = {};
  for (const answer of answers) {
    const key = answer.citationSupport ?? "unknown";
    citationSupport[key] = (citationSupport[key] ?? 0) + 1;
  }
  return {
    answers: answers.length,
    failedAnswers: answers.filter(
      (answer) => answer.status === "failed" || answer.status === "timed-out",
    ).length,
    citedAnswers,
    citedAnswerShare: share(citedAnswers, answers.length),
    medianSeconds: median(answers.map((answer) => answer.seconds)),
    citations: citations.length,
    outcomes,
    foundShare: share(outcomes.found, citations.length),
    falseNotFoundShare: share(outcomes["false-not-found"], citations.length),
    sentences: sentences.length,
    citedSentences: cited,
    coverage: share(cited, sentences.length),
    droppedMarkers: answers.reduce((sum, answer) => sum + answer.droppedMarkers, 0),
    droppedRecords: answers.reduce((sum, answer) => sum + answer.droppedRecords, 0),
    citationSupport,
  };
}

const percent = (value: number | null) => (value === null ? "–" : `${(value * 100).toFixed(1)}%`);

/** Why a gating run misses the targets, per language. */
function citationFailures(
  summary: Record<CitationGroup, GroupSummary>,
  minCitations: number,
): string[] {
  const failures: string[] = [];
  for (const [group, label] of [
    ["en", "English"],
    ["zh", "Chinese"],
  ] as const) {
    const { citations, foundShare, falseNotFoundShare } = summary[group];
    if (citations < minCitations) {
      failures.push(`Citations, ${label}: only ${citations}, needs at least ${minCitations}.`);
    }
    if (foundShare === null || foundShare < CITATION_TARGETS.found) {
      failures.push(
        `Citations, ${label}: ${percent(foundShare)} show "Quote found", target ${percent(CITATION_TARGETS.found)}.`,
      );
    }
    if (falseNotFoundShare === null || falseNotFoundShare > CITATION_TARGETS.falseNotFound) {
      failures.push(
        `Citations, ${label}: ${percent(falseNotFoundShare)} false "not found", target at most ${percent(CITATION_TARGETS.falseNotFound)}.`,
      );
    }
  }
  return failures;
}

/**
 * The Questions to ask: those `ids` names (INCARNAMIND_EVAL_QUESTIONS), in
 * the set's order, or all of them when `ids` is null.
 */
export function questionsToAsk(
  questions: readonly EvalQuestion[],
  ids: readonly string[] | null,
): EvalQuestion[] {
  return ids === null ? [...questions] : questions.filter((question) => ids.includes(question.id));
}

/** Throws when an id names no Question of the sets the run asks from. */
export function checkQuestionIds(
  ids: readonly string[],
  sets: readonly (readonly EvalQuestion[])[],
): void {
  const known = new Set(sets.flatMap((questions) => questions.map((question) => question.id)));
  const unknown = ids.filter((id) => !known.has(id));
  if (unknown.length > 0) {
    throw new Error(
      `INCARNAMIND_EVAL_QUESTIONS names Questions that aren't in the evaluation sets this run asks from: ${unknown.join(", ")}.`,
    );
  }
}

/**
 * Sets up the chat model on the library's core and asks the evaluation's
 * Questions: every Question once, then more rounds of the gating Questions in
 * a language with too few Citations, up to `maxRounds`. With `questionIds`,
 * only those Questions, each once: a short check, which never gates.
 */
export async function runCitations(
  library: Library,
  questions: readonly EvalQuestion[],
  chat: ChatSettings,
  config: Pick<EvalConfig, "minCitations" | "maxRounds" | "answerTimeoutMs"> &
    Partial<Pick<EvalConfig, "questionIds">>,
  log: Log,
): Promise<CitationRun> {
  const subset = config.questionIds ?? null;
  const asked = questionsToAsk(questions, subset);
  // More rounds only gather Citations for the gating targets, which a subset doesn't meet.
  const maxRounds = subset ? 1 : config.maxRounds;
  const { core } = library;
  // Setting the variables is the consent to send Questions and Passages to the chat
  // model. Automatic tagging would send Document excerpts as well, so it is declined.
  const stopConsent = core.on("consent.requested", (request) => {
    core
      .respondToConsent(request.requestId, request.flow.id === "chat")
      .catch((error: unknown) => console.error(error));
  });
  try {
    const provider = await core.saveChatProvider({
      kind: chat.kind,
      modelId: chat.modelId,
      ...(chat.apiKey !== null && { apiKey: chat.apiKey }),
      ...(chat.baseUrl !== null && { baseUrl: chat.baseUrl }),
    });
    const model = `${chat.kind}/${chat.modelId}`;
    const gating = provider.service !== null && subset === null;
    const overrides = [
      ...(chat.numCtx !== null ? [`num_ctx ${chat.numCtx}`] : []),
      ...(chat.citing !== null ? [`citing mode "${chat.citing}"`] : []),
    ];
    const where = provider.service ? `, sent to ${provider.service.name}` : " (local, not gating)";
    log(
      `Asking with ${model}${where}${overrides.length > 0 ? `, ${overrides.join(", ")}` : ""}${subset ? `; asking only ${asked.map((question) => question.id).join(", ")}, once each (not gating)` : ""}`,
    );

    const answers: AnswerRecord[] = [];
    const citationsIn = (group: CitationGroup) =>
      answers
        .filter((answer) => groupOf(answer) === group)
        .reduce((sum, answer) => sum + answer.citations.length, 0);
    let rounds = 0;
    for (let round = 1; round <= maxRounds; round++) {
      const asking =
        round === 1
          ? asked
          : asked.filter(
              (question) =>
                !question.crossLingual && citationsIn(question.language) < config.minCitations,
            );
      if (asking.length === 0) break;
      rounds = round;
      for (const question of asking) {
        const answer = await askOne(library, question, round, config.answerTimeoutMs);
        answers.push(answer);
        const found = answer.citations.filter((citation) => citation.outcome === "found").length;
        log(
          `${question.id} (round ${round}): ${answer.status}, ${answer.citationSupport ?? "unknown"}, ${answer.citations.length} Citations, ${found} found, ${answer.seconds.toFixed(0)} s`,
        );
      }
    }

    const summary: Record<CitationGroup, GroupSummary> = {
      en: summariseGroup(answers.filter((answer) => groupOf(answer) === "en")),
      zh: summariseGroup(answers.filter((answer) => groupOf(answer) === "zh")),
      crossLingual: summariseGroup(answers.filter((answer) => groupOf(answer) === "crossLingual")),
    };
    return {
      model,
      service: provider.service?.name ?? null,
      gating,
      subset: subset ? asked.map((question) => question.id) : null,
      overrides,
      minCitations: config.minCitations,
      rounds,
      answers,
      summary,
      failures: gating ? citationFailures(summary, config.minCitations) : [],
    };
  } finally {
    stopConsent();
  }
}
